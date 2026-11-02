# Optional HyperParallel backend

VeOmni can use HyperParallel as an optional FSDP2/HSDP implementation without changing its training entry points or topology configuration. The default remains PyTorch FSDP2.

```yaml
model:
  accelerator:
    dp_replicate_size: 1
    dp_shard_size: 8
    fsdp_config:
      fsdp_mode: fsdp2
      fsdp_backend: hyper

train:
  checkpoint:
    save_hf_weights: false
```

Set `dp_replicate_size` above one to use the same configuration as HSDP. VeOmni continues to own DeviceMesh creation, ExtraParallel/EP, Ulysses, model loading, gradient clipping, trainer lifecycle, and checkpoint selection. The adapter reuses those process groups and maps existing `fully_shard` calls to HyperParallel.

HyperParallel is imported lazily. Leaving `fsdp_backend` unset is equivalent to `fsdp_backend: torch` and does not require or import the package. `fsdp_backend: hyper` is valid only with `fsdp_mode: fsdp2`; unsupported combinations fail during configuration validation.

## Internal wrap and prefetch policy

Selecting the Hyper backend also selects its internal device policy; no wrap
declarations or prefetch-depth settings are required.

On CUDA and NPU, Qwen3-MoE decoder blocks and Qwen3-VL-MoE text/vision blocks use the
validated attention and MLP/MoE boundaries. Attention prefetches the current
MLP and next attention; MLP prefetches the next attention and MLP. Residual
parameters such as norms in split blocks remain on the root unit. Existing EP and FP32
exception units retain their mesh and precision policy and follow their
enclosing compute stage.

Unrecognized block types, compiled models and blocks with parameters shared
across boundaries retain whole-block wrapping. Models with a mixed-precision
`output_dtype` retain their original block and child boundaries, preserving
their input and output casts. Models with frozen parameters retain whole-block
wrapping (and the original CUDA child boundaries when `output_dtype` is set)
and use no manual Hyper prefetch, since frozen units may not participate in
backward. Persistent ExtraParallel blocks
also retain their existing wrapping. Models without eligible split blocks use
the existing prefetch flow; mixed split/unsplit blocks
participate in the same internal stage chain.
The framework does not infer arbitrary execution order from child registration
order or split every small module. The normal Torch backend is unchanged.

The existing `forward_prefetch` switch still controls explicit forward and
backward prefetch targets. Turning it off does not change wrapping boundaries.
Backward links follow reverse compute-stage order; this policy does not change
Hyper's recomputation handling or the reduction waiting interval.

For the patched Qwen3-MoE causal LM, an untied, bias-free standard output
projection without pre-existing module hooks is also a separate FSDP unit.
Its `FusedCrossEntropy` forward invokes the existing causal loss after the
head's FSDP pre-forward hook has made the weight available. The original
`lm_head.weight` parameter name and ordinary logits path are preserved.
The head remains unsharded between its forward and backward so losses that
save the weight for backward can safely read it. Tied, customized or hooked
heads and heads of other models retain the root-managed behavior. When the
embedding and output projection share parameters, both remain with the root.
The embedding is not a backward prefetch target, since `inputs_embeds` may
bypass it.
This does not change the Torch FSDP2 path or enable fused communication.

## HyperParallel Muon

HyperParallel's distributed Muon optimizer can be selected independently:

```yaml
model:
  optimizer:
    type: muon
    use_hyper_optimizer: true
    muon_ns_implementation: std
```

VeOmni retains its existing Muon/AdamW parameter classification and passes the resulting groups to HyperParallel. The Newton-Schulz coefficients, step count, and epsilon are mapped to HyperParallel's custom-coefficient implementation. The default `use_hyper_optimizer: false` keeps VeOmni's existing optimizer unchanged.

HyperParallel requires `muon_adjust_lr_fn: match_rms_adamw`. On CUDA, `muon_ns_implementation: gram` or `gram_quack` uses Hyper's `zeropower_fn` callback to run VeOmni's Gram/QuACK implementation on the owner-batched matrices, retaining Hyper's ownership and communication scheduling. Coefficients, epsilon, iteration count and `muon_gram_ns_reset_iterations` are forwarded. QuACK uses the same capability checks and warning-backed Gram fallback as native VeOmni Muon. Explicit `std` retains Hyper's existing implementation; non-CUDA devices still require `std`. Head-split Muon options remain unsupported. Native VeOmni Muon is unchanged.

For a CUDA Hyper Muon experiment, set:

```yaml
model:
  optimizer:
    type: muon
    use_hyper_optimizer: true
    muon_ns_implementation: gram_quack
    muon_gram_ns_reset_iterations: [2]
```

The optional `gram-newton-schulz` package and its QuACK dependencies must be available to execute the QuACK kernels. Select the same NS settings for both optimizers when comparing performance.

CUDA Gram/QuACK optimizer checkpoints record the callback settings. Resuming with different coefficients, epsilon, iteration count, implementation or reset schedule is rejected before loading the leaf optimizer state. A checkpoint without the callback metadata cannot be resumed into this new callback path; use its original optimizer configuration instead.

## Current limitations

- HyperParallel checkpointing does not yet support trainable-only LoRA checkpoints or `dcp_save_to_lowest_rank`.
- Hugging Face export and checkpoint staging are not yet supported by the HyperParallel backend.
- Model and optimizer checkpoints contain rank-local shards and must be resumed with the same world size, rank mapping, and parallel topology.
- HyperParallel checkpoint saving is currently synchronous.
- HyperParallel must be installed separately from its source repository; it is not part of VeOmni's default dependency set.

## Gradient reduction waiting interval

Set `model.accelerator.fsdp_config.hyper_reduce_comm_interval: 2` to keep
gradient reductions in flight for two FSDP units before waiting. This calls
Hyper's `set_reduce_comm_interval(2)` on the wrapped root before the first
forward. It changes waiting distance, not gradient reduction frequency; the
root drains outstanding reductions at backward completion. The default is
`1`, which does not call the setter and preserves the existing behavior.
Values above `1` require the Hyper backend and a HyperParallel version with
this API. With Hyper communication fusion enabled, Hyper falls back to
interval `1`. A larger interval retains more in-flight gradient buffers.
## FP32 expert weight gradients

For CUDA Qwen3 MoE with `moe_implementation: fused_triton`, `ep_size: 1`,
BF16 compute parameters, and FP32 gradient reduction, set
`model.accelerator.fsdp_config.hyper_expert_grad_dtype_fp32: true`. This is
opt-in and requires both PyTorch `Tensor.grad_dtype` and a HyperParallel revision
supporting `MixedPrecisionPolicy(grad_dtype_fn=...)`, such as the revision in
[HyperParallel PR #1](https://github.com/Cui-yshoho/hyper-parallel/pull/1).
The unsharded `gate_up_proj`
and `down_proj` expert leaves then accept FP32 gradients. The merged Triton
grouped GEMM accumulates BF16 products in FP32 and writes that accumulator
directly into the FP32 gradient buffer, which can be used for ReduceScatter.
Other parameters and the default configuration retain their existing behavior.
The option does not support expert parallelism or LoRA.
