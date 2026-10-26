# Optional HyperParallel runtime

Hyper-specific runtime execution, wrapping/prefetch, loading, optimizer execution
and checkpointing live in `veomni/hyper_parallel/`. Native runtime and loading
implementations remain unchanged. See the [integration design](fsdp2_backends.md).

For VLM/DiT/DPO, an explicit launcher selects thin Hyper-local subclasses
of the native trainers. Text and Omni instead branch at their existing training and
distributed-inference construction sites, importing the distinctly named
Hyper builder only inside its selected branch. Native constructors and both
original Omni builder functions remain unchanged. Dedicated runtime
subclasses retain native modality methods and their
`super()` chains, including VLM parameter groups. There is no public tensor
registry or new execution hook on the native runtime.

FSDP and Muon remain independently selectable. Optimizer state and scheduler
ownership remain with the underlying optimizers. Hyper checkpoint managers select
their own persistence; auto-resume retains completion-marker compatibility.

Keep the normal configuration and launch through the optional entry:

```bash
export HYPER_PARALLEL_PLATFORM=torch
torchrun --standalone --nproc-per-node=8 --module veomni.hyper_parallel.train text config.yaml
```

Use `vlm`, `dit`, `dpo`, `omni` or `omni_infer` instead of `text` for the
corresponding native lifecycle. Existing Text and Omni task scripts also recognize
the Hyper flags through lazy call-site branches. VLM/DiT/DPO native task scripts
remain unchanged; use this entry to enable Hyper there. Custom Python callers may use
the Hyper trainer entry described in the integration design.


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

On CUDA, Qwen3-MoE decoder blocks and Qwen3-VL-MoE text/vision blocks use the
validated attention and MLP/MoE boundaries. Attention prefetches the current
MLP and next attention; MLP prefetches the next attention and MLP. Residual
parameters such as norms remain on the enclosing block. Existing EP and FP32
exception units retain their mesh and precision policy and follow their
enclosing compute stage.

NPU, unrecognized block types, compiled models and blocks with parameters
shared across boundaries retain whole-block wrapping. Persistent ExtraParallel
blocks also retain their existing wrapping. NPU and models without eligible
split blocks use the existing prefetch flow; mixed split/unsplit blocks
participate in the same internal stage chain.
The framework does not infer arbitrary execution order from child registration
order or split every small module. The normal Torch backend is unchanged.

The existing `forward_prefetch` switch still controls explicit forward and
backward prefetch targets. Turning it off does not change wrapping boundaries.
Backward links follow reverse compute-stage order; this policy does not change
Hyper's recomputation handling or the reduction waiting interval.

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

## Loading and gradient clipping

With CPU parameter offload, local gradient norms are computed on CPU and only
scalar statistics move to the accelerator for the FSDP reduction. Clipping then
applies the resulting coefficient to the CPU gradients. With ExtraParallel,
shared LoRA parameters replicated across EP ranks are counted once in both
Muon construction paths.

Persistent 2D ExtraParallel parameters retain both shard placements during
loading. Rank0 chunk loading sends each destination its local rectangle, just
as per-rank streaming reads that rectangle, without redistributing it again.
This path requires a world-complete 2D mesh and preserves parameter identity.

## Current limitations

- HyperParallel checkpointing does not yet support trainable-only LoRA checkpoints or `dcp_save_to_lowest_rank`.
- Hugging Face export and checkpoint staging are not yet supported by the HyperParallel backend.
- Model and optimizer checkpoints contain rank-local shards and must be resumed with the same world size, rank mapping, and parallel topology.
- HyperParallel checkpoint saving is currently synchronous.
- HyperParallel must be installed separately from its source repository; it is not part of VeOmni's default dependency set.
- Reentrant gradient checkpointing retains VeOmni's existing process-global Torch CheckpointFunction replacement. The default non-reentrant path does not use that replacement; trainer selection introduces no global method patch.

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
