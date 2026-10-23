# Optional HyperParallel backend

The integration uses the existing model runtime and optimizer factory, not a
separate trainer. Internally, `distributed/fsdp2/backend.py` resolves FSDP API
types, mesh conversion, root reshard policy and weight-loading arguments;
the shared FSDP builder retains model traversal, EP/SP and initialization.
`distributed/hyper_parallelize.py` owns Hyper API resolution, device-specific
wrapping/prefetch, EP gradient scaling, reduction intervals and clipping.
`optim/hyper_optimizer.py` owns the local-shard execution adapter and Hyper
DTensor layout conversion used by native Muon; the native Muon update algorithm
is shared rather than copied.
`optim/hyper_muon.py` owns Hyper Muon construction and its optional NS callback,
while `optim/optimizer.py` retains common parameter grouping and optimizer
finalization. FSDP and Muon backend selection remain independent. These are
internal helpers, not a third-party plugin registration interface.

Both selections resolve directly to the same backend descriptor. Hyper imports
are cached only after selection; there is no intermediate API dictionary.

Optimizer finalization returns a local-shard execution adapter with an explicit
`step()` method. It does not replace the underlying optimizer's or its children's
methods. Schedulers bind to the underlying optimizers; checkpoint state and hooks
remain owned by those optimizers. Hyper checkpoint selection uses the registered
DCP factory, retaining the default Torch factory call unchanged.

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
