# Optional HyperParallel backend

VeOmni can use HyperParallel as an optional FSDP2/HSDP implementation without changing its training entry points or topology configuration. The default remains PyTorch FSDP2.

```yaml
train:
  accelerator:
    dp_replicate_size: 1
    dp_shard_size: 8
    fsdp_config:
      fsdp_mode: fsdp2
      fsdp_backend: hyper
```

Set `dp_replicate_size` above one to use the same configuration as HSDP. VeOmni continues to own DeviceMesh creation, ExtraParallel/EP, Ulysses, model loading, gradient clipping, trainer lifecycle, and checkpoint selection. The adapter reuses those process groups and maps existing `fully_shard` calls to HyperParallel.

HyperParallel is imported lazily. Leaving `fsdp_backend` unset is equivalent to `fsdp_backend: torch` and does not require or import the package. `fsdp_backend: hyper` is valid only with `fsdp_mode: fsdp2`; unsupported combinations fail during configuration validation.

## HyperParallel Muon

HyperParallel's distributed Muon optimizer can be selected independently:

```yaml
train:
  optimizer:
    type: muon
    use_hyper_optimizer: true
    muon_ns_implementation: std
```

VeOmni retains its existing Muon/AdamW parameter classification and passes the resulting groups to HyperParallel. The Newton-Schulz coefficients, step count, and epsilon are mapped to HyperParallel's custom-coefficient implementation. The default `use_hyper_optimizer: false` keeps VeOmni's existing optimizer unchanged.

HyperParallel currently requires `muon_ns_implementation: std` and `muon_adjust_lr_fn: match_rms_adamw`. Gram/Quack reset schedules and head-split Muon options are rejected with explicit errors instead of being silently ignored.

## Current limitations

- HyperParallel checkpointing does not yet support trainable-only LoRA checkpoints or `dcp_save_to_lowest_rank`.
- Optimizer checkpoints contain rank-local shards and must be resumed with the same parallel topology.
- HyperParallel async checkpoint persistence is saved synchronously on NPU.
- HyperParallel must be installed separately from its source repository; it is not part of VeOmni's default dependency set.
