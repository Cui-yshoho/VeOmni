# Optional HyperParallel runtime

Training/runtime assembly and checkpoint managers live in `veomni/hyper_parallel/`,
reusing native lifecycles. Wrapping/prefetch, layout-aware loading, optimizer
execution and low-level DCP persistence live in HyperParallel's optional VeOmni
adapter. Native runtime and loading implementations remain unchanged.
See the [integration design](fsdp2_backends.md).

Text, VLM, DiT, DPO and Omni use their original task scripts. Each trainer
selects Hyper at its runtime-construction site through the same configuration
flags, importing the Hyper builder only inside its selected branch. Native
constructors and both original Omni builder functions remain unchanged. Dedicated runtime
subclasses retain native modality methods and their
`super()` chains, including VLM parameter groups. There is no public tensor
registry or new execution hook on the native runtime.

FSDP and Muon remain independently selectable. Optimizer state and scheduler
ownership remain with the underlying optimizers. Checkpoint persistence is
selected independently: `train.checkpoint.manager: dcp` (the default) uses native
VeOmni/Torch DCP, including with Hyper FSDP. Only explicit `manager: hyper` uses
Hyper DCP. Auto-resume retains completion-marker compatibility.

Keep the normal configuration and original task entry:

```bash
export HYPER_PARALLEL_PLATFORM=torch
torchrun --standalone --nproc-per-node=8 tasks/train_text.py config.yaml
```

Use `tasks/train_vlm.py`, `tasks/train_dit.py`, `tasks/train_text_dpo.py`,
`tasks/omni/train_omni.py` or `tasks/omni/infer_omni.py` for the corresponding task.
All entries recognize `fsdp_backend: hyper` and `use_hyper_optimizer: true`
independently. The optional `--module veomni.hyper_parallel.train <task>` launcher
remains available and constructs the same trainer as the task scripts.


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
    manager: dcp
    save_hf_weights: false
```

Set `dp_replicate_size` above one to use the same configuration as HSDP. VeOmni continues to own configuration, DeviceMesh creation, model definitions, Ulysses and the native trainer lifecycle. Hyper's VeOmni adapter owns Hyper wrapping/loading, optimizer, clipping and checkpoint execution, reusing native helpers and existing process groups.

HyperParallel is imported lazily. Leaving `fsdp_backend` unset is equivalent to `fsdp_backend: torch` and does not require or import the package. `fsdp_backend: hyper` is valid only with `fsdp_mode: fsdp2`; unsupported combinations fail during configuration validation.

## Internal wrap and prefetch policy

Selecting Hyper FSDP or Hyper Muon lazily loads `hyper_parallel.integration.veomni` from
the optional HyperParallel installation. Install a HyperParallel revision that
provides this adapter; older installations fail with an explicit update error.
The external adapter owns loading, optimizer execution, low-level persistence,
model boundary declarations and scheduling. Runtime subclasses and checkpoint
managers belong to VeOmni. VeOmni's public
`ParallelPlan` and model definitions are unchanged. No additional YAML settings
are required.

On CUDA and NPU, Qwen3-MoE decoder blocks and Qwen3-VL-MoE text/vision blocks
declare attention and MLP/MoE boundaries. Attention prefetches the current
MLP and next attention; MLP prefetches the next attention and MLP. Split
decoder blocks are not wrapped again; residual norms belong to root. EP and FP32
exception units retain their mesh and precision policy and follow their
enclosing compute stage.

Undeclared block types, compiled models and blocks with parameters
shared across boundaries retain whole-block wrapping. Persistent ExtraParallel
blocks also retain their existing wrapping. Models without eligible
split blocks use the existing prefetch flow; mixed split/unsplit blocks
participate in the same internal stage chain.
The framework does not infer arbitrary execution order from child registration
order or split every small module. The normal Torch backend is unchanged.

The existing `forward_prefetch` switch still controls explicit forward and
backward prefetch targets. Turning it off does not change wrapping boundaries.
Backward links follow a whole-block replay-aware order: both attention and
MLP prefetch the preceding block's ordered stages. Independent input embeddings
are wrapped separately when the model supports splitting. Output projections
remain on the root FSDP unit, which retains gathered weights until backward.
This supports fused losses that read the output weight directly without calling
the projection's forward, without changing model classes or loss dispatch.
With explicit output dtype, the enclosing CUDA block retains the cast boundary;
NPU uses whole-block wrapping. Frozen models disable explicit prefetch, and
without an output-cast boundary also retain whole-block wrapping. No checkpoint
algorithm or reduction interval is changed. This does not imply that prefetch
is fully hidden on every device; hardware profiling remains necessary.

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

### Optional Hyper Muon scheduling and NS backend

The following fields belong to `model.optimizer` and require
`type: muon` and `use_hyper_optimizer: true`:

| Field | Default | Effect |
| --- | --- | --- |
| `hyper_muon_fsdp_overlap` | `false` | Pipeline supported FSDP gather/NS/update buckets; unsupported layouts use the serial path. |
| `hyper_muon_fsdp_overlap_memory_budget_bytes` | `null` | Positive byte budget for the overlap pipeline, including its live buffers. |
| `hyper_muon_momentum_mode` | `legacy` | `vanilla` selects EMA-style momentum; it changes the update formula, not only scheduling. |
| `hyper_muon_hsdp_replicate_dedup` | `true` | Set false to compute on each replica instead of owner-compute plus replica broadcast. |
| `hyper_muon_ns_backend` | `veomni` | Keep the existing VeOmni callback selection; `native` selects Hyper NS and `gram` selects Hyper's optional Gram adapter. |

`hyper_muon_ns_backend: gram` requires the optional `gram-newton-schulz`
package, CUDA SM90+ and CUDA 12.9+. It is distinct from the existing
`muon_ns_implementation: gram_quack` callback route. Select one backend, not
both callbacks. Native/Gram backend selection bypasses VeOmni's implementation
selector; Hyper still receives the configured coefficients, epsilon and steps.
All defaults preserve the previous optimizer behavior. Overlap is opt-in and
needs device-specific profiling before any throughput claim.
The adapter records the momentum mode in optimizer checkpoints and rejects
restoring into a different formula; older unmarked checkpoints imply `legacy`.
Keep replica-deduplication settings and rank topology unchanged when resuming.

### Optional FP32 expert weight gradients

`model.accelerator.fsdp_config.hyper_expert_grad_dtype_fp32: true` selects
FP32 autograd gradients for Qwen3-MoE expert `gate_up_proj` and `down_proj`
parameters while retaining BF16 parameter compute. It requires CUDA, EP=1,
`moe_implementation: fused_triton`, no LoRA, BF16 mixed-precision parameters,
FP32 gradient reduction, Hyper's `grad_dtype_fn` policy and a Torch version
supporting `Tensor.grad_dtype`. It is off by default and does not change the
Torch-native path or force every model parameter's gradient to FP32.
Composed Omni model-scope wrapping rejects this model-specific option.

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

- Explicit `manager: hyper` currently requires Hyper FSDP and does not support
  trainable-only LoRA checkpoints, `dcp_save_to_lowest_rank`, Hugging Face export
  or staging. It saves synchronously and requires the same world size, rank
  mapping and parallel topology when resuming. These are Hyper DCP limitations,
  not a reason to select it automatically for Hyper FSDP.
- Default `manager: dcp` retains native DCP orchestration, planners and storage.
  Hyper shards are exposed as checkpoint-only Torch DTensor views, preserving
  parameter identity and global shard metadata. Training tensors are not replaced.
  Native and Hyper DCP formats are selected explicitly, not auto-converted.
  The optional bridge wraps payloads at the existing `execute_save` boundary,
  inheriting native save, async and staging orchestration. Its load method reuses
  native state wrappers, readers, strict planners and extra-state restoration for
  both split and legacy directories. Native checkpoint implementation files are
  unchanged; no global state-class patch is installed.
- HyperParallel must be installed separately from its source repository; it is not part of VeOmni's default dependency set.
- Reentrant gradient checkpointing retains VeOmni's existing process-global Torch CheckpointFunction replacement. The default non-reentrant path does not use that replacement; trainer selection introduces no global method patch.

To opt into Hyper DCP instead of native DCP:

```yaml
train:
  checkpoint:
    manager: hyper
    save_hf_weights: false
```

Restore a checkpoint with the manager used to save it. Model weights written by
the native DCP bridge use Torch DTensor metadata; optimizer compatibility across
different optimizer implementations is a separate concern, not implied by the
choice of storage backend.

With `manager: hyper`, `load_path: auto` resolves during checkpoint argument
construction, before the native training-argument path resolver runs. Discovery
requires the trainer manifest and a valid `.hyper_complete` marker for every
model module, and skips incomplete newer steps. The optional integration owns
this discovery; native DCP continues to require genuine `.metadata` files.

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
