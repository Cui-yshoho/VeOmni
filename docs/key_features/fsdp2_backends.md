# Optional HyperParallel integration

VeOmni owns trainer selection, model/runtime construction, data, training loops,
callbacks and checkpoint lifecycle. Hyper-specific execution capabilities live
in the separately installed
[HyperParallel VeOmni adapter](https://github.com/Cui-yshoho/hyper-parallel/tree/feat/veomni-integration/hyper_parallel/integration/veomni).
Install matching VeOmni and HyperParallel revisions; an older Hyper installation
without this adapter cannot run the Hyper path. HyperParallel is not a mandatory
VeOmni dependency.

## Ownership and lazy construction

- [Configuration predicate](../../veomni/hyper_parallel/__init__.py#L51):
  `_uses_hyper` is configuration-only and imports no optional library.
- [Text construction entry](../../veomni/hyper_parallel/model_runtime.py#L384):
  `text_runtime(args, model_name="base", *, train)` constructs a VeOmni-owned
  runtime that lazily selects Hyper execution methods.
- [Omni construction entry](../../veomni/hyper_parallel/model_runtime.py#L443):
  `build_hyper_omni_model_runtime(args, *, train=None, for_inference=False)`
  composes resolved module runtimes inside VeOmni for training or inference.
- [Task launcher](../../veomni/hyper_parallel/train.py#L32): `build_trainer`
  constructs the same native trainer as each original task script.
  Runtime selection follows the trainer's configuration flags in either entry.

These entries are framework-provided; model implementations do not override them.
They return normal runtime/trainer objects, do not replace model classes or
forwards, and propagate dependency or construction failures without falling back
to Torch silently. The external adapter can import VeOmni explicitly; Hyper's
core and root must not import the adapter automatically.

Text, VLM, DiT, DPO and Omni use explicit branch-local imports at their runtime
construction sites. Their native branches keep the original construction calls.
All original task scripts and the optional launcher honor the same Hyper flags.
Importing the launcher or runtime adapter alone does not import external Hyper.

## Configuration selection

The read-only `ModelArguments.use_hyper_parallel` property selects Hyper when this
model uses Hyper FSDP or Hyper Muon. Omni's property resolves model-scope/global
versus module-scope/child accelerators and each child's independent optimizer
choice. Frozen modules may have no optimizer; empty module mappings return false.
Neither property is a new YAML field or imports Hyper.

| FSDP | Muon | Execution |
| --- | --- | --- |
| Torch | Native | Original runtime and constructors; Hyper is not imported |
| Torch | Hyper | Hyper optimizer; native FSDP, clipping and checkpoint format |
| Hyper | Native | Hyper FSDP and layout-compatible native Muon |
| Hyper | Hyper | Hyper FSDP and Hyper Muon |

The native runtime, loaders, FSDP, optimizer, scheduler and model/Omni builders
remain unchanged. Native task behavior and the independent flags are preserved.

## Training ownership and external execution

VeOmni's optional `hyper_parallel` directory owns:

- `train.py`: optional task launcher, reusing the original trainer classes.
- `model_runtime.py`: runtime lifecycle subclasses, model construction and Omni
  composition, preserving native modality `super()` chains and scoping.
- `checkpoint.py`: model/Omni managers inheriting native save/load lifecycle and
  path handling. Existing `manager: dcp` keeps native persistence by default;
  `manager: hyper` explicitly selects Hyper DCP. Includes checkpoint-only Torch
  DTensor views for Hyper shards;
  native DCP remains responsible for orchestration, planning and file storage.

Hyper's external adapter owns only execution capabilities:

- FSDP/HSDP wrapping and prefetch, preserving EP meshes and FP32 exceptions.
- Materialization and ordinary/rank0/EP weight loading, reusing native readers,
  converters, buffer finalization and no-weight-read initialization paths.
- Optimizer/Muon construction, Gram/QuACK callbacks, scheduler adaptation and
  layout adaptation, reusing compatible native factories and NS/update kernels.
- Gradient clipping and low-level Hyper DCP protocol, reusing native group/norm
  helpers. It does not own trainers, runtime factories or checkpoint managers.

VLM retains visual/language parameter grouping; DiT retains condition-model setup;
DPO retains separate policy/reference roles and frozen reference precision. Omni
retains module names, resolved arguments, frozen handling and model-scope deferred
wrapping. Native Torch children are still built by native factories.

A VLM invocation follows:

- The task script or optional launcher constructs the native VLM trainer.
  - Native initialization selects Hyper at its runtime-construction method.
    - VeOmni runtime composes native VLM lifecycle with lazy Hyper execution.
    - Native configuration/model building and Hyper wrapping/loading run in the
      same named ParallelState scopes as before migration.
  - Native data/scheduler/callback setup continues.
- Native training loop runs unchanged.
- VeOmni's checkpoint manager uses native DCP unless Hyper DCP is explicitly selected.

This relocation adds no global patch or new backend framework. The existing
reentrant-checkpoint branch still follows native VeOmni's process-global
CheckpointFunction replacement; the default non-reentrant path does not.
Checkpoint, wrapping, recomputation, communication and optimizer algorithms are
not changed by relocation.

## Persistence compatibility

Native `veomni/checkpoint/dcp_checkpointer.py` and `layout.py` remain unchanged.
The optional checkpointer adapts payloads at the existing `execute_save`
boundary and reuses native readers and planners for loading. Explicit
`manager: hyper` auto-resume is resolved during checkpoint argument construction
by a lightweight helper in `veomni.hyper_parallel`, without importing external
Hyper. It requires a trainer manifest and each module's `.hyper_complete`
marker, written only after all its components are saved. Incomplete steps are
skipped; restore still validates the saved parallel topology. Native discovery
does not recognize Hyper markers or change its `.metadata` requirements.

## Usage and validation

See [Hyper FSDP2 and Muon configuration](hyper_fsdp2.md) for flags and limitations.
For example:

```bash
export HYPER_PARALLEL_PLATFORM=torch
torchrun --standalone --nproc-per-node=8 tasks/train_text.py config.yaml
torchrun --standalone --nproc-per-node=8 tasks/train_vlm.py config.yaml
```

DiT, DPO and Omni likewise use their original task scripts. The optional module
launcher supports `text`, `vlm`, `dit`, `dpo`, `omni` and `omni_infer` with the same
selection behavior. Multi-node runs use normal torchrun rendezvous/node options
instead of `--standalone`.

Migration validation compares import-normalized implementation ASTs and reruns
native/runtime/config, stage scheduling, loading, optimizer and checkpoint
checks, plus fresh-process optional-import isolation. External diagnostics are
not additional CI coverage. Prior hardware measurements do not validate a newly
relocated implementation; real GPU/NPU training and QuACK kernels need separate
validation.
