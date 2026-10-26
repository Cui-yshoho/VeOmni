# HyperParallel independent integration

Hyper execution lives in `veomni/hyper_parallel/`. The native model runtime,
weight loader, checkpoint manager, checkpointer factory, FSDP implementation,
Muon, optimizer and scheduler stay unchanged from upstream.

## Construction boundary

For VLM/DiT/DPO, the [explicit launcher](../../veomni/hyper_parallel/train.py#L34)
selects [thin Hyper-local trainer subclasses](../../veomni/hyper_parallel/trainer.py#L20).
They inherit native initializers, data handling, callbacks and training loops,
overriding only construction. Imports are local to the selected modality.
Public trainer constructors retain native signatures, without factory parameters.

Text and Omni instead branch at their existing construction call
sites. The Hyper branch locally imports and directly calls
`text_runtime` or `build_hyper_omni_model_runtime`; the native branch keeps its
original runtime construction call. Neither is hidden behind a common alias.
Native-only and eager-inference jobs do not import Hyper integration.

Text reads the configuration-only
[ModelArguments.use_hyper_parallel](../../veomni/arguments/arguments_types.py#L1806)
property after job setup and before data construction. It returns a boolean
from this model's FSDP backend and independent Hyper Muon flag; a missing
optimizer selects only by FSDP. The native branch retains the original
BaseTrainer construction call. Omni overrides the same property as
[OmniModelRuntimeArguments.use_hyper_parallel](../../veomni/models/seed_omni/accelerated/omni_model/omni_model_config.py#L76)
and reads it at both call sites. It returns a boolean from the resolved modules:
model-scope FSDP uses the global accelerator, module-scope FSDP uses each
module's accelerator, and Hyper Muon is independently selected per module.
Frozen modules may have no optimizer; an empty module mapping returns false.
This framework-provided property is read-only, not a YAML/CLI field, and imports
neither Hyper integration nor the external library. Callers read it after
model/module arguments have been resolved; both properties are recomputed rather than cached.

| Caller | Hyper entry | Result |
| --- | --- | --- |
| Text / DPO policy | `text_runtime` | HyperModelRuntime |
| VLM | `vlm_runtime` | Native VLM lifecycle with Hyper execution |
| DiT | `dit_runtime` | Native DiT lifecycle with Hyper execution |
| DPO reference | `reference_runtime` | Frozen reference, no optimizer |
| Omni module | `build_hyper_omni_module_runtime` | HyperModuleRuntime with its named mesh |
| Omni composed model | `build_hyper_omni_model_runtime` | HyperOmniRuntime only for Hyper model-scope FSDP |

The [runtime entries](../../veomni/hyper_parallel/model_runtime.py#L353)
validate optional dependencies and persistence limits before constructing a
Hyper model. Construction adds no backend framework, public tensor registry,
global method replacement or execution hook on VeOmniModelRuntime.

### Existing construction methods and minimal extensions

BaseTrainer, VLM, DiT and DPO retain their upstream files. Their private trainer
subclasses override existing construction methods, not initializers or loops.
DPO retains separate policy/reference roles and the reference precision.

| Location | Ownership and invocation | Contract |
| --- | --- | --- |
| [TextTrainer.__init__](../../veomni/trainer/text_trainer.py#L38) | Existing initializer selects construction after distributed setup, before data construction | Hyper import occurs only when selected. The native branch calls BaseTrainer._build_model_runtime directly; no delegate method or extra TextTrainer subclass is added |
| [OmniTrainer._build_model_runtime](../../veomni/trainer/omni/omni_trainer.py#L265) | Existing method resolves module arguments and explicitly selects the builder | Hyper import occurs only for an effective Hyper FSDP or module optimizer choice. The original empty-optimizer guard remains |
| [OmniInferencer._build_model](../../veomni/trainer/omni/omni_inferencer.py#L119) | Existing method selects the builder only for distributed inference | Eager loading is unchanged. No extra _build_model_runtime method is added |

Runtime construction returns a fully built runtime, never a bare model or
uninitialized class. It preserves the native interface, model name and
ParallelState ownership and propagates failures without silent fallback.

The original Omni model/module builders retain their names, signatures and
bodies. The Hyper model builder keeps a small local assembly sequence instead
of modifying them: load config sidecars, construct children, validate LoRA
matches, assemble OmniModel, wrap, then build the training optimizer. It reuses
native configuration, the original builder for Torch children, the native LoRA
validator and existing runtime lifecycles. This repeats bounded composition
wiring, not the training loop.

Model-scope FSDP uses the global accelerator; module-scope uses each child's
accelerator. Hyper Muon remains independently selectable. Only Hyper
model-scope FSDP selects HyperOmniRuntime; optimizer-only or module-scope Hyper
keeps the native composed runtime. Child selection is per module, including
frozen children with no optimizer.

A Python caller can use the same entry as the command-line launcher:

```python
from veomni.hyper_parallel.train import build_trainer

trainer = build_trainer("vlm", args)
trainer.train()
```

The VLM flow is:

- The explicit entry selects the private VLMTrainer subclass.
  - The inherited native initializer sets up the job and calls its existing construction method.
    - The override calls `vlm_runtime` with the original arguments.
    - That entry selects native or Hyper execution from this model's flags.
  - Native data, scheduler and callback initialization continues.
- The inherited native training loop runs unchanged.

Existing runtime/checkpoint tests cover native lifecycle. External checks
also cover thin-class inheritance, explicit Omni routing, mixed module/model
scope, independent FSDP/Muon choices, frozen modules, eager/distributed
inference, CLI dispatch and import isolation. These are local checks, not
additional CI coverage.

## Reusing modality lifecycle

[HyperModelRuntime](../../veomni/hyper_parallel/model_runtime.py#L33) overrides existing
parallelization, optimizer, scheduler, clipping and checkpoint construction
methods. It does not require a new method on the native base.

[Explicit modality subclasses](../../veomni/hyper_parallel/model_runtime.py#L197)
place the native modality class before HyperModelRuntime in their inheritance
order. VLM's original optimizer method groups visual/language parameters, then
its existing `super()` reaches Hyper construction. ModuleRuntime retains frozen
module handling, named ParallelState scopes and deferred model-scope wrapping.
DiT retains its offline-embedding guards and condition-model state. DPO keeps
the native frozen reference constructor. No class methods are replaced at run time.

A VLM build flows through:

- Original VLM construction, model creation and freeze/LoRA.
  - Hyper parallelization and private weight loading.
  - Original VLM parameter grouping.
    - Hyper optimizer construction with those groups.
  - Hyper-owned checkpoint manager when Hyper owns FSDP.

Unknown or custom model behavior is not declared validated by this composition.
The same initialization-operator restrictions of the installed Hyper version apply.
The unchanged reentrant-checkpoint branch in `parallelize.py` still follows
VeOmni's native implementation by replacing the process-global Torch
CheckpointFunction. This is an existing execution-side limitation, not part of
the thin-trainer selection mechanism. The default non-reentrant path does not
perform that replacement; this refactor does not change checkpoint algorithms.

## Hyper-owned operations

- `parallelize.py`: FSDP/HSDP, lazy Hyper API resolution, FP32 exceptions and existing
  device-specific wrapping/prefetch. This structural change does not tune
  recomputation, wrapping boundaries or communication algorithms.
- `module_utils.py`: materialization, every-rank loading,
  rank0 broadcasting, EP chunking and opt-in EP streaming. The orchestration is
  maintained locally to keep public loaders unchanged; file readers,
  conversion helpers and buffer finalization are reused.
- The same `module_utils.py` owns private Torch/Hyper tensor interoperability and type dispatch.
  No registration is installed into a shared VeOmni registry.
- `optimizer.py`, `muon.py`: local-shard execution,
  Gram/QuACK integration, scheduler binding and optimizer state ownership.
  Construction delegates directly to VeOmni for ordinary AdamW/AnyPrecision,
  native Muon on Torch layouts and Torch-layout EP groups. Hyper-specific
  construction remains only for Hyper Muon or layout-sensitive Hyper EP/Muon
  handling. Muon head-block grouping is the original VeOmni helper.
  The native Muon subclass for Hyper layouts inherits native parameter updates
  and delegates NS/head-block computation to the native implementation; it
  retains only layout adaptation and step scheduling. This does not change
  HyperParallel Muon's own update or communication implementation.
- `clip_grad_norm.py`, `checkpoint.py`: clipping and Hyper persistence.
  Private checkpoint managers reuse native save/load operations and Omni paths
  but select the Hyper checkpointer in their own constructors.
  HSDP unique-shard group selection, norm finalization and non-finite checks
  reuse native helpers; local Hyper gradient extraction stays in the adapter.

Private loading must preserve parameter identity, checkpoint converters, dtype,
logical EP shape and adapter key mapping. A locally read shard must not be
distributed a second time. Schedulers retain the actual optimizer owners;
adapters must not replace optimizer methods or state formats.
Random initialization and full-checkpoint resume materialization delegate to
the native materializer's no-weight-read branch; actual Hyper weight loading
and composed-model child selection remain private.

## Auto-resume compatibility

One deliberate shared compatibility point remains in
[checkpoint layout](../../veomni/checkpoint/layout.py#L237): it recognizes the
Hyper completion marker in addition to native DCP markers. Discovery runs during
argument resolution, before runtimes exist; moving selection there would require
additional changes to argument resolution and checkpoint discovery.

The marker is written only after all model components have been saved. It does
not replace the trainer manifest or fabricate Torch DCP metadata. Incomplete
steps remain ineligible for auto-resume. Existing Hyper topology validation is
performed by the Hyper checkpointer at restore.

## Independent FSDP and Muon selection

| FSDP | Muon | Execution |
| --- | --- | --- |
| Torch | Native | Original runtime and constructors |
| Torch | Hyper | Hyper optimizer; native FSDP, clipping and checkpoint format |
| Hyper | Native | Hyper FSDP and layout-compatible native Muon |
| Hyper | Hyper | Hyper FSDP and Hyper Muon |

`fsdp_backend` defaults to `torch`; `use_hyper_optimizer` defaults to `false`.
Configure, for example:

```yaml
model:
  accelerator:
    fsdp_config:
      fsdp_mode: fsdp2
      fsdp_backend: hyper
  optimizer:
    type: muon
    use_hyper_optimizer: true
train:
  checkpoint:
    save_hf_weights: false
```

Run Hyper training through its explicit entry (use the same config path and
overrides as the corresponding native task):

```bash
export HYPER_PARALLEL_PLATFORM=torch
torchrun --standalone --nproc-per-node=8 --module veomni.hyper_parallel.train text config.yaml
torchrun --standalone --nproc-per-node=8 --module veomni.hyper_parallel.train vlm config.yaml
```

Other modes are `dit`, `dpo`, `omni` and `omni_infer`. Omni uses its original
trainer/inferencer with explicit lazy routing, so existing
`tasks/omni/train_omni.py` and `tasks/omni/infer_omni.py` also recognize Hyper
flags. For multiple nodes use
the normal torchrun rendezvous/node arguments instead of `--standalone`; the
module command and config do not change. Do not pass this module as a script
path to `train.sh`. Text's original task also recognizes Hyper flags through
the explicit construction branch. VLM/DiT/DPO `tasks/train_*.py` commands
remain native: use this entry for Hyper execution on those modalities.
Importing the launcher itself does not load
the optional external package; constructor validation loads it only when needed.

Validation uses existing runtime/checkpoint tests plus CPU/mock/Gloo diagnostics
for selection, modality ordering, private loading and optimizer DCP round trips.
Full-model 910C 2K/32K training and performance comparisons passed before this
construction-only refactor. Those measurements do not validate the new launcher;
its routing checks and the native regression results are recorded separately.
GPU kernels and distributed Hyper checkpoint hardware remain unverified.
