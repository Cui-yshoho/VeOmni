# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model construction entries and Hyper execution under native modality lifecycles."""

from dataclasses import fields
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from ..distributed.parallel_state import use_parallel_state
from ..models.model_runtime import VeOmniModelRuntime
from ..models.seed_omni.accelerated.omni_model.omni_model_runtime import OmniModelRuntime, _scoped_no_split_modules
from ..models.seed_omni.accelerated.omni_module.omni_module_runtime import ModuleRuntime
from ..trainer.vlm_trainer import VLMModelRuntime
from ..utils import logging
from . import _uses_hyper


logger = logging.get_logger(__name__)


class HyperModelRuntime(VeOmniModelRuntime):
    """Execution lifecycle for either Hyper FSDP or Hyper Muon.

    Modality runtimes inherit their native type before this class so their
    existing super() calls retain grouping, scoping and model-specific setup.
    """

    @property
    def _hyper_fsdp_enabled(self):
        accelerator = getattr(self, "mesh_accelerator", self.args.accelerator)
        return accelerator.fsdp_config.fsdp_backend == "hyper"

    def _build_parallelized_model(self) -> None:
        """FSDP2/DDP-wrap the model and load its weights.

        The wrap preserves ``requires_grad`` (the shard inherits it) and the
        loader writes weights in place, so a freeze applied in
        :meth:`_freeze_model_module` survives and is not re-asserted here.
        """
        if not self._hyper_fsdp_enabled:
            return super()._build_parallelized_model()
        args = self.args
        if args.accelerator.fsdp_config.fsdp_mode == "eager":
            # ``build_parallelize_model`` has no unwrapped branch, so an eager
            # mode reaching it would be handed to DDP instead.
            raise ValueError(
                "model.accelerator.fsdp_config.fsdp_mode='eager' is only supported by SeedOmni "
                "module inference (ModuleRuntime._init_eager_inference)."
            )

        self._apply_async_activation_offload()

        # Customized parallelize model.
        customized_parallelize_model_function = getattr(self.model, "build_parallelize_model", None)
        if callable(customized_parallelize_model_function):
            parallelized_model = customized_parallelize_model_function(
                weights_path=self.args.model_path, args=self.args
            )
            if parallelized_model is not None:
                self.model = parallelized_model
                self.model.train()
                logger.info_rank0("Built customized parallelized model.")
                return

        # Default parallelize model.
        kwargs: Dict[str, Any] = {}
        cpu_load_param_name = None
        if hasattr(self.model, "get_parallel_plan"):
            cpu_load_param_name = getattr(self.model.get_parallel_plan(), "cpu_load_param_name", None)
        kwargs["cpu_load_param_name"] = cpu_load_param_name
        if bool(args.lora_config):
            kwargs["adapter_path"] = args.lora_config.get("lora_adapter", None)
            kwargs["is_peft_model"] = True

        muon_expert_zero_comm = args.optimizer.type == "muon" and args.optimizer.muon_expert_zero_comm

        if args.fqn_to_index_mapping is not None:
            kwargs["fqn_to_index_mapping"] = args.fqn_to_index_mapping

        skip_hf_weight_load = self.skip_hf_weight_load
        if skip_hf_weight_load:
            logger.info_rank0(
                f"Checkpoint resume enabled (load_path={self.train_args.checkpoint.load_path}); "
                "skipping HF weight materialization before checkpoint restore."
            )

        from ..distributed.torch_compile import CompileConfig
        from . import parallelize as torch_parallelize

        compile_config = CompileConfig(
            **{field.name: getattr(args.accelerator.torch_compile, field.name) for field in fields(CompileConfig)}
        )
        self.model = torch_parallelize.build_parallelize_model(
            self.model,
            init_device=args.accelerator.init_device,
            weights_path=args.model_path,
            should_skip_hf_weight_load=skip_hf_weight_load,
            enable_reshard_after_forward=args.accelerator.fsdp_config.reshard_after_forward,
            mixed_precision=args.accelerator.fsdp_config.mixed_precision,
            enable_gradient_checkpointing=args.accelerator.gradient_checkpointing.enable,
            basic_modules=list(set(getattr(self.model, "_no_split_modules", None) or []) | set(args.basic_modules)),
            enable_reentrant=args.accelerator.gradient_checkpointing.enable_reentrant,
            early_stop=args.accelerator.gradient_checkpointing.early_stop,
            enable_forward_prefetch=args.accelerator.fsdp_config.forward_prefetch,
            enable_fsdp_offload=args.accelerator.fsdp_config.offload,
            fsdp_offload_pin_memory=args.accelerator.fsdp_config.offload_pin_memory,
            broadcast_model_weights_from_rank0=args.broadcast_model_weights_from_rank0,
            ep_sharded_stream_load=args.ep_sharded_stream_load,
            max_load_broadcast_size=args.accelerator.fsdp_config.max_load_broadcast_size,
            muon_expert_zero_comm=muon_expert_zero_comm,
            low_precision_reduce_scatter_comm=args.accelerator.fsdp_config.low_precision_reduce_scatter_comm,
            compile_config=compile_config,
            hyper_reduce_comm_interval=args.accelerator.fsdp_config.hyper_reduce_comm_interval,
            **kwargs,
        )
        self.model.train()

    def _build_optimizer(self, param_groups: Optional[List[Dict[str, Any]]] = None) -> None:
        """Build the optimizer over this model's still-trainable params.

        A distributed optimizer (Muon) reads ``get_parallel_state()`` at build
        time, so it must resolve to *this* model's mesh.

        Args:
            param_groups: Split the parameters across groups with their own
                hyperparameters — a VLM giving its vision tower a separate lr,
                say. Which parameters belong together is knowledge the model's
                trainer has, so it is passed in rather than derived here.
        """
        opt = self.args.optimizer
        from .optimizer import build_optimizer

        self.optimizer = build_optimizer(
            self.model,
            lr=opt.lr,
            betas=opt.betas,
            weight_decay=opt.weight_decay,
            fused=True,
            optimizer_type=opt.type,
            param_groups=param_groups,
            no_decay_modules=opt.no_decay_modules,
            no_decay_params=opt.no_decay_params,
            optimizer_config=opt,
        )

    def _build_lr_scheduler(self, total_steps: int) -> None:
        """Build the lr-scheduler over ``total_steps``.

        Takes the step count rather than reading it off a training config: it
        is only known once the dataset has been sized, which is job-bound.
        """
        opt = self.args.optimizer
        from .optimizer import build_lr_scheduler

        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            train_steps=total_steps,
            lr=opt.lr,
            lr_min=opt.lr_min,
            lr_decay_style=opt.lr_decay_style,
            lr_decay_ratio=opt.lr_decay_ratio,
            lr_warmup_ratio=opt.lr_warmup_ratio,
            lr_start=opt.lr_start,
        )

    def clip_grad_norm(self, max_norm: Optional[float] = None):
        """Clip this model's gradients under its own parallelism and return the norm."""
        if not self._hyper_fsdp_enabled:
            return super().clip_grad_norm(max_norm)
        from .clip_grad_norm import clip_grad_norm

        if max_norm is None:
            max_norm = self.args.optimizer.max_grad_norm
        with use_parallel_state(self.model_name):
            return clip_grad_norm(self.model, max_norm).item()

    def build_checkpoint(self):
        if not self._hyper_fsdp_enabled:
            return super().build_checkpoint()
        from .checkpoint import HyperModelCheckpointManager

        self.checkpoint = HyperModelCheckpointManager(self)


class HyperVLMRuntime(VLMModelRuntime, HyperModelRuntime):
    """Keep VLM grouping and freezing before Hyper execution."""


class HyperModuleRuntime(ModuleRuntime, HyperModelRuntime):
    """Keep module scope, frozen-module handling and composed-wrap ordering."""

    def clip_grad_norm(self, max_norm=None, norm_type=2.0):
        if not self._hyper_fsdp_enabled:
            return super().clip_grad_norm(max_norm, norm_type)
        from .clip_grad_norm import clip_omni_module_grad_norm

        if max_norm is None:
            max_norm = self.args.optimizer.max_grad_norm
        with self._scoped():
            return clip_omni_module_grad_norm(self.model, max_norm, norm_type)

    def build_checkpoint(self):
        if not self._hyper_fsdp_enabled:
            return super().build_checkpoint()
        from .checkpoint import HyperOmniCheckpointManager

        if not self.has_trainable_parameters:
            self.checkpoint = None
            return
        self.checkpoint = HyperOmniCheckpointManager(self)

    def _model_reshard(self, reshard: bool) -> None:
        """Set ``set_reshard_after_backward`` on this module's FSDP2 units.

        A gradient-accumulation optimization owned per-module: the orchestrator
        decides *when* (``reshard=False`` on the first micro-step to keep params
        gathered across the window and skip the reshard → re-all-gather churn;
        ``reshard=True`` on the last so the final backward frees the full
        params). This method only *applies* that intent to this module — trading
        param memory for communication.

        Read the wrap-time ``fsdp_config`` (not the orchestrator's): under
        ``fsdp_scope='model'`` that is the top-level accelerator so a module
        YAML ``ddp`` overlay does not skip reshard on an FSDP-wrapped child;
        otherwise it is the module's own merged ``accelerator``. A DDP module
        has no FSDP2 units to toggle (skipped by the ``isinstance`` check), and
        a module that keeps ``reshard_after_backward=True`` opts out here. No
        ``ParallelState`` is read (unlike :meth:`clip_grad_norm`), so this needs no
        scoping — ``set_reshard_after_backward`` just flips a flag on the unit.
        """
        fsdp_cfg = self.mesh_accelerator.fsdp_config
        if fsdp_cfg.fsdp_mode != "fsdp2" or fsdp_cfg.reshard_after_backward:
            return
        # ``set_reshard_after_backward`` recurses into every nested FSDP unit by
        # default, so one call on the root-sharded model covers them all (the
        # generic ``parallelize_model_fsdp2`` ``fully_shard``s the root). A model
        # that declares its own ``build_parallelize_model`` may leave the root
        # un-sharded — it then owns its reshard policy, so skip rather than
        # assume a root FSDP unit.
        model = self.model
        if callable(getattr(model, "set_reshard_after_backward", None)):
            model.set_reshard_after_backward(reshard)


class HyperOmniRuntime(OmniModelRuntime):
    """Own the Hyper model-scope wrap; module-scope models use their adapters."""

    def _parallelize_composed_model(self, *, for_inference: bool = False) -> None:
        """``fully_shard`` the composed :class:`OmniModel` when ``fsdp_scope='model'``.

        Each :class:`ModuleRuntime` has already meta-initialized, frozen, and
        bound assets, but skipped its own wrap. One FSDP2 tree over the parent:
        wrap each child's ``_no_split_modules`` **scoped to that child**
        (``janus_llama.LlamaDecoderLayer``, ``janus_text_encoder.Embedding``, …),
        then ``fully_shard`` the OmniModel root.

        HF ``post_init``'s union is replaced rather than reused: it is a flat
        class-name set, so the text encoder's ``Embedding`` would also match the
        VQ codebook (see :func:`_scoped_no_split_modules`). Leftover params
        (aligner, final norm, anything not in a nested unit) unshard on
        :meth:`OmniModel.forward`.
        """
        from ..models.seed_omni.accelerated.omni_module.omni_module_runtime import composed_model_owns_wrap

        args = self.omni_model_runtime_args
        acc = args.accelerator
        if not composed_model_owns_wrap(acc):
            return

        modules = self.module_runtimes
        wrap_units = _scoped_no_split_modules(modules)
        self.model._no_split_modules = wrap_units

        kwargs: dict[str, Any] = {
            "cpu_load_param_name": None,
            "module_skip_hf_weight_load": {name: runtime.skip_hf_weight_load for name, runtime in modules.items()},
            "module_is_peft_model": {name: bool(runtime.args.lora_config) for name, runtime in modules.items()},
            "module_adapter_path": {
                name: (runtime.args.lora_config or {}).get("lora_adapter") for name, runtime in modules.items()
            },
        }
        if any(kwargs["module_is_peft_model"].values()):
            kwargs["is_peft_model"] = True

        from ..distributed.torch_compile import CompileConfig
        from .parallelize import build_parallelize_model

        compile_config = CompileConfig(
            **{field.name: getattr(acc.torch_compile, field.name) for field in fields(CompileConfig)}
        )
        opt = args.optimizer
        muon_expert_zero_comm = bool(opt) and opt.type == "muon" and opt.muon_expert_zero_comm
        weights_path = {name: runtime.args.model_path for name, runtime in modules.items()}

        logger.info_rank0(f"OmniModelRuntime: wrapping composed OmniModel (fsdp_scope='model') over {list(modules)}.")
        # ``OmniTrainer._setup`` registered ``base`` from the same top-level accelerator.
        with use_parallel_state("base"):
            self.model = build_parallelize_model(
                self.model,
                init_device=acc.init_device,
                weights_path=weights_path,
                should_skip_hf_weight_load=False,
                enable_reshard_after_forward=acc.fsdp_config.reshard_after_forward,
                mixed_precision=acc.fsdp_config.mixed_precision,
                enable_gradient_checkpointing=False,
                basic_modules=wrap_units,
                enable_reentrant=acc.gradient_checkpointing.enable_reentrant,
                early_stop=acc.gradient_checkpointing.early_stop,
                enable_forward_prefetch=acc.fsdp_config.forward_prefetch,
                enable_fsdp_offload=acc.fsdp_config.offload,
                fsdp_offload_pin_memory=acc.fsdp_config.offload_pin_memory,
                broadcast_model_weights_from_rank0=args.broadcast_model_weights_from_rank0,
                ep_sharded_stream_load=args.ep_sharded_stream_load,
                max_load_broadcast_size=acc.fsdp_config.max_load_broadcast_size,
                muon_expert_zero_comm=muon_expert_zero_comm,
                compile_config=compile_config,
                hyper_reduce_comm_interval=acc.fsdp_config.hyper_reduce_comm_interval,
                low_precision_reduce_scatter_comm=acc.fsdp_config.low_precision_reduce_scatter_comm,
                **kwargs,
            )

        if not for_inference:
            for runtime in modules.values():
                runtime.build_after_omni_model_wrap()


def validate_runtime(args, train, global_accelerator=None):
    try:
        import hyper_parallel  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Hyper options require the optional hyper_parallel package.") from exc
    accelerator = args.accelerator
    if global_accelerator is not None and global_accelerator.fsdp_config.fsdp_scope == "model":
        accelerator = global_accelerator
    if accelerator.fsdp_config.fsdp_backend == "hyper" and train is not None:
        from .checkpoint import HyperParallelCheckpointer

        HyperParallelCheckpointer.validate_config(SimpleNamespace(model=args, train=train))


def text_runtime(args, model_name="base", *, train):
    if not _uses_hyper(args):
        return VeOmniModelRuntime(args, model_name, train=train)
    validate_runtime(args, train)

    return HyperModelRuntime(args, model_name, train=train)


def vlm_runtime(args, model_name="base", *, train):
    if not _uses_hyper(args):
        return VLMModelRuntime(args, model_name, train=train)
    validate_runtime(args, train)

    return HyperVLMRuntime(args, model_name, train=train)


def dit_runtime(args, model_name="base", *, train):
    from ..trainer.dit_trainer import DiTModelRuntime

    if not _uses_hyper(args):
        return DiTModelRuntime(args, model_name, train=train)
    validate_runtime(args, train)

    class HyperDiTRuntime(DiTModelRuntime, HyperModelRuntime):
        """Preserve condition-model setup and state via native super() calls."""

    return HyperDiTRuntime(args, model_name, train=train)


def reference_runtime(args, model_name="reference", *, train, **kwargs):
    from ..trainer.text_dpo_trainer import DPOReferenceModelRuntime

    if not _uses_hyper(args):
        return DPOReferenceModelRuntime(args, model_name, train=train, **kwargs)
    validate_runtime(args, train)

    class HyperReferenceRuntime(DPOReferenceModelRuntime, HyperModelRuntime):
        """Keep the native reference constructor: no optimizer or checkpoint."""

    return HyperReferenceRuntime(args, model_name, train=train, **kwargs)


def build_hyper_omni_module_runtime(args, **kwargs):
    if not _uses_hyper(args, global_accelerator=kwargs.get("global_accelerator")):
        from ..models.seed_omni.accelerated.omni_module.omni_module_runtime import build_omni_module_runtime

        return build_omni_module_runtime(args, **kwargs)
    validate_runtime(args, kwargs.get("train"), kwargs.get("global_accelerator"))

    return HyperModuleRuntime(args, **kwargs)


def build_hyper_omni_model_runtime(args, *, train=None, for_inference=False):
    """Compose Hyper modules without modifying the native assembly function."""
    from ..models.seed_omni.accelerated.omni_model.omni_model_runtime import _reject_lora_that_matched_nothing
    from ..models.seed_omni.accelerated.omni_module.omni_module_runtime import build_omni_module_runtime
    from ..models.seed_omni.modeling_omni import OmniModel

    omni_config = args.to_hf_config()
    omni_config.load_checkpoint_sidecars(args.resolved_model_path)
    module_runtimes = {}
    for name in omni_config.module_names:
        module_args = args.modules[name]
        module_kwargs = {
            "module_name": name,
            "module_config": omni_config._module_configs[name],
            "train": train,
            "for_inference": for_inference,
            "global_accelerator": args.accelerator,
        }
        if _uses_hyper(module_args, global_accelerator=args.accelerator):
            module_runtime = build_hyper_omni_module_runtime(module_args, **module_kwargs)
        else:
            module_runtime = build_omni_module_runtime(module_args, **module_kwargs)
        module_runtimes[name] = module_runtime
        logger.info_rank0(f"OmniModelRuntime: built ModuleRuntime '{name}' from {module_args.model_path}")

    logger.info_rank0(
        f"OmniModelRuntime: composed OmniModel with {len(module_runtimes)} module(s) ({list(module_runtimes)})."
    )
    if not for_inference:
        _reject_lora_that_matched_nothing(module_runtimes)
    model = OmniModel(omni_config, {name: runtime.omni_module for name, runtime in module_runtimes.items()})
    runtime_kwargs = {
        "module_runtimes": module_runtimes,
        "omni_model_runtime_args": args,
        "train_args": train,
    }
    fsdp = args.accelerator.fsdp_config
    if fsdp.fsdp_scope == "model" and fsdp.fsdp_backend == "hyper":
        runtime = HyperOmniRuntime(model, **runtime_kwargs)
    else:
        runtime = OmniModelRuntime(model, **runtime_kwargs)
    runtime._parallelize_composed_model(for_inference=for_inference)
    if not for_inference:
        runtime._build_optimizer()
    return runtime
