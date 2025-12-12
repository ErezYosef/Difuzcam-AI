from .mycontrol import MyControlNetModel

import numpy as np

from diffusers.models.modeling_utils import _load_state_dict_into_model
from diffusers.models.modeling_utils import *
_LOW_CPU_MEM_USAGE_DEFAULT = False
import os
from diffusers.utils import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    SAFETENSORS_FILE_EXTENSION,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    _add_variant,
    _get_model_file,
    deprecate,
    is_accelerate_available,
    is_torch_version,
    logging,
)
from diffusers import __version__


from diffusers.models.controlnet import ControlNetOutput


from diffusers.utils import BaseOutput, logging



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class FlatControlNetModel(MyControlNetModel):

    _supports_gradient_checkpointing = True

    #@register_to_config
    def __init__(self, *args, **kwargs):
        kwargs['conditioning_channels'] = 4
        super().__init__(*args, **kwargs)

        self.flat_inverse = Separable_matmult(L_size=(500, 512), R_size=(620, 512))
        self.conv = torch.nn.Conv2d(4, 3, 3, padding=1)


    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def forward(
        self,
        *args, **kwargs,
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.FloatTensor, ...], torch.FloatTensor]]:

        # controlnet_cond: torch.FloatTensor,
        controlnet_cond = kwargs.pop('controlnet_cond')
        #print('b', controlnet_cond.shape)
        controlnet_cond = self.flat_inverse(controlnet_cond)
        #print('a', controlnet_cond.shape, args[0].shape)
        #print(controlnet_cond.shape)
        if self.training:
            return self.conv(controlnet_cond), super().forward(controlnet_cond=controlnet_cond, *args, **kwargs)

        return super().forward(controlnet_cond=controlnet_cond, *args, **kwargs)

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict: OrderedDict,
        resolved_archive_file,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        ignore_mismatched_sizes: bool = False,
    ):
        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        loaded_keys = list(state_dict.keys())

        expected_keys = list(model_state_dict.keys())

        original_loaded_keys = loaded_keys

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))
        print('missing_keys', missing_keys)
        print('unexpected_keys', unexpected_keys)
        print('erez', model.down_blocks[0].attentions[0].proj_in.weight.shape)

        # Make sure we are able to load base models as well as derived models (with heads)
        model_to_load = model

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        if state_dict is not None:
            # Whole checkpoint
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                ignore_mismatched_sizes,
            )
            error_msgs = _load_state_dict_into_model(model_to_load, state_dict)

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task"
                " or with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly"
                " identical (initializing a BertForSequenceClassification model from a"
                " BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the"
                f" checkpoint was trained on, you can already use {model.__class__.__name__} for predictions"
                " without further training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be"
                " able to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a pretrained PyTorch model from a pretrained model configuration.

        The model is set in evaluation mode - `model.eval()` - by default, and dropout modules are deactivated. To
        train the model, set it back in training mode with `model.train()`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`~ModelMixin.save_pretrained`].

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info (`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if `device_map` contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            variant (`str`, *optional*):
                Load weights from a specified `variant` filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the `safetensors` weights are downloaded if they're available **and** if the
                `safetensors` library is installed. If set to `True`, the model is forcibly loaded from `safetensors`
                weights. If set to `False`, `safetensors` weights are not loaded.

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
        `huggingface-cli login`. You can also activate the special
        ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a
        firewalled environment.

        </Tip>

        Example:

        ```py
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        ```

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```bash
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_flax = kwargs.pop("from_flax", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        unet = kwargs.pop("unet", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if device_map is not None and not is_accelerate_available():
            raise NotImplementedError(
                "Loading and dispatching requires `accelerate`. Please make sure to install accelerate or set"
                " `device_map=None`. You can install accelerate with `pip install accelerate`."
            )

        # Check if we can handle device_map and dispatching the weights
        if device_map is not None and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `device_map=None`."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to `False` while using device_map={device_map} for loading and"
                " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # load config
        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            user_agent=user_agent,
            **kwargs,
        )

        # load model
        model_file = None

        if use_safetensors:
            try:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    commit_hash=commit_hash,
                )
            except IOError as e:
                if not allow_pickle:
                    raise e
                pass
        if model_file is None:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=_add_variant(WEIGHTS_NAME, variant),
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                commit_hash=commit_hash,
            )

        if low_cpu_mem_usage:
            raise ValueError("Flat_controlnet do not support : low_cpu_mem_usage")
        if unet is not None:
            model = cls.from_unet(unet)
        else:
            model = cls.from_config(config, **unused_kwargs)

        state_dict = load_state_dict(model_file, variant=variant)
        model._convert_deprecated_attention_blocks(state_dict)

        model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
            model,
            state_dict,
            model_file,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )

        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
            "error_msgs": error_msgs,
        }

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        elif torch_dtype is not None:
            model = model.to(torch_dtype)

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        if output_loading_info:
            return model, loading_info

        return model


class FlatControlNetModel_E(FlatControlNetModel):

    _supports_gradient_checkpointing = True

    #@register_to_config
    def __init__(self, *args, **kwargs):
        kwargs['conditioning_channels'] = 4
        super().__init__(*args, **kwargs)

        self.flat_inverse = Separable_matmult(L_size=(952//2, 512), R_size=(1276//2, 512))
        self.conv = torch.nn.Conv2d(4, 3, 3, padding=1)

class FlatControlNetModel_Efull(FlatControlNetModel_E):

    _supports_gradient_checkpointing = True

    #@register_to_config
    def __init__(self, *args, **kwargs):
        kwargs['conditioning_channels'] = 4
        super().__init__(*args, **kwargs)

        self.flat_inverse = Separable_matmult(L_size=(952, 512), R_size=(1276, 512))
        self.conv = torch.nn.Conv2d(4, 3, 3, padding=1)


class Separable_matmult(nn.Module):
    def __init__(self, channel_dim=4, L_size=(500, 256), R_size=(620, 256), std_randn=0.001):
        super().__init__()
        l, r = [], []

        for i in range(channel_dim):
            phil, phir = Separable_matmult.get_mats_init('True_randn', L_size, R_size, std_randn)
            l.append(phil)
            r.append(phir)
        l = np.stack(l, 0)
        r = np.stack(r, 0)
        self.PhiL = nn.Parameter(torch.from_numpy(l))
        self.PhiR = nn.Parameter(torch.from_numpy(r))

    def forward(self, x):
        x = torch.einsum('bchw,chv-> bcvw', x, self.PhiL)
        x = torch.einsum('bcvw,cwu-> bcvu', x, self.PhiR)
        return x


    @staticmethod
    def get_mats_init(init_transform, L_size, R_size, std_randn=0.001):
        if init_transform == 'Transpose':
            print('Loading calibrated files')
            import scipy.io as sio
            d = sio.loadmat('data/flatcam_prototype2_calibdata.mat')
            phil = np.zeros(L_size)
            phir = np.zeros(R_size)
            phil[:, :] = d['P1gb']
            phir[:, :] = d['Q1gb']
            phil = phil.astype('float32')
            phir = phir.astype('float32')
        elif init_transform == 'True_rand':
            print('Load True_rand')
            phil = np.random.rand(*L_size) - 0.5
            phir = np.random.rand(*R_size) - 0.5
            phil = phil.astype('float32') / 100
            phir = phir.astype('float32') / 100
        elif init_transform == 'True_randn':
            print('Load True_randn')
            phil = np.random.randn(*L_size) * std_randn
            phir = np.random.randn(*R_size) * std_randn
            phil = phil.astype('float32')
            phir = phir.astype('float32')
        elif init_transform == 'eye':
            print('Load eye')
            phil = torch.eye(256)
            phir = torch.eye(256)
        elif init_transform == 'randn_diag':
            a = np.random.randn(1, L_size[0] + 1) * np.ones((L_size[1], 1)) * 0.003
            b = a.flatten()
            phil = (b[:(len(b) // L_size[0]) * L_size[0]]).reshape(
                (len(b) // L_size[0], L_size[0])).T
            a = np.random.randn(1, R_size[0] + 1) * np.ones((R_size[1], 1)) * 0.003
            b = a.flatten()
            phir = (b[:(len(b) // R_size[0]) * R_size[0]]).reshape(
                (len(b) // R_size[0], R_size[0])).T
            phil = phil.astype('float32')
            phir = phir.astype('float32')
        else:
            import scipy.io as sio
            print('Loading Random Toeplitz')
            phil = np.zeros(L_size)
            phir = np.zeros(R_size)
            pl = sio.loadmat('data/phil_toep_slope22.mat')
            pr = sio.loadmat('data/phir_toep_slope22.mat')
            phil[:, :] = pl['phil'][:, :, 0]
            phir[:, :] = pr['phir'][:, :, 0]
            phil = phil.astype('float32')
            phir = phir.astype('float32')
        return phil, phir