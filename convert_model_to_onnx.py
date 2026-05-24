import torch
import hydra
import logging
import transformers.masking_utils

# Monkey-patch transformers.masking_utils.sdpa_mask to fix IndexError: tuple index out of range for 0D tensors during ONNX export
def patched_sdpa_mask(
    batch_size: int,
    q_length,
    kv_length,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function = transformers.masking_utils.causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    local_size: int | None = None,
    allow_is_causal_skip: bool = True,
    allow_is_bidirectional_skip: bool = False,
    allow_torch_fix: bool = True,
    use_vmap: bool = False,
    device = "cpu",
    **kwargs,
):
    if isinstance(q_length, torch.Tensor) and q_length.dim() > 0:
        q_length, q_offset = q_length.shape[0], q_length[0].to(device)

    padding_mask = transformers.masking_utils.prepare_padding_mask(attention_mask, kv_length, kv_offset)

    if allow_is_causal_skip and transformers.masking_utils._ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None
    if allow_is_bidirectional_skip and transformers.masking_utils._ignore_bidirectional_mask_sdpa(padding_mask, kv_length, local_size):
        return None

    if padding_mask is not None:
        mask_function = transformers.masking_utils.and_masks(mask_function, transformers.masking_utils.padding_mask_function(padding_mask))

    batch_arange = torch.arange(batch_size, device=device)
    head_arange = torch.arange(1, device=device)
    q_arange = torch.arange(q_length, device=device) + q_offset
    kv_arange = torch.arange(kv_length, device=device) + kv_offset

    if not use_vmap:
        attention_mask = mask_function(*transformers.masking_utils._non_vmap_expansion_sdpa(batch_arange, head_arange, q_arange, kv_arange))
        attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
    elif transformers.masking_utils._is_torch_greater_or_equal_than_2_6:
        with transformers.masking_utils.TransformGetItemToIndex():
            attention_mask = transformers.masking_utils._vmap_expansion_sdpa(mask_function)(batch_arange, head_arange, q_arange, kv_arange)
    else:
        raise ValueError(
            "The vmap functionality for mask creation is only supported from torch>=2.6."
        )

    if not transformers.masking_utils._is_torch_greater_or_equal_than_2_5 and allow_torch_fix:
        attention_mask = attention_mask | torch.all(~attention_mask, dim=-1, keepdim=True)

    return attention_mask

transformers.masking_utils.sdpa_mask = patched_sdpa_mask
transformers.masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping["sdpa"] = patched_sdpa_mask

from model import ColaModel
from data import DataModule

logger = logging.getLogger(__name__)


@hydra.main(config_path="./configs", config_name="config")
def convert_model(cfg):
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/best-checkpoint.ckpt"
    logger.info(f"Loading pre-trained model from: {model_path}")
    cola_model = ColaModel.load_from_checkpoint(model_path, map_location="cpu")
    cola_model.to("cpu")
    cola_model.eval()

    data_model = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    data_model.prepare_data()
    data_model.setup()
    input_batch = next(iter(data_model.train_dataloader()))
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0),
    }

    # Export the model
    logger.info(f"Converting the model into ONNX format")
    torch.onnx.export(
        cola_model,  # model being run
        (
            input_sample["input_ids"],
            input_sample["attention_mask"],
        ),  # model input (or a tuple for multiple inputs)
        f"{root_dir}/models/model.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=10,
        input_names=["input_ids", "attention_mask"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input_ids": {0: "batch_size"},  # variable length axes
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    logger.info(
        f"Model converted successfully. ONNX format model is at: {root_dir}/models/model.onnx"
    )


if __name__ == "__main__":
    convert_model()
