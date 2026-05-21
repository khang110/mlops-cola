import torch
import hydra
import logging
import warnings

from omegaconf.omegaconf import OmegaConf

from model import ColaModel
from data import DataModule

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="triton not found; flop counting will not work for triton kernels",
)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def convert_model(cfg):
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/best-checkpoint.ckpt"
    logger.info(f"Loading pre-trained model from: {model_path}")
    cola_model = ColaModel.load_from_checkpoint(model_path)
    cola_model.eval()
    cola_model.cpu()

    data_model = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    data_model.prepare_data()
    data_model.setup()
    input_batch = next(iter(data_model.train_dataloader()))
    export_device = next(cola_model.parameters()).device
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0).to(export_device),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0).to(export_device),
    }

    # Export the model
    logger.info(f"Converting the model into ONNX format")
    with torch.no_grad():
        torch.onnx.export(
            cola_model,  # model being run
            (
                input_sample["input_ids"],
                input_sample["attention_mask"],
            ),  # model input (or a tuple for multiple inputs)
            f"{root_dir}/models/model.onnx",  # where to save the model
            export_params=True,
            opset_version=18,
            input_names=["input_ids", "attention_mask"],  # the model's input names
            output_names=["output"],  # the model's output names
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

    logger.info(
        f"Model converted successfully. ONNX format model is at: {root_dir}/models/model.onnx"
    )


if __name__ == "__main__":
    convert_model()
