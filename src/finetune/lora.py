from omegaconf import DictConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from src.models.base import Model


def add_lora(model: Model, cfg: DictConfig) -> PeftModel:
    """Add a trainable LoRA adapter on top of a loaded model's weights.

    Args:
        model (Model): The loaded model; it keeps the adapter from now on.
        cfg (DictConfig): The LoRA settings.

    Returns:
        PeftModel: A model with a LoRA adapter on top of the loaded weights.

    Raises:
        TypeError: If an adapter is already attached.
    """
    base = model.model

    if isinstance(base, PeftModel):
        raise TypeError(f"{model.name} already has a LoRA adapter attached")

    if model.quantization_config() is not None:
        base = prepare_model_for_kbit_training(base)

    lora_config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=cfg.target_modules,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, lora_config)
    trainable, total = peft_model.get_nb_trainable_parameters()
    model.model = peft_model

    model.logger.info(
        f"LoRA trains {trainable:,} of {total:,} parameters ({trainable / total:.2%})"
    )

    return peft_model


def load_adapter(model: Model, path: str) -> None:
    """Load a trained LoRA adapter on top of a loaded model's weights.

    Args:
        model (Model): The loaded model; it keeps the adapter from now on.
        path (str): The folder the adapter was saved to.
    """
    model.model = PeftModel.from_pretrained(model.model, path)
    model.model.eval()

    model.logger.info(f"Loaded the LoRA adapter from {path}")


def detach_adapter(model: Model) -> None:
    """Detach an attached LoRA adapter, restoring the loaded weights as they were.

    Args:
        model (Model): The loaded model; it keeps the restored weights from now on.
    """
    if not isinstance(model.model, PeftModel):
        return

    model.model = model.model.unload()

    model.logger.info(f"Removed the LoRA adapter from {model.name}")
