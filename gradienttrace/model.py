from __future__ import annotations

import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class AIDetector(PreTrainedModel):
    """Encoder-only backbone (DeBERTa-v3 by default) + single-logit AI head.

    Kept as a PreTrainedModel so Trainer/Hub save, load and push work natively.
    config_class is AutoConfig, so the backbone family (DeBERTa-v2/v3,
    ModernBERT, ...) is a config choice, not a code change.
    """

    config_class = AutoConfig
    base_model_prefix = "encoder"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config)
        self.head_dropout = float(getattr(config, "head_dropout", 0.1))
        self.pooling = str(getattr(config, "pooling", "cls"))
        self.dropout = nn.Dropout(self.head_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.post_init()

    @classmethod
    def from_backbone(cls, name: str, head_dropout: float = 0.1, pooling: str = "cls") -> "AIDetector":
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        config = AutoConfig.from_pretrained(name)
        config.head_dropout = head_dropout
        config.pooling = pooling
        config.backbone_name = name
        model = cls(config)
        pretrained = AutoModel.from_pretrained(name, config=config)
        model.encoder.load_state_dict(pretrained.state_dict())
        del pretrained
        return model

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=getattr(self.config, "initializer_range", 0.02))
            if module.bias is not None:
                module.bias.data.zero_()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        encoder_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_inputs)
        hidden = outputs.last_hidden_state

        if self.pooling == "mean" and attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden[:, 0]

        logits = self.classifier(self.dropout(pooled)).squeeze(-1)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.to(logits.dtype))
        return SequenceClassifierOutput(loss=loss, logits=logits)


def build_tokenizer(name: str):
    return AutoTokenizer.from_pretrained(name, use_fast=True)


def build_model(name: str, head_dropout: float = 0.1, pooling: str = "cls") -> AIDetector:
    return AIDetector.from_backbone(name, head_dropout=head_dropout, pooling=pooling)
