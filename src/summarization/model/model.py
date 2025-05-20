from transformers import (
    Adafactor,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    get_linear_schedule_with_warmup
)
import pytorch_lightning as pl
from torchsummary import summary
import torch

class NewsSummaryModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4, accumulate_grad_batches=4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.accumulate_grad_batches = accumulate_grad_batches
        self.save_hyperparameters(ignore=['model'])  # Save hyperparameters except the model

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, _ = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, _ = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, _ = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )
        self.log("test_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adafactor(
            self.model.parameters(),
            lr=self.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False
        )
        # Configure scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,  # Adjust based on dataset size
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def setup(self, stage=None):
        # Enable mixed precision training
        # self.trainer.precision = "16-mixed" if torch.cuda.is_available() else "32-true"
        pass