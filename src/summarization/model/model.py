from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
import pytorch_lightning as pl
import torch

class NewsSummaryModel(pl.LightningModule):
  MODEL_BASE = T5ForConditionalGeneration
  OPTIM = AdamW
  def __init__(self,
               model_name:str='t5-base',
               lr:int= 0.0001,
    ):
    super().__init__()
    
  
    self.model_name = model_name
    self.model = self.MODEL_BASE.from_pretrained(self.model_name, return_dict=True)
    
  
    self.lr = lr

  def forward(self, input_ids, attention_mask, decoder_attention_mask, labels = None):
    output = self.model(
        input_ids,
        attention_mask = attention_mask,
        labels = labels,
        decoder_attention_mask=decoder_attention_mask
    )
    return output.loss, output.logits

  def training_step(self, batch, batch_idx):
    
    input_ids = batch["text_input_ids"]
    attention_mask = batch["text_attention_mask"]
    labels = batch["labels"]
    labels_attention_mask = batch["labels_attention_mask"]

    loss, outputs = self(input_ids=input_ids, 
                         attention_mask=attention_mask,
                         decoder_attention_mask=labels_attention_mask,
                         labels=labels
                         )
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    input_ids = batch["text_input_ids"]
    attention_mask = batch["text_attention_mask"]
    labels = batch["labels"]
    labels_attention_mask = batch["labels_attention_mask"]

    loss, outputs = self(input_ids=input_ids, 
                         attention_mask=attention_mask,
                         decoder_attention_mask=labels_attention_mask,
                         labels=labels
                         )
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss
  
  def test_step(self, batch, batch_idx):
    input_ids = batch["text_input_ids"]
    attention_mask = batch["text_attention_mask"]
    labels = batch["labels"]
    labels_attention_mask = batch["labels_attention_mask"]

    loss, outputs = self(input_ids=input_ids, 
                         attention_mask=attention_mask,
                         decoder_attention_mask=labels_attention_mask,
                         labels=labels
                         )
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):
      optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr
        )
      scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1
            ),
            'interval': 'step',
            'frequency': 1
        }
      
      return [optimizer], [scheduler]
