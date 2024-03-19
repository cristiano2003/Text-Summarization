from transformers import (
    AdamW,
    Adafactor,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    get_linear_schedule_with_warmup
)
import pytorch_lightning as pl


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
      return Adafactor(
        self.model.parameters(),
        lr=1e-6,
        eps = (1e-30, 1e-3),
        clip_threshold = 1.0,
        decay_rate = -0.8,
        beta1 = None,
        weight_decay = 0.0,
        relative_step = False,
        scale_parameter = False,
        warmup_init = False
      )
