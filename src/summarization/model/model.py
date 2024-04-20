from transformers import (
    AdamW,
    Adafactor,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    get_linear_schedule_with_warmup
)
import pytorch_lightning as pl
from torchsummary import summary

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
      return AdamW(self.model.parameters(), lr = 1e-4)


if __name__ == "__main__":


# Load the model and create example input as before
  model = T5ForConditionalGeneration.from_pretrained('t5-small')
  tokenizer =  T5Tokenizer.from_pretrained('t5-small')
  input_ids = tokenizer.encode("This is an example input", return_tensors="pt")

  # Print the model summary
  summary(model, input_ids, device='cuda')