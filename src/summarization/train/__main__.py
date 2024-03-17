from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb
import pytorch_lightning as pl
from pathlib import Path
from rouge import Rouge
import datasets
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
# from ..dataset.datamodule import *
# from ..model.model import *


pl.seed_everything(42)

project_path = Path('../')
data_path = project_path / 'data'
data_path.mkdir(parents=True, exist_ok=True)

#Train Data Path
train_ds_path = data_path / 'train'
train_ds_path.mkdir(parents=True, exist_ok=True)

#Test Data Path
test_ds_path = data_path / 'test'
test_ds_path.mkdir(parents=True, exist_ok=True)


#Val Data Path
val_ds_path = data_path / 'val'
val_ds_path.mkdir(parents=True, exist_ok=True)

#Model Checkpoint
chkpt_path = project_path / "checkpoints"
chkpt_path.mkdir(parents=True, exist_ok=True)

train_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train[:30%]")    #300k
val_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="validation[:30%]")
     
train_data.save_to_disk(train_ds_path/'30_percent_train')
val_data.save_to_disk(train_ds_path/'30_percent_val')

train_data = datasets.load_from_disk(train_ds_path/'30_percent_train')
val_data = datasets.load_from_disk(train_ds_path/'30_percent_val')

MODEL_NAME = 't5-base'

N_EPOCHS = 2
BATCH_SIZE = 8

learning_rate = 0.0001

log_steps = 500
valid_steps = 500
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length=512)
# data_module = NewsSummaryDataModule(train_data, val_data, tokenizer, batch_size=BATCH_SIZE, num_workers=4)

# model = NewsSummaryModel(lr=learning_rate)

# chkpt_path = "../checkpoints"
# checkpoint_callback = ModelCheckpoint(
#     dirpath = str(chkpt_path),
#     filename="Best-T5-{epoch}-{step}-{val_loss:.2f}",
#     save_top_k=1,
#     verbose=True,
#     monitor="val_loss",
# )

# logger =  logger = WandbLogger(project="text-summarization",
#                              name="",
#                              log_model="all")

# trainer = pl.Trainer(
#     logger=logger,
#     callbacks=[checkpoint_callback],
#     max_epochs=N_EPOCHS,
#     gpus=-1,
#     log_every_n_steps=log_steps,
#     val_check_interval=valid_steps
# )