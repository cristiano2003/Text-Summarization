# Abstractive Text Summarization



## Clone

```bash
!git clone https://github.com/cristiano2003/Text-Summarization.git
%cd Text-Summarization
```

## Setup, Build Package and download Checkpoint, Dataset and Demo

```bash
source ./scripts/setup.sh
```

## Train

```bash
python -m dl_project.train --model resnet --batch_size 64
```

### Train all

```bash
python -m dl_project.train --model all --batch_size 64
```

### Train Demo

```bash
python -m dl_project.train --model resnet --batch_size 64 --max_epochs 10 --folder generate --name demo
```


```
