# ECSAnet – BreakHis Classification Pipeline

EfficientNet-V2-S + CBAM for histopathology image classification on the BreakHis dataset.

## Project Structure

```
ecsanet/
├── prepare_data.py        ← Run on Kaggle: PNG → .npy
├── main.py                ← Main training entry point
├── requirements.txt
└── src/
    ├── preprocess.py      ← Reinhard stain normalisation + base transforms
    ├── augment.py         ← Augmentation strategies
    ├── dataset.py         ← Dataset classes (loads from .npy)
    ├── model.py           ← CBAM + ECSAnet architecture
    ├── train.py           ← Single-epoch training loop
    ├── validate.py        ← Validation / test evaluation loop
    └── metrics.py         ← Metrics computation, MLflow/TB logging, plots
```

## Step 1 – Data Preparation (run on Kaggle)

Open a Kaggle notebook, upload `prepare_data.py`, then run:

```bash
# Add ambarish/breakhis dataset via the Data panel, then:
!python prepare_data.py \
    --output_dir /kaggle/working/npy_data \
    --img_size 384
```

This produces:
```
npy_data/
├── metadata.json
├── target_image.png          ← stain-norm reference
├── 40X/
│   ├── train_images.npy      ← (N, 384, 384, 3) uint8
│   ├── train_labels.npy      ← (N,) int64
│   ├── val_images.npy
│   ├── val_labels.npy
│   ├── test_images.npy
│   └── test_labels.npy
├── 100X/ ...
├── 200X/ ...
└── 400X/ ...
```

Download `npy_data/` to your local machine.

## Step 2 – Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3 – Train

### Default (all magnifications, paper settings)
```bash
python main.py --data_dir ./npy_data
```

### Custom run
```bash
python main.py \
    --data_dir        ./npy_data \
    --magnification   40X 200X \
    --model           ecsamet \
    --optimizer       sgd \
    --lr              0.001 \
    --weight_decay    0.01 \
    --augment         standard \
    --preprocess      reinhard \
    --epochs          50 \
    --batch_size      32 \
    --patience        25 \
    --output_dir      ./outputs \
    --experiment      BreakHis-ECSAnet \
    --run_name        sgd_standard_40X
```

## CLI Arguments

| Argument | Default | Options / Notes |
|---|---|---|
| `--data_dir` | *(required)* | Path to `npy_data/` folder |
| `--magnification` | `all` | `40X` `100X` `200X` `400X` `all` |
| `--model` | `ecsamet` | see `src/model.py MODEL_REGISTRY` |
| `--optimizer` | `sgd` | `sgd` `adam` `adamw` |
| `--lr` | `0.001` | learning rate |
| `--momentum` | `0.9` | SGD only |
| `--weight_decay` | `0.01` | |
| `--preprocess` | `reinhard` | `reinhard` `none` |
| `--augment` | `standard` | `standard` `augmix` `combined` `none` |
| `--img_size` | `384` | |
| `--epochs` | `50` | |
| `--batch_size` | `32` | |
| `--patience` | `25` | early-stopping patience |
| `--no_amp` | off | disable mixed precision |
| `--output_dir` | `./outputs` | |
| `--experiment` | `ECSAnet-BreakHis` | MLflow experiment name |
| `--run_name` | auto | MLflow run name |

## Output Structure

```
outputs/
├── models/{mag}/best_model.pth
├── checkpoints/{mag}/ckpt_epoch_*.pth
├── figures/{model}_{mag}_cm.pdf
│           {model}_{mag}_roc.pdf
├── metrics/{model}_{mag}_test_metrics.txt
└── runs/{run_name}/{mag}/...     ← TensorBoard logs
```

## MLflow UI

```bash
mlflow ui --port 5000
# open http://localhost:5000
```

## Class Labels

| Short | Full |
|---|---|
| A | adenosis |
| DC | ductal_carcinoma |
| F | fibroadenoma |
| LC | lobular_carcinoma |
| MC | mucinous_carcinoma |
| PC | papillary_carcinoma |
| PT | phyllodes_tumor |
| TA | tubular_adenoma |
