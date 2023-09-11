# Leaf Segmentation/Counting

## Data preparation

Run `scripts/train_test_split.py` as follows:
```
python scripts/train_test_split.py --root_dir path/to/dataset
```

## Training
Run `main.py` as follows:
```
python main.py --root_dir path/to/dataset
```
The code will finetune a pretrained segmentation model (`Unet` for now), and outputs some figures:
- `data.png` shows some samples of images and corresponding masks in the training dataset.
- `prediction.png` shows some predictions from the model given images and masks.

The code also saves checkpoints for each epoch at `ckpts` directory.