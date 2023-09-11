# Leaf Segmentation/Counting

## Data preparation

Run `scripts/train_test_split.py` as follows:
```
python scripts/train_test_split.py --root_dir path/to/dataset
```

## Training
Run `main.py` as follows:
```
python main.py --root_dir path/to/dataset --model resnet
```
Available models to train:
- Modified ResNet for leaf counting
- Faster R-CNN

The code will finetune a selected pretrained segmentation model, and outputs some figures:
- `data.png` shows some samples of images and corresponding masks in the training dataset.
- `binary_masks.png` shows individual binary mask for each leaf in some images
- `{model_name}_losses.png` shows plots for train and validation losses over epochs

The code also saves checkpoints every 10 epochs at `ckpts` directory.

## Inference
Run `predict.py` as follows:
```
python predict.py --model resnet
```
Available models to infer:
- Modified ResNet for leaf counting
- Faster R-CNN

The code will evaluate the model and output corresponding evaluation metrics on the validation set.

For Faster R-CNN model, bounding box predictions of each image in the validation set are also generated. 

## LIME explanations
Run `lime_explainer.py` as follows:
```
python lime_explainer.py --root_dir path/to/project
```
The code will generate LIME explanations for trained ResNetCounter model, and outputs relevant figures:
- `explanation_{i}.png` where i is the index of the LIME explanation.
- `explanations.csv` which contains actual predicted labels for the generated LIME explanations.
