import albumentations as albu

def train_augmentation():
    train_transform = [
        albu.RandomSizedBBoxSafeCrop(height=320, width=320, always_apply=True),
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.Flip(p=0.5),

        albu.RandomRotate90(p=0.5),

        albu.GaussNoise(p=0.2),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        )
    ]
    return albu.Compose(train_transform, bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(384, 480)
    ]
    return albu.Compose(test_transform, bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn)
    ]
    return albu.Compose(_transform, bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['class_labels']))