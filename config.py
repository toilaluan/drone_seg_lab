import albumentations as A
import cv2



train_img_aug = A.Compose(
    (
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p = 0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.5),
        A.RandomBrightnessContrast(),
        A.GaussNoise(5),
    )
    
)
train_map_aug = A.Compose(
    [
        A.RandomRotate90(),
        A.GridDistortion(3),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.OneOf([
                A.CLAHE (clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, interpolation=cv2.INTER_NEAREST, p=0.5),
        ], p=1.0),
    ]
)
