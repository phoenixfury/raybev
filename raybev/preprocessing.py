"""Camera preprocessing matching the resize/letterbox used for LUT generation."""
import torch
import torch.nn.functional as F
from .geometry import letterbox_params


def resize_camera_image(image,target_width=800,target_height=640):
    """Host preprocessing shared by native-image training and validation.

    PIL bilinear downsampling, exact integer letterbox and black RGB padding.
    Projection uses the identical transform in resized_camera_intrinsics.
    """
    from PIL import Image
    width,height,left,top=letterbox_params(image.width,image.height,target_width,target_height)
    resized=image.convert('RGB').resize((width,height),Image.Resampling.BILINEAR)
    if (width,height)==(target_width,target_height):return resized
    output=Image.new('RGB',(target_width,target_height),(0,0,0))
    output.paste(resized,(left,top))
    return output


def letterbox_images(images: torch.Tensor, target_width=800, target_height=640):
    """Resize unnormalized float raw camera images without anisotropic stretching.

    For the calibrated 1920x1536 sources, 800x640 needs no padding.

    images: [B, 4, 3, raw_height, raw_width], float in [0,1].
    output: [B, 4, 3, target_height, target_width].
    ImageNet normalization must be applied separately and consistently.
    """
    if images.ndim != 5 or images.shape[1:3] != (4, 3):
        raise ValueError('Expected images [B,4,3,H,W]')
    b, n, c, h, w = images.shape
    out_w, out_h, left, top = letterbox_params(w, h, target_width, target_height)
    resized = F.interpolate(images.reshape(b*n,c,h,w), size=(out_h,out_w),
                            mode='bilinear', align_corners=False)
    padded = F.pad(resized, (left, target_width-left-out_w,
                             top, target_height-top-out_h))
    return padded.reshape(b,n,c,target_height,target_width)


def normalize_imagenet(images: torch.Tensor):
    mean = images.new_tensor((.485, .456, .406)).reshape(1,1,3,1,1)
    std = images.new_tensor((.229, .224, .225)).reshape(1,1,3,1,1)
    return (images - mean) / std
