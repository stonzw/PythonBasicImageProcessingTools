import numpy as np
import cv2
from PIL import Image


def get_image_wh(image):
    t = type(image)
    if t == np.ndarray:
        w = image.shape[1]
        h = image.shape[0]
        return (w, h)
    if t == Image.Image:
        return image.size

def resize_image(image, wh):
    t = type(image)
    if t == Image.Image:
        return image.resize(wh)
    if t == np.ndarray:
        return cv2.resize(image,(wh[0],wh[1]))

def padding_image(image, wh):
    cnum = image.shape[2]
    blank = np.zeros(wh[1],wh[0], cnum)
    return blank

def resize_width(image, resize_width):
    w, h = get_image_wh(image)
    ratio = resize_width / w
    new_wh = (resize_width, int(h * ratio))
    return resize_image(image, new_wh)

def resize_height(image, resize_height):
    w, h = get_image_wh(image)
    ratio = resize_height / h
    new_wh = (int(w * ratio), resize_height)
    return resize_image(image, new_wh)

def imread(image_path, is_pil=False):
    image = cv2.imread(image_path)
    image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if is_pil:
        return Image.fromarray(image)
    return image

def sampleDestWH(image_width, image_height, alpha=1, beta=0.3):
    w = int(image_width * np.random.beta(alpha, beta))
    h = int(image_height * np.random.beta(alpha, beta))
    return [
        (w, h),
        (image_width - w, h),
        (w, image_height - h),
        (image_width  - w, image_height - h)
    ]
    return w,h

def recapCrop(recapWH, images, destW, destH):
    crop_images = []
    for image, dest_wh in zip(images, recapWH):
        src_wh = get_image_wh(image)
        if src_wh[0] < dest_wh[0] and src_wh[1] < dest_wh[1]:
            image = resize_image(image, dest_wh)
        elif src_wh[0] < dest_wh[0]:
            image = resize_width(image, dest_wh[0])
        elif src_wh[1] < dest_wh[1]:
            image = resize_height(image, dest_wh[1])
        src_wh = get_image_wh(image)
        try:
            x = np.random.randint(0, 1 + src_wh[0] - dest_wh[0])
            y = np.random.randint(0, 1 + src_wh[1] - dest_wh[1])
            crop_images.append(image[y:y + dest_wh[1], x:x+dest_wh[0]])
        except:
            print(src_wh)
            print(dest_wh)
            raise
    return crop_images

def recap(images, alpha=1, beta=0.3):
    recapWHs = sampleDestWH(images[0].shape[1], images[0].shape[0],alpha,beta)
    recapImages = recapCrop(recapWHs, images, images[0].shape[1], images[0].shape[0])
    imageUpper = np.hstack([recapImages[0],recapImages[1]])
    imageLower = np.hstack([recapImages[2],recapImages[3]])
    recapImage = np.vstack([imageUpper,imageLower])
    return recapImage

def crop(image, x, y, w, h):
    return image[y:y+h, x:x+w]

def cv2pil(image):
    return Image.fromarray(image)

def pil2cv(image):
    return np.array(image)

def isPIL(image):
    return type(image) == Image.Image

def paddingImage(image, top=0, bottom=0, left=0, right=0, color=(0, 0, 0)):
    if isPIL(image):
        _image = cv2.copyMakeBorder(pil2cv(image), top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return cv2pil(_image)
    else:
        _image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return _image

def resizeImagePadding(image, wh):
    image_w,image_h = get_image_wh(image)
    padding_width = wh[0] - image_w
    padding_height = wh[1] - image_h
    if 0 <= padding_width and 0 <= padding_height:
        left_padding = padding_width // 2
        right_padding = padding_width - left_padding
        top_padding = padding_height // 2
        bottom_padding = padding_height - top_padding
        return paddingImage(image, top=top_padding,bottom=bottom_padding,left=left_padding, right=right_padding)
    if padding_height < padding_width:
        image_resized = resize_height(image, wh[1])
        resized_width = get_image_wh(image_resized)[0]
        padding_width = wh[0] - resized_width
        assert 0 < padding_width
        left_padding = padding_width // 2
        right_padding = padding_width - left_padding
        return paddingImage(image_resized, left=left_padding, right=right_padding)
    else:
        image_resized = resize_width(image, wh[0])
        resized_height = get_image_wh(image_resized)[1]
        padding_height = wh[0] - resized_height
        assert 0 < padding_height
        top_padding = padding_height // 2
        bottom_padding = padding_height - top_padding
        return paddingImage(image_resized, top=top_padding, bottom=bottom_padding)

def patch_row_patch_height(image, patch_height, ignore_fraction_size_ratio = 0.1):
    if image.shape[0] < patch_height:
        return []
    row_images = []
    i = 0
    while (i + 1) * patch_height < image.shape[0]:
        row_images.append(image[patch_height * i: patch_height * (i+1)])
        i += 1
    rem = patch_height * (i+1) - image.shape[0]
    if ignore_fraction_size_ratio * patch_height < rem:
        row_images.append(image[image.shape[0] - patch_height: image.shape[0]])
    return row_images

def patch_col_patch_width(image, patch_width, ignore_fraction_size_ratio = 0.1):
    if image.shape[1] < patch_width:
        return []
    col_images = []
    i = 0
    while (i + 1) * patch_width < image.shape[1]:
        col_images.append(image[::, patch_width * i: patch_width * (i+1)])
        i += 1
    rem = patch_width * (i+1) - image.shape[1]
    if ignore_fraction_size_ratio * patch_width < rem:
        col_images.append(image[::, image.shape[1] - patch_width: image.shape[1]])
    return col_images

