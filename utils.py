import numpy as np
import scipy.misc

# Center crop as much as crop_size, then resize to 64
def transform(image, crop_size, resize_w=64):
    height, width = image.shape[:2]
    j = int((height-crop_size)/2)
    i = int((width-crop_size)/2)
    cropped_image = scipy.misc.imresize(image[j:j+crop_size, i:i+crop_size], [resize_w, resize_w])
    # Make image array value between -1 and 1
    return np.asarray(cropped_image)/127.5  - 1

def get_image(image_file, image_size, resize_w):
    img = scipy.misc.imread(image_file)
    return transform(img, image_size, resize_w)

def save_image(imgs, size, path):
    print('Image merging')
    # imgs : [batch, height, width, channel], size : how to arrange images
    height, width = imgs.shape[1], imgs.shape[2]
    merged_image = np.zeros([size[0]*height, size[1]*width, imgs.shape[3]])
    for image_index, image in enumerate(imgs):
        j = image_index % size[1]
        i = image_index // size[1]
        merged_image[i:i+height, j:j+width, :]= image
    merged_image += 1
    merged_image *=127.5
    merged_image = np.clip(merged_image, 0, 255).astype(np.uint8)

    scipy.misc.imsave(path, merged_image)

if __name__ == "__main__":
    print('a')
