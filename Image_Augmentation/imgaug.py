import numpy as np
import imgaug.augmenters as iaa
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./data/sample_data_01/train/snow/0830.jpg')

image_arr = np.array(image)

images = [image_arr, image_arr, image_arr, image_arr] # images[0].shape: (518, 920, 3)

#

rotate = iaa.Affine(rotate=(-25, 25))

images_rotate = rotate(images=images) # images_rotate[0].shape: (518, 920, 3)

plt.figure(figsize=(12, 12))
plt.imshow(np.hstack(images_rotate)) # ~ np.hstack([(518, 920, 3), (518, 920, 3), (518, 920, 3), (518, 920, 3)])
plt.show()

#

crop = iaa.Crop(percent=(0, 0.2))

images_crop = crop(images=images)

plt.figure(figsize=(12, 12))
plt.imshow(np.hstack(images_crop))
plt.show()

#

rotate_crop = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.Crop(percent=(0, 0.2))
])

images_rotate_crop = rotate_crop(images=images)

plt.figure(figsize=(12, 12))
plt.imshow(np.hstack(images_rotate_crop))
plt.show()

#

crop_rotate = iaa.Sequential([
    iaa.Crop(percent=(0, 0.2)),
    iaa.Affine(rotate=(-25, 25))
], random_order=True)

images_crop_rotate = crop_rotate(images=images)

plt.figure(figsize=(12, 12))
plt.imshow(np.hstack(images_crop_rotate))
plt.show()

#

other2 = iaa.Sequential([
    iaa.Sometimes(0.6, iaa.AddToSaturation((-50, 50))),
    iaa.Sometimes(0.2, iaa.Grayscale(alpha=(0.0, 1.0)))
])

images_other2 = other2(images=images)

plt.figure(figsize=(12, 12))
plt.imshow(np.hstack(images_other2))
plt.show()