import cv2
import numpy as np

image_path = r"D:\Projects\AI\Data\Shadow_4input\Train\shadow\validation_images\images\Bistro_Width4Camera0Light1.png"

image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

width, height, _ = image.shape

def ProcessImage(l):
    new_image = np.zeros((width, height, 1), dtype=np.uint8)

    gap = 256 / l

    for i in range(width):
        for j in range(height):
            value = image[i, j, 0]
            rate = (int)(value / gap)
            new_image[i, j, 0] = gap * rate

    cv2.imwrite("new_image{}.png".format(l), new_image)


R = [16, 32, 64, 128]
for i in R:
    ProcessImage(i)
