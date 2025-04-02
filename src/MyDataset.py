import os

import torch
from PIL import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class TrainData(Dataset):
    def __init__(self, data_list):
        self.transform = transforms.ToTensor()

        self.sample_list = data_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, item):
        conditioning_image_name, image_name = self.sample_list[item]
        conditioning_image_name0 = conditioning_image_name + "_0.png"
        conditioning_image_name1 = conditioning_image_name + "_1.png"
        conditioning_image_name2 = conditioning_image_name + "_2.png"
        conditioning_image_name3 = conditioning_image_name + "_3.png"
        conditioning_image_name4 = conditioning_image_name + "_d.exr"
        conditioning_image0 = self.transform(cv2.imread(conditioning_image_name0, cv2.IMREAD_UNCHANGED))
        conditioning_image1 = self.transform(cv2.imread(conditioning_image_name1, cv2.IMREAD_UNCHANGED))
        conditioning_image2 = self.transform(cv2.imread(conditioning_image_name2, cv2.IMREAD_UNCHANGED))
        conditioning_image3 = self.transform(cv2.imread(conditioning_image_name3, cv2.IMREAD_UNCHANGED))
        conditioning_image4 = self.transform(cv2.imread(conditioning_image_name4, cv2.IMREAD_UNCHANGED)[:,:,2])
        conditioning_image = torch.cat(
            [conditioning_image0, conditioning_image1, conditioning_image2, conditioning_image3, conditioning_image4], dim=0)

        image_name = image_name + ".png"
        image = self.transform(cv2.imread(image_name, cv2.IMREAD_UNCHANGED)[:, :, 2])

        return conditioning_image, image


class TestData(Dataset):
    def __init__(self, data_path):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.data_path = data_path
        self.test_list = list()
        image_path = os.path.join(data_path, "images")
        for root, dirs, files in os.walk(image_path):
            for file in files:
                file_name = os.path.splitext(file)[0]
                self.test_list.append([file_name])

    def __len__(self):
        return len(self.test_list)

    def __getitem__(self, index):
        file_name = self.test_list[index]
        conditioning_image_path0 = os.path.join(self.data_path, "conditioning_images", file_name[0] + "_0.png")
        conditioning_image_path1 = os.path.join(self.data_path, "conditioning_images", file_name[0] + "_1.png")
        conditioning_image_path2 = os.path.join(self.data_path, "conditioning_images", file_name[0] + "_2.png")
        conditioning_image_path3 = os.path.join(self.data_path, "conditioning_images", file_name[0] + "_3.png")
        conditioning_image_path4 = os.path.join(self.data_path, "conditioning_images", file_name[0] + "_d.exr")

        conditioning_image0 = self.transform(cv2.imread(conditioning_image_path0, cv2.IMREAD_UNCHANGED))
        conditioning_image1 = self.transform(cv2.imread(conditioning_image_path1, cv2.IMREAD_UNCHANGED))
        conditioning_image2 = self.transform(cv2.imread(conditioning_image_path2, cv2.IMREAD_UNCHANGED))
        conditioning_image3 = self.transform(cv2.imread(conditioning_image_path3, cv2.IMREAD_UNCHANGED))
        conditioning_image4 = self.transform(cv2.imread(conditioning_image_path4, cv2.IMREAD_UNCHANGED)[:,:,2])

        conditioning_image = torch.cat(
            [conditioning_image0, conditioning_image1, conditioning_image2, conditioning_image3,conditioning_image4], dim=0)

        image_path = os.path.join(self.data_path, "images", file_name[0] + ".png")
        image = self.transform(cv2.imread(image_path, cv2.IMREAD_UNCHANGED))

        return conditioning_image, image, file_name[0]


if __name__ == '__main__':
    exr_path = r"D:\Projects\AI\Data\Shadow_4input\Train\shadow\conditioning_images\Bistro_Width0Camera0Light0_d.exr"
    image = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    image = image[:,:,2]
    print(image)
