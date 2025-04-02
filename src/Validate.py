from MyDataset import *
from Unet import *
import Unet
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import time
import sys
import json


def CHECK_PATH(path_):
    if not os.path.exists(path_):
        os.makedirs(path_)
        print("Create folder:{}".format(path_))
    return path_

sys.modules["Unet"] = Unet

with open("../parameter.json", "r") as file:
    config = json.load(file)

# find model
train_path = config["train"]["train_path"]
folder_name = config["train"]["folder_name"]
model_name = folder_name
model_path = os.path.join(train_path, "models", folder_name, "Unet", model_name+"_epoch124.pth")
# model_path = os.path.join(train_path, "models", folder_name, "Unet", model_name+"_best.pth")

# find output path
validate_manual_path = CHECK_PATH(os.path.join(train_path, "models", folder_name, "Validate", "Validate_manual"))

# find data path
validate_data_path = os.path.join(train_path, "shadow\\validation_images")

net = torch.load(model_path)

test_data = TestData(validate_data_path)
test_loader = DataLoader(test_data, batch_size=1)

total_num = 0

mse_fn = torch.nn.MSELoss()
json_file = list()

net.eval()
for conditioning_image, image, fileName in test_loader:
    conditioning_image = conditioning_image.to(device='cuda', dtype=torch.float32)
    image = image.to(device='cuda', dtype=torch.float32)

    start_time = time.perf_counter()
    pred = net(conditioning_image)
    end_time = time.perf_counter()

    print("%f ms" % ((end_time - start_time) * 1000))

    mse_loss = mse_fn(image, pred)
    json_file.append([fileName[0], mse_loss.item()])

    save_path = os.path.join(validate_manual_path, fileName[0] + ".png")
    save_image(pred[0], save_path)


with open(os.path.join(validate_manual_path, "mse.json"), "w") as f:
    json.dump(json_file, f, indent=2)
