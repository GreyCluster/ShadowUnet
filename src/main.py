import cv2

from Unet import *
from MyDataset import *
from Loss import *

import torch
from torch.utils.data import DataLoader
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
import os
import json


def CHECK_PATH(path_):
    if not os.path.exists(path_):
        os.makedirs(path_)
        print("Create folder:{}".format(path_))
    return path_


def validation(writer, net, device, validation_path, predict_path, epoch):
    image_path = os.path.join(validation_path, "images")
    conditioning_path = os.path.join(validation_path, "conditioning_images")

    trans = transforms.ToTensor()

    validation_images = list()
    for root, dirs, files in os.walk(image_path):
        for file in files:
            file_name, _ = os.path.splitext(file)
            validation_images.append(file_name)

    step = 0
    for image_name in validation_images:
        conditioning_image0 = trans(
            cv2.imread(os.path.join(conditioning_path, image_name + "_0.png"), cv2.IMREAD_UNCHANGED))
        conditioning_image1 = trans(
            cv2.imread(os.path.join(conditioning_path, image_name + "_1.png"), cv2.IMREAD_UNCHANGED))
        conditioning_image2 = trans(
            cv2.imread(os.path.join(conditioning_path, image_name + "_2.png"), cv2.IMREAD_UNCHANGED))
        conditioning_image3 = trans(
            cv2.imread(os.path.join(conditioning_path, image_name + "_3.png"), cv2.IMREAD_UNCHANGED))
        conditioning_image4 = trans(
            cv2.imread(os.path.join(conditioning_path, image_name + "_d.exr"), cv2.IMREAD_UNCHANGED)[:,:,2])
        conditioning_image = torch.cat([conditioning_image0,
                                        conditioning_image1,
                                        conditioning_image2,
                                        conditioning_image3,
                                        conditioning_image4], dim=0).to(device)

        conditioning_image = torch.unsqueeze(conditioning_image, dim=0)
        result = net(conditioning_image)
        result = torch.squeeze(result, dim=0)

        # tensorboard
        writer.add_image("validation" + str(step), result, epoch)

        # save to predict folder
        CHECK_PATH(predict_path)
        save_path = os.path.join(predict_path, "epoch{}".format(epoch) + image_name + ".png")
        save_image(result, save_path)

        step += 1


def train_net(net, device, config, epochs, batch_size, lr, last_epoch=0):
    # train
    train_path           = config["train"]["train_path"]
    data_path            = os.path.join(train_path, "shadow")
    conditioning_path    = os.path.join(data_path, "conditioning_images")
    validation_data_path = os.path.join(data_path, "validation_images")

    images_path          = os.path.join(data_path, "images")
    images_train_path = os.path.join(images_path, "train")
    images_val_path = os.path.join(images_path, "val")

    # models
    folder_name = config["train"]["folder_name"]
    model_name  = folder_name

    folder_path          = CHECK_PATH(os.path.join(train_path, "models", folder_name))
    log_path             = CHECK_PATH(os.path.join(folder_path, "logs"))
    model_path           = CHECK_PATH(os.path.join(folder_path, "Unet", ))
    validata_output_path = CHECK_PATH(os.path.join(folder_path, "Validate"))
    predict_path         = CHECK_PATH(os.path.join(validata_output_path, "Predict"))

    # train\validation\test
    train_list = []
    for root, dirs, files in os.walk(images_train_path):
        for file in files:
            name, _ = os.path.splitext(file)
            train_list.append([os.path.join(conditioning_path, name), os.path.join(images_train_path, name)])
    val_list = []
    for root, dirs, files in os.walk(images_val_path):
        for file in files:
            name, _ = os.path.splitext(file)
            val_list.append([os.path.join(conditioning_path, name), os.path.join(images_val_path, name)])

    # train data
    train_data = TrainData(data_list=train_list)
    val_data = TrainData(data_list=val_list)
    # data loader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # loss
    ls_fn = VGG(device)
    # optim
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    writer = SummaryWriter(log_path)
    total_train_step = 0
    min_loss = 1e5
    for epoch in range(last_epoch, epochs):
        print("-"*25 + "epoch% d train start!!!" % epoch + "-"*25)
        # train pass
        train_total_loss = 0
        train_step = 0
        train_val_loss = 0
        train_val_step = 0
        for condition, img in train_loader:
            net.train()
            optimizer.zero_grad()

            condition = condition.to(device=device)
            img = img.to(device=device)

            pred = net(condition)

            loss = ls_fn(img, pred)
            train_total_loss += loss
            train_step += 1
            train_val_loss += loss
            train_val_step += 1
            total_train_step += 1

            loss.backward()
            optimizer.step()

            # tensorboard
            if total_train_step % validation_step == 0:
                train_avg_loss = train_total_loss / train_step
                print("epoch:%d step:%d loss=%f" % (epoch, total_train_step, train_avg_loss))
                train_total_loss = 0
                train_step = 0
                writer.add_scalar(tag="loss", scalar_value=train_avg_loss, global_step=total_train_step)

        # validate image for each epoch
        validation(writer, net, device, validation_data_path, predict_path, epoch)

        # save model
        if epoch % 4 == 0:
            save_path = os.path.join(model_path, model_name + "_epoch" + str(epoch) + ".pth")
            torch.save(net, save_path)

        # validation pass
        print("epoch %d validation start!!!" % epoch)
        val_total_loss = 0
        net.eval()
        val_count = 0
        with torch.no_grad():
            for condition, img in val_loader:
                condition = condition.to(device=device)
                img = img.to(device=device)

                pred = net(condition)

                if img.size(1) == 1 and pred.size(1) == 1:
                    loss = ls_fn(img.repeat(1, 3, 1, 1), pred.repeat(1, 3, 1, 1))
                else:
                    loss = ls_fn(img, pred)
                val_total_loss += loss
                val_count += 1

        val_avg_loss = val_total_loss / val_count
        print("epoch:%d val_avg_loss=%f" % (epoch, val_avg_loss))
        writer.add_scalars(main_tag="train/val loss",
                           tag_scalar_dict={"val": val_avg_loss,
                                            "train": train_val_loss / train_val_step},
                           global_step=epoch)
        if val_avg_loss < min_loss:
            min_loss = val_avg_loss

            save_path = os.path.join(model_path, model_name + "_best.pth")
            torch.save(net, save_path)

    writer.close()


if __name__ == '__main__':
    # super parameters
    with open("../parameter.json", "r") as file:
        config = json.load(file)

    epoch = config["parameters"]["epoch"]
    learning_rate = config["parameters"]["learning_rate"]
    batch_size = config["parameters"]["batch_size"]
    validation_step = config["parameters"]["validation_step"]

    # main
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = Unet_avgpool_add_level012345_depth()

    last_epoch = config["train"]["last_epoch"]
    if last_epoch==0:
        net.to(device=device)
    else:
        model_path = os.path.join(config["train"]["train_path"], "models", config["train"]["folder_name"], "Unet",
                                  config["train"]["folder_name"] + "_epoch{}.pth".format(last_epoch))
        net = torch.load(model_path)

    train_net(net, device=device, config=config, epochs=epoch, batch_size=batch_size, lr=learning_rate, last_epoch=last_epoch)
