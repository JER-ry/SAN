import os
import cv2
import argparse
import torch
import json
from tqdm import tqdm

from utils import load_config, load_checkpoint
from infer.Backbone import Backbone
from dataset import Words

os.chdir(os.getcwd())

parser = argparse.ArgumentParser(description="Spatial channel attention")
parser.add_argument("--config", default="copy.yaml", type=str, help="配置文件路径")
parser.add_argument(
    "--image_path",
    default="/home/jerry/ocr-for-edu/math-ocr/crop/crop_images",
    type=str,
    help="测试image路径",
)
args = parser.parse_args()

if not args.config:
    print("请提供config yaml路径！")
    exit(-1)

"""加载config文件"""
params = load_config(args.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params["device"] = device

words = Words(params["word_path"])
params["word_num"] = len(words)
params["struct_num"] = 7
params["words"] = words

model = Backbone(params)
model = model.to(device)

load_checkpoint(model, None, params["checkpoint"])

model.eval()

word_right, node_right, exp_right, length, cal_num = 0, 0, 0, 0, 0


def convert(nodeid, gtd_list):
    isparent = False
    child_list = []
    for i in range(len(gtd_list)):
        if gtd_list[i][2] == nodeid:
            isparent = True
            child_list.append([gtd_list[i][0], gtd_list[i][1], gtd_list[i][3]])
    if not isparent:
        return [gtd_list[nodeid][0]]
    else:
        if gtd_list[nodeid][0] == "\\frac":
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == "Above":
                    return_string += ["{"] + convert(child_list[i][1], gtd_list) + ["}"]
            for i in range(len(child_list)):
                if child_list[i][2] == "Below":
                    return_string += ["{"] + convert(child_list[i][1], gtd_list) + ["}"]
            for i in range(len(child_list)):
                if child_list[i][2] == "Right":
                    return_string += convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2] not in ["Right", "Above", "Below"]:
                    return_string += ["illegal"]
        else:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] in ["l_sup"]:
                    return_string += ["["] + convert(child_list[i][1], gtd_list) + ["]"]
            for i in range(len(child_list)):
                if child_list[i][2] == "Inside":
                    return_string += ["{"] + convert(child_list[i][1], gtd_list) + ["}"]
            for i in range(len(child_list)):
                if child_list[i][2] in ["Sub", "Below"]:
                    return_string += (
                        ["_", "{"] + convert(child_list[i][1], gtd_list) + ["}"]
                    )
            for i in range(len(child_list)):
                if child_list[i][2] in ["Sup", "Above"]:
                    return_string += (
                        ["^", "{"] + convert(child_list[i][1], gtd_list) + ["}"]
                    )
            for i in range(len(child_list)):
                if child_list[i][2] in ["Right"]:
                    return_string += convert(child_list[i][1], gtd_list)
        return return_string


with torch.no_grad():
    result = {}
    for item in tqdm(os.listdir(args.image_path)):
        if item.endswith(".png") or item.endswith(".jpg") or item.endswith(".bmp"):
            img = cv2.imread(args.image_path + "/" + item)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = torch.Tensor(img) / 255
            image = image.unsqueeze(0).unsqueeze(0)

            image_mask = torch.ones(image.shape)
            image, image_mask = image.to(device), image_mask.to(device)

            prediction = model(image, image_mask)

            latex_list = convert(1, prediction)
            latex_string = " ".join(latex_list)
            result[item] = latex_string

with open("result.json", "w") as f:
    json.dump(result, f, ensure_ascii=False)
