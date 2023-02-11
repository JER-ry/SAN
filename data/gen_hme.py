#%% import libs
import re
import os
import random
import glob
import pickle as pkl
import cv2
from tqdm import tqdm
from pathlib import Path

random.seed(0)

# out_path = "/home/jerry/ocr-for-edu/math-ocr/SAN/data/hmini"
out_path = "/home/jerry/ocr-for-edu/math-ocr/SAN/data/hme"
word_ori_path = "/home/jerry/ocr-for-edu/math-ocr/SAN/data/word_ori.txt"
data_path = "/home/jerry/ocr-for-edu/math-ocr/HME100K"

excludes1 = [
    "\\xrightarrow",
    "\\dot",
    "\\textcircled",
    "\\widehat",
    "\\overrightarrow",
    "\\mathop",
    "\\xlongequal",
    "\\boxed",
    "\\ddot",
    "\\overline",
    "甲",
    "次",
    "足",
    "种",
    "面",
    "即",
    "将",
    "自",
    "然",
    "明",
    "元",
    "和",
    "丙",
    "入",
    "式",
    "的",
    "满",
    "所",
    "时",
    "以",
    "袋",
    "把",
    "积",
    "大",
    "是",
    "方",
    "乙",
    "解",
    "个",
    "本",
    "周",
    "可",
    "该",
    "倍",
    "原",
    "法",
    "妈",
    "或",
    "组",
    "求",
    "岁",
    "公",
    "行",
    "第",
    "年",
    "代",
    "因",
    "页",
    "最",
    "数",
    "人",
    "天",
    "得",
    "能",
    "分",
    "为",
    "小",
    "项",
    "负",
    "答",
]
excludes2 = ["\t. {"]

#%% def gen(label, out)
position = set(["^", "_"])
math = set(["\\frac", "\\sqrt"])


def gen(label, out):
    class Tree:
        def __init__(self, label, parent_label="None", id=0, parent_id=0, op="none"):
            self.children = []
            self.label = label
            self.id = id
            self.parent_id = parent_id
            self.parent_label = parent_label
            self.op = op

    def convert(root: Tree, f):
        if root.tag == "N-T":
            f.write(
                f"{root.id}\t{root.label}\t{root.parent_id}\t{root.parent_label}\t{root.tag}\n"
            )
            for child in root.children:
                convert(child, f)
        else:
            f.write(
                f"{root.id}\t{root.label}\t{root.parent_id}\t{root.parent_label}\t{root.tag}\n"
            )

    with open(label) as f:
        lines = f.readlines()
        num = 0
        for line in tqdm(lines):
            # line = 'RIT_2014_178.jpg x ^ { \\frac { p } { q } } = \sqrt [ q ] { x ^ { p } } = \sqrt [ q ] { x ^ { p } }'
            name, *words = line.split()
            name = name.split(".")[0]

            parents = []
            root = Tree("root", parent_label="root", parent_id=-1)

            struct_list = ["\\frac", "\\sqrt"]

            labels = []
            id = 1
            parents = [Tree("<sos>", id=0)]
            parent = Tree("<sos>", id=0)

            for i in range(len(words)):
                a = words[i]
                if a == "\\limits":
                    continue
                if i == 0 and words[i] in ["_", "^", "{", "}"]:
                    print(name)
                    break
                elif words[i] == "{":
                    if words[i - 1] == "\\frac":
                        labels.append([id, "struct", parent.id, parent.label])
                        parents.append(Tree("\\frac", id=parent.id, op="above"))
                        id += 1
                        parent = Tree("above", id=parents[-1].id + 1)
                    elif (
                        words[i - 1] == "}"
                        and parents[-1].label == "\\frac"
                        and parents[-1].op == "above"
                    ):
                        parent = Tree("below", id=parents[-1].id + 1)
                        parents[-1].op = "below"
                    elif words[i - 1] == "\\sqrt":
                        labels.append([id, "struct", parent.id, "\\sqrt"])
                        parents.append(Tree("\\sqrt", id=parent.id))
                        parent = Tree("inside", id=id)
                        id += 1
                    elif words[i - 1] == "]" and parents[-1].label == "\\sqrt":
                        parent = Tree("inside", id=parents[-1].id + 1)
                    elif words[i - 1] == "^":
                        if words[i - 2] != "}":
                            if words[i - 2] == "\\sum":
                                labels.append([id, "struct", parent.id, parent.label])
                                parents.append(Tree("\\sum", id=parent.id))
                                parent = Tree("above", id=id)
                                id += 1
                            else:
                                labels.append([id, "struct", parent.id, parent.label])
                                parents.append(Tree(words[i - 2], id=parent.id))
                                parent = Tree("sup", id=id)
                                id += 1
                        else:
                            # labels.append([id, 'struct', parents[-1].id, parents[-1].label])
                            if parents[-1].label == "\\sum":
                                parent = Tree("above", id=parents[-1].id + 1)
                            else:
                                parent = Tree("sup", id=parents[-1].id + 1)
                        # id += 1
                    elif words[i - 1] == "_":
                        if words[i - 2] != "}":
                            if words[i - 2] == "\\sum":
                                labels.append([id, "struct", parent.id, parent.label])
                                parents.append(Tree("\\sum", id=parent.id))
                                parent = Tree("below", id=id)
                                id += 1
                            else:
                                labels.append([id, "struct", parent.id, parent.label])
                                parents.append(Tree(words[i - 2], id=parent.id))
                                parent = Tree("sub", id=id)
                                id += 1
                        else:
                            # labels.append([id, 'struct', parents[-1].id, parents[-1].label])
                            if parents[-1].label == "\\sum":
                                parent = Tree("below", id=parents[-1].id + 1)
                            else:
                                parent = Tree("above", id=parents[-1].id + 1)
                        # id += 1
                    else:
                        print(f"unknown word {words[i - 1]} before {{", name, i)
                elif words[i] == "[" and words[i - 1] == "\\sqrt":
                    labels.append([id, "struct", parent.id, "\\sqrt"])
                    parents.append(Tree("\\sqrt", id=parent.id))
                    parent = Tree("L-sup", id=id)
                    id += 1
                elif words[i] == "]" and parents[-1].label == "\\sqrt":
                    labels.append([id, "<eos>", parent.id, parent.label])
                    id += 1
                elif words[i] == "}":
                    if words[i - 1] != "}":
                        labels.append([id, "<eos>", parent.id, parent.label])
                        id += 1
                    if (
                        i + 1 < len(words)
                        and words[i + 1] == "{"
                        and parents[-1].label == "\\frac"
                        and parents[-1].op == "above"
                    ):
                        continue
                    if i + 1 < len(words) and words[i + 1] in ["_", "^"]:
                        continue
                    elif i + 1 < len(words) and words[i + 1] != "}":
                        parent = Tree("right", id=parents[-1].id + 1)
                    parents.pop()
                else:
                    if words[i] in ["^", "_"]:
                        continue
                    labels.append([id, words[i], parent.id, parent.label])
                    parent = Tree(words[i], id=id)
                    id += 1

            parent_dict = {0: []}
            for i in range(len(labels)):
                parent_dict[i + 1] = []
                parent_dict[labels[i][2]].append(labels[i][3])

            with open(f"{out}/{name}.txt", "w") as f:
                for line in labels:
                    id, label, parent_id, parent_label = line
                    if label != "struct":
                        f.write(
                            f"{id}\t{label}\t{parent_id}\t{parent_label}\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n"
                        )
                    else:
                        tem = f"{id}\t{label}\t{parent_id}\t{parent_label}"
                        tem = (
                            tem + "\tabove"
                            if "above" in parent_dict[id]
                            else tem + "\tNone"
                        )
                        tem = (
                            tem + "\tbelow"
                            if "below" in parent_dict[id]
                            else tem + "\tNone"
                        )
                        tem = (
                            tem + "\tsub"
                            if "sub" in parent_dict[id]
                            else tem + "\tNone"
                        )
                        tem = (
                            tem + "\tsup"
                            if "sup" in parent_dict[id]
                            else tem + "\tNone"
                        )
                        tem = (
                            tem + "\tL-sup"
                            if "L-sup" in parent_dict[id]
                            else tem + "\tNone"
                        )
                        tem = (
                            tem + "\tinside"
                            if "inside" in parent_dict[id]
                            else tem + "\tNone"
                        )
                        tem = (
                            tem + "\tright"
                            if "right" in parent_dict[id]
                            else tem + "\tNone"
                        )
                        f.write(tem + "\n")
                if label != "<eos>":
                    f.write(
                        f"{id+1}\t<eos>\t{id}\t{label}\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n"
                    )


#%% train_hyb
os.makedirs(f"{out_path}/train_hyb", exist_ok=True)

with open(f"{data_path}/train/train_labels.txt", "r") as f, open(
    f"{out_path}/train_labels.txt", "w"
) as f2:
    f_list = [
        i
        for i in f.readlines()
        if all(
            re.compile(".+\t").sub("", j).replace("\n", "") not in excludes1
            for j in i.split(" ")
        )
        and all(not re.compile(j).search(i) for j in excludes2)
    ]
    # f2.write("".join(random.sample(f_list, int(len(f_list) * 0.08))))
    f2.write("".join(f_list))

gen(
    f"{out_path}/train_labels.txt",
    f"{out_path}/train_hyb",
)

#%% test_hyb
os.makedirs(f"{out_path}/test_hyb", exist_ok=True)

with open(f"{data_path}/test/test_labels.txt", "r") as f, open(
    f"{out_path}/test_labels.txt", "w"
) as f2:
    f_list = [
        i
        for i in f.readlines()
        if all(
            re.compile(".+\t").sub("", j).replace("\n", "") not in excludes1
            for j in i.split(" ")
        )
        and all(not re.compile(j).search(i) for j in excludes2)
    ]
    # f2.write("".join(random.sample(f_list, int(len(f_list) * 0.04))))
    f2.write("".join(f_list))

gen(
    f"{out_path}/test_labels.txt",
    f"{out_path}/test_hyb",
)

#%% def gen2(image_path, image_out, label_path, label_out)
def gen2(image_path, image_out, label_path, label_out):
    labels = [
        os.path.basename(item).replace(".txt", "")
        for item in glob.glob(os.path.join(label_path, "*.txt"))
    ]
    image_dict = {}
    label_dict = {}
    for item in tqdm(labels):
        img = cv2.imread(os.path.join(image_path, item + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_dict[item] = img
        with open(os.path.join(label_path, item + ".txt")) as f:
            lines = f.readlines()
        label_dict[item] = lines
    with open(image_out, "wb") as f, open(label_out, "wb") as f2:
        pkl.dump(image_dict, f)
        pkl.dump(label_dict, f2)


#%% train pkls
gen2(
    f"{data_path}/train/train_images",
    f"{out_path}/train_image.pkl",
    f"{out_path}/train_hyb",
    f"{out_path}/train_label.pkl",
)

#%% test pkls
gen2(
    f"{data_path}/test/test_images",
    f"{out_path}/test_image.pkl",
    f"{out_path}/test_hyb",
    f"{out_path}/test_label.pkl",
)

#%% word.txt
label_path1 = f"{out_path}/train_hyb"
label_path2 = f"{out_path}/test_hyb"
labels = glob.glob(os.path.join(label_path1, "*.txt")) + glob.glob(
    os.path.join(label_path2, "*.txt")
)
words_dict = set(["<eos>", "<sos>", "struct"])
with open(f"{out_path}/word_temp.txt", "w") as writer:
    writer.write("<eos>\n<sos>\nstruct\n")
    i = 3
    for item in tqdm(labels):
        with open(item) as f:
            lines = f.readlines()
        for line in lines:
            cid, c, pid, p, *r = line.strip().split()
            if c not in words_dict:
                words_dict.add(c)
                writer.write(f"{c}\n")
                i += 1
    writer.write("above\nbelow\nsub\nsup\nL-sup\ninside\nright")
with open(word_ori_path, "r") as ori, open(
    f"{out_path}/word_temp.txt", "r"
) as new, open(f"{out_path}/word.txt", "w") as new_fixed:
    ori_list = [i.replace("\n", "") for i in ori.readlines()]
    new_list = [i.replace("\n", "") for i in new.readlines()]
    new_fixed_list = ori_list
    for i in new_list:
        if i not in ori_list:
            new_fixed_list.append(i)
    new_fixed.write("\n".join(new_fixed_list))

# %%
