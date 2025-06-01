import os
import shutil

# 源文件夹路径
source_folder = "UIEB/reference-890"
# 目标文件夹路径
destination_folder = "UIEB/test"
# test.txt 文件路径
txt_file = "UIEB/test.txt"

# 读取 test.txt 文件
with open(txt_file, "r") as file:
    lines = file.readlines()

# 遍历每一行，复制对应的图片到目标文件夹
for line in lines:
    # 去除行末尾的换行符
    line = line.strip()
    # 构建图片的完整路径
    image_path = os.path.join(source_folder, line)
    destination_path = os.path.join(destination_folder, line)
    # 检查图片是否存在
    if os.path.exists(image_path):
        # 复制图片到目标文件夹中
        shutil.copy(image_path, destination_path)
        print(f"复制图片 {line} 成功")
    else:
        print(f"图片 {line} 不存在")


# 源文件夹路径
source_folder = "LSUI/input"
# 目标文件夹路径
destination_folder = "LSUI/test"
# test.txt 文件路径
txt_file = "LSUI/test.txt"

# 读取 test.txt 文件
with open(txt_file, "r") as file:
    lines = file.readlines()

# 遍历每一行，复制对应的图片到目标文件夹
for line in lines:
    # 去除行末尾的换行符
    line = line.strip()
    # 构建图片的完整路径
    image_path = os.path.join(source_folder, line)
    destination_path = os.path.join(destination_folder, line)
    # 检查图片是否存在
    if os.path.exists(image_path):
        # 复制图片到目标文件夹中
        shutil.copy(image_path, destination_path)
        print(f"复制图片 {line} 成功")
    else:
        print(f"图片 {line} 不存在")



# 源文件夹路径
source_folder = "UIID/input"
# 目标文件夹路径
destination_folder = "UIID/test"
# test.txt 文件路径
txt_file = "UIID/test.txt"

# 读取 test.txt 文件
with open(txt_file, "r") as file:
    lines = file.readlines()

# 遍历每一行，复制对应的图片到目标文件夹
for line in lines:
    # 去除行末尾的换行符
    line = line.strip()
    # 构建图片的完整路径
    image_path = os.path.join(source_folder, line)
    destination_path = os.path.join(destination_folder, line)
    # 检查图片是否存在
    if os.path.exists(image_path):
        # 复制图片到目标文件夹中
        shutil.copy(image_path, destination_path)
        print(f"复制图片 {line} 成功")
    else:
        print(f"图片 {line} 不存在")