import os
import shutil


# 读取test.txt中的文件名
with open("datasets/LSUI/test.txt", 'r') as f:
    files_to_keep = [line.strip() for line in f]

# 遍历文件夹中的所有文件
folder_path = 'results/01_UIEC2Net_LSUI_UIID_test/visualization/LSUI'
for filename in os.listdir(folder_path):
    if filename not in files_to_keep:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")


# 读取test.txt中的文件名
with open("datasets/UIID/test.txt", 'r') as f:
    files_to_keep = [line.strip() for line in f]

# 遍历文件夹中的所有文件
folder_path = 'results/01_UIEC2Net_LSUI_UIID_test/visualization/UIID'
for filename in os.listdir(folder_path):
    if filename not in files_to_keep:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")