import os
import random

random.seed(1)

# # split UIEB dataset
# folder_path = "./UIEB/raw-890/"

# image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")]

# random.shuffle(image_files)

# train_count = 800 
# test_count = len(image_files) - train_count
# print(train_count, test_count)

# with open("./UIEB/train.txt", "w") as train_file:
#     for i in range(train_count):
#         train_file.write(image_files[i] + "\n")

# with open("./UIEB/test.txt", "w") as test_file:
#     for i in range(train_count, train_count + test_count):
#         test_file.write(image_files[i] + "\n")


# # split LSUI dataset
# folder_path = "./LSUI/input/"

# image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")]

# random.shuffle(image_files)

# train_count = 3879 
# test_count = len(image_files) - train_count
# print(train_count, test_count)

# with open("./LSUI/train.txt", "w") as train_file:
#     for i in range(train_count):
#         train_file.write(image_files[i] + "\n")

# with open("./LSUI/test.txt", "w") as test_file:
#     for i in range(train_count, train_count + test_count):
#         test_file.write(image_files[i] + "\n")


# split UIID dataset
folder_path = "./UIID/input/"

image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")]

random.shuffle(image_files)

train_count = 3486-350 
test_count = len(image_files) - train_count
print(train_count, test_count)

with open("./UIID/train.txt", "w") as train_file:
    for i in range(train_count):
        train_file.write(image_files[i] + "\n")

with open("./UIID/test.txt", "w") as test_file:
    for i in range(train_count, train_count + test_count):
        test_file.write(image_files[i] + "\n")