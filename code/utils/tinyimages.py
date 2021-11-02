import glob
import os
from shutil import move
from os import rmdir

target_folder = r"E:\Task\Dataset\tiny-imagenet-200\val"

val_dict = {}
with open(r'E:\Task\Dataset\tiny-imagenet-200\val\val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]

paths = glob.glob(r'E:\Task\Dataset\tiny-imagenet-200\val\images\*')
for path in paths:
    file = path.split('\\')[-1]
    folder = val_dict[file]
    if not os.path.exists(os.path.join(target_folder , str(folder))):
        os.mkdir(os.path.join(target_folder , str(folder)))
        # os.mkdir(target_folder + str(folder) + '/images')

for path in paths:
    file = path.split('\\')[-1]
    folder = val_dict[file]
    dest = os.path.join(target_folder , str(folder) , str(file))
    move(path, dest)

rmdir(r'E:\Task\Dataset\tiny-imagenet-200\val\images')
