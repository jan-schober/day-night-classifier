import json
import os
from shutil import copyfile

json_path = '/home/schober/Day-Night-Classifier/export_train.json'
with open(json_path) as f:
    data = json.load(f)

night_images = data['Day_Images']

dest_path_img = '/home/schober/bdd100k/images/10k/train_day/'
dest_path_label = '/home/schober/bdd100k/labels/sem_seg/masks/train_day/'
source_path_label = '/home/schober/bdd100k/labels/sem_seg/masks/train/'
for file in night_images:
    file_name = file.split('/')[-1]
    dst_img = dest_path_img + file_name

    label_name = file_name.replace('.jpg', '.png')
    label = source_path_label + label_name
    dst_label = dest_path_label + label_name

    copyfile(file, dst_img)
    copyfile(label, dst_label)
