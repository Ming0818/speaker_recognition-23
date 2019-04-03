import os

root_path = "D:\\vox1_dev_wav\\wav"
for dir_first in os.listdir(root_path):
    for dir_second in os.listdir(os.path.join(root_path, dir_first)):
        for file in os.listdir(os.path.join(root_path, dir_first, dir_second)):
            os.rename(os.path.join(root_path, dir_first, dir_second, file),
                      os.path.join(root_path, dir_first, dir_second + "_" + file))
