# 导入需要的模块
from glob import glob

import tqdm
from PIL import Image
import os

# 图片路径
# 使用 glob模块 获得文件夹内所有jpg图像
img_path = glob("/media/weifeng/B2AF92FAB69D5E90/Experiments/Students/2021/zengwf/animegan2-train/image/*.jpg")
# 存储（输出）路径
path_save = "./redpi_results/"

for file in tqdm.tqdm(img_path):
    name = os.path.join(path_save, os.path.basename(file))
    im = Image.open(file)
    # im.thumbnail((720,1280))
    reim = im.resize((256, 256))
    print(im.format, reim.size, reim.mode)
    reim.save(name, im.format)