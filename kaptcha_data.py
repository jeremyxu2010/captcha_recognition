import random
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 验证码的可选字符是从kaptcha得到的默认值
captcha_chars = 'abcde2345678gfynmnpwx'

base_dir = '/Volumes/SSDHOME/IdeaProjects/demos'
pics_dir = base_dir + '/pics'
processed_pics_dir = base_dir + '/processed_pics'

char_idx_mappings = {}
idx_char_mappings = {}

for idx, c in enumerate(list(captcha_chars)):
    char_idx_mappings[c] = idx
    idx_char_mappings[idx] = c

IMAGE_HEIGHT = 50
IMAGE_WIDTH = 200
MAX_CAPTCHA = 5
CHAR_SET_LEN = len(captcha_chars)

# 验证码转化为向量
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长%d个字符'%MAX_CAPTCHA)

    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char_idx_mappings[c]
        vector[idx] = 1
    return vector

# 向量转化为验证码
def vec2text(vec):
    text = []
    vec[vec<0.5] = 0
    char_pos = vec.nonzero()[0]
    for i, c in enumerate(char_pos):
        char_idx = c % CHAR_SET_LEN
        text.append(idx_char_mappings[char_idx])
    return ''.join(text)

# 将图片灰度化以减少计算压力
def preprocess_pics():
    for (dirpath, dirnames, filenames) in os.walk(pics_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                with open(pics_dir + '/' + filename, 'rb') as f:
                    image = Image.open(f)
                    image = image.convert('L')
                    with open(processed_pics_dir + '/' + filename, 'wb') as of:
                        image.save(of)



img_idx_filename_mappings = {}
img_idx_text_mappings = {}
img_idxes = []

# 首先遍历目录，根据文件名初始化idx->filename, idx->text的映射，同时初始化idx列表
for (dirpath, dirnames, filenames) in os.walk(processed_pics_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            idx = int(filename[0:filename.index('_')])
            text = filename[int(filename.index('_')+1):int(filename.index('.'))]
            img_idx_filename_mappings[idx] = filename
            img_idx_text_mappings[idx] = text
            img_idxes.append(idx)

# 为避免频繁读入文件，将images及labels缓存起来
sample_idx_image_mappings = {}
sample_idx_label_mappings = {}

# 提供给外部取得一批训练数据的接口
def get_batch_data(batch_size):
    images = []
    labels = []
    target_idxes = random.sample(img_idxes, batch_size)
    for target_idx in target_idxes:
        image = None
        if target_idx in sample_idx_image_mappings:
            image = sample_idx_image_mappings[target_idx]
        else:
            with open(processed_pics_dir + '/' + img_idx_filename_mappings[target_idx], 'rb') as f:
                image = Image.open(f)
                image = np.array(image)/255
            sample_idx_image_mappings[target_idx] = image
        label = None
        if target_idx in sample_idx_label_mappings:
            label = sample_idx_label_mappings[target_idx]
        else:
            label = text2vec(img_idx_text_mappings[target_idx])
            sample_idx_label_mappings[target_idx] = label
        images.append(image)
        labels.append(label)
    x = np.array(images)
    y = np.array(labels)
    return (x, y)

def get_single_image(filename):
    images = []
    with open(filename, 'rb') as f:
        image = Image.open(f)
        image = image.convert('L')
        images.append(np.array(image)/255)
    return np.array(images)



if __name__ == '__main__':
    # vec = text2vec('abcde')
    # print(vec)
    # text = vec2text(vec)
    # print(text)
    #
    # with open(processed_pics_dir + '/320_e32a3.jpg', 'rb') as f:
    #     image = Image.open(f)
    #     image = image.convert('L')
    #     image = np.array(image)
    #     plt.imshow(image)
    #     plt.show()

    preprocess_pics()

    # samples = get_batch_data(1)
    # for (image, label) in samples:
    #     print(image.shape, label.shape)
    #     print(image, label)
    pass