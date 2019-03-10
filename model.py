from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.layers.core import Reshape, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, concatenate
from keras.models import Model

import kaptcha_data

model = Sequential()

# 首先对输入数据reshape一下，因为输入的数据是(-1, kaptcha_data.IMAGE_HEIGHT, kaptcha_data.IMAGE_WIDTH), 要把它变为(-1, kaptcha_data.IMAGE_HEIGHT, kaptcha_data.IMAGE_WIDTH, 1)这样才能方便后面卷积层处理
model.add(InputLayer(input_shape=(kaptcha_data.IMAGE_HEIGHT, kaptcha_data.IMAGE_WIDTH)))
model.add(Reshape((kaptcha_data.IMAGE_HEIGHT, kaptcha_data.IMAGE_WIDTH, 1)))

# 四组卷积逻辑，每组包括两个卷积层及一个池化层
for i in range(4):
    model.add(Conv2D(4*2**i, 3, strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(8*2**i, 3, strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

# 马上要接上全连接层，要将数据展平
model.add(Flatten())


image_input = Input(shape=(kaptcha_data.IMAGE_HEIGHT, kaptcha_data.IMAGE_WIDTH))
encoded_image = model(image_input)

# 全连接层，输出维数是kaptcha_data.MAX_CAPTCHA * kaptcha_data.CHAR_SET_LEN
encoded_softmax = []
for i in range(kaptcha_data.MAX_CAPTCHA):
    encoded_softmax.append(Dense(kaptcha_data.CHAR_SET_LEN, use_bias=True, activation='softmax')(encoded_image))
output = concatenate(encoded_softmax)

model = Model(inputs=[image_input], outputs=output)

# 编译模型，损失函数使用categorical_crossentropy， 优化函数使用adam，每一次epoch度量accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 模型可视化
# from keras.utils import plot_model
# plot_model(model, to_file=captcha_preprocess.base_dir + '/captcha_recognition_model.png')

# 加载之前模型的权值
if Path(kaptcha_data.base_dir + '/kaptcha_recognition.h5').is_file():
    model.load_weights(kaptcha_data.base_dir + '/kaptcha_recognition.h5')

batch_size = 512
epoch = 0
while True:
    print("epoch {}...".format(epoch + 1))
    (x_batch, y_batch) = kaptcha_data.get_batch_data(batch_size)
    train_result = model.train_on_batch(x=x_batch, y=y_batch)
    print(' loss: %.6f, accuracy: %.6f' % (train_result[0], train_result[1]))
    if epoch % 50 == 0:
        # 保存模型的权值
        model.save_weights(kaptcha_data.base_dir + '/kaptcha_recognition.h5')
    # 当准确率大于0.5时，说明学习到的模型已经可以投入实际使用，停止计算
    if train_result[1] > 0.5:
        break
    epoch += 1

# 计算某一张图片的验证码
predicts = model.predict(kaptcha_data.get_single_image(kaptcha_data.base_dir + '/val_pics/362_gmdyp.jpg'), batch_size=1)
print('predict: %s' % kaptcha_data.vec2text(predicts[0]))

