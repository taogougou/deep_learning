import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Activation,BatchNormalization,Dropout,Flatten
from keras.initializers import he_normal
from keras.datasets import cifar10
from keras.utils.data_utils import get_file
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
#组建alexNet网络
def build_model(x_train,weight_decay,dropout):
    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape = x_train.shape[1:], padding='valid', activation='relu',
                     kernel_initializer='uniform',kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'
                     ,kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'
                     , kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'
                     , kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'
                     , kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu',kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(Dropout(dropout))
    model.add(Dense(4096, activation='relu',kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(Dropout(dropout))
    model.add(Dense(10,kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(Activation('softmax'))
    return model
def scheduler(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 160:
        return 0.01
    return 0.001
if __name__ == "__main__":
    num_classes = 10
    batch_size = 128
    epochs = 200
    iterations = 391
    dropout = 0.5
    weight_decay = 0.0001
    #log保存的目录
    log_filepath = r'./vgg19_retrain_logs/'
    #vgg19与训练模型的位置，可以下载下来放在一个目录下
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
    filepath = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models')

    # 下载数据集
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # 对原始数据做前期的处理
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - 123.680)
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - 116.779)
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - 103.939)
    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - 123.680)
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - 116.779)
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - 103.939)

    #建立模型并训练
    model = build_model(x_train,weight_decay,dropout)

    # 加载与训练模型
    model.load_weights(filepath, by_name=True)

    # 设置优化器
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    #损失函数是交叉熵
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)

    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, tb_cb]

    print('数据增强。。。')
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant', cval=0.)

    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        callbacks=cbks,
                        validation_data=(x_test, y_test))

    #将训练后的模型保存
    model.save('alexNet_retrain.h5')
