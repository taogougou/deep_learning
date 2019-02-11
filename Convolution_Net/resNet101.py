'''
resNet101的keras版本实现
'''
import keras.models as KM
import keras.layers as KL
from keras.layers import BatchNormalization
#resNet101的主结构
def resNet101_main(input_image):
    #相当于tf中的placehold
    input_image = KL.Input(input_image)
    #要求输入的图片尺寸是224*224
    # Stage 1
    #在图片边缘填充0
    x = KL.ZeroPadding2D((3, 3))(input_image)
    #卷积尺寸7*7*64，步数是2，输出是112*112*64的特征图
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    #上面卷积后输出112*112*64的特征图，经过3*3和步数为2的池化后输出为56*56*64的特征图
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    #输入的特征图为56*56*64，通道数为64，输出通道数为256，通道数不一致，所以要用conv_block()进行shortcut，最终输出
    #通道数为256
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    #x的通道数变为256，最终的输出通道数也为256，所以用identity_block()进行shortcut
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # Stage 3
    #同理，输入通道数为256，输出通道数为512，所以用conv_block()进行shortcut
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    block_count = {"resnet101": 22}["resnet101"]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
    C4 = x
    # Stage 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    #平均池化层
    x = KL.AveragePooling2D((2, 2), name='avg_pool')(x)
    #将数据摊平，以便输入全连接层
    x = KL.Flatten()(x)
    #输入全连接层
    x = KL.Dense(units=1000,activation = 'softmax', name = 'fc1000',kernel_initializer='glorot_uniform')

    #构建resnet101模型
    model = KM.Model(inputs=input_image, outputs=x, name='ResNet101')
    return model

    # return [C1, C2, C3, C4, C5]

#在shortcut的时候输入和输出的通道数不同，这里要使通道数一致
def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True):
    nb_filter1, nb_filter2, nb_filter3 = filters#[64,64,256]等
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                                           '2c', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

#如果输入和输出的通道数一致，这里不需要做转换，直接shortcut
def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

#做训练
if __name__ == '__main__':
    #构建模型
    model = resNet101_main()
    model.compile()
    model.fit()