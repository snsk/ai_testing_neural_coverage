import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K
import codecs
import sys

np.set_printoptions(threshold=sys.maxsize)

#colab_root = '/content/drive/MyDrive/colab_root/'
colab_root = './'
filename = 'hiyashi_1.jpg'

# 画像を読み込む。
img = image.load_img(colab_root+filename,  target_size=(100,100))
img = np.array(img)

#plt.imshow(img)
#plt.show()

# flow に渡すために配列を四次元化
img = img[None, ...]

# 画像変形のジェネレータを作成
datagen = image.ImageDataGenerator(rotation_range=150)
gen = datagen.flow(img, batch_size = 1)

# バッチの実行
batches = next(gen)
g_img = batches[0]/255
g_img = g_img[None, ...]

# modelの読み込み
model = load_model(colab_root + 'ramen_hiyashi_acc0.9675.h5')
# 正解ラベルの定義
label=['ramen', 'hiyashi']
model.summary()

# 判別
pred = model.predict(g_img, batch_size=1, verbose=0)
score = np.max(pred)
pred_label = label[np.argmax(pred[0])]
print('name:',pred_label)
print('score:',score)

layer_name=['cond2d_28', 'conv2d_29', 'maxpooling', 'dropout',
'conv2d_30', 'conv2d31', 'maxpooling', 'dropout', 
'flatten7', 'dence', 'dropout', 'dense(softmax)']

# with a Sequential model
for i in range(12):
    get_3rd_layer_output = K.function([model.layers[0].input],
                                    [model.layers[i].output])
    layer_output = get_3rd_layer_output(g_img)[0]
    # ファイルへの書き出し
    print(layer_output, file=codecs.open(filename + '_numpy_l' + 
        str(i) + '_' + layer_name[i] + '.txt', 'w', 'utf-8'))
    
    # neuron coverage の計算
    active_neurons_count = np.count_nonzero(layer_output)
    total_neurons_count = layer_output.size
    print('nuron_cov({}): {} {}/{}'.format(layer_name[i], '{:.2%}'.format(
        active_neurons_count/total_neurons_count), 
        active_neurons_count, 
        total_neurons_count)
        )
