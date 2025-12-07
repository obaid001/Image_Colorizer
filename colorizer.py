import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import tensorflow as tf
for device in tf.config.get_visible_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Conv2DTranspose, Add
from tensorflow.keras.callbacks import ModelCheckpoint
from skimage.color import *
import matplotlib.pyplot as plt
tf.config.run_functions_eagerly(True)
from utils import img_rebuilder, data_gen, predict_disp

img_w = 256
img_h = 256
dim = (img_h, img_w, 3)

dirr = r'D:\Datasets\colorize_data/'
train_files = []
test_files = []
for x in os.listdir(dirr):
    path = dirr + x
    if 'train' in x:
        train_files += [f'{path}/{x}' for x in os.listdir(path)]    
    if 'test' in x:
        test_files += [f'{path}/{x}' for x in os.listdir(path)]    
train_files = pd.DataFrame(train_files)
test_files = pd.DataFrame(test_files)
len(train_files), len(test_files)

bs = 10
train_data = data_gen(train_files, bs, True)
val_data = data_gen(test_files, bs)

X, Y = next(train_data)
Y1, Y2, Y3 = Y
print('Gray:',np.max(X[0, ...]), np.min(X[0, ...]))
print('R:',np.max(Y1[0, ...]), np.min(Y1[0, ...]))
print('G:',np.max(Y2[0, ...]), np.min(Y2[0, ...]))
print('B:',np.max(Y3[0, ...]), np.min(Y3[0, ...]))
print(X.shape, Y1.shape, Y2.shape, Y3.shape)

n = np.random.randint(0, len(X), 1)[0]
Y = Y1[n], Y2[n], Y3[n]
fig, ax = plt.subplots(1,2)
ax[0].imshow(img_rebuilder(X[n], Y))
ax[0].axis('off')
ax[0].set_title('Ground Truth')
ax[1].imshow(X[n], cmap = 'gray')
ax[1].axis('off')
ax[1].set_title('Input')
fig.set_figheight(15)
fig.set_figwidth(12)

output_path = "comparison_image.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Saved to: {output_path}")

vgg = tf.keras.applications.VGG19(include_top = False, input_shape = dim)
vgg.trainable = False

vgg_outputs = []
for l in ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv4', 'block5_conv4', 'block5_pool']:
    vgg_outputs.append(vgg.get_layer(l).output)

inp = vgg.input
feat = vgg_outputs[5]

feat = BatchNormalization()(feat)
up1 = Conv2DTranspose(512, kernel_size = 3, strides = 2, padding = 'same', name = 'up1', activation = 'relu')(feat)    
conv11 = Conv2D(512, kernel_size = 3, padding = 'same', name = 'conv11', activation = 'relu')(up1)    
conv12 = Conv2D(512, kernel_size = 3, padding = 'same', name = 'conv12', activation = 'relu')(conv11)   
add11 = Add(name = 'add11')([up1, conv11, conv12, vgg_outputs[4]])

add11 = BatchNormalization()(add11)
up2 = Conv2DTranspose(512, kernel_size = 3, strides = 2, padding = 'same', name = 'up2', activation = 'relu')(add11)    
conv21 = Conv2D(512, kernel_size = 3, padding = 'same', name = 'conv21', activation = 'relu')(up2)    
conv22 = Conv2D(512, kernel_size = 3, padding = 'same', name = 'conv22', activation = 'relu')(conv21)   
add21 = Add(name = 'add21')([up2, conv21, conv22, vgg_outputs[3]])

add21 = BatchNormalization()(add21)
up3 = Conv2DTranspose(256, kernel_size = 3, strides = 2, padding = 'same', name = 'up3', activation = 'relu')(add21)    
conv31 = Conv2D(256, kernel_size = 3, padding = 'same', name = 'conv31', activation = 'relu')(up3)    
conv32 = Conv2D(256, kernel_size = 3, padding = 'same', name = 'conv32', activation = 'relu')(conv31)   
add31 = Add(name = 'add31')([up3, conv31, conv32, vgg_outputs[2]])

add31 = BatchNormalization()(add31)
up4 = Conv2DTranspose(128, kernel_size = 3, strides = 2, padding = 'same', name = 'up4', activation = 'relu')(add31)    
conv41 = Conv2D(128, kernel_size = 3, padding = 'same', name = 'conv41', activation = 'relu')(up4)    
conv42 = Conv2D(128, kernel_size = 3, padding = 'same', name = 'conv42', activation = 'relu')(conv41)

add41 = Add(name = 'add41')([up4, conv41, conv42, vgg_outputs[1]])

add41 = BatchNormalization()(add41)
up5 = Conv2DTranspose(64, kernel_size = 3, strides = 2, padding = 'same', name = 'up5', activation = 'relu')(add41)    
conv51 = Conv2D(64, kernel_size = 3, padding = 'same', name = 'conv51', activation = 'relu')(up5)    
conv52 = Conv2D(64, kernel_size = 3, padding = 'same', name = 'conv52', activation = 'relu')(conv51)   
add51 = Add(name = 'add51')([up5, conv51, conv52, vgg_outputs[0]])

add51 = BatchNormalization()(add51)
r = Conv2D(1, kernel_size = 3, padding = 'same', name = 'R', activation = 'sigmoid')(add51)    
g = Conv2D(1, kernel_size = 3, padding = 'same', name = 'G', activation = 'sigmoid')(add51)    
b = Conv2D(1, kernel_size = 3, padding = 'same', name = 'B', activation = 'sigmoid')(add51)    

model = Model(inp, [r, g, b])
optimizer = tf.keras.optimizers.Adam(lr = 0.001)
loss = {'R' : 'mae', 'G': 'mae', 'B': 'mae'}
model.compile(optimizer, loss)
print(model.summary())

try:
    model = load_model('colorizer_model.h5', compile = False)
    loss = {'R' : 'mae', 'G': 'mae', 'B': 'mae'}
    optimizer = tf.keras.optimizers.Adam(lr = 0.001)
    model.compile(optimizer, loss)
except:
    pass

disp = predict_disp()
check = ModelCheckpoint('colorizer_model.h5', save_best_only = True, verbose = True)
logger = tf.keras.callbacks.CSVLogger('log.csv', separator = ",", append = True)
model.fit(train_data, epochs = 3000, validation_data = (val_data), steps_per_epoch = 200, 
          callbacks = [disp, check, logger], validation_steps = 100)

result = pd.read_csv('log.csv')[['epoch', 'loss', 'val_loss']]
print(result.head(2))

plt.figure(figsize=(8, 5))
plt.plot(result['epoch'] + 1, result['loss'], label='Train Loss')
plt.plot(result['epoch'] + 1, result['val_loss'], label='Validation Loss')

plt.tight_layout()
plt.legend()

output_path = "loss_curve.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved loss curve → {output_path}")


model = load_model('colorizer_model.h5')

save_dir = "comparison_samples"
os.makedirs(save_dir, exist_ok=True)

for i in range(4):
    f, axarr = plt.subplots(1, 3)

    X, Y = next(val_data)
    n = np.random.randint(0, len(X), 1)[0]
    Y1, Y2, Y3 = Y
    Y = (Y1[n], Y2[n], Y3[n])

    pred = model.predict(np.expand_dims(X[n], axis=0))

    # Input image
    axarr[0].imshow(X[n], cmap='gray')
    axarr[0].set_title('Input')
    axarr[0].axis('off')

    # Ground truth
    gt = img_rebuilder(X[n], Y)
    axarr[1].imshow(gt)
    axarr[1].set_title('Ground Truth')
    axarr[1].axis('off')

    # Prediction
    pred_img = img_rebuilder(X[n], pred)[0]
    axarr[2].imshow(pred_img)
    axarr[2].set_title('Prediction')
    axarr[2].axis('off')

    f.set_figwidth(12)
    f.set_figheight(12)
    f.tight_layout()

    save_path = os.path.join(save_dir, f"comparison_{i}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(f)

    print(f"Saved → {save_path}")


test = r"C:\Users\Admin\Downloads\folwer.jpg"
test = 'sun.jpg'

X = np.array(load_img(test, target_size=(img_h, img_w))) / 255
test_gray = rgb2gray(X)
test_rgb = gray2rgb(test_gray)

f, axarr = plt.subplots(1, 3)

# Input (grayscale → RGB)
axarr[0].imshow(test_rgb)
axarr[0].set_title('Input')
axarr[0].axis('off')

# Ground Truth (original color)
axarr[1].imshow(X)
axarr[1].set_title('Ground Truth')
axarr[1].axis('off')

# Prediction
test_batch = np.expand_dims(test_rgb, 0)
pred = model.predict(test_batch)
pred_img = img_rebuilder(test_rgb, pred)[0]

axarr[2].imshow(pred_img)
axarr[2].set_title('Prediction')
axarr[2].axis('off')

f.set_figwidth(12)
f.set_figheight(12)
f.tight_layout()

save_path = "test_comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(f)

print(f"Saved → {save_path}")
