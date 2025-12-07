import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
for device in tf.config.get_visible_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import *
from skimage.color import *
import matplotlib.pyplot as plt
tf.config.run_functions_eagerly(True)

def data_gen(data, batch_size = 32, augment = False): # Image Generator
    while True:
        X, Y1, Y2, Y3 = [], [], [], []
        sample = data.sample(batch_size).values
        for i in range(batch_size):
            x = load_img(sample[i][0], target_size = (img_h, img_w))
            x = np.array(x)/255
            if augment:
                ops = np.random.choice([None, np.fliplr, np.flipud])
                if ops:
                    x = ops(x)
            X.append(gray2rgb(rgb2gray(x)))
            r, g, b = np.split(x, 3, -1)
            Y1.append(r)
            Y2.append(g)
            Y3.append(b)

        X = np.array(X)
        Y1 = np.array(Y1)
        Y2 = np.array(Y2)
        Y3 = np.array(Y3)
        yield X, [Y1, Y2, Y3]

def img_rebuilder(L, Y):
    result = np.concatenate(Y, -1)
    return result

class predict_disp(tf.keras.callbacks.Callback):
    def __init__(self, data=False, save_dir="epoch_predictions"):
        super(predict_disp, self).__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        n = np.random.randint(0, bs, 1)[0]

        X, Y = next(val_data)
        Y1, Y2, Y3 = Y
        Y = Y1[n], Y2[n], Y3[n]
        
        X = tf.expand_dims(X[n], 0)
        pred = self.model.predict(X)
        pred = img_rebuilder(X[0], pred)[0]
        gt   = img_rebuilder(X[0], Y)
        
        fig, ax = plt.subplots(1, 3)
        
        ax[0].imshow(X[0])
        ax[0].axis('off')
        ax[0].set_title('Input')
        
        ax[1].imshow(gt)
        ax[1].axis('off')
        ax[1].set_title('Ground Truth')
        
        ax[2].imshow(pred)
        ax[2].axis('off')
        ax[2].set_title('Prediction')
        
        fig.set_figheight(15)
        fig.set_figwidth(12)

        save_path = os.path.join(self.save_dir, f"epoch_{epoch+1}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved prediction visualization → {save_path}")
