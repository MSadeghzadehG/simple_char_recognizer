from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte
from kegit ras.models import load_model
# from skimage.transform import rescale, resize, downscale_local_mean
from skimage.transform import resize
import numpy as np

import os,cv2
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



def binarization(input_image):
    threshold_value = threshold_otsu(input_image)
    return input_image > threshold_value

car_image = imread("captcha.jpg", as_grey=True)

binary_car_image = binarization(car_image * 255)
# print(car_image.shape)




model = load_model('my_model.h5')
for i in range(3):
    char = binary_car_image[:, i*15:(i+1)*15]
    # print(char.shape)
    char = resize(char, (28, 28), anti_aliasing=False)
    # imshow(char)
    char = binarization(char)
    char = img_as_ubyte(char)
    # cv2.imshow('image', char)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    char = char.reshape(1, 28, 28, 1)
    char = char.astype('float32')
    # char /= 255    
    out = model.predict(char)
    # print(out)
    print(str(np.argmax(out)))