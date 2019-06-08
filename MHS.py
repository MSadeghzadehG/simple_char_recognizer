from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte
from keras.models import load_model
from skimage.transform import rescale, resize, downscale_local_mean


def binarization(input_image):
    threshold_value = threshold_otsu(input_image)
    return input_image > threshold_value

car_image = imread("captcha.jpg", as_grey=True)

binary_car_image = binarization(car_image * 255)
# print(car_image.shape)

cv_image = img_as_ubyte(binary_car_image)



model = load_model('my_model.h5')
for i in range(3):
    char = cv_image[:, i*15:(i+1)*15]
    print(char.shape)
    char = char.reshape(1, 28, 28, 1)/255    
    out = model.predict(char)
    print(out)