# Refer the solution of below post, reisse the image to 16X32, use LeNet, grayImage
# https://medium.com/@xslittlegrass/self-driving-car-in-a-simulator-with-a-tiny-neural-network-13d33b871234


# Readd the image for the training data
import csv
import matplotlib.pyplot as plt
# import scipy.misc
import cv2
import pickle
from tqdm import tqdm

import numpy as np

images = []
steerings = []

def read_data(path, split):
    lines = []
    with open(path+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line) # line content: center	left	right	steering	throttle	brake	speed

    lines = lines[1:] # get rid of first line which inlcude colum label

    for indx, line in tqdm(enumerate(lines)):

    # skip the the loop after load 500 images, for debug
        # if indx == 1:
        #     break      
        image_source = line[0] # the image path in the cvs file(recorded path)
        filename = image_source.split(split)[-1]
        image_path = path + 'IMG/' + filename

        image = cv2.imread(image_path)
        
        if image.any() == None: # if cv2 return None
            print('read file error')
            continue

        # plt.imshow(image)
        # plt.show()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # plt.imshow(image, cmap='gray')
        # plt.show()
        # image = cv2.resize(image, (32,32))
        # image = image.reshape(32,32,1)
        # plt.imshow(image, cmap='gray')
        # plt.show()       


        # if image.shape != (160,320,3):
        #     print("read file problem, not the correct shape")
        #     continue
        
        steering = float(line[3])
        # check the steering and decide if need append the data to the list, if zero, skip
        if abs(steering) < 0.02:
            # if np.random.randint(-1, 2) == 0 # control the percent of zero get rid of
            continue

        images.append(image)
        steerings.append(steering)



def read_data_sides(path, split, offset):
    
    lines = []
    with open(path+'driving_log.csv') as csvfile:
      reader = csv.reader(csvfile)
      for line in reader:
        lines.append(line) # line content: center   left    right   steering    throttle    brake   speed

    lines = lines[1:] # get rid of first line which inlcude colum label

    for indx, line in tqdm(enumerate(lines)):

        # skip the the loop after load 500 images, for debug
#         if indx == 500:
#             break

        image_source_left = line[1] # the image path in the cvs file(recorded path)
        filename_left = image_source_left.split(split)[-1]
        image_path_left = path + 'IMG/' + filename_left

        image_source_right = line[2] # the image path in the cvs file(recorded path)
        filename_right = image_source_right.split(split)[-1]
        image_path_right = path + 'IMG/' + filename_right

        try:  
            image_left = cv2.imread(image_left_path)
            image_left = cv2.cvtColor(image_left, cv2.COLOR.BGR2RGB)
            # image_left = cv2.resize((32,32))

            image_right = cv2.imread(image_right_path)
            image_right = cv2.cvtColor(image_right, cv2.COLOR.BGR2RGB)
            # image_right = cv2.resize((32,32))
            # if image_left.shape != (160,320,3):
            #     print("read file problem, not the correct shape")
            #     continue
            # if image_right.shape != (160,320,3):
            #     print("read file problem, not the correct shape")
            #     continue

            steering = float(line[3])
            # check the steering and decide if need append the data to the list, if zero, skip
#             if steering == 0.0:
#                 # if np.random.randint(-1, 2) == 0 # control the percent of zero get rid of
#                 continue
            if abs(steering + offset) > 0.02:         
                images.append(image_left)
                steerings.append(steering + offset)
            if abs(steering - offset) > 0.02:
                images.append(image_right)
                steerings.append(steering - offset)

        except:
            print('read file error')
            continue

# read data
print('read data')
read_data(path='../data/',split= '/') # data provid by udacity, collect from unix system, the file split use '/'
# print('read data, left and right')
# read_data_sides(path='./data/',split= '/', offset=0.08) # data provid by udacity, collect from unix system, the file split use '/'
# read_data_sides(path='../data/',split= '/', offset=0.15) # data provid by udacity, collect from unix system, the file split use '/'

# print('read data_local')
# read_data(path='/opt/carnd_p3/data_local/', split='\\') # datat collection from local PC which use windows, the file split use '\'
# read_data_sides(path='/opt/carnd_p3/data_local/', split='\\', offset=0.15)
# print('read data_local_reverse')
# read_data(path='/opt/carnd_p3/data_local_reverse/', split='\\') 
# read_data_sides(path='/opt/carnd_p3/data_local_reverse/', split='\\', offset=0.15)
# print('read data_t2')
# read_data(path='/opt/carnd_p3/data_t2/', split='\\') 
# read_data_sides(path='/opt/carnd_p3/data_t2/', split='\\', offset=0.15)
# print('read data_t2_reverse')
# read_data(path='/opt/carnd_p3/data_t2_reverse/', split='\\') 
# read_data_sides(path='/opt/carnd_p3/data_t2_reverse/', split='\\', offset=0.15)

def flip_images():
    print("Flip the images...")
    images_flipped = []
    steerings_flipped = []
    for image, steering in zip(images, steerings):
        image_flipped = np.fliplr(image)
        steering_flipped = -steering
        images_flipped.append(image_flipped)
        steerings_flipped.append(steering_flipped)

    # add the flipped image to images
    for image_filpped, steering_flipped in zip(images_flipped, steerings_flipped):
        images.append(image_flipped)
        steerings.append(steering_flipped)

# Data Augmentation
# flip_images() 

# num_samples = len(images)

print("Image data loaded: ", len(images))

print("Gen X_train, y_train array from readed data")
X_train = np.array(images)
y_train = np.array(steerings)

# plt.imshow(images[9])
# plt.show()

print('sample number: ', len(X_train))
print(X_train.shape)
print(X_train[0].shape)
print('sample label number: ', len(y_train))
print(y_train.shape)


def model_lenet(save_name, epoch=10):
    '''
    the name should be *.h5
    '''
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Cropping2D    

    model = Sequential()

    def image_process(image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (32,32))
        image = image.reshape(32,32,1)
        image = image / 255 - 0.5

    # data preprocess
    model.add(Lambda(image_process, input_shape=(160,320,3)))
    # Layer 1 Conv
    model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2)) # keras dropout para is the problity of get rid of the nero
    # # Layer 2 Conv
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))
    # Layer 3 FC
    model.add(Flatten())
    model.add(Dense(500))
    # model.add(Dropout(0.5))
    # Layer 4 FC
    model.add(Dense(100))
    # model.add(Dropout(0.5))
    # layer 5
    model.add(Dense(1))    

    model.compile(loss='mse', optimizer='adam')
    history=model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epoch)

    model.save('model_lenet.h5')

# model_lenet('model_try.h5')  
# model_lenet('model_lenet.h5')  
# model_lenet('model_lenet_add_flip.h5')  

# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def model_nvidia(save_name, epoch=10):
    '''
    the name should be *.h5
    '''
    # Built the model
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Cropping2D
    
    from keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
                featurewise_center=True,
                featturewise_std_normalization=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range =0.2,
                brightness_range=(0.2, 0.3))

    datagen.fit(X_train)


    model = Sequential()

    # data preprocess
    # model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320, 3))) # crop the image to, bot
    # model.add(Lambda(lambda x: x / 255.0 - 0.5)) # normalize
    # Layer 1 Conv
    model.add(Conv2D(24, (5,5), activation='relu'), input_shape=(160,320, 3))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2)) # keras dropout para is the problity of get rid of the nero
    # Layer 2 Conv
    model.add(Conv2D(36, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    # Layer 3 Conv
    model.add(Conv2D(48, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    # Layer 4 Conv
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))        
    # Layer 5 FC
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(0.5))
    # Layer 6 FC
    model.add(Dense(100))
    model.add(Dropout(0.5))
    # Layer 7 FC
    model.add(Dense(50))
    model.add(Dropout(0.5))
    # layer 7
    model.add(Dense(1))    

    model.compile(loss='mse', optimizer='adam')
    # history=model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epoch)

    model.fit_generator

    model.save(save_name)
    
# model_nvidia('model_nvidia_add_flip_10.h5', 10) 
# model_nvidia('model_nvidia_cpu.h5', 1) 
# model_nvidia('model_nvidia_local_data.h5', 5) 
# model_nvidia('model_nvidia_data_local_data_reverse_flip_10.h5', 10) 
# model_nvidia('model_nvidia_data_local_data_reverse_flip.h5', 5) 
# model_nvidia('model_nvidia_nonzero.h5', 5) 
# model_nvidia('model_nvidia_sides_0.08.h5', 10) 
# model_nvidia('model_nvidia_sides_0.15.h5', 10) 
# model_nvidia('model_nvidia_sides_0.15_nonzeros.h5', 10) 
# model_nvidia('model_nvidia_sides_0.15_nonzeros_flip.h5', 10) 
# model_nvidia('model_nvidia_sides_0.08_nonzeros.h5', 10) 
# model_nvidia('model_nvidia_sides_0.3_nonzeros.h5', 10) 
# model_nvidia('model_nvidia_t1_t2_sides15_nonzeros.h5', 10) 
# model_nvidia('model_nvidia_data_local_data_reverse_sides15_nonzeros.h5', 10) 

model_lenet('model_lenet.h5',5)