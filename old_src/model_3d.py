
# CarND Project
# model_3: create the train_model function to put all the parameter change in this funciton, 
#          rewrite the model_nvidia, name navidia2(input size from (90, 320, 3) to (66, 100, 3), use strides 2x2 at kernel 5x5, get rid of maxpooling. 2018/11/27
# model_3_yuv: load_images/load_sides_image load the yuv image, process_yuv changed, model_nvidia not change, work with drive_yuv.py 2018//11/28
# model_3a_yuv: data_path in train_model function became a tuple to le the train_modle function could load more than one data folder.
#               del the data_split parameter in train_model(), load_images(), load_sides_images(), auto detect split is '/' or '\' 2018/11/29
# model_3b_yuv: add sampling rate. (to sovle the problem training memeroy issue.) 1/2 sample rate
# model_3c_yuv: the drive_yuv change, the output is (66,200,3) yuv, no need for pre-porcess, to save memeray in workspace to load more data and aug the data
# model_3c: add image size choice, let the image color/size could be choice at the input of train_function(). 
#               add navidia3() function, which only use Y channel and the filter numbers is also adjust

# import lib, the tensorflow and keras imported in the function which used them
import csv
import matplotlib.pyplot as plt
import cv2
import pickle
from tqdm import tqdm
import numpy as np
import sys

def load_log_file(path, sample_rate=1):
    '''
    read the log file in the path folder
    return a list contain the log information
    '''
    sample_num = 0
    lines = []
    with open(path+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for indx, line in enumerate(reader):
            if indx % sample_rate== 0:
                lines.append(line) # line content: center   left    right   steering    throttle    brake   speed
            else:
                pass
            sample_num = indx
    
    print('totlal sample number: {}'.format(sample_num), 'sample rate = 1 / {}'.format(sample_rate))
    print('actual loade sample will be: {}'.format(sample_num//sample_rate))
    print()

    lines = lines[1:] # get rid of first line which inlcude colum label

    return lines

def process_image(image_file, image_color, image_size):
    """
    read the image from image_file and preprocess the image according the image_color and image_size
    image_file: image file name include the path
    image_color: 'yuv', 'y', ('rgb', 'hsv', 'gray' is not impemented)
    image_size: (160, 320) : original size, no change
                (66, 200) : size could feed to nvidia model direclty
                (32, 32) : not implement(feed to lenet)
    return: image with required size and color.
    """
    image = cv2.imread(image_file)
    
    # image_size process
    if image_size == (160,320):
        pass
    elif image_size == (66, 200):
        image = cv2.resize(image, (200,100)) # (160,320,3)-->(100,200,3)
        image = image[25:91,:,:] # (100,200,3)-->(66,200,3)
    else:
        print("Not valid size, (160, 320), (66, 200) is valid size !!!")
        sys.exit()

    if image_color == 'YUV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) # Note, the drive_yuv.py feed simulator RGB image(160,320,3)
    elif image_color == 'Y':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image = image[:,:,0] # only use y channel
        image = image[:,:,None] # change shape (66, 200) to (66,200, 1)

    return image

def load_images(images, steerings, path, color, size, sample_rate, sub_read, sample_balance):
    '''
    images: the list which contain the images
    steerings: the list which contain the steering data
    path: the data folder which contain the IMG(folder holder the center/left/right image) and 'driving_log.csv' file
    image_color: 
    image_size:
    sample_rate:
    sub_read: False or int, if number inputted, will just read the number recevied
    sample_balance: False/0 or int (1, 101), balance the steering data will >50% is zero or <0.02 (steering is -1.0-1.0, 0.02 is 1%, just cosider the number < 1% is noise)
                    another view of point, steering < 0.02, the car will keep straight forward, 
                    this is affect the cars perfomace in curve lane, when the >50% data feed to the model is just keep the car straight forword
    '''
    print("Loading Centeral images from: ", path)
    print("The image loaded color: {}, size: {}".format(color, size))
    lines = load_log_file(path, sample_rate)

    # check the split
    if ('/' in lines[0][0]):
        split = '/'
    else:
        split = '\\'

    num_before = len(images)

    if sub_read:
        print("Will loading {} images".format(sub_read))
    else:
        print("Will loading {} images".format(len(lines)))

    if sample_balance:
        print("Sample balance, {} percent steering<0.02 image will be keeped.".format(sample_balance))

    for indx, line in enumerate(lines):
        # skip the the loop after load __ images, for debug/tester
        if sub_read:
            if indx == sub_read:
                break      
        image_source = line[0] # the image path in the cvs file(recorded path)
        filename = image_source.split(split)[-1]
        image_path = path + 'IMG/' + filename

        image = process_image(image_path, color, size)
        
        steering = float(line[3])
        # check the steering and decide if need append the data to the list, if zero, skip
        if sample_balance:
            if abs(steering) < 0.02 and np.random.randint(0, 99) < 100 - sample_balance: #and np.random.randint(0,10) < 10: # controal the percent, 10 100% get rid of
                # if np.random.randint(-1, 2) == 0 # control the percent of zero get rid of
                continue

        images.append(image)
        steerings.append(steering)
    
    num_after = len(images)
    print("Actual loading {} images.".format(num_after-num_before))
    print()

def load_sides_images(images, steerings, path, color, size, sample_rate, sub_read, sample_balance, offset):
    '''
    images: the list which contain the images
    steerings: the list which contain the steering data
    path: the data folder which contain the IMG(folder holder the center/left/right image) and 'driving_log.csv' file
    image_color: 
    image_size:
    sample_rate:
    offset: the left/right image steering offset
    sub_read: False or int, if number inputted, will just read the number recevied
    sample_balance: False/0 or int (1, 101), balance the steering data will >50% is zero or <0.02 (steering is -1.0-1.0, 0.02 is 1%, just cosider the number < 1% is noise)
                    another view of point, steering < 0.02, the car will keep straight forward, 
                    this is affect the cars perfomace in curve lane, when the >50% data feed to the model is just keep the car straight forword
    '''
    
    print("Loading Left/Right images from: ", path)
    print("The images loaded color: {}, size: {}".format(color, size))
    print("Left images will offset: {}, Right images will offset: {}".format(-offset, offset))
    
    lines = load_log_file(path, sample_rate)   
    # check the split
    if ('/' in lines[0][0]):
        split = '/'
    else:
        split = '\\' 
    
    num_before = len(images)
    
    if sub_read:
        print("Will loading {} images".format(sub_read*2))
    else:
        print("Will loading {} images".format(len(lines)*2))

    if sample_balance:
        print("Sample balance, {} percent steering<0.02 image will be keeped.".format(sample_balance))    

    
    for indx, line in tqdm(enumerate(lines)):
        # skip the the loop after load 500 images, for debug
        if sub_read:
            if indx == sub_read:
                break 

        image_source_left = line[1] # the image path in the cvs file(recorded path)
        filename_left = image_source_left.split(split)[-1]
        image_path_left = path + 'IMG/' + filename_left

        image_source_right = line[2] # the image path in the cvs file(recorded path)
        filename_right = image_source_right.split(split)[-1]
        image_path_right = path + 'IMG/' + filename_right


        image_left = process_image(image_path_left, color, size)

        image_right = process_image(image_path_right, color, size)

        steering = float(line[3])
       
        if sample_balance:
            if abs(steering + offset) < 0.02 and np.random.randint(0, 99) < 100 - sample_balance: 
                continue
        images.append(image_left)
        steerings.append(steering + offset)

        if sample_balance:
            if abs(steering - offset) < 0.02 and np.random.randint(0, 99) < 100 - sample_balance: 
                continue
        images.append(image_right)
        steerings.append(steering - offset)
    
    num_after = len(images)
    print("Actual loading {} images.".format(num_after-num_before))
    print()      

def image_rotate(image, angle):
    """
    rotate the image 
    """
    image_shape = image.shape
    center = (image_shape[1]//2, image_shape[0]//2)
    scale = 1.0
    
    M = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine(image, M, (image_shape[1], image_shape[0]))
    
    if len(image.shape)==2: # handle the one channel image
        image = image[:,:,None]
    
    return image

def image_shift(image, dx, dy):
    '''
    shift the image dx and dy
    '''
    image_shape = image.shape
    
    M = np.float32([[1,0,dx], [0,1,dy]])
    image = cv2.warpAffine(image, M, (image_shape[1], image_shape[0]))
    
    if len(image.shape)==2: # handle the one channel image
        image = image[:,:,None]
    
    return image

def images_augmentation(images, steerings, factor=1.0, rotate=15, shift=0.2):
    """
    rotate and shift the images randomly
    factor: the muli number of augmentation numbers 1.0 = 1.0x, 2.0=2.0x
    rotate: rotate angle range >0
    shift: shift range, muli the width and height
    """
    print("Begin augemenation the images {}X".format(factor))
    print('rotate: {}, shift: {}'.format(rotate, shift))
    image_shape = images[0].shape
    num_images = len(images)
    num_aug = int(num_images * factor)

    # (160, 320, 3)
    width = int(image_shape[1] * shift)
    height = int(image_shape[0] * shift)

    for indx in range(num_aug):
        if indx % 2 == 0: # rotate
            angle = np.random.randint(-rotate, rotate+1)
            image = image_rotate(images[indx%num_images], angle)
            images.append(image)
            steerings.append(steerings[indx%num_images])
        else:
            dx = np.random.randint(-width, width+1)
            dy = np.random.randint(-height, height+1)
            image = image_shift(images[indx%num_images], dx, dy)
            images.append(image)
            steerings.append(steerings[indx%num_images])

    print("Data augmentation Done!!!") 
    print()           

def flip_images(images, steerings):
    '''
    flip the image left/right, the steering will be mul (-1)
    '''
    print("Flip the images...")
    images_flipped = []
    steerings_flipped = []
    for image, steering in zip(images, steerings):
        image_flipped = np.fliplr(image)
        steering_flipped = steering * (-1)
        images_flipped.append(image_flipped)
        steerings_flipped.append(steering_flipped)

    # add the flipped image to images
    for image_filpped, steering_flipped in zip(images_flipped, steerings_flipped):
        images.append(image_flipped)
        steerings.append(steering_flipped)
 
def model_lenet(save_name):
    '''
    the name should be *.h5
    need change, input should gray (32, 32, 1) 2018/11/30
    '''
    model = Sequential()

    # data preprocess
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    # Layer 1 Conv
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2)) # keras dropout para is the problity of get rid of the nero
    # # Layer 2 Conv
    # model.add(Conv2D(64, (3,3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))
    # Layer 3 FC
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Dropout(0.5))
    # Layer 4 FC
    model.add(Dense(100))
    model.add(Dropout(0.5))
    # layer 5
    model.add(Dense(1))    

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    history=model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)

    model.save('model_lenet.h5')

def cvt_hsv(x):
    '''
    use tensorflow function to convert image. (input is tensor, opencv just hand images)
    '''
    import tensorflow as tf
    return tf.image.rgb_to_hsv(x)

def cvt_yuv(x):
    '''
    use tensorflow function to convert image. (input is tensor, opencv just hand images)
    rgb_to_yuv not worked at tensorflow 1.3.0, it should be worked at tensorflow 1.8.0
    '''
    import tensorflow as tf
    return tf.image.rgb_to_yuv(x)

def preprocess_rgb(x):
    '''
    preprocess the image from drive.py and feed to nvidia network.
    input (160, 320, 3) RGB
    output (66, 200, 3) RGB
    '''
    import tensorflow as tf
    x = tf.image.resize_images(x, (100, 200))
    x = tf.image.crop_to_bounding_box(x, offset_height=25, offset_width=0, target_height=66, target_width=200)   
    
    return x

def preprocess_hsv(x):
    '''
    preprocess the image from drive.py and feed to nvidia network.
    input (160, 320, 3) RGB
    output (66, 200, 3) hsv
    '''
    import tensorflow as tf
    
    # x = cvt_hsv(x)
    x = tf.image.rgb_to_hsv(x) # training model is ok when use the funtion cvt_hsv(x), but when use drive.py to load the model, error said "cvt_hsv" not defined.2018/11/28
    x = tf.image.resize_images(x, (100, 200))
    x = tf.image.crop_to_bounding_box(x, offset_height=25, offset_width=0, target_height=66, target_width=200)     
    return x

def preprocess_yiq(x):
    '''
    preprocess the image from drive.py and feed to nvidia network.
    input (160, 320, 3) RGB
    output (66, 200, 3) yiq
    '''
    import tensorflow as tf
    
    x = tf.image.rgb_to_yiq(x) # try yiq, since all the lumi of yiq and yuv in the first channel.
    x = tf.image.resize_images(x, (100, 200))
    x = tf.image.crop_to_bounding_box(x, offset_height=25, offset_width=0, target_height=66, target_width=200)     
    return x

def preprocess_yuv(x):
    '''
    preprocess the image from drive.py and feed to nvidia network.
    input (160, 320, 3) yuv
    output (66, 200, 3) yuv
    '''
    import tensorflow as tf
    # x = cvt_yuv(x)
    x = tf.image.resize_images(x, (100, 200))
    x = tf.image.crop_to_bounding_box(x, offset_height=25, offset_width=0, target_height=66, target_width=200)     
    return x

def model_nvidia(X_train, y_train, save_name, color='RGB', epoch=10):
    '''
    the name should be *.h5
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    '''
    # Built the model
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Cropping2D
    
    print("Loading the model...")

    model = Sequential()

    # data preprocess
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320, 3))) # crop the image
    # choice color
    if color == "RGB":
        print("Using RGB color channel")
        pass
    elif color == "YUV":
        model.add(Lambda(cvt_yuv)) # yuv not worked at tensorflow 1.3.0, it should be worked at tensorflow 1.8.0
        print("Using YUV color channel")
    elif color == "HSV":
        model.add(Lambda(cvt_hsv))
        print("Using HSV color channel")
    else:
        print('Wrong color choice, valid color is "RGB", "YUV", "HSV"')
        sys.exit()
    model.add(Lambda(lambda x: x / 255.0 - 0.5)) # normalize
    
    # Layer 1 Conv
    model.add(Conv2D(24, (5,5), activation='relu'))
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
    # layer 8
    model.add(Dense(1))    

    model.compile(loss='mse', optimizer='adam')
    print("Loss is MSE, optimizer is adam")
    history=model.fit(X_train, y_train, batch_size=256, validation_split=0.2, shuffle=True, epochs=epoch)

    model.save(save_name)
    print(save_name, 'saved!!!')

def model_nvidia2(X_train, y_train, save_name, color='YUV', epoch=10):
    '''
    2018/11/27 rewrite the nvidia model according the papaer. (add three color choice, RGB/HSV/YUV)
    Trid tensorflow 1.8.0, but the speed of Workspace is not acceptable.
    the name should be *.h5
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    '''
    # Built the model
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Cropping2D
    
    print("Loading the model...")

    model = Sequential()

    # choice color
    if color == "RGB":
        print("Using RGB color channel")
        print("Not Implement, exit!!")
        sys.exit()
    elif color == "YUV":
        # model.add(Lambda(preprocess_yuv, input_shape=(160,320,3)))
        print("Using YUV color channel")
    elif color == "HSV":
        print("Using HSV color channel")
        print("Not Implement, exit!!")
        sys.exit()
    elif color == "YIQ":
        print("Using YIQ color channel")
        print("Not Implement, exit!!")
        sys.exit()
    else:
        print('Wrong color choice, valid color is "RGB", "YUV", "HSV", "YIQ"')
        sys.exit()

    # Normalize Input (66, 200, 3)
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66,200,3))) # normalize, from drive_yuv.py is (66,200,3) YUV image
    
    # Conv1 (66, 200, 3) --> (31, 98, 24)
    model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2)) # keras dropout para is the problity of get rid of the nero
    # Conv2 (31, 98, 24) --> (14, 47, 36)
    model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))
    # Conv3 (14, 47, 36) --> (48, 5, 22)
    model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))
    # Conv4 (48, 5, 22) --> (64, 3, 20)
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))  
    # Conv5, (64, 3, 20) --> (64, 1, 18)
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))   
    # FC1
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(0.5))
    # FC2
    model.add(Dense(100))
    model.add(Dropout(0.5))
    # FC3
    model.add(Dense(50))
    model.add(Dropout(0.5))
    # FC4
    model.add(Dense(10))
    model.add(Dropout(0.5))
    # Out put
    model.add(Dense(1))
        

    model.compile(loss='mse', optimizer='adam')
    print("Loss is MSE, optimizer is adam")
    history=model.fit(X_train, y_train, batch_size=256, validation_split=0.2, shuffle=True, epochs=epoch)

    model.save(save_name)
    print(save_name, 'saved!!!')

def model_nvidia3(X_train, y_train, save_name, epoch=10):
    '''
    2018/11/30
    nvidia3 is a nvida model changed version, only use y channel, the filter numbers is also adjust
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    '''
    # Built the model
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Cropping2D
    
    print("Loading the model...")

    model = Sequential()

    # Normalize Input (66, 200, 1)
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66,200,1))) # normalize, (66,200,1) Y channel (gray/HSV-V ?? should try also??)
    
    # Conv1 (66, 200, 1) --> (31, 98, 8)
    model.add(Conv2D(8, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2)) # keras dropout para is the problity of get rid of the nero
    # Conv2 (31, 98, 8) --> (14, 47, 12)
    model.add(Conv2D(12, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))
    # Conv3 (14, 47, 12) --> (5, 22, 16)
    model.add(Conv2D(16, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))
    # Conv4 (5, 22, 16) --> (5, 22, 20)
    model.add(Conv2D(20, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))  
    # Conv5, (5, 22, 20) --> (1, 18, 20)
    model.add(Conv2D(20, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))   
    # FC1
    model.add(Flatten()) # (18*20=360)
    model.add(Dropout(0.5))
    # FC2
    model.add(Dense(100))
    model.add(Dropout(0.5))
    # FC3
    model.add(Dense(50))
    model.add(Dropout(0.5))
    # FC4
    model.add(Dense(10))
    model.add(Dropout(0.5))
    # Out put
    model.add(Dense(1))
        

    model.compile(loss='mse', optimizer='adam')
    print("Loss is MSE, optimizer is adam")
    history=model.fit(X_train, y_train, batch_size=256, validation_split=0.2, shuffle=True, epochs=epoch)

    model.save(save_name)
    print(save_name, 'saved!!!')

def train_model(data_path, color='Y', size=(66, 200),sample_rate=1, sub_read=False, sample_balance=False, sides_image=True, offset=0.22, data_augmentation=1, 
                model='nvidia', train_epoch=10):
    '''
    main function training the model
    ''' 

    # cread the list to hold the features and labels
    images = []
    steerings = []

    # read data
    for path in data_path:
        # load center image
        load_images(images, steerings, path, color, size, sample_rate, sub_read, sample_balance)        
        # load sides image if the flag 'sides_image' is True
        if sides_image:
            load_sides_images(images, steerings, path, color, size, sample_rate, sub_read, sample_balance, offset)
    # Data Augmentation
    # flip_images()
    if data_augmentation:
        images_augmentation(images, steerings, factor=data_augmentation) 

    # Generate the training data
    print(images[0].shape)

    print("Gen X_train, y_train array from loaded images")
    X_train = np.array(images)
    y_train = np.array(steerings)

    # print the samples information
    print('Sample number: ', len(X_train))
    print("Sample shape: ", X_train.shape)
    print("Image shape: ", X_train[0].shape)
    print('Sample label number: ', len(y_train))
    print()

    # training the model
    if model == 'nvidia':
        model_name="m-{}_s-{}_lr-{}-{}_db{}_{}_aug{}_ep{}.h5".format(
                        model,sample_rate, sides_image,offset,sample_balance,color,data_augmentation,train_epoch)
        print(model_name, 'will be training and saved!!')
        print()
        model_nvidia(X_train, y_train, model_name)
    elif model == 'nvidia2':
        model_name="m-{}_s-{}_lr-{}-{}_db{}_{}_aug{}_ep{}.h5".format(
                        model,sample_rate, sides_image,offset,sample_balance,color,data_augmentation,train_epoch)
        print(model_name, 'will be training and saved!!')
        print()
        model_nvidia2(X_train, y_train, model_name)
    elif model == 'nvidia3':
        model_name="m-{}_s-{}_lr-{}-{}_db{}_{}_aug{}_ep{}.h5".format(
                        model,sample_rate, sides_image,offset,sample_balance,color,data_augmentation,train_epoch)
        print(model_name, 'will be training and saved!!')
        print()
        model_nvidia3(X_train, y_train, model_name, train_epoch)

    elif model == 'lenet':
        pass
    else:
        print('Wrong model choice, valid color is "nvidia", "lenet", "???"')
        sys.exit()

if __name__ == '__main__':

    train_model(data_path=('./data/', ), color='Y', size=(66, 200), sample_rate=1, sub_read=False, sample_balance=100, sides_image=False, offset=0.22, data_augmentation=1, 
                model='nvidia3', train_epoch=1)