
# CarND Project
# model_3: create the train_model function to put all the parameter change in this funciton, 
#          rewrite the model_nvidia, name navidia2(input size from (90, 320, 3) to (66, 100, 3), use strides 2x2 at kernel 5x5, get rid of maxpooling. 2018/11/27
# model_3_yuv: load_images/load_sides_image load the yuv image, process_yuv changed, model_nvidia not change, work with drive_yuv.py 2018//11/28
# model_3a_yuv: data_path in train_model function became a tuple to le the train_modle function could load more than one data folder.
#               del the data_split parameter in train_model(), load_images(), load_sides_images(), auto detect split is '/' or '\' 2018/11/29
# model_3b_yuv: add sampling rate. (to sovle the problem training memeroy issue.) 1/2 sample rate
# model_3c_yuv: the drive_yuv change, the output is (66,200,3) yuv, no need for pre-porcess, to save memeray in workspace to load more data and aug the data
# model_3c_color: implement the RGB, HSV for compare.
# model.py: clearn up, del unneccsary functions. 2018/12/1

# import lib, the tensorflow and keras imported in the function which used them
import csv
import matplotlib.pyplot as plt
import cv2
import pickle
from tqdm import tqdm
import numpy as np
import sys

def load_log_file(path, sample_rate=2):
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

def process_image(image_file, color):
    """
    read the image from image_file and preprocess the image
    """
    image = cv2.imread(image_file)
    
    if color == "YUV":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) # Note, the drive_yuv.py feed simulator RGB image(160,320,3)
    elif color == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color == "HSV":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        print("Wrong color choice, valid color is 'YUV', 'RGB', 'HSV'.")
        sys.exit()

    image = cv2.resize(image, (200,100)) # (160,320,3)-->(100,200,3)
    image = image[25:91,:,:] # (100,200,3)-->(66,200,3)

    return image

def load_images(images, steerings, path, color, sample_rate, sub_read=False, sample_balance=False):
    '''
    append the center image to the list 'images'.
    append the steering data to list 'steerings'
    path: the data folder
    split: "/" lod file from  Linux system;  "\\" log file  from windows system
    sub_read: False or int, if number inputted, will just read the number recevied
    sample_balance: False/0 or int (1, 101), balance the steering data will >50% is zero or <0.02 (steering is -1.0-1.0, 0.02 is 1%, just cosider the number < 1% is noise)
                    another view of point, steering < 0.02, the car will keep straight forward, 
                    this is affect the cars perfomace in curve lane, when the >50% data feed to the model is just keep the car straight forword
    '''
    print("Loading Centeral image from: ", path)
    print("The image loaded color: {}, size: (66,220,3)".format(color))
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

    for indx, line in tqdm(enumerate(lines)):
        # skip the the loop after load __ images, for debug/tester
        if sub_read:
            if indx == sub_read:
                break      
        image_source = line[0] # the image path in the cvs file(recorded path)
        filename = image_source.split(split)[-1]
        image_path = path + 'IMG/' + filename

        image = process_image(image_path, color)
        
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

def load_sides_images(images, steerings, path, color, sample_rate, offset, sub_read=False, sample_balance=False):
    '''
    append the left/right image to the list 'images'.
    append the offseted steering data to list 'steerings'
    path: the data folder
    split: "/" lod file from  Linux system;  "\\" log file keeped from windows system
    offset: offset the steering of light/right images
    sub_read: False or int, if number inputted, will just read the number recevied
    sample_balance: False or int (0, 101), balance the steering data will >50% is zero or <0.02 (steering is -1.0-1.0, 0.02 is 1%, just cosider the number < 1% is noise)
                    another view of point, steering < 0.02, the car will keep straight forward, 
                    this is affect the cars perfomace in curve lane, when the >50% data feed to the model is just keep the car straight forword
                    Note: seems sample_balnce for side images is not neccessary. (Decided if get rid of later)
    '''
    
    print("Loading Left/Right image from: ", path)
    print("The image loaded color: {}, size: (66,220,3)".format(color))
    print("Left image will offset: {}, Right images will offset: {}".format(-offset, offset))
    
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


        image_left = process_image(image_path_left, color)

        image_right = process_image(image_path_right, color)

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
    return image

def image_shift(image, dx, dy):
    '''
    shift the image dx and dy
    '''
    image_shape = image.shape
    M = np.float32([[1,0,dx], [0,1,dy]])
    image = cv2.warpAffine(image, M, (image_shape[1], image_shape[0]))
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

    for indx in tqdm(range(num_aug)):
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

def show_loss(history):
    """
    show the training and validation loss
    """
    # plot the training and validation loss for each epoch
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def model_lenet(save_name):
    '''
    the name should be *.h5
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

def model_nvidia2(X_train, y_train, save_name, epoch=10):
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

def model_nvidia2a(X_train, y_train, save_name, learn_par=(0.001,256,20)):
    '''
    X_train: training features
    y_train: training labels
    save_name: saved model name
    learn_part: learning rate, batch_size, epochs
    2018/11/27 rewrite the nvidia model according the papaer. (add three color choice, RGB/HSV/YUV)
    Trid tensorflow 1.8.0, but the speed of Workspace is not acceptable.
    the name should be *.h5
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    2018/12/1: add visulize polt of loss, 
               early stop, 
               change parameter "epoch" to learn_par
    '''
    # Built the model
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Cropping2D
    from keras import optimizers, callbacks
    
    print("Loading the model...")

    model = Sequential()

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
        
    adam = optimizers.Adam(lr=learn_par[0], beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)

    early_stop=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

    model.compile(loss='mse', optimizer=adam)
    print("Loss is MSE, optimizer is adam, callback: early_stop")
    history=model.fit(X_train, y_train, batch_size=learn_par[1], validation_split=0.2, shuffle=True, epochs=learn_par[2], callbacks=[early_stop])

    model.save(save_name)
    print(save_name, 'saved!!!')

    show_loss(history)

def model_nvidia2b(X_train, y_train, save_name, learn_par=(0.001,256,20)):
    '''
    X_train: training features
    y_train: training labels
    save_name: saved model name
    learn_part: learning rate, batch_size, epochs
    2018/11/27 rewrite the nvidia model according the papaer. (add three color choice, RGB/HSV/YUV)
    Trid tensorflow 1.8.0, but the speed of Workspace is not acceptable.
    the name should be *.h5
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    2018/12/1: add visulize polt of loss, 
               early stop, 
               change parameter "epoch" to learn_par
               add drop at conv layers (dif with nvidia2a)
    '''
    # Built the model
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Cropping2D
    from keras import optimizers, callbacks
    
    print("Loading the model...")

    model = Sequential()

    # Normalize Input (66, 200, 3)
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66,200,3))) # normalize, from drive_yuv.py is (66,200,3) YUV image
    
    # Conv1 (66, 200, 3) --> (31, 98, 24)
    model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2)) # keras dropout para is the problity of get rid of the nero
    # Conv2 (31, 98, 24) --> (14, 47, 36)
    model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    # Conv3 (14, 47, 36) --> (48, 5, 22)
    model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    # Conv4 (48, 5, 22) --> (64, 3, 20)
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))  
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
        
    adam = optimizers.Adam(lr=learn_par[0], beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)

    early_stop=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

    model.compile(loss='mse', optimizer=adam)
    print("Loss is MSE, optimizer is adam, callback: early_stop")
    history=model.fit(X_train, y_train, batch_size=learn_par[1], validation_split=0.2, shuffle=True, epochs=learn_par[2], callbacks=[early_stop])

    model.save(save_name)
    print(save_name, 'saved!!!')

    show_loss(history)

def train_model(data_path, sample_rate=1, data_sub_read=False, data_sample_balance=False, sides_image=True, sides_image_offset=0.22, data_augmentation=1, 
                model='nvidia2a', input_color='RGB', learn_par=(0.001, 256, 20)):
    '''
    data_path: a tuple include the data folder, ('./data/', ), or ('./data_t2_2/')
    sample_rate: int, 1, 100% sample rate; 2, 50% samle rate; 3 33.3% sample rate
    data_sub_read: False/int, if False, load all the data in the folder; int, for example, 50, only load 50 images in the folder (it is useful only load 50/100 image in the debug)
    data_sample_balance: False/int, if False, load all the image in the folder; int, for example 2o, only keep the 20% images which's steering is <0.02, adjust the data's perect of straight and curve image
    sides_image: True/False, if load the left/right camera images
    sides_image_offset: the offset of steering of left and right image compare with center image
    data_augmentation: False/Float, rotate, shift the image, for example, 1.5, expand the data for 1x to (1+1.5)x
    model: choice the model for training, only navidia2 is availiable now
    input_color: choice the color of training image, "RGB", "YUV", "HSV" is available
    train_epoch: 
    ''' 

    # cread the list to hold the features and labels
    images = []
    steerings = []
   
    # read data
    for path in data_path:
        # load center image
        load_images(images, steerings, path=path, color=input_color, sample_rate=sample_rate, sub_read=data_sub_read, sample_balance=data_sample_balance)        
        # load sides image if the flag 'sides_image' is True
        if sides_image:
            load_sides_images(images, steerings, path=path, color=input_color, sample_rate=sample_rate,sub_read=data_sub_read, sample_balance=data_sample_balance, offset=sides_image_offset)   

    # Data Augmentation
    # flip_images()
    if data_augmentation:
        images_augmentation(images, steerings, factor=data_augmentation) 

    # Generate the training data
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
    if model == 'nvidia2':
        model_name="m-{}_s-{}_lr-{}-{}_db{}_{}_aug{}_ep{}.h5".format(
                        model,sample_rate, sides_image,sides_image_offset,data_sample_balance,input_color,data_augmentation,learn_par[2])
        print(model_name, 'will be training and saved!!')
        print()
        model_nvidia2(X_train, y_train, model_name, epoch=train_epoch)
    elif model == 'nvidia2a':
        model_name="m-{}_s-{}_lr-{}-{}_db{}_{}_aug{}_ep{}_batch{}_rate{}.h5".format(
                        model,sample_rate, sides_image,sides_image_offset,data_sample_balance,input_color,data_augmentation,learn_par[2], learn_par[1], learn_par[0])
        print(model_name, 'will be training and saved!!')
        print()
        model_nvidia2a(X_train, y_train, model_name, learn_par)
    elif model == 'nvidia2b':
        model_name="m-{}_s-{}_lr-{}-{}_db{}_{}_aug{}_ep{}_batch{}_rate{}.h5".format(
                        model,sample_rate, sides_image,sides_image_offset,data_sample_balance,input_color,data_augmentation,learn_par[2], learn_par[1], learn_par[0])
        print(model_name, 'will be training and saved!!')
        print()
        model_nvidia2b(X_train, y_train, model_name, learn_par)
    elif model == 'lenet':
        print('Not Implemented!!, Exit:')
        sys.exit()
    else:
        print('Wrong model choice, valid color is "nvidia2", "lenet",')
        sys.exit()

if __name__ == '__main__':

    train_model(data_path=('./data/', './data_t2_2/', './data_t2_reverse/'), sample_rate=1, data_sub_read=False, data_sample_balance=0.01, sides_image=True, sides_image_offset=0.3, data_augmentation=3, 
                model='nvidia2a', input_color='RGB', learn_par=(0.001, 256, 20))