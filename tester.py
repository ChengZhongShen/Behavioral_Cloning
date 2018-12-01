# test file for model.py, test the image_aug function
def image_aug_tester():
    
    image_shape = (160, 320, 3) # image shape date will be used in other functions

    # cread the list to hold the features and labels
    images = []
    steerings = []

    # images load parameter
    offset_input = 0.22
    sub_read_input = 12
    sample_balance_input = False
   
    # read data
    load_images(images, steerings, path='../data/',split= '/', sub_read=sub_read_input, sample_balance=sample_balance_input)
    load_sides_images(images, steerings, path='../data/',split= '/', offset=offset_input, sub_read=sub_read_input, sample_balance=sample_balance_input) 

    from sklearn.utils import shuffle
    images, steerings = shuffle(images, steerings) 

    from matplotlib.gridspec import GridSpec
    samples_fig = plt.figure(figsize=(16,6))
    rows = 3
    cols = 4
    count = 0
    gs = GridSpec(rows,cols, left=0.02,right=0.98, top=0.98, bottom=0.02, wspace=0.02,hspace=0.02)
    for i in range(rows):
        for j in range(cols):
            ax = samples_fig.add_subplot(gs[i,j])
            ax.imshow(images[count])
            ax.set_xticks([])
            ax.set_yticks([])
            count += 1

    images_augmentation(images, steerings, factor=1.0, rotate=15, shift=0.2)

    aug_fig = plt.figure(figsize=(16,6))
    rows = 3
    cols = 4
    count = 0
    gs = GridSpec(rows,cols,left=0.02,right=0.98, top=0.98, bottom=0.02, wspace=0.02,hspace=0.02)
    for i in range(rows):
        for j in range(cols):
            ax = aug_fig.add_subplot(gs[i,j])
            ax.imshow(images[count+sub_read_input*3])
            ax.set_xticks([])
            ax.set_yticks([])
            count += 1
    
    print(steerings[:12])
    print(steerings[sub_read_input*3:sub_read_input*3+12])
    
    plt.show()

def image_aug_tester_v2():
    '''
    rewrite the tester for latest model.py file
    '''

    # cread the list to hold the features and labels
    images = []
    steerings = []

    # images load parameter
    offset_input = 0.22
    sub_read_input = 12
    sample_balance_input = False
    color = 'RGB'
    sample_rate=1
   
    # read data
    load_images(images, steerings, path='../data/',color=color, sample_rate=1, sub_read=sub_read_input, sample_balance=sample_balance_input)
    load_sides_images(images, steerings, path='../data/',color=color, sample_rate=1, sub_read=sub_read_input, sample_balance=sample_balance_input, offset=offset_input) 

    from sklearn.utils import shuffle
    images, steerings = shuffle(images, steerings) 

    from matplotlib.gridspec import GridSpec
    samples_fig = plt.figure(figsize=(8,3))
    rows = 3
    cols = 3
    count = 0
    gs = GridSpec(rows,cols, left=0.02,right=0.98, top=0.98, bottom=0.02, wspace=0.02,hspace=0.02)
    for i in range(rows):
        for j in range(cols):
            ax = samples_fig.add_subplot(gs[i,j])
            ax.imshow(images[count])
            ax.set_xticks([])
            ax.set_yticks([])
            count += 1

    images_augmentation(images, steerings, factor=1.0, rotate=15, shift=0.2)

    aug_fig = plt.figure(figsize=(8,3))
    rows = 3
    cols = 3
    count = 0
    gs = GridSpec(rows,cols,left=0.02,right=0.98, top=0.98, bottom=0.02, wspace=0.02,hspace=0.02)
    for i in range(rows):
        for j in range(cols):
            ax = aug_fig.add_subplot(gs[i,j])
            ax.imshow(images[count+sub_read_input*3])
            ax.set_xticks([])
            ax.set_yticks([])
            count += 1
    
    print(steerings[:12])
    print(steerings[sub_read_input*3:sub_read_input*3+12])
    
    plt.show()

if __name__ == "__main__":

    # from model_3 import * # copy mode_3 from old_src to current folder  
    # image_aug_tester() # this old version tester for model_3.py file

    from model import *
    image_aug_tester_v2()
