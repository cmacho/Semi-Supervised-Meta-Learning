
import numpy as np
import random
import os
import pathlib
import PIL.Image as Image
from abc import ABC, abstractmethod


def load_data_from_folders(directory):
    """ directory .. path to a directory where each subfolder corresponds 
                     to a class and contains images of that class
        returns:
                X .. a numpy array of shape (N, height, width, channels) containing images
                Y .. a numpy array of shape (1,N) containing labels"""
    data_dir = pathlib.Path(directory)
    list_data_dir = list(data_dir.glob("*/*"))
    image_count = len(list_data_dir)

    class_folders = np.array([item.name for item in data_dir.glob('*')])

    def get_label(file_path):
        # convert the path to a list of path components
        parts = file_path.split("/")
        # The second to last is the class-directory
        return np.where(class_folders == parts[-2])[0][0]

    img1 = Image.open(list_data_dir[0])
    img1_array = np.array(img1)
    assert(img1_array.ndim == 3)
    height = img1_array.shape[0]
    width = img1_array.shape[1]
    channels = img1_array.shape[2]


    X = np.zeros((image_count, height, width, channels))
    Y = np.zeros((1,image_count))

    for ii in range(image_count):
        img = Image.open(list_data_dir[ii])
        img_array = np.array(img) / 255.
        label = get_label( str(list_data_dir[ii]) )
        X[ii] = img_array
        Y[0,ii] = label

    total_classes = np.array([len(class_folders)])
    total_classes = total_classes.reshape((1,))
    return X, Y, total_classes


class BaseDataLoader(ABC):
    @abstractmethod
    def sample_batch(self, num_classes, samples_per_class, batch_size):
        pass


class DataLoader(object):
    def __init__(self, X, Y, total_classes):
        """X of shape (N, height, width, channels)
        Y of shape (num_rows, N)
        total_classes (num_rows,)
        """
        self.X = X
        self.Y = Y
        # total_classes is no longer used in current version
        # self.total_classes = total_classes
        self.used = np.zeros(Y.shape)
        self.height = X.shape[1]
        self.width = X.shape[2]
        self.channels = X.shape[3]
        self.num_images = X.shape[0]
        # for i in range(Y.shape[0]):
        #     assert(int(np.max(Y[i])) == total_classes[i] - 1)

    def select_classes(self, num_classes, row):
        return random.sample(list(np.unique(self.Y[row])), num_classes)


    def sample_meta_task(self, num_classes, samples_per_class):
        meta_task_data = np.zeros((num_classes,samples_per_class,self.height,self.width,self.channels))

        # decide which row of Y to use labels from:
        row = random.randint(0,self.Y.shape[0] - 1)

        classes = self.select_classes(num_classes, row)

        for ii in range(num_classes):
            curr_class = classes[ii]
            #print(f"ii is {ii}, curr_class is {curr_class}")

            relevant = np.where((self.Y[row,:] == curr_class) & (self.used[row,:] == 0))[0]
            if relevant.size < samples_per_class:
                self.used[row, self.Y[row, :] == curr_class] = 0
                relevant = np.where((self.Y[row,:] == curr_class))[0]
                #print(f"resetted used[{row},:]. relevant is now", relevant)
            while relevant.size < samples_per_class:
                #print("(in loop) relevant is now", relevant)
                relevant = np.concatenate((relevant, relevant))
                #print(f"class {curr_class} in row {row} had not enough examples")
                #print("doubling the np-array relevant")

            chosen_examples = np.random.choice(relevant, samples_per_class, replace=False)

            self.used[row, chosen_examples] = 1
            meta_task_data[ii,:,:,:,:] = self.X[chosen_examples]

        return meta_task_data

    def sample_batch(self, num_classes, samples_per_class, batch_size):
        batch = np.zeros((batch_size, num_classes, samples_per_class, self.height, self.width, self.channels))
        for ii in range(batch_size):
            #print(f"current batch is {ii}")
            batch[ii] = self.sample_meta_task(num_classes, samples_per_class)
        return batch


class DataLoaderFromSmallNumOfTasks(object):
    def __init__(self, tasks):
        self.tasks = tasks
        self.num_tasks = tasks.shape[0]
        self.used = 0
        self.num_classes = tasks.shape[1]
        self.samples_per_class = tasks.shape[2]

    def sample_batch(self,num_classes, samples_per_class, batch_size):
        assert(num_classes == self.num_classes)
        assert(samples_per_class == self.samples_per_class)

        while(batch_size > self.num_tasks):
            print("doubling tasks because batch_size > num_tasks")
            self.tasks = np.concatenate((self.tasks,self.tasks), axis=0)
            self.num_tasks = self.tasks.shape[0]

        if self.used + batch_size > self.num_tasks:
            self.used = 0
            np.random.shuffle(self.tasks)

        batch = self.tasks[self.used:self.used + batch_size]
        self.used = self.used + batch_size
        return batch


class DataLoaderCombiner(object):
    def __init__(self, data_loader1, data_loader2, num_samples_list1, num_samples_list2):
        self.data_loader1 = data_loader1
        self.data_loader2 = data_loader2
        self.num_samples_list1 = num_samples_list1
        self.num_samples_list2 = num_samples_list2
        self.period = len(num_samples_list1)
        assert len(num_samples_list2) == self.period
        self.count_batches = 0

    def sample_batch(self, num_classes, samples_per_class, batch_size):
        idx = self.count_batches % self.period
        batch_size1 = self.num_samples_list1[idx]
        batch_size2 = self.num_samples_list2[idx]
        assert batch_size == batch_size1 + batch_size2

        batch1 = self.data_loader1.sample_batch(num_classes, samples_per_class, batch_size1)
        batch2 = self.data_loader2.sample_batch(num_classes, samples_per_class, batch_size2)
        batch = np.concatenate([batch1, batch2])
        self.count_batches = self.count_batches + 1
        return batch


def augment_data(batch, shuffle = True, flip = True, crop = False, crop_height = 74, crop_width = 74):
    batch_size, num_classes, samples_per_class, h, w, c = batch.shape
     
    if crop: 
        new_batch = np.zeros((batch_size, num_classes, samples_per_class, crop_height, crop_width,c))
    else:
        new_batch = np.copy(batch)
    for ii in range(batch_size):
        for jj in range(num_classes):
            for kk in range(samples_per_class):
                img = Image.fromarray(np.uint8(batch[ii,jj,kk] * 255))
                if flip == True and random.random() < 0.5:
                    img = img.transpose(method = Image.FLIP_LEFT_RIGHT)
                if crop == True:
                    left = random.randint(0, w - crop_width)
                    upper = random.randint(0, h - crop_height)
                    right = left + crop_width
                    lower = upper + crop_height
                    img = img.crop((left, upper, right, lower))
                new_batch[ii, jj, kk, :, :, :] = np.array(img) / 255.
            if shuffle == True:
                np.random.shuffle(new_batch[ii, jj])
    return new_batch


if __name__ == '__main__':
    import datetime
    image_directory = "/home/clemens/Pictures/data_set_for_test_dataloader"
    parent_directory = "/home/clemens/cs330_proj/test_dataloader/test1"
    X, Y, total_num_classes = load_data_from_folders(image_directory)

    print("max X is", np.max(X))
    print("min X is ", np.min(X))
    print("avg X is", np.mean(X))
    print(X[0])

    output_dir = os.path.join(parent_directory, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.mkdir(output_dir)

    dataloader = DataLoader(X,Y,total_num_classes)

    e = dataloader.sample_batch(3,2,8)

    for ii in range(e.shape[0]):
        batch_folder = os.path.join(output_dir, str(ii))
        os.mkdir(batch_folder)
        for jj in range(e.shape[1]):
            subfolder = os.path.join(batch_folder, str(jj))
            os.mkdir(subfolder)
            for kk in range(e.shape[2]):
                im = Image.fromarray(np.uint8(e[ii,jj,kk] * 255))
                im.save( os.path.join(subfolder, str(kk) + ".png"))

    print("augmenting data")
    new_e = augment_data(e)
    print("finished augmenting data")

    for ii in range(new_e.shape[0]):
        batch_folder = os.path.join(output_dir, str(ii) + "_augmented")
        os.mkdir(batch_folder)
        for jj in range(new_e.shape[1]):
            subfolder = os.path.join(batch_folder, str(jj) + "_augmented")
            os.mkdir(subfolder)
            for kk in range(new_e.shape[2]):
                im = Image.fromarray(np.uint8(new_e[ii,jj,kk] * 255))
                im.save( os.path.join(subfolder, str(kk) + ".png"))





