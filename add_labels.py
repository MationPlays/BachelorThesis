# imports
from tqdm import tqdm
import numpy as np
import cv2
import os
# set to true to one once, then back to false unless you want to change something in your training data.
REBUILD_DATA = True

# class for the different kernel size and Testing picture folders and LABELS dictionary


class gblur_ks():
    IMG_SIZE = 32
    ks0 = "gblurred_pictures\gblur\ks0"
    ks1 = "gblurred_pictures\gblur\ks1"
    ks3 = "gblurred_pictures\gblur\ks3"
    ks5 = "gblurred_pictures\gblur\ks5"
    ks7 = "gblurred_pictures\gblur\ks7"
    ks9 = "gblurred_pictures\gblur\ks9"

    LABELS = {ks0: 0, ks1: 1, ks3: 2, ks5: 3, ks7: 4, ks9: 5}
    training_data = []

# count at zero
    ks0count = 0
    ks1count = 0
    ks3count = 0
    ks5count = 0
    ks7count = 0
    ks9count = 0

# make training data method
    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        # do something like print(np.eye(2)[1]), just makes one_hot
                        self.training_data.append(
                            [np.array(img), np.eye(6)[self.LABELS[label]]])
                        # print(np.eye(2)[self.LABELS[label]])

                # if its a ks0 for example increase counter for ks0 with 1
                        if label == self.ks0:
                            self.ks0count += 1
                        elif label == self.ks1:
                            self.ks1count += 1
                        elif label == self.ks3:
                            self.ks3count += 1
                        elif label == self.ks5:
                            self.ks5count += 1
                        elif label == self.ks7:
                            self.ks7count += 1
                        elif label == self.ks9:
                            self.ks9count += 1

                    except Exception as e:
                        pass
                        #print(label, f, str(e))

# print the amounts of the different kernel sizes
        np.save("training_data_v2.npy", self.training_data)
        print('amount with ks0:', gblurks.ks0count)
        print('amount with ks1:', gblurks.ks1count)
        print('amount with ks3:', gblurks.ks3count)
        print('amount with ks5:', gblurks.ks5count)
        print('amount with ks7:', gblurks.ks7count)
        print('amount with ks9:', gblurks.ks9count)


if REBUILD_DATA:
    gblurks = gblur_ks()
    gblurks.make_training_data()


training_data = np.load("training_data_v2.npy", allow_pickle=True)
print(len(training_data))
