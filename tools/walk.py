import os
from scipy.misc import imresize, imsave, imread
for dirName, subdirList, fileList in os.walk('./Latin/'):
    print('Found directory: ', dirName)
    for fname in fileList:
        x = imread(dirName+'/'+fname)
        x = x[:, :, 3]
        x = imresize(x, (28, 28, 1))
        imsave(dirName+'/'+fname, x)
