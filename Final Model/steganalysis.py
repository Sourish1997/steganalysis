import numpy as np
from scipy import ndimage

########################################################################
# Same code for feature extraction as that in 'Image Preprocessing.py' #
########################################################################

def getPlaneBits(planeId, binary_image):
    return [int(b[planeId]) for b in binary_image]

def getBitPlanes(img):
    bin_image = []
    bit_planes = []

    for i in range(0, 512):
        for j in range(0, 512):
            bin_image.append(np.binary_repr(img[i][j], width = 8))
            
    for i in range(0, 8):
        bit_planes.append(np.array(getPlaneBits(i, bin_image)).reshape(512, 512))
            
    return bit_planes

from scipy.stats import pearsonr

def autocor(A, k, l):
    Xk = A[0:512 - k, 0:512 - l]
    Xl = A[k:512, l:512]
    return pearsonr(Xk.flatten(), Xl.flatten())

def getHl1(img_hist, l):
    return img_hist[0:256 - l]

def getHl2(img_hist, l):
    return img_hist[l:256]

def getCHl(img_hist, l):
    return pearsonr(getHl1(img_hist, l), getHl2(img_hist, l))

def getModifiedWavelet(C, t):
    for i, row in enumerate(C):
        for j, val in enumerate(row):
            if abs(val) < t:
                C[i][j] = 0
    return C

def getE(img, t):
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs

    LH = getModifiedWavelet(LH, t)
    HL = getModifiedWavelet(HL, t)
    HH = getModifiedWavelet(HH, t)

    img_denoised = pywt.idwt2((LL, (LH, HL, HH)), 'haar')

    E = img - img_denoised
    
    return E

def getCE(img, t, k, l):
    E = getE(img, t)
    return autocor(E, k, l)

import pywt

def getFeatures(filename):
    features = []
    
    img = ndimage.imread(filename, mode = 'L')
    bit_planes = getBitPlanes(img)

    autocor_kl_pairs = [[1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 1], [2, 2], [3, 3], [4, 4],
                       [1, 2], [2, 1]]

    M1 = bit_planes[0]
    M2 = bit_planes[1]

    features.append(pearsonr(M1.flatten(), M2.flatten())[0])

    for pair in autocor_kl_pairs:
        features.append(autocor(M1, pair[0], pair[1])[0])

    img_hist, bin_edges = np.histogram(img.flatten(), bins = list(range(0, 257)), density = True)

    He = [img_hist[i] for i in range(0, 256, 2)]
    Ho = [img_hist[i] for i in range(1, 256, 2)]

    features.append(pearsonr(He, Ho)[0])

    for i in range(1, 5):
        features.append(getCHl(img_hist, i)[0])
        
    autocor_tkl_triplets = [[1.5, 0, 1], [1.5, 1, 0], [1.5, 1, 1], [1.5, 0, 2], [1.5, 2, 0], [1.5, 1, 2], [1.5, 2, 1],
                       [2, 0, 1], [2, 1, 0], [2, 1, 1], [2, 0, 2], [2, 2, 0], [2, 1, 2], [2, 2, 1], [2.5, 0, 1],
                       [2.5, 1, 0], [2.5, 1, 1], [2.5, 0, 2], [2.5, 2, 0], [2.5, 1, 2], [2.5, 2, 1]]

    for triplet in autocor_tkl_triplets:
        features.append(getCE(img, triplet[0], triplet[1], triplet[2])[0])

    return features

######################################
# Image Preprocessing code ends here #
######################################

from sklearn.externals import joblib
from os import sys

# Path of input image file
filename = sys.argv[1]

# Load persisted models
models = [joblib.load('original_abc.pkl'), joblib.load('original_mlp.pkl'),
          joblib.load('good_abc.pkl'), joblib.load('good_mlp.pkl')]

# Extract CF feature set for chosen file
features = [getFeatures(filename)]

preds = []

# Let each model make a prediction
for model in models:
    preds.append(model.predict(features))

cover_count = 0
steg_count = 0
for pred in preds:
    if pred[0] == 1:
        steg_count += 1
    else:
        cover_count += 1

# Assign a label based on majority vote
if cover_count > steg_count:
    print("No steganography detected in the image.")
else:
    print("Steganography detected in the image!")
