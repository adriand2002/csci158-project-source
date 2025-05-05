import fingerprint_feature_extractor
import numpy as np
import cv2

import warnings
warnings.simplefilter(action='ignore', category = FutureWarning)

IMG_SIZE = (328,356)

################################################################################################################################
### Helper feature extraction and normalization methods ########################################################################
################################################################################################################################

# Extracts minutiae from a single fingerprint sample.
def extract_sample_minutiae(imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    endingsRaw, forksRaw = fingerprint_feature_extractor.extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=True, showResult=False, saveResult=False)

    endings = []
    for ending in endingsRaw:
        pos = (
            int(ending.locX), 
            int(ending.locY), 
            ending.Orientation
        )
        endings.append(pos)

    forks = []
    for fork in forksRaw:
        tineOrientation = [tine for tine in fork.Orientation if not np.isnan(tine)]
            
        pos = (
            int(fork.locX), 
            int(fork.locY), 
            tineOrientation
        )
        forks.append(pos)

    minutiae = {
        "endings": endings,
        "forks": forks
    }

    return minutiae

# Normalizes a sample/template's minutiae data:
# - Centers around coordinate median of minutiae
# - Rotates to align minutiae along standard principal component
# - Rescales to smaller image side length
def normalize_minutiae(minutiae):

    # Hacky workaround; need to flatten the data so it works with numpy
    endings = [ [x, y, angles] for x, y, angles in minutiae["endings"] ]
    forks = [ [x, y, angles] for x, y, angles in minutiae["forks"] ]

    # check if dataset is empty
    if len(endings) == 0 and len(forks) == 0:
        return minutiae
    
    # combines minutiae coordinates into one array
    coords = []
    if len(endings) > 0:
        coords.extend([e[:2] for e in endings])
    if len(forks) > 0:
        coords.extend([f[:2] for f in forks])

    # center at mean
    center = np.mean(coords, axis=0)
    centeredCoords = coords - center

    # rotate to align with PC
    if len(centeredCoords) >= 2:
        covMat = np.cov(centeredCoords.T)
        _, eigVecs = np.linalg.eigh(covMat)
        pcAxis = eigVecs[:,-1]
        axisAngle = -np.arctan2(pcAxis[1], pcAxis[0])

        rotMat = np.array([
            [np.cos(axisAngle), -np.sin(axisAngle)],
            [np.sin(axisAngle), np.cos(axisAngle)]
        ])

        rotatedCoords = centeredCoords @ rotMat.T
    else:
        rotatedCoords = centeredCoords
        axisAngle = 0

    # rescale to standard size
    minXY = rotatedCoords.min(axis=0)
    maxXY = rotatedCoords.max(axis=0)
    rangeXY = maxXY - minXY

    if rangeXY[0] == 0 or rangeXY[1] == 0:
        scale = 1
    else:
        targetSize = IMG_SIZE[0]
        scale = min(targetSize / rangeXY[0], targetSize / rangeXY[1])

    scaledCoords = (rotatedCoords - minXY) * scale

    # create normalized minutiae data dictionary
    normEndings = []
    normForks = []

    index = 0
    for i in range(len(endings)):
        x, y = scaledCoords[index]
        theta = endings[i][2][0]
        theta = (theta + np.degrees(axisAngle)) % 360
        normEndings.append([x, y, theta])
        index += 1

    for i in range(len(forks)):
        x, y = scaledCoords[index]
        orientations = forks[i][2]
        orientations = [(orientation + np.degrees(axisAngle)) % 360 for orientation in orientations]
        normForks.append([x, y, orientations])
        index += 1

    normMinutiae = {
        "endings" : normEndings,
        "forks" : normForks
    }

    return normMinutiae