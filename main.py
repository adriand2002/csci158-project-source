import fingerprint_feature_extractor
import json
import cv2
import os
import numpy as np
import random
import warnings
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category = FutureWarning)

IMG_SIZE = (328,356)    # Standard image size in CASIA dataset
HIST_GRID_SIZE = 4      # N in spatial_histogram()
EUCL_THRESH = 1.02       # Euclidean distance threshold for matching
SPATIAL_WEIGHT = 0.7    # Affects how much spatial vs orientation data affects euclidean distance in classifier 1

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

################################################################################################################################
### CLASSIFICATION METHOD 1: EUCLIDEAN MATCHING ############################################################
################################################################################################################################

# Computes the minutiae spatial histogram.
# Breaks the image into a NxN grid,
# and counts the number of endings
# and bifurcations in each.
def spatial_histogram(fingerprint):
    grid = np.zeros((2, HIST_GRID_SIZE, HIST_GRID_SIZE)) # 2 Layers, one for endings and one for bifurcations
    cellWidth = IMG_SIZE[0] / HIST_GRID_SIZE
    cellHeight = IMG_SIZE[1] / HIST_GRID_SIZE

    # Get count of ridge endings in each grid square
    endings = fingerprint["endings"]
    for ending in endings:
        gridCol = min(int(ending[0] // cellWidth), HIST_GRID_SIZE - 1)
        gridRow = min(int(ending[1] // cellHeight), HIST_GRID_SIZE - 1)
        grid[0, gridRow, gridCol] += 1

    # Get count of bifurcations in each grid square
    forks = fingerprint["forks"]
    for fork in forks:
        gridCol = min(int(fork[0] // cellWidth), HIST_GRID_SIZE - 1)
        gridRow = min(int(fork[1] // cellHeight), HIST_GRID_SIZE - 1)
        grid[1, gridRow, gridCol] += 1

    # Flatten grid into 1D vector
    vector = grid.flatten()

    return vector

def orientation_histogram(fingerprint):
    # flatten orientations
    orientations = []
    for x, y, theta in fingerprint["endings"]:
        orientations.append(theta)
    for x, y, thetaList in fingerprint["forks"]:
        orientations.extend(thetaList)

    # Create histogram
    orientHist, _ = np.histogram(orientations, bins=36, range=(0,360), density = True)
    return orientHist

def classification_one(sample, template):
    # Compute histograms
    sampleSpatialHist = spatial_histogram(sample)
    sampleOrientHist = orientation_histogram(sample)
    templateSpatialHist = spatial_histogram(template)
    templateOrientHist = orientation_histogram(template)

    # Normalizing histogram vectors
    sampleSpatialHist /= np.linalg.norm(sampleSpatialHist) + 1e-8 # 1e-8 added as hacky workaround to prevent division by zero
    sampleOrientHist /= np.linalg.norm(sampleOrientHist) + 1e-8
    templateSpatialHist /= np.linalg.norm(templateSpatialHist) + 1e-8
    templateOrientHist /= np.linalg.norm(templateOrientHist) + 1e-8

    # Compute euclidean distances between histograms
    spatialDist = np.linalg.norm(sampleSpatialHist - templateSpatialHist)
    orientDist = np.linalg.norm(sampleOrientHist - templateOrientHist)
    combinedDist = SPATIAL_WEIGHT * spatialDist + (1 - SPATIAL_WEIGHT) * orientDist

    print(f'  => Euclidean Distance: {combinedDist}')
    return combinedDist, (combinedDist <= EUCL_THRESH)

################################################################################################################################
### Main execution #############################################################################################################
################################################################################################################################

if __name__ == "__main__":
    cwd = os.getcwd()
    subjects = os.listdir(f'{cwd}/dataset/')

    classOneTruePositives = 0
    classOneTrueNegatives = 0
    classOneFalsePositives = 0
    classOneFalseNegatives = 0

    rocLabels = []
    rocDists = []

    print("[INIT] Grabbing templates")

    with open("templates.json","r") as jsonStr:
        templates = json.load(jsonStr)

    print(f' => Loaded {len(templates)} from dataset JSON')

    # Testing 5 random subjects
    for i in range(0,5):
        
        template = random.choice(templates)
        subject = template["subject"]
        finger = template["finger"]
        endings = template["endings"]
        forks = template["forks"]
        print(f"[TESTING] Testing with subject {subject}'s right thumb.")

        print(f' => Loaded finger {finger} of subject {subject} ({len(endings)} ridge endings, {len(forks)} bifurcations)')

        # Test against samples of the same finger to test acceptance accuracy
        for j in range(0,5):
            print(f'[SAMPLE PROCESSING] Testing against sample {j} of the same finger.')
            imgPath = f'{cwd}/dataset/{subject}/R/{subject}_{finger}_{j}.bmp'

            #print(f' => Extracting from {imgPath}...')
        
            minutiae = extract_sample_minutiae(imgPath)

            foundEndings = minutiae["endings"]
            foundForks = minutiae["forks"]

            #print(f' => Extracted {len(foundEndings)} ridge endings and {len(foundForks)} bifurcations')

            print(f' [MATCHING 1] Minutiae Histogram')

            normTemplate = normalize_minutiae(template)
            normSample = normalize_minutiae(minutiae)

            distance, accepted = classification_one(normSample,normTemplate)
        
            if accepted:
                print('  => ACCEPTED')
                classOneTruePositives += 1
                rocLabels.append(1)
            else:
                print('  => REJECTED')
                classOneFalseNegatives += 1
                rocLabels.append(0)

            rocDists.append(distance)

        
        # Choose new subjects to test rejection accuracy
        for j in range(1,5):
            randomSubject = templates[random.choice(range(0,len(templates)))]["subject"]

            print(f'[SAMPLE PROCESSING] Testing against subject {randomSubject}\'s finger.')
            imgPath = f'{cwd}/dataset/{randomSubject}/R/{randomSubject}_{finger}_0.bmp'

            #print(f' => Extracting from {imgPath}...')
        
            minutiae = extract_sample_minutiae(imgPath)

            foundEndings = minutiae["endings"]
            foundForks = minutiae["forks"]

            #print(f' => Extracted {len(foundEndings)} ridge endings and {len(foundForks)} bifurcations')

            print(f' [MATCHING 1] Minutiae Histogram')

            normTemplate = normalize_minutiae(template)
            normSample = normalize_minutiae(minutiae)

            distance, accepted = classification_one(normSample,normTemplate)

            if accepted:
                print('  => ACCEPTED')
                classOneFalsePositives += 1
            else:
                print('  => REJECTED')
                classOneTrueNegatives += 1
            
    print(f'\n[RESULTS]')
    print(f' => Classification Method One (Histogram + Euclidean Distance)')
    print(f'    -> True positives: {classOneTruePositives}')
    print(f'    -> False positives: {classOneFalsePositives}')
    print(f'    -> True negatives: {classOneTrueNegatives}')
    print(f'    -> False negatives: {classOneFalseNegatives}')

    classOneAccuracy = (classOneTruePositives + classOneTrueNegatives) / (classOneTruePositives + classOneTrueNegatives + classOneTrueNegatives + classOneFalseNegatives)
    print(f'    -> Accuracy: {classOneAccuracy}')