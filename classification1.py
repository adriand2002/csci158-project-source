from helpers import extract_sample_minutiae, normalize_minutiae
import json
import os

import random
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = (328,356)    # Standard image size in CASIA dataset
HIST_GRID_SIZE = 4      # N in spatial_histogram()
EUCL_THRESH = 0.9066338893965881       # Euclidean distance threshold for matching; calculated by mismatch distance average
SPATIAL_WEIGHT = 0.7    # Affects how much spatial vs orientation data affects euclidean distance in classifier 1

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

    # Testing first 10 subjects from dataset
    for i in range(0,5):
        
        template = templates[i]
        subject = template["subject"]
        finger = template["finger"]
        endings = template["endings"]
        forks = template["forks"]
        print(f"\n[TESTING] Testing with subject {subject}'s right thumb.")

        print(f' => Loaded subject {subject}\'s right thumb ({len(endings)} ridge endings, {len(forks)} bifurcations)')

        sum = 0
        count = 0

        # Test against samples of the same finger to test acceptance accuracy
        for j in range(0,5):
            print(f'[SAMPLE PROCESSING] Testing against sample {j} of the same finger.')
            imgPath = f'{cwd}/dataset/{subject}/R/{subject}_{finger}_{j}.bmp'

            #print(f' => Extracting from {imgPath}...')
        
            minutiae = extract_sample_minutiae(imgPath)

            foundEndings = minutiae["endings"]
            foundForks = minutiae["forks"]

            #print(f' => Extracted {len(foundEndings)} ridge endings and {len(foundForks)} bifurcations')

            # print(f' [MATCHING 1] Minutiae Histogram')

            normTemplate = normalize_minutiae(template)
            normSample = normalize_minutiae(minutiae)

            distance, accepted = classification_one(normSample,normTemplate)

            print(f' => Euclidean Distance: {distance}')
        
            if accepted:
                print(' => ACCEPTED')
                classOneTruePositives += 1
                rocLabels.append(1)
            else:
                print(' => REJECTED')
                classOneFalseNegatives += 1
                rocLabels.append(1)

            rocDists.append(distance)

        
        # Choose new subjects to test rejection accuracy
        for j in range(0,195):
            if j == i:
                continue # Avoiding testing same subject

            newSubject = templates[j]

            print(f'[SAMPLE PROCESSING] Testing against subject {newSubject["subject"]}\'s right thumb.')
            #imgPath = f'{cwd}/dataset/{newSubject}/R/{newSubject}_{finger}_0.bmp'
        
            #minutiae = extract_sample_minutiae(imgPath)

            #foundEndings = minutiae["endings"]
            #foundForks = minutiae["forks"]

            foundEndings = newSubject["endings"]
            foundForks = newSubject["forks"]

            #print(f' [MATCHING 1] Minutiae Histogram')

            normTemplate = normalize_minutiae(template)
            normSample = normalize_minutiae(newSubject)

            distance, accepted = classification_one(normSample,normTemplate)

            print(f' => Euclidean Distance: {distance}')

            if accepted:
                print(' => ACCEPTED')
                classOneFalsePositives += 1
                rocLabels.append(0)
            else:
                print(' => REJECTED')
                classOneTrueNegatives += 1
                rocLabels.append(0)
            
            rocDists.append(distance)

            sum += distance
            count += 1

    print(f'=> mismatch distance avg: {sum/count}')

    # Invert score (lower distance = "higher" score)
    rocDists = [-x for x in rocDists]
    fpr, tpr, thresholds = roc_curve(rocLabels, rocDists)
    roc_auc = auc(fpr, tpr)

    print(f'\n[RESULTS]')
    print(f' => Classification Method One (Histogram + Euclidean Distance)')
    print(f'    -> TPR: {classOneTruePositives / (classOneTruePositives + classOneFalseNegatives)}')
    print(f'    -> FPR: {classOneFalsePositives / (classOneFalsePositives + classOneTrueNegatives)}')
    #print(f'    -> True negatives: {classOneTrueNegatives}')
    #print(f'    -> False negatives: {classOneFalseNegatives}')

    classOneAccuracy = (classOneTruePositives + classOneTrueNegatives) / (classOneTruePositives + classOneTrueNegatives + classOneFalsePositives + classOneFalseNegatives)
    print(f'    -> Accuracy: {classOneAccuracy}')

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.grid(True)
    plt.show()