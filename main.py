from helpers import extract_sample_minutiae, normalize_minutiae
from classification1 import classification_one
import json
import os

import random
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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
    for i in range(0,2):
        
        template = templates[i]
        subject = template["subject"]
        finger = template["finger"]
        endings = template["endings"]
        forks = template["forks"]
        print(f"[TESTING] Testing with subject {subject}'s right thumb.")

        print(f' => Loaded subject {subject}\'s right thumb ({len(endings)} ridge endings, {len(forks)} bifurcations)')

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
                rocLabels.append(1)

            rocDists.append(distance)

        
        # Choose new subjects to test rejection accuracy
        for j in range(0,3):
            if j == i:
                continue # Avoiding testing same subject

            newSubject = templates[j]["subject"]

            print(f'[SAMPLE PROCESSING] Testing against subject {newSubject}\'s right thumb.')
            imgPath = f'{cwd}/dataset/{newSubject}/R/{newSubject}_{finger}_0.bmp'

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
                rocLabels.append(0)
            else:
                print('  => REJECTED')
                classOneTrueNegatives += 1
                rocLabels.append(0)
            
            rocDists.append(distance)

    print(f'\n[RESULTS]')
    print(f' => Classification Method One (Histogram + Euclidean Distance)')
    print(f'    -> True positives: {classOneTruePositives}')
    print(f'    -> False positives: {classOneFalsePositives}')
    print(f'    -> True negatives: {classOneTrueNegatives}')
    print(f'    -> False negatives: {classOneFalseNegatives}')

    classOneAccuracy = (classOneTruePositives + classOneTrueNegatives) / (classOneTruePositives + classOneTrueNegatives + classOneTrueNegatives + classOneFalseNegatives)
    print(f'    -> Accuracy: {classOneAccuracy}')

    # Invert score (lower distance = "higher" score)
    rocDists = [-x for x in rocDists]
    fpr, tpr, thresholds = roc_curve(rocLabels, rocDists)
    roc_auc = auc(fpr, tpr)

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