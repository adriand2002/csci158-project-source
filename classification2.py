from helpers import extract_sample_minutiae, normalize_minutiae
import json
import os
from sklearn.metrics import roc_curve, auc
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = (328,356)    # Standard image size in CASIA dataset
K = 10                   # Number of nearest-neighbors to consider in graph
LENGTH_THRESH = 10      # Threshold for matching graph edges based on distance
ANGLE_THRESH = 20       # Ditto but for relative orientation between points
MATCH_THRESH = 0.981      # Threshold for determining acceptance based on how many edges match between sample/template graphs

################################################################################################################################
### CLASSIFICATION METHOD 2: MINUTIAE GRAPH MATCHING + KNN ANALYSIS ############################################################
################################################################################################################################

# Constructs a graph of the fingerprint's
# minutiae, and then finds the K nearest
# neighbors for each vertex in the graph.
def minutiae_graph(fingerprint):

    vertices = np.array(
        [(x,y) for x, y, theta in fingerprint["endings"]] 
      + [(x,y) for x, y, thetas in fingerprint["forks"]],
        dtype=np.float32 # workaround; kdTree function doesn't work with integers
    )

    # Construct a kdtree to efficiently find KNNs
    kdTree = KDTree(vertices)
    edges = []

    # find K nearest neighbors through KD tree search
    for i, vertex in enumerate(vertices):
        dists, indices = kdTree.query(vertex, k=K+1) # skip self

        # Building edges between nodes and nearest neighbors
        # Each edge holds info about distance and angle
        for j in range(1, len(indices)):
            neighbor = vertices[indices[j]]
            length = np.linalg.norm(vertex - neighbor)
            angle = np.degrees(np.arctan2(neighbor[1] - vertex[1], neighbor[0] - vertex[0])) % 360
            edges.append((i, indices[j], length, angle))

    return edges

# Compares edges in the graphs of two
# fingerprints, matching them if they
# have distances and angles within the
# target threshold when compared
def compare_edges(edgesA, edgesB):
    edgeMatches = 0

    for x1, y1, l1, a1 in edgesA:
        for x2, y2, l2, a2 in edgesB:
            lengthDiff = abs(l1 - l2)
            angleDiff = min(abs(a1 - a2), 360 - abs(a1 - a2))

            # Check if under thresholds
            if lengthDiff < LENGTH_THRESH and angleDiff < ANGLE_THRESH:
                # Likely match found
                edgeMatches += 1
                break # Stop searching; likely match already found

    # Normalize to bounds of edge count
    edgeMatches = edgeMatches / max(len(edgesA), 1)
    return edgeMatches

def classification_two(sample, template):

    # Generate minutiae graphs of input fingerprints
    sampleGraph = minutiae_graph(sample)
    templateGraph = minutiae_graph(template)

    # Compare graphs and decide on match based on score threshold
    matchScore = compare_edges(sampleGraph, templateGraph)
    
    return matchScore, (matchScore > MATCH_THRESH)

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
    rocScores = []

    print("[INIT] Grabbing templates")

    with open("templates.json","r") as jsonStr:
        templates = json.load(jsonStr)

    print(f' => Loaded {len(templates)} from dataset JSON')

    for i in range(0,len(templates)):
        
        template = templates[i]
        subject = template["subject"]
        finger = template["finger"]
        endings = template["endings"]
        forks = template["forks"]
        print(f"\n[TESTING] Testing with subject {subject}'s right thumb.")

        print(f' => Loaded subject {subject}\'s right thumb ({len(endings)} ridge endings, {len(forks)} bifurcations)')

        sum = 0
        sum2 = 0
        count = 0
        count2 = 0

        # Test against samples of the same finger to test acceptance accuracy
        for j in range(0,5):
            print(f'[SAMPLE PROCESSING] Testing against sample {j} of the same finger.')
            imgPath = f'{cwd}/dataset/{subject}/R/{subject}_{finger}_{j}.bmp'

            #print(f' => Extracting from {imgPath}...')
        
            minutiae = extract_sample_minutiae(imgPath)

            foundEndings = minutiae["endings"]
            foundForks = minutiae["forks"]

            #print(f' => Extracted {len(foundEndings)} ridge endings and {len(foundForks)} bifurcations')

            normTemplate = normalize_minutiae(template)
            normSample = normalize_minutiae(minutiae)

            score, accepted = classification_two(normSample,normTemplate)

            print(f' => Match Score: {score}')
        
            if accepted:
                print(' => ACCEPTED')
                classOneTruePositives += 1
                rocLabels.append(1)
            else:
                print(' => REJECTED')
                classOneFalseNegatives += 1
                rocLabels.append(1)

            rocScores.append(score)
            sum2 += score
            count2 += 1

        
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

            score, accepted = classification_two(normSample,normTemplate)

            print(f' => Match Score: {score}')

            if accepted:
                print(' => ACCEPTED')
                classOneFalsePositives += 1
                rocLabels.append(0)
            else:
                print(' => REJECTED')
                classOneTrueNegatives += 1
                rocLabels.append(0)
            
            rocScores.append(score)

            sum += score
            count += 1

    print(f'=> real score avg: {sum2/count2}')
    print(f'=> mismatch score avg: {sum/count}')

    fpr, tpr, thresholds = roc_curve(rocLabels, rocScores)
    roc_auc = auc(fpr, tpr)

    print(f'\n[RESULTS]')
    print(f' => Classification Method Two (Graph + Edge Matching)')
    print(f'    -> TPR: {classOneTruePositives / (classOneTruePositives + classOneFalseNegatives)}')
    print(f'    -> FPR: {classOneFalsePositives / (classOneFalsePositives + classOneTrueNegatives)}')
    print(f'    -> TPs: {classOneTruePositives}')
    print(f'    -> FPs: {classOneFalsePositives}')
    print(f'    -> TNs: {classOneTrueNegatives}')
    print(f'    -> FNs: {classOneFalseNegatives}')

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