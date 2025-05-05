import numpy as np

IMG_SIZE = (328,356)    # Standard image size in CASIA dataset
HIST_GRID_SIZE = 4      # N in spatial_histogram()
EUCL_THRESH = 1.02       # Euclidean distance threshold for matching
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

    print(f'  => Euclidean Distance: {combinedDist}')
    return combinedDist, (combinedDist <= EUCL_THRESH)