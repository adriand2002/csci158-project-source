# Fingerprint Recognition System

Source code for the CSCI 158 fingerprint recognition project.

This project aims to demonstrate a fingerprint recognition system using pre-existing feature extraction functions and 2 hand-implemented classification methods; the performance of the latter 2 will be compared in terms of accuracy.

This project uses the Casia V5 Dataset, whose documentation and contents can be found here: http://biometrics.idealtest.org/downloadDesc.do?id=7#/datasetDetail/7. While this is a complete dataset, only each subject's right thumb is considered for feature extraction and template matching.

Packages used in this project:
- OpenCV: `opencv-python`
- Scikit-Image: `scikit-image`
- Numpy: `numpy`
- Fingerprint Feature Extractor: `fingerprint-feature-extractor`

## Running this Program

1. Open a terminal in the directory of your choice and clone this repository using `git clone https://github.com/adriand2002/csci158-project-source.git`
2. Change your working directory using `cd csci158-project-source`
3. Download the `CASIA-FingerprintV5(BMP).zip`, extract it into this directory, and rename the folder to `dataset`.
4. Install Python3.xx and pip if not done so already.
5. Install required packages using `pip install opencv-python matplotlib scikit-image`. For Linux users, you may be prompted to first create a virtual environment which can be done by running `python -m venv venv; source venv/bin/activate` before you use the pip install command.
6. Run using `python3 main.py`.

Feature extraction and template generation will take you a while to perform, so a json file with templates of each subject's right thumb has already been pregenerated for testing using gentemplate.py. If you wish, you can regenerate it by running `python3 gentemplate.py`.