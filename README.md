# Fingerprint Recognition System

Source code for the CSCI 158 fingerprint recognition project.

This project aims to demonstrate a fingerprint recognition system using pre-existing feature extraction functions and 2 hand-implemented classification methods; the performance of the latter 2 will be compared 

This project uses the Sokoto Coventry Fingerprint Dataset, which can be found here: https://www.kaggle.com/datasets/ruizgara/socofing

Packages used in this project:
- OpenCV: `opencv-python`
- Matplot: `matplotlib`
- Scikit-Image: `scikit-image`

## Running this Program

1. Open a terminal in the directory of your choice and clone this repository using `git clone  https://github.com/adriand2002/csci158-project-source.git`
2. Change your working directory using `cd csci158-project-source`
3. Download the dataset zip, unzip it in this directory, and rename the folder to `dataset`.
4. Install Python3.xx and pip if not done so already.
5. Install required packages using `pip install opencv-python matplotlib scikit-image`
6. Run using `python3 main.py`. 

## For Linux users on Wayland
The process might lock up on Linux desktop environments running in Wayland due to a Qt plugin dependency. The best workaround we've come up with is to create a child process of your terminal (i.e., running `bash` in your bash terminal) and then running the program as a background process with `python3 main.py &`. This will let you kill the child process normally.