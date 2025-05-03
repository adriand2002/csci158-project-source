# Fingerprint Recognition System

Source code for the CSCI 158 fingerprint recognition project.

This project aims to demonstrate a fingerprint recognition system using pre-existing feature extraction functions and 2 hand-implemented classification methods; the performance of the latter 2 will be compared 

Dataset used can be found here: https://www.kaggle.com/datasets/ruizgara/socofing

Packages used in this project:
- OpenCV: `opencv-python`
- Matplot: `matplotlib`
- Scikit-Image: `scikit-image`

## Notes:
- FOR LINUX USERS: The process might lock up on Linux desktop environments running in Wayland due to a Qt plugin dependency. The best workaround we've come up with is to create a child process of your terminal (i.e., running `bash` in your bash terminal) and then running the program with `python3 main.py &`. This will let you kill the child process normally.