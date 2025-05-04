import fingerprint_feature_extractor
import json
import cv2
import os
import numpy as np

if __name__ == "__main__":
    fingerprintData = []

    cwd = os.getcwd()
    subjects = os.listdir(f'{cwd}/dataset/')

    count = 0

    for subject in subjects:
        count += 1
        
        rightPrints = os.listdir(f'{cwd}/dataset/{subject}/R/')

        for fprint in rightPrints:

            # Breaks the filename up into its constituent parts.
            # Each filename follows the format XXX_YY_Z.bmp
            # where X is the subject ID, Y is the finger, and Z is the kth sample of that finger.

            # Only considers the right thumb
            # Only considers the first fingerprint for template creation as 
            # these are the "control" prints with upright orientation
            # and normal pressure.
            
            sample = fprint[7]
            finger = fprint[5]

            if sample != "0" or finger != "0":
                continue

            print (f"[PROCESSING] {fprint} (subject {count}/499)")
            
            id = fprint[0:3]
            finger = fprint[4:6]
            hand = finger[0]


        ## IMAGE PREPROCESSING
            path = f'{cwd}/dataset/{id}/{hand}/{fprint}'

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            _, img = cv2.threshold(img, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #cv2.imshow("", img)

            endingsRaw, forksRaw = fingerprint_feature_extractor.extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=True, showResult=False, saveResult=False)

            # Converting the MinutiaeFeature object from fingerprint feature extractor to dictionaries
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

            dictionary = {
                "subject" : id,
                "finger" : finger,
                "endings": endings,
                "forks": forks
            }

            fingerprintData.append(dictionary)

    jsonObj = json.dumps(fingerprintData, indent=2)

    with open("templates.json","w") as outfile:
        outfile.write(jsonObj)