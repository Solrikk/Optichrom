
import numpy as np
import cv2
from matplotlib import pyplot as plt

def match_features(image1_path, image2_path):
    img1 = cv2.imread(image1_path, 0)
    img2 = cv2.imread(image2_path, 0)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

    plt.imsave('orb_visualization.png', img3)
    
    return len(matches)

if __name__ == "__main__":
    num_matches = match_features('745439676.jpeg', 'photo_2024-08-05_10-51-54.jpg')
    print(f"ORB visualization saved as 'orb_visualization.png'")
    print(f"Number of matches: {num_matches}")
