import cv2
import imageio
from descriptor import detectAndDescribe
from Matcher import createMatcher, matchKeyPointsBF, matchKeyPointsKNN
from homography import getHomography
import numpy as np
import matplotlib.pyplot as plt

'List of inputs'

feature_extractor = 'orb'
feature_matching = 'bf'
cutoff = 2.4 #how many columns at the end of the panoramic image need to be cropped percent/100 (this should be less than overlap percentage)
images = ['Images/11.jpg','Images/4.jpg'] #Input the images to be stitched together

queryImg = imageio.imread(images[0])
queryImg0 = queryImg[:,0:2750]
queryImg = queryImg[:,2750:] #remove this line later

rows_filter = queryImg.shape[0]
column_filter = int(cutoff*queryImg.shape[1])

for i in range(1,len(images)):
    
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)
    trainImg = imageio.imread(images[i])

    trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)

    kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)
    
    if feature_matching == 'bf':
        matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
    elif feature_matching == 'knn':
        matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)

    M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
    if M is None:
        print("Error!, homography matrix is empty")
    (matches, H, status) = M
    
    'Panoramic corrections'
    
    width = trainImg.shape[1] + queryImg.shape[1]
    height = trainImg.shape[0] + queryImg.shape[0]

    result = cv2.warpPerspective(trainImg, H, (width, height))
    result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg
    result = result[:rows_filter,0:result.shape[1]-column_filter]
    
    queryImg = np.concatenate((queryImg0,result),axis=1)
    
    #del result
    
    
    plt.figure(figsize=(20,10))
    plt.imshow(queryImg)
    plt.axis('off')
    plt.show()
    
    
    
