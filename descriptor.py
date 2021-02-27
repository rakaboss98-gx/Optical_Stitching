import cv2

def detectAndDescribe(image,method=None):
    
    '''
    Compute key points and feature descriptors using specific points
    
    '''
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.xfeatures2d.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    #Get keypoints and descriptors
    
    kps, features = descriptor.detectAndCompute(image,None)
    
    return(kps, features)