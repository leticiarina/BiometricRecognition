import numpy as np 
import pandas as pd
import imageio
import math
import cv2
import matplotlib.pyplot as plt

# Converts a RGB image to gray scale using colorimetric conversion
def grayTransform(img):
	return np.dot(img, [0.299, 0.587, 0.114])

# Calculates a weigh for a class
def weigh(M, x1, x2, hist):
	return (1/M)*np.sum(hist[x1:x2])

# Calculates mean for a class
def mean(x1, x2, hist):

	a = 0 

	for i in range(x1, x2):
		a = a+(i*hist[i])

	if np.sum(hist[x1:x2]) != 0:
		return a/(np.sum(hist[x1:x2]))
	else: 
		return 0

# Calculates variance for a class
def variance(x1, x2, mean, hist):

	a = 0

	for i in range(x1, x2):
		a = a+(math.pow(i-mean, 2)*hist[i])

	if np.sum(hist[x1:x2]) != 0:
		return a/(np.sum(hist[x1:x2]))
	else: 
		return 0

# Computes the optimal thresholding using Otsu algorithm
def otsuThresholding(img):

	# Computing histogram
	hist, bin_edges = np.histogram(img, bins='auto');
	# Converts array with edges to int
	bin_edges = bin_edges.astype(int)

	# New array to histogram
	histogram = np.zeros(255)

	# Fills new histogram
	for x in range(hist.size):
		histogram[bin_edges[x]] = hist[x]

	# Converts to int
	histogram = histogram.astype(int)

	intraclassVar = np.zeros(histogram.size)

	# Intra-class variance for each intensity
	for L in range(histogram.size):
		
		# Weigh for class A, sum of frequencies from 0 to L-1
		weighA = weigh(img.shape[0]*img.shape[1], 0, L, histogram)
		# Mean for class A
		meanA = mean(0, L+1, histogram)
		# Variance for class A
		varA = variance(0, L+1, meanA, histogram)

		# Weigh for class B, sum of frequencies from L to last item
		weighB = weigh(img.shape[0]*img.shape[1], L, histogram.size, histogram)
		# Mean for class B
		meanB = mean(L, histogram.size, histogram)
		# Variance for class B
		varB = variance(L, histogram.size, meanB, histogram)
		
		# Stores intraclass variance calculated
		intraclassVar[L] = weighA*varA + weighB*varB

		# Set the min value 
		if L is 0:
			minValue = intraclassVar[L]
		elif L != 0 and minValue > intraclassVar[L]: 
			minValue = intraclassVar[L] 

	# Returns the optimal thresholding
	return list(intraclassVar).index(minValue)


# Converts a grey scale image to binary (only black and white)
def binaryTransform(img, thresholding):

    binImg = np.zeros(img.shape)

    # Sets image values according to chosen thresholding
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x,y] > thresholding: binImg[x,y] = 1

    return binImg

"""
Extracts the biggest object from a binary image with multiple objects
Parameters:
    binImg: binary image with multiple objects
Returns:
    mask: the binary mask with the biggest object
    contour: contour points of the biggest object
"""
def selectBiggestObject(binImg):

    binary = binImg.astype(np.uint8)

    # Reads the binary image
    img = image.astype(np.uint8)

    # Finding contours
    im2, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # Finding the biggest contour
    cnt = None
    cntArea = 0
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > cntArea:
            cnt = contours[i]
            cntArea = cv2.contourArea(contours[i])

    # Drawing the contour in the original image
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [cnt], -1, (255), -1)
    
    return (mask, cnt)

"""
Calculates the convex hull defects of a binary mask with one object
Parameters:
    contour: the contour points of the object in the mask
Returns:
    defects: the defects found in the convex hull of the object
"""
def convexHullDefects(contour):
    #generating convex hull and convexity defects
    hull = cv2.convexHull(contour, returnPoints = False)
    old_defects = cv2.convexityDefects(contour, hull)

    #iterating through defects and checking for defects that are too small
    defects = []
    for i in range(old_defects.shape[0]):
        s,e,f,d = old_defects[i,0]
        # if distance from defect to hull is smaller than 4000, continue
        if d < 4000:
            continue;
        # adding the big defects to a new array
        defects.append(old_defects[i])
        
    return np.array(defects)

""" 
Cuts the palm from the binary mask of the hand
Parameters:
    mask: binary mask of the hand
    cnt: contour points of the binary mask
    defects: convex hull defects of the binary mask contour
Returns:
    cut_hand_mask: binary mask without the palm
    palm_mask: binary mask of the palm cut from mask
"""
def cutPalm(mask, cnt, defects):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Finding cut points
    points = []
    for i in range(defects.shape[0]):
        start = tuple( cnt[defects[i  ,0,2]][0] )
        points.append(start)
    
    # Adjusting the array for minEnclosingCircle
    # Needs to be an array of format (ROWSx1x2), where each row has a tuple with (1, (x,y) ), x and y coordinates of each point
    array = np.array(points)
    array = array.reshape((-1,1,2))
    
    # calculates the minimum enclosing circle of the hand
    (x,y),radius = cv2.minEnclosingCircle(array)
    center = (int(x),int(y))
    radius = int(radius)
    
    # Cutting hand from fingers_mask
    cut_hand_mask = np.copy(mask)
    cv2.circle(cut_hand_mask,center,radius,(0),-1)
    
    # Drawing palm_mask. Palm mask is an AND between the minEnclosingCircle and mask
    palm_mask = np.zeros_like(mask)
    cv2.circle(palm_mask,center,radius,(255),-1)
    palm_mask = cv2.bitwise_and(palm_mask, mask)
    
    return (cut_hand_mask, palm_mask)

"""
Removes the arm and the thumb from the cut_hand_mask
Parameters:
    cut_hand_mask: binary mask of the hand with the palm cut
Returns:
    five_fingers_mask: mask with all five fingers
    four_fingers_mask: mask with the four fingers and thumb removed
"""
def removeArmThumb(cut_hand_mask):
    
    # Finding contours
    im2, contours, hierarchy = cv2.findContours(cut_hand_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 10000]
    
    # Generates a Bounding Rectangle for each contour
    rects = [cv2.boundingRect(c) for c in contours]
    
    # Removes Contour with smallest with minimum Bouding Rectangle height (arm)
    min_h = float('inf')
    min_rect = -1
    for i in range(len(rects)):
        if(rects[i][3] < min_h):
            min_rect = i
            min_h = rects[i][3]
    rects.remove(rects[min_rect])
    contours.remove(contours[min_rect])
    
    # generates the five fingers mask
    five_fingers_mask = np.zeros_like(cut_hand_mask)
    for c in contours:
        cv2.drawContours(five_fingers_mask, [c], -1, (255) , -1)
            
    # Removes Contour with biggest Bounding Rectangle area (thumb)
    max_a = float('-inf')
    max_rect = -1
    for i in range(len(rects)):
        if((rects[i][3] * rects[i][2]) > max_a):
            max_rect = i
            max_a = (rects[i][3] * rects[i][2])
    rects.remove(rects[min_rect])
    contours.remove(contours[min_rect])
        
    # generates the four fingers mask
    four_fingers_mask = np.zeros_like(cut_hand_mask)
    for c in contours:
        cv2.drawContours(four_fingers_mask, [c], -1, (255), -1)
        
    return (four_fingers_mask, five_fingers_mask)


"""
Isolates each finger in the parameter mask into four binary masks,
cropped to the fingers minimum bounding rectangle and rotated so the finger faces up
Parameters:
    fingers_mask: mask with the four fingers that will be split intro separeted masks
Returns:
    fingers_masks: an array with 4 elements, each is an image with a finger binary mask
"""
def isolateFingers(fingers_mask):
    
    # Applying morphological opening for thin edges reduction
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    fingers_mask = cv2.morphologyEx(fingers_mask, cv2.MORPH_OPEN, kernel)
    
    #finding new contours after opening
    im2, contours, hierarchy = cv2.findContours(fingers_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    # Fingers_masks contains a binary mask of each finger
    fingers_masks = []
    
    for c, i in zip(contours, range(len(contours))):
        
        # Creating the new images
        fingers_masks.append(np.zeros_like(fingers_mask))
        
        # Drawing finger mask
        cv2.drawContours(fingers_masks[i], [c], -1, (255), -1) 
        
        # Generating rotated bounding rectangle
        rect = cv2.minAreaRect(c)
        
        # Changes rotation angle so fingers are always on horizontal.
        # This code might be wrong, need to check corner cases
        rot = rect[2]
        if (rect[1][0] > rect[1][1]):
            rot = 90 + rot
        rotation_matrix = cv2.getRotationMatrix2D(rect[0], rot, 1.0)
        
        # Applies rotation to mask and image
        rotated_mask = cv2.warpAffine(fingers_masks[i], rotation_matrix, fingers_masks[i].shape[:2])
        
        # Finding new contour after rotation
        cont = cv2.findContours(rotated_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        # Getting new bouding rect
        x, y, w, h = cv2.boundingRect(cont[0])
        
        # Cropping finger mask
        fingers_masks[i] = np.zeros((w,h))
        fingers_masks[i] = rotated_mask[y:y+h , x:x+w]
        
    #returns a list containing each finger tuple(mask, image, contour)
    return fingers_masks


"""
Calculates the lenght and width of a finger mask
Parameters:
    finger mask: binary mask with a finger image
Returns:
    a tuple with height, width of bottom middle and top parts of the finger
"""
def fingerMeasure(finger_mask):
    
    # Finds the fingers contours
    im2, contours, hierarchy = cv2.findContours(finger_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # Calculating the bounding box
    x, y, width, height = cv2.boundingRect(contours[0])
    
    # Bottom width is calculated at the 25% percentile of the height
    bot_width = np.count_nonzero(finger_mask[int(height * 0.25)])
    mid_width = np.count_nonzero(finger_mask[int(height * 0.50)])
    top_width = np.count_nonzero(finger_mask[int(height * 0.75)])
    
    return (height, bot_width, mid_width, top_width)

"""
Calculates the lenght and width of a object within a binary mask
Parameters:
    obj_mask: binary mask with the object
Returns:
    a tuple with height and width of the object
"""
def objMeasure(obj_mask):

    # Finds the fingers contours
    im2, contours, hierarchy = cv2.findContours(obj_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # Find biggest contour
    maxArea = -1
    maxC = None
    for c in contours:
        if(cv2.contourArea(c) > maxArea):
            maxC = c
            maxArea = cv2.contourArea(c)

    # Calculating the bounding box
    x, y, width, height = cv2.boundingRect(maxC)

    return (height, width)

"""
Generates a numpy array with the values from palm_measure, hand_measure and fingers_measures
Parameters:
    palm_measure: tuple with (heigth, width) of a palm
    hand_measure: tuple with (heigth, width) of a hand
    fingers_measures: list of tuples with (heigth, width_bottom, width_middle, width_top) of a finger
Returns:
    numpy array with values. The indexes are:
    'palm_length', 'palm_width', 'hand_length', 'hand_width', 'finger0_length',
    'finger0_bot_width', 'finger0_mid_width', 'finger0_top_width', 'finger1_length',
    'finger1_bot_width', 'finger1_mid_width', 'finger1_top_width', 'finger2_length',
    'finger2_bot_width', 'finger2_mid_width', 'finger2_top_width', 'finger3_length',
    'finger3_bot_width', 'finger3_mid_width', 'finger3_top_width'
"""
def generateAttributesList(palm_measure, hand_measure, fingers_measures):
    values = [palm_measure[0], palm_measure[1], hand_measure[0], hand_measure[1]]
    for finger in fingers_measures:
        values.append(finger[0])
        values.append(finger[1])
        values.append(finger[2])
        values.append(finger[3])
    return values

"""
Does the whole processing described above for an image
Parameters:
    filename: image filename
Returns:
    an numpy array with all attributes extracted
"""
def processImage(filename):
    
    
    indexes = ['palm_length', 'palm_width', 'hand_length', 'hand_width', 'finger0_length', 'finger0_bot_width', 'finger0_mid_width', 'finger0_top_width', 'finger1_length', 'finger1_bot_width', 'finger1_mid_width', 'finger1_top_width', 'finger2_length', 'finger2_bot_width', 'finger2_mid_width', 'finger2_top_width', 'finger3_length', 'finger3_bot_width', 'finger3_mid_width', 'finger3_top_width', 'foto_id', 'person']
    
    #generates name and 
    name = filename.split('/')[-1].split('_')[0]
    photo_id = int(filename.split('/')[-1].split('_')[1].replace('.jpg',''))
    
    # Opens image, turns image to grayscale
    gray = grayTransform(imageio.imread(filename)).astype(int)
    # Applies otsu threshold to create a binary mask for image
    binImg = binaryTransform(gray, otsuThresholding(gray))
    # generates a mask and a contour for the object in the binary image
    mask, contour = selectBiggestObject(binImg)
    # finds biggest defects in contour convex hull
    defects = convexHullDefects(contour)
    # separates the palm from the hand in the binary mask
    cut_hand_mask, palm_mask = cutPalm(mask, contour, defects)
    # separates the fingers from the cut hand mask
    four_fingers_mask, five_fingers_mask = removeArmThumb(cut_hand_mask)
    # separates each finger in a separate mask and rotates them so all fingers are pointing up
    fingers_masks = isolateFingers(four_fingers_mask)
    # calculates all measures for fingers
    fingers_measures = [fingerMeasure(f) for f in fingers_masks]
    # calculates all measures for palm
    palm_measure = objMeasure(palm_mask)
    # calculates all measures for hand
    hand_measure = objMeasure(palm_mask + five_fingers_mask)
    
    values = generateAttributesList(palm_measure, hand_measure, fingers_measures)
    values.append(photo_id)
    values.append(name)

    #generates a numpy array with all measurements
    return np.array(values)

"""
Generates a database_filename.csv file with all images on images_folder extracted
Parameters:
    images_fodler: folder where images are
    database_filename: database filename. The file will be named <database_filename>.csv
Returns:
    a pandas dataframe containing the database generated
"""
def generateDatabase(images_folder='handDatabase', database_filename=None):
    files = os.listdir(images_folder)
    try:
        files.remove('notWorking')
    except:
        pass
        
    df = pd.DataFrame(columns=indexes)
    for f, i in zip(files, range(len(files))):
        df.loc[i] = processImage(images_folder +'/' + f)
        print('file {0}:{1} done'.format(i,f))
    if database_filename:
        #guaranteed to have .csv extension
        database_filename.replace('.csv','')
        df.to_csv(database_filename + '.csv')
    return df
    
