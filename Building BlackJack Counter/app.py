import numpy as np
import cv2

def reorder(pts):
    """Reorder points in a rectangle in clockwise order to be consistent with OpenCV!
    """
    pts = pts.reshape((4,2))
    pts_new = np.zeros((4,2), np.float32)

    add = pts.sum(1)
    pts_new[0] = pts[np.argmin(add)]
    pts_new[2] = pts[np.argmax(add)]

    diff = np.diff(pts, 1)
    pts_new[1] = pts[np.argmin(diff)]
    pts_new[3] = pts[np.argmax(diff)]

    return pts_new

def preprocess(img):
    """Image cleaning
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 1)
    return thresh
  
def img_compare(A, B):
    """Metric of image comparison
    """
    A = cv2.GaussianBlur(A, (5,5), 5)
    B = cv2.GaussianBlur(B, (5,5), 5)    
    diff = cv2.absdiff(A, B)  
    _, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY) 
    return np.sum(diff)  

def closest_card(model, img):
    """Find the playing card that is the closest to img
    """
    features = preprocess(img)
    closest_match = sorted(model.values(), key=lambda x:img_compare(x[1], features))[0]
    return closest_match[0]
  
def extract_cards(img, num_cards=4):
    cards = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1,1), 1000)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_cards]
    new_dim = np.array([[0,0], [449, 0], [449, 449], [0, 449]], np.float32)
    
    for card in contours:
    	epsilon = 0.1*cv2.arcLength(card, True)
    	approx = cv2.approxPolyDP(card, epsilon, True)
    	approx = reorder(approx)
    	
    	transform = cv2.getPerspectiveTransform(approx, new_dim)
    	warp = cv2.warpPerspective(img, transform, (450, 450))
    	
    	cards.append(warp)
    
    return cards

def train(training_labels_filename='train.tsv', training_image_filename='train.png', num_training_cards=56):
    """Collect training information for model
    """
    model= {}
  
    labels = {}
    for line in file(training_labels_filename): 
        key, num, suit = line.strip().split()
        labels[int(key)] = (num, suit)
    
    training_img = cv2.imread(training_image_filename)
    for i, card in enumerate(extract_cards(training_img, num_training_cards)):
        model[i] = (labels[i], preprocess(card))
  
    return model 

if __name__ == '__main__':
    filename = 'test.jpg'
    num_cards = 4

    model = train()

    img = cv2.imread(filename)

    width = img.shape[0]
    height = img.shape[1]
    if width < height:
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)

    # See images
    for i,c in enumerate(extract_cards(img, num_cards)):
        card = closest_card(model, c)
        cv2.imshow(str(card), c)
    cv2.waitKey(0) 

    cards = [closest_card(model, c) for c in extract_cards(img, num_cards)]
    print cards
    
