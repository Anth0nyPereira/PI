import json
from os.path import join

import numpy as np
from numpyencoder import NumpyEncoder
from manage import es
from app.VGG import model
from app.models import ImageNeo, Person, Tag, Location, Country, City, ImageES
from app.utils import get_imlist, ImageFeature
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

from imutils.object_detection import non_max_suppression
import cv2
import pytesseract
features = []
imageFeatures = []

def uploadImages(uri):
    print("----------------------------------------------")
    print("            featrue extraction starts         ")
    print("----------------------------------------------")

    img_list = get_imlist(uri)
    s = set(imageFeatures)
    for index, img_path in enumerate(img_list):
        img_name = os.path.split(img_path)[1]
        i = ImageFeature(img_path)
        if i in s:
            print("Image " + img_path + " has already been processed")
            continue

        norm_feat, height, width = model.vgg_extract_feat(img_path)  # extrair infos
        f = json.dumps(norm_feat, cls=NumpyEncoder)
        i.features = f
        iJson = json.dumps(i.__dict__)

        image = ImageNeo(folder_uri=os.path.split(img_path)[0],
                         name=img_name,
                         processing=iJson,
                         format=img_name.split(".")[1],
                         width=width,
                         height=height)
        image.save()
        p = Person(name="wei")
        p.save()
        image.person.connect(p, {'coordinates':0.0})

        place = getPlaces(img_path)
        if place:
            t = Tag(name=place)
            t.save()
            image.tag.connect(t)

        l = Location(name="UA")
        l.save()
        image.location.connect(l, {"latitude":10.0, "longitude":20.0, "altitude":30.0})

        c = City(name="Aveiro")
        c.save()
        l.city.connect(c)

        ct = Country(name="PT")
        ct.save()
        ct.city.connect(c)

        # add features to "cache"
        features.append(norm_feat)
        i.features = norm_feat
        imageFeatures.append(i)
        s.add(i)

        print("extracting feature from image No. %d , %d images in total " % ((index + 1), len(img_list)))


def findSimilarImages(uri):
    norm_feat, height, width = model.vgg_extract_feat(uri)  # extrair infos
    feats = np.array(features)
    scores = np.dot(norm_feat, feats.T)
    rank = np.argsort(scores)[::-1]
    rank_score = scores[rank]

    maxres = 40  # 40 imagens com maiores scores

    imlist = []
    for i, index in enumerate(rank[0:maxres]):
        imlist.append(imageFeatures[index])
        print("image names: " + str(imageFeatures[index].name) + " scores: %f" % rank_score[i])


def getPlaces(img_name):
    # th architecture to use
    arch = 'resnet18'

    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the test image
    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    return classes[idx[0]]

def getOCR(img_path):
        #load installed tesseract-ocr from users pc
    pytesseract.pytesseract.tesseract_cmd = r'D:\\OCR\\tesseract'
    custom_config = r'--oem 3 --psm 6'
    east = "frozen_east_text_detection.pb"
    min_confidence = 0.6

    results = []
    #These must be multiple of 32
    newW = 128
    newH = 128
    net = cv2.dnn.readNet(east)
    image = cv2.imread(img_path)
    orig = image.copy()
    (H, W) = image.shape[:2]
    
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    rW = W / float(newW)
    rH = H / float(newH)
    
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    
    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
    
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            # loop over the number of columns
            for x in range(0, numCols):
                    # if our score does not have sufficient probability, ignore it
                    if scoresData[x] < min_confidence:
                            continue
                    # compute the offset factor as our resulting feature maps will
                    # be 4x smaller than the input image
                    (offsetX, offsetY) = (x * 4.0, y * 4.0)
                    # extract the rotation angle for the prediction and then
                    # compute the sin and cosine
                    angle = anglesData[x]
                    cos = np.cos(angle)
                    sin = np.sin(angle)
                    # use the geometry volume to derive the width and height of
                    # the bounding box
                    h = xData0[x] + xData2[x]
                    w = xData1[x] + xData3[x]
                    # compute both the starting and ending (x, y)-coordinates for
                    # the text prediction bounding box
                    endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                    endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                    startX = int(endX - w)
                    startY = int(endY - h)
                    # add the bounding box coordinates and probability score to
                    # our respective lists
                    rects.append((startX, startY, endX, endY))
                    confidences.append(scoresData[x])
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    # loop over the bounding boxes
    (maxH, maxW) = orig.shape[:2]
    for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW) - 20
            if(startX <= 0):
                startX = 1
            startY = int(startY * rH) - 20
            if(startY <= 0):
                startY = 1
            endX = int(endX * rW) + 20
            if(endX >= maxW):
                endX = maxW-1
            endY = int(endY * rH) + 20
            if(endY >= maxH):
                endY = maxH-1
            # draw the bounding box on the image
            ROI = orig[startY:endY, startX:endX]
            imageText = pytesseract.image_to_string(ROI, config=custom_config)
            result = imageText.replace("\x0c", " ").replace("\n", " ")
            results += result.split(" ")

  
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Load image, grayscale, Gaussian blur, adaptive threshold
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        if area > 10000:
            x,y,w,h = cv2.boundingRect(c)
            ROI = orig[y:y+h, x:x+w]
            imageText = pytesseract.image_to_string(ROI, config=custom_config)
            result = imageText.replace("\x0c", " ").replace("\n", " ")
            results += result.split(" ")

    print(set(results))

    
# load all images to memory
def setUp():
    images = ImageNeo.nodes.all()

    for image in images:
        i = ImageFeature(**json.loads(image.processing))
        if i.features is None:
            continue
        i.features = np.array(json.loads(i.features))
        features.append(i.features)
        imageFeatures.append(i)

        tags = []
        for tag in image.tag:
            tags.append(tag.name)

        persons = []
        for p in image.person:
            persons.append(p.name)

        locations = set()
        for l in image.location:
            locations.add(l.name)

            for c in l.city:
                locations.add(c.name)

                for ct in c.country:
                    locations.add(ct.name)

        locations = list(locations)

        uri = join(image.folder_uri, image.name)
        ImageES(meta={'id': uri}, uri=uri,
                tags=tags, locations=locations, persons=persons)\
            .save(using=es)

    # load the class label for scene recognition
    file_name = 'categories_places365.txt'
    global classes
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

setUp()
