from django.shortcuts import render
from rest_framework import generics, status, views, permissions
from .serializers import *
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.views import *

from django.contrib.sites.shortcuts import get_current_site
from django.urls import reverse
import jwt
from django.conf import settings


from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.utils.encoding import smart_str, force_str, smart_bytes, DjangoUnicodeDecodeError
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.contrib.sites.shortcuts import get_current_site
from django.urls import reverse

from django.shortcuts import redirect
from django.http import HttpResponsePermanentRedirect
import os

from .testImage import *

from ultralytics import YOLO
import cv2
import cvzone
import math
import time

import numpy as np
import base64





# decoded_data = base64.b64decode(image)
# np_data = np.fromstring(decoded_data,np.uint8)
# img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)



# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
#cap = cv2.VideoCapture("images.jpg")  # For Video

class PeopleDetection(APIView):

    def get(self, request, *args, **krags):
       
        model = YOLO("../Yolo-Weights/yolov8l.pt")

        classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

        prev_frame_time = 0
        new_frame_time = 0
        count = 0
        #while True:
        new_frame_time = time.time()
        #img = cv2.imread("fu.jpeg")
        img = decodeImg()
        results = model(img, show=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),1)
                w, h = x2 - x1, y2 - y1
                #cvzone.cornerRect(img, (x1, y1, w, h),l=0, rt=1, colorR=(255, 0, 255))
                # Confidence
                #print('conf ',box.conf[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                if currentClass == "person" and conf > 0.5:
                    cvzone.cornerRect(img, (x1, y1, w, h),l=0, rt=1, colorR=(255, 0, 255))
                    cvzone.putTextRect(img, f'{classNames[cls]}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1,offset=1)
                    count = count+1


                    print('people count ',count)
                #cv2.imshow(img)
        return Response({'people_count':count},status=status.HTTP_204_NO_CONTENT)


class VehicleDetection(APIView):

    def get(self, request, *args, **krags):
        vehicle_name = []
       
        model = YOLO("../Yolo-Weights/yolov8l.pt")

        classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

        prev_frame_time = 0
        new_frame_time = 0
        count = 0
        #while True:
        new_frame_time = time.time()
        #img = cv2.imread("fu.jpeg")
        img = decodeImg() #call function in the testImg file
        results = model(img, show=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),1)
                w, h = x2 - x1, y2 - y1
                #cvzone.cornerRect(img, (x1, y1, w, h),l=0, rt=1, colorR=(255, 0, 255))
                # Confidence
                
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                #bicycle = int(box.cls[1])
                cls = int(box.cls[0])
                print('conf ',conf)
                #motorbike = int(box.cls[3])
                #bus = int(box.cls[5])
                #truck = int(box.cls[7])
                currentClass = classNames[cls]
                if currentClass == "bicycle" or currentClass == "car" or currentClass == "bus" or currentClass == "truck" or currentClass == "motorbike" and conf > 0.5:
                    cvzone.cornerRect(img, (x1, y1, w, h),l=0, rt=1, colorR=(255, 0, 255))
                    cvzone.putTextRect(img, f'{classNames[cls]}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1,offset=1)
                    count = count+1


                    print('classNames ',classNames[cls])
                    vehicle_name.append(classNames[cls])
                    print('vehicle count ',count)
                #cv2.imshow(img)
        return Response({'vehicle_count':count,'vehicle_name':vehicle_name},status=status.HTTP_204_NO_CONTENT)


class ObjectDetection(APIView):

    def get(self, request, *args, **krags):
        object_detect = []
       
        model = YOLO("../Yolo-Weights/yolov8l.pt")

        classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

        prev_frame_time = 0
        new_frame_time = 0
        count = 0
        #while True:
        new_frame_time = time.time()
        #img = cv2.imread("fu.jpeg")
        img = decodeImg() #call function in the testImg file
        results = model(img, show=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),1)
                w, h = x2 - x1, y2 - y1
                #cvzone.cornerRect(img, (x1, y1, w, h),l=0, rt=1, colorR=(255, 0, 255))
                # Confidence
                
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                #bicycle = int(box.cls[1])
                cls = int(box.cls[0])
                print('conf ',conf)
                #motorbike = int(box.cls[3])
                #bus = int(box.cls[5])
                #truck = int(box.cls[7])
                currentClass = classNames[cls]
                if conf >= 0.45:
                    cvzone.cornerRect(img, (x1, y1, w, h),l=0, rt=1, colorR=(255, 0, 255))
                    cvzone.putTextRect(img, f'{classNames[cls]}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1,offset=1)
                    count = count+1


                    print('classNames ',classNames[cls])
                    object_detect.append(classNames[cls])
                    print('object count ',count)
                #cv2.imshow(img)
        return Response({'object_count':count,'object_name':object_detect},status=status.HTTP_204_NO_CONTENT)

