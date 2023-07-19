######## Webcam Object Detection Using Tensorflow-trained Classifier ##########################################################################################
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.
# 
# Modified by: Shawn Hymel ###################################################################################################################################
# Date: 09/22/20
# Description:
# Added ability to resize cv2 window and added center dot coordinates of each detected object.
# Objects and center coordinates are printed to console.
#
#
# Modified by: Sebastian Hintner #############################################################################################################################
# Date: 12/12/2021
# Description:
# Get Visitor Flow - SPS WS21: Use the basic code from source above for object detection.
# Additions:
# Only detect definded objects, not all objects which are possible for TensorFlow
# Count the detected objects in each direction with splitting the screen and count the objects on each side
# Save data in a csv - continious with timestamp and all together in another List when programm stops
# Start programm by './start_object_detection.sh' by using a small other programm at the terminal (change the programm with 'nano start_object_detection.sh'



# Import packages
import os
import argparse
import cv2
import csv
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

#Define List with objects wich should be detected (not all objects should be recognized and be count)
select_objects = ["person", "bicycle", "car"]

#now initializing variables to count object and which direction, value at beginn: 0
counter_person_moved_left = 0
counter_person_moved_right = 0
counter_bicycle_moved_left = 0
counter_bicycle_moved_right = 0
counter_car_moved_left = 0
counter_car_moved_right = 0

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

############################# end of class ##################################

#new code from Sebastian
        
#csv saving of data
from datetime import datetime #Import time for use as Timestamp
directory_csv = str("data/")  #Path where the csv-files will be stored
#one dir(ectory), and two csv files:
#1) continous file with timestamps = data_continous.csv
#2) summary file = data_summary.csv
csv_file_continous_data = open(directory_csv + "data_continous.csv", 'a') #'a' = add data in same file
csv_writer_continous_data = csv.writer(csv_file_continous_data) #insert data into csv with the csv.writer() -funtion
csv_file_summary_data = open(directory_csv + "data_summary_" + str(datetime.now()) + ".csv", 'w') #'w' = create a new file
csv_writer_summary_data = csv.writer(csv_file_summary_data)

counter_for_continous_data_file_flushing = 0 #Datei muss geflusht (aktuallisiert) werden. Dazu wird sie weiter unten geflusht wenn der Zähler hochgeht

#write header for continous csv
csv_writer_continous_data.writerow(['Object_Type', 'moved_left', 'moved_right', "Time"])
#end new code from Sebastian


# Define and parse input arguments #Auslesen und bewerten der Komando-Zeilen
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt') #labelmap = Textdatei mit Objekten
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args() 

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels #Labelmap = Textdatei mit Objekten
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)  #Variablen für Höhe und Breite des Fensters/der Aufnahme
use_TPU = args.edgetpu

#start new code from Sebastian
#Define location of lines on the camera-view for object counting. 
# region of interest_left | Middle cross line position | region of interest_right
region_of_interest_left_boundary_x_position = 150
region_of_interest_right_boundary_x_position = 1050 # max resolution 1280(x720) see Line 116
middle_line_x_position = int((region_of_interest_left_boundary_x_position + region_of_interest_right_boundary_x_position) / 2)  #middle of Region of Interest
#note: kann angepasst werden, falls das Sichtfeld der Weitwinkelkamera am Aufstellungsort eingeschränkt werden muss - Außenlinien verschieben, Mittellinie wird
#automatisch mittig platziert
#end new code from Sebastian


# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model (TPU = Tensor Processing Unit, speziell von Google entworfene Chips zur Softwaresammlung von TensorFlow) 
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME) #Labelmap = Textdatei mit Objekten

# Load the label map # Zugriff auf die Textdatei mit den Objekten
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()


# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start() #Initialisierung und Starten des Threat zum Starten der Kamera (Oben definintiert)
time.sleep(1)

# Create window
window_name = 'Get Visitor Flow - SPS WS21' #kann angepasst werden
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#start new code from Sebastian   
#init all counters for the states of the old frame, the counters are used to measure the changes between the object-counts
#(wie viele Personen haben sich gerade in der linken oder rechten Bildhälfte befunden - Erfassung über diese Counter)
pre_frame_counter_person_currently_in_left_field = 0
pre_frame_counter_person_currently_in_right_field = 0
pre_frame_counter_bicycle_currently_in_left_field = 0
pre_frame_counter_bicycle_currently_in_right_field = 0
pre_frame_counter_car_currently_in_left_field = 0
pre_frame_counter_car_currently_in_right_field = 0
#end new code from Sebastian


#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True: #Wihle Schleife führt Kameraaufnahme aus - Definiert oben in der Video-Klasse, Schleife wird für jeden Frame durchlaufen

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std


    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke() #invoke = calling a command --> interpreter = TensorFlow-Library

    # Retrieve detection results (Abrufen der Erkennungsergebnisse)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    #Num funktioniert nicht wenn Raute gelöscht wird ..
    
    #start new code from Sebastian
    #init all current-object-counters
    counter_person_currently_in_left_field = 0
    counter_person_currently_in_right_field = 0
    counter_bicycle_currently_in_left_field = 0
    counter_bicycle_currently_in_right_field = 0
    counter_car_currently_in_left_field = 0
    counter_car_currently_in_right_field = 0
    #end new code from Sebastian
    
    # Loop over all detections and draw detection box if confidence is above minimum threshold #######Rechteck + Label + Punkt#########################
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index - siese Objekt wurde gerade erkannt!
            
            if object_name in select_objects: #select_objects - Liste ganz oben was Detektiert werden soll!

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
            
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            
                # Draw label           
                #if object_name in select_objects: #select_objects
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                # Draw circle in center
                xcenter = xmin + (int(round((xmax - xmin) / 2)))  #<--- xcenter nehmen um zu Bestimmen ob Objekt über die Linie ist!
                ycenter = ymin + (int(round((ymax - ymin) / 2)))
                cv2.circle(frame, (xcenter, ycenter), 5, (0,0,255), thickness=-1)
                

                #start new code from Sebastian
                #Wie viele Center-Punkte sind im linken Feld und im rechten Feld, ausführung für jeden Objektnamen
                if (object_name == 'person'): #object_name = Objekt das gerade erkannt wurde, definiert in Zeile ~288
                    if (xcenter > region_of_interest_left_boundary_x_position and xcenter < middle_line_x_position): #Befindet sich die Person zwischen den Linien im linken Feld?
                        counter_person_currently_in_left_field += 1 #counter um 1 erhöhen
                    elif (xcenter > middle_line_x_position and xcenter < region_of_interest_right_boundary_x_position): #Befindet sich die Person zwischen den Linien im rechten Feld?
                        counter_person_currently_in_right_field += 1
                if (object_name == 'bicycle'):
                    if (xcenter > region_of_interest_left_boundary_x_position and xcenter < middle_line_x_position): #Befindet sich das Fahrrad zwischen den Linien im linken Feld?
                        counter_bicycle_currently_in_left_field += 1
                    elif (xcenter > middle_line_x_position and xcenter < region_of_interest_right_boundary_x_position): #Befindet sich das Fahrrad zwischen den Linien im rechten Feld?
                        counter_bicycle_currently_in_right_field += 1
                if (object_name == 'car'):
                    if (xcenter > region_of_interest_left_boundary_x_position and xcenter < middle_line_x_position): #Befindet sich das Auto zwischen den Linien im linken Feld?
                        counter_car_currently_in_left_field += 1
                    elif (xcenter > middle_line_x_position and xcenter < region_of_interest_right_boundary_x_position): #Befindet sich das Auto zwischen den Linien im rechten Feld?
                        counter_car_currently_in_right_field += 1        
                #end code from Sebastian 
                
                
                # Print info: "Objekt 0: person at (xxxx, xxx)"  - wird ins Comand-Window geschrieben solange das Program läuft
                    #print('Object ' + str(i) + ': ' + object_name + ' at (' + str(xcenter) + ', ' + str(ycenter) + ')')
            
            
    #code from Sebastian
    #Logikcode um von den counter der jeweiligen objekt-Punkte auf den Werte zu kommen, welche Objekte sich wohin bewegt haben (über Wahrheitstabelle)
    #counting logic:
                
    #               L | R                        
    #pre Frame      5 | 2
    #current Frame  4 | 3
    #-> 1 object moved to right
    
    #               L | R                        
    #pre Frame      8 | 2
    #current Frame  4 | 5
    #-> 3 objects moved to right and 1 object left the left_region_of_interest to the left
    #(Grenzfall: zwei Wechsel der Felder passieren genau gleichzeitig in den 250ms, Bedeutet: 1 bewegt sich von L->R und in der gleichen Zeit bewegt sich 1 von R->L |
    #In diesem Fall wird nichts gezählt da sich die Objekte aufheben, die Wahrscheinlichkeit, dass der Wechsel genau gleich auftritt wird als gering angenommen,
    #wodurch diese Verfälschung der Zählung in Kauf genommen wird.)
    
    #               L | R                        
    #pre Frame      7 | 3
    #current Frame  8 | 2
    #-> 1 object moved to left
    
    #               L | R                        
    #pre Frame      7 | 3
    #current Frame  9 | 1
    #-> 2 objects moved to left
    
    #               L | R                        
    #pre Frame      2 | 1
    #current Frame  1 | 1
    #-> do not count because L hast left the region of interest
    #Auch hier kann ein Grenzfall eintreten: L moved to R und 1-R moved to the right (verlässt die region_of_interest - die Zählung würde in die falsche Richtung gehen |
    #Wie oben muss auch hier der Wechsel genau in einem Frame passieren (250ms) - Wahrscheinlichkeit sehr gering, die Verfälschung der Zählung wird in Kauf genommen.)
    
    #persons
    diff_L = pre_frame_counter_person_currently_in_left_field - counter_person_currently_in_left_field  #Differnz der Felder berechnen
    diff_R = pre_frame_counter_person_currently_in_right_field - counter_person_currently_in_right_field
    if (diff_L < 0 and diff_R > 0): #moved from L to R (linke Seite einer weniger, rechte Seite einer mehr)
        value = min(abs(diff_L), abs(diff_R)) #Wert berechnen um den Hochgezählt werden soll: abs=betrag, kleinerer Wert wird gezählt da auch jemand von außen in die Felder kommen kann
        counter_person_moved_right += value #Wert hinzuzählen
        csv_writer_continous_data.writerow(['Person', str(0), str(value), str(datetime.now())]) #In csv eine Zeile schreiben str(0)=keine Person moved left, str(value)= so viele Personen moved right, +aktuelle Zeit
    elif (diff_L > 0 and diff_R < 0): #moved from R to L # das Ganze nochmal in die andere Richtung!
        value = min(abs(diff_L), abs(diff_R))
        counter_person_moved_left += value
        csv_writer_continous_data.writerow(['Person', str(value), str(0), str(datetime.now())])
    
    #bicycle (gleich wie bei Person, nur für die Fahrräder)
    diff_L = pre_frame_counter_bicycle_currently_in_left_field - counter_bicycle_currently_in_left_field
    diff_R = pre_frame_counter_bicycle_currently_in_right_field - counter_bicycle_currently_in_right_field
    if (diff_L < 0 and diff_R > 0): #moved from L to R
        value = min(abs(diff_L), abs(diff_R))
        counter_bicycle_moved_right += value
        csv_writer_continous_data.writerow(['Bicycle', str(0), str(value), str(datetime.now())])
    elif (diff_L > 0 and diff_R < 0): #moved from R to L
        value = min(abs(diff_L), abs(diff_R))
        counter_bicycle_moved_left += value
        csv_writer_continous_data.writerow(['Bicycle', str(value), str(0), str(datetime.now())])
    
    #car (gleich wie bei Person und Fahrräder, nur für die Autos)
    diff_L = pre_frame_counter_car_currently_in_left_field - counter_car_currently_in_left_field
    diff_R = pre_frame_counter_car_currently_in_right_field - counter_car_currently_in_right_field
    if (diff_L < 0 and diff_R > 0): #moved from L to R
        value = min(abs(diff_L), abs(diff_R))
        counter_car_moved_right += value
        csv_writer_continous_data.writerow(['Car', str(0), str(value), str(datetime.now())])
    elif (diff_L > 0 and diff_R < 0): #moved from R to L
        value = min(abs(diff_L), abs(diff_R))
        counter_car_moved_left += value
        csv_writer_continous_data.writerow(['Car', str(value), str(0), str(datetime.now())])
    
    #print("Persons left field = " + str(counter_person_currently_in_left_field) + " | Persons right field = " + str(counter_person_currently_in_right_field)) #Ausgabe am Command-Fenster
    
    #copy values from the current evaluated frame to the pre_frame counters for the next while-run - schreibt die Werte in die pre-Zählung für den nächsten Schleifendurchlauf
    pre_frame_counter_person_currently_in_left_field = counter_person_currently_in_left_field
    pre_frame_counter_person_currently_in_right_field = counter_person_currently_in_right_field
    pre_frame_counter_bicycle_currently_in_left_field = counter_bicycle_currently_in_left_field
    pre_frame_counter_bicycle_currently_in_right_field = counter_bicycle_currently_in_right_field
    pre_frame_counter_car_currently_in_left_field = counter_car_currently_in_left_field
    pre_frame_counter_car_currently_in_right_field = counter_car_currently_in_right_field
    
    #Flushen der Datei die die kontinuierliche Zählung beinhaltet. (bereits oben als 0 definiert.
    #um eines hochzählen
    counter_for_continous_data_file_flushing += 1
    #Flush alle 30 Sekunden (30 x 4 FPS = 120)
    if (counter_for_continous_data_file_flushing % 120 == 0): #Über Modulo (%) abgleichen, ob der Counter schon auf 120 ist
        csv_file_continous_data.flush() #Eigentliche Funktion für das Aktualisieren (Flushen)
        counter_for_continous_data_file_flushing = 0 #Zähler wieder auf 0 zurücksetzen für nächsten Druchlauf
    #end code from Sebastian
                
                
    # Draw into Camera-Picture on Screen
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(1100,50),cv2.FONT_HERSHEY_SIMPLEX,1,(10,255,0),2,cv2.LINE_AA) #30,50
    
    #start code from Sebastian
    # Draw the crossing lines
    #cv2.line(img, (Start x, Start y), (Ende x, Ende y), (Farbe BGR-Code: 255, 0, 255), Dicke in px)
    #region_of_interest_left_boundary line:
    cv2.line(frame, (region_of_interest_left_boundary_x_position, 0), (region_of_interest_left_boundary_x_position, imH), (0, 0, 255), 2)
    #region_of_interest_right_boundary line:
    cv2.line(frame, (region_of_interest_right_boundary_x_position, 0), (region_of_interest_right_boundary_x_position, imH), (0, 0, 255), 2)
    #region_of_interest_middle line:
    cv2.line(frame, (middle_line_x_position, 0), (middle_line_x_position, imH), (255, 0, 255), 2)
    # Draw the other text in the image (Counter, Lines, Info)
    cv2.putText(frame,'left field',(int((region_of_interest_left_boundary_x_position + middle_line_x_position) / 2 - 60),50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA) #Definition Up im Kamerabild + Darstellung
    cv2.putText(frame,'right field',(int((region_of_interest_right_boundary_x_position + middle_line_x_position) / 2 - 70),50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA) #Definition Down im Kamerabild + Darstellung
    cv2.putText(frame,'press "q" for exit',(970 ,690),cv2.FONT_HERSHEY_SIMPLEX,1,(10,255,0),2,cv2.LINE_AA) #Info zum Schließen
    cv2.putText(frame,'Get Visitor Flow - SPS WS21',(30 ,690),cv2.FONT_HERSHEY_SIMPLEX,1,(10,255,0),2,cv2.LINE_AA) #Copyright Get Visitor Flow - SPS WS21
    # Draw tabel texts in the frame
    cv2.putText(frame,'Counter',(30,90),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA) #Tabellenbeschriftung
    cv2.putText(frame,'<-',(180,90),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA) #Tabellenbeschriftung
    cv2.putText(frame,'->',(270,90),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA) #Tabellenbeschriftung
    # Draw tabel lines
    cv2.line(frame, (30, 100), (340, 100), (255,255,0), 2) #Tabellenlinie horrizontal oben (top-rule)
    cv2.line(frame, (30, 220), (340, 220), (255,255,0), 2) #Tabellenlinie horrizontal unten (bottom-rule)
    cv2.line(frame, (165, 70), (165, 220), (255,255,0), 2) #Tabellenlinie vertikal 1
    cv2.line(frame, (250, 70), (250, 220), (255,255,0), 2) #Tabellenlinie vertikal 2
    # Draw table content 
    cv2.putText(frame,'Person:',(30,130),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA) 
    cv2.putText(frame,str(counter_person_moved_left),(175,130),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA) 
    cv2.putText(frame,str(counter_person_moved_right),(270,130),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA) 
    
    cv2.putText(frame,'Bicycle:',(30,170),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    cv2.putText(frame,str(counter_bicycle_moved_left),(175,170),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA) 
    cv2.putText(frame,str(counter_bicycle_moved_right),(270,170),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA) 
    
    cv2.putText(frame,'Car:',(30,210),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    cv2.putText(frame,str(counter_car_moved_left),(175,210),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA) 
    cv2.putText(frame,str(counter_car_moved_right),(270,210),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    #end code from Sebastian


    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow(window_name, frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
#Ende While-Schleife Videostream
    
#start code from Sebastian
#output summary csv #Erstellung der Einträge in die csv-Datei der gesamten Zählung
csv_writer_summary_data.writerow(['Object_Type', 'moved_left', 'moved_right'])
csv_writer_summary_data.writerow(['Person', str(counter_person_moved_left), str(counter_person_moved_right)])
csv_writer_summary_data.writerow(['Bicycle', str(counter_bicycle_moved_left), str(counter_bicycle_moved_right)])
csv_writer_summary_data.writerow(['Car', str(counter_car_moved_left), str(counter_car_moved_right)])
#end code from Sebastian

# Clean up
csv_file_continous_data.close()
csv_file_summary_data.close()
cv2.destroyAllWindows()
videostream.stop()