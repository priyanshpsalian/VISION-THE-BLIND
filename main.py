import speech_recognition as sr
from time import sleep
import pyttsx3
r = sr.Recognizer()
mic = sr.Microphone()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
def talk(text):
    engine.say(text)
    engine.runAndWait()

import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
import imp
import cv2
import time
import os
import io,requests
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
import re
from utils import *

def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            print(name)

        face_names.append(name)

        # for (top, right, bottom, left), name in zip(face_locations, face_names):
        #     # Draw a box around the face
        #     cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

        #     # Draw a label with a name below the face
        #     cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
        #     font = cv2.FONT_HERSHEY_DUPLEX
        #     cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    # Display the resulting image
    # while True:

        # cv2.imshow('Video', img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            
        return face_names


def take_image(string):

    # name of person
    # string = input("Enter your string: ")
    #splitting the string
    words = string.split()
    #slicing the list (negative index means index from the end)
    #-1 means the last element of the list
    print(words[-1])


    # taking image
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        return
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)

    # time.sleep(5) # after every 5 sec take ss
    img_name = words[-1]+".png".format(img_counter)
        # img_name = words[-1]+"opencv_frame_{}.png".format(img_counter)

    path= 'C:/Users/Admin/Desktop/PRIYANSH/MPR/PRIYANSH/faces'
    # cv2.imwrite(img_name, frame)
    cv2.imwrite(os.path.join(path , img_name), frame)
    print("{} written!".format(img_name))
    img_counter += 1

    # time.sleep(2) # ML program

    if os.path.exists(os.path.join("absolute path",img_name)): #delete the file
        os.remove(os.path.join("absolute path",img_name))
    else:
        print("The file does not exist")
    return
def check_image():

    # taking image
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        return
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)

    # time.sleep(5) # after every 5 sec take ss
    img_name = "test.jpg".format(img_counter)
        # img_name = words[-1]+"opencv_frame_{}.png".format(img_counter)

    # path= 'C:/Users/Admin/Desktop/PRIYANSH/MPR/PRIYANSH/faces'
    # cv2.imwrite(img_name, frame)
    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))
    img_counter += 1

    # time.sleep(2) # ML program

    if os.path.exists(os.path.join("absolute path",img_name)): #delete the file
        os.remove(os.path.join("absolute path",img_name))
    else:
        print("The file does not exist")
    name=classify_face("test.jpg")
    return name

# gives the name of all the objects near by 
def check_surrounding():
   

    net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(320, 320), scale=1/255)

    # Load class lists
    classes = []
    with open("dnn_model/classes.txt", "r") as file_object:
        for class_name in file_object.readlines():
            class_name = class_name.strip()
            classes.append(class_name)


    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # FULL HD 1920 x 1080


    ret, frame = cap.read()
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
    obj_names = []
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        print(class_name)
        obj_names.append(class_name)
    return obj_names  


def read():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        return
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)

    # time.sleep(5) # after every 5 sec take ss
    img_name = "read.jpg".format(img_counter)
        # img_name = words[-1]+"opencv_frame_{}.png".format(img_counter)

    # path= 'C:/Users/Admin/Desktop/PRIYANSH/MPR/PRIYANSH/faces'
    # cv2.imwrite(img_name, frame)
    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))
    img_counter += 1

    # time.sleep(2) # ML program

    if os.path.exists(os.path.join("absolute path",img_name)): #delete the file
        os.remove(os.path.join("absolute path",img_name))
    else:
        print("The file does not exist")
    
    
    img1="read.jpg"
    img = cv2.imread(img1)
    _, compressedimg = cv2.imencode(".jpg", img)
    file_bytes = io.BytesIO(compressedimg)

    url_api = "https://api.ocr.space/parse/image"
    response = requests.post(url_api, 
        files = {img1 : file_bytes}, 
        data = {"apikey" : "K85285375488957"})
    result = response.json()
    output = result["ParsedResults"][0]["ParsedText"]   
    print(output)
    return output



def bill():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        return
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)

    # time.sleep(5) # after every 5 sec take ss
    img_name = "bill.png".format(img_counter)
        # img_name = words[-1]+"opencv_frame_{}.png".format(img_counter)

    # path= 'C:/Users/Admin/Desktop/PRIYANSH/MPR/PRIYANSH/faces'
    # cv2.imwrite(img_name, frame)
    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))
    img_counter += 1

    # time.sleep(2) # ML program

    if os.path.exists(os.path.join("absolute path",img_name)): #delete the file
        os.remove(os.path.join("absolute path",img_name))
    else:
        print("The file does not exist")
    
    
    # img1="bill.jpg"
    image=cv2.imread('bill.png',0)
    #convert it into text
    text=(pytesseract.image_to_string(image)).lower()
    # print(text)
    match=re.findall(r'\d+[/.-]\d+[/.-]\d+', text)

    st=" "
    st=st.join(match)
    print(st)
    talk("Bill on the date of")
    talk(st)
    #lets find the price of the category.
    price=re.findall(r'[\$\£\€](\d+(?:\.\d{1,2})?)',text)
    price = list(map(float,price)) 
    print(max(price))
    x=max(price)
    talk("price is")  
    talk(x)
    print(output)
    return output





def currency():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        return
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)

    # time.sleep(5) # after every 5 sec take ss
    img_name = "currency.png".format(img_counter)
        # img_name = words[-1]+"opencv_frame_{}.png".format(img_counter)

    # path= 'C:/Users/Admin/Desktop/PRIYANSH/MPR/PRIYANSH/faces'
    # cv2.imwrite(img_name, frame)
    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))
    img_counter += 1

    # time.sleep(2) # ML program

    if os.path.exists(os.path.join("absolute path",img_name)): #delete the file
        os.remove(os.path.join("absolute path",img_name))
    else:
        print("The file does not exist")
    
    max_val = 8
    max_pt = -1
    max_kp = 0

    orb = cv2.ORB_create()
    test_img = read_img('currency.png')
    # test_img = read_img('currency.png')
    # resizing must be dynamic
    original = resize_img(test_img, 0.4)
    # display('original', original)

    # keypoints and descriptors
    # (kp1, des1) = orb.detectAndCompute(test_img, None)
    (kp1, des1) = orb.detectAndCompute(test_img, None)

    training_set = ['files/20.jpg', 'files/50.jpg', 'files/100.jpg', 'files/500.jpg','files/2000.jpg']

    for i in range(0, len(training_set)):
        # train image
        train_img = cv2.imread(training_set[i])

        (kp2, des2) = orb.detectAndCompute(train_img, None)

        # brute force matcher
        bf = cv2.BFMatcher()
        all_matches = bf.knnMatch(des1, des2, k=2)

        good = []
        # give an arbitrary number -> 0.789
        # if good -> append to list of good matches
        for (m, n) in all_matches:
            if m.distance < 0.789 * n.distance:
                good.append([m])

        if len(good) > max_val:
            max_val = len(good)
            max_pt = i
            max_kp = kp2

        print(i, ' ', training_set[i], ' ', len(good))

    if max_val != 8:
        print(training_set[max_pt])
        print('good matches ', max_val)

        train_img = cv2.imread(training_set[max_pt])
        img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
        
        note = str(training_set[max_pt])[6:-4]
        print('\nDetected denomination: Rs. ', note)
        talk('\nDetected denomination: Rs. ')
        talk(note)
        # audio_file = 'audio/{}.mp3'.format(note)
        #audio_file = "value.mp3"
        #tts = gTTS(text=speech_out, lang="en")
        #tts.save(audio_file)
        #return_code = subprocess.call(["afplay", audio_file])
        #playsound(audio_file)
        # (plt.imshow(img3), plt.show())

    else:
        print('No Matches')
        talk('No Matches')
    return


talk('listening...')
while True:
    
    try:
    # connecting to microphone
    # while i<6:
        with sr.Microphone() as source:
            print('listening...')
            
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            command = r.recognize_google(audio)
            command = command.lower()
            print(command)
        # if alexa is in command the only
        if 'alexa' in command:
            words = command.replace('alexa', '')
            
            if words == " hello":
                talk('Hello How are you')

            # testing
            if words==" identify":
                talk("okay")
                name=classify_face("test.jpg")
                
                for a in name:
                    talk(a)
            # testing 
            if words == " take ":
                talk("okay")
                take_image(words)

            # save a new face
            if "save" in words: 
                talk("okay")
                take_image(words)
            # prints the name of person   
            if "hu" in words: 
                talk("okay")
                name=check_image()  
                if name is None:
                    # 👇️ this runs
                    talk('not recognizable')
                else:
                    for a in name:
                        talk(a)
                # classify_face("test.jpg")   
            # checks surrounding
            if "surrounding" in words: 
                talk("okay")
                name=check_surrounding()
                if name is None:
                    # 👇️ this runs
                    talk('not recognizable')
                else:
                    for a in name:
                        talk(a) 
            if "read" in words: 
                talk("okay")
                name=read()
                talk(name)   

            if "bill" in words:
                talk("okay") 
                name=bill()
                # talk(name)   
            if "currency" in words: 
                talk("okay")
                name=currency()
                # talk(name)                   
            if words == "exit":
                print("...")
                sleep(1)
                print("...")
                sleep(1)
                print("...")
                sleep(1)
                print("Goodbye")
                break
    except:
            pass                 

    
