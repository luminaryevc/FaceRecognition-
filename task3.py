#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:45:44 2023

@author: shahadsami
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:22:39 2023

@author: shahadsami
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import numpy as np

img_height = 180
img_width = 180
class_map = {0: 'Besan',
 1: 'HalaF',
 2: 'Omar',
 3: 'Tareq',
 4: 'abdulaziz',
 5: 'abdulrahman',
 6: 'abdulrahmanN',
 7: 'afnan',
 8: 'ahmad',
 9: 'anas',
 10: 'hassan',
 11: 'hazem',
 12: 'marwh',
 13: 'mohammad',
 14: 'mohammadA',
 15: 'sanad',
 16: 'shahadpic',
 17: 'shooq'}

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained face recognition model
face_model = load_model('/Users/shahadsami/Desktop/faces_model.h5')

# Define a function for real-time face recognition
def real_time_face_recognition():
    video_capture = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        ret, frame = video_capture.read()
        print(ret, frame)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) from the original frame
            roi_color = frame[y:y+h, x:x+w]
            
            # Preprocess the ROI for face recognition
            resized_roi = cv2.resize(roi_color, (img_width, img_height))
            expanded_frame = np.expand_dims(resized_roi, axis=0)

            image_float = np.array(expanded_frame, dtype=np.float32)


            image_normalized = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float))
            image_normalized *= 255.0
            
            # Make predictions using the face recognition model
            print(image_normalized)

            predictions = face_model.predict(image_normalized)
            predicted_label =class_map[np.argmax(predictions)]
            
            # Draw a rectangle and label around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Real-Time Face Recognition', frame)
        
        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        def saveToExel(day, date, name, image, time):
            filepath = '/Users/shahadsami/Desktop/EVC_team.xlsx'
    
            workbook = openpyxl.load_workbook(filepath)
            sheet = workbook.active  # workbook['Sheet1']

    # Find last row of written data
            row = sheet.max_row + 1

    # Writing your data at the bottom
            sheet.cell(row=row, column=1).value = day
            sheet.cell(row=row, column=2).value = date
            sheet.cell(row=row, column=3).value = name
            sheet.cell(row=row, column=4).value = time
            sheet.cell(row=row, column=5).value = image

            workbook.save(filepath)
            
    

    # Release the video capture and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

# Run the real-time face recognition function
real_time_face_recognition()
