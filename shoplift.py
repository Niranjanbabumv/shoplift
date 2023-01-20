import cv2
from google.colab.patches import cv2_imshow
import numpy as np
from sklearn import svm
import smtplib
from email.message import EmailMessage
 

def email_alert():
    email = EmailMessage()
    TEST_EMAIL = 'niranjan.ibm@gmail.com'
    email['from'] = TEST_EMAIL
    email['to'] = 'niranjanbabu.mv@cognizant.com'
    email['subject'] = 'Test - Abnormal Event Detected'
    email.set_content('Abnomal Event Detected in the Video Processing')
    with smtplib.SMTP(host='smtp.gmail.com', port=587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(TEST_EMAIL, 'xxxxxx')
        smtp.send_message(email)
        # print('The mail was sent!')
# Load the video
cap = cv2.VideoCapture("/content/video2.mp4")

# Extract frames
frames = []
while True:
    ret, frame = cap.read()    
        
    if not ret:
        break
    frames.append(frame)

# Convert frames to grayscale and reshape for analysis
#gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
#gray_frames = [f.reshape(-1) for f in gray_frames]

# Convert frames to grayscale and reshape for analysis
#gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
#gray_frames = [f.reshape(1, -1) for f in gray_frames]
    
# Convert frames to grayscale and reshape for analysis
gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
gray_frames = np.array(gray_frames)
print("Printing2",gray_frames.shape, gray_frames.dtype)
if len(gray_frames.shape) == 3:
    gray_frames = gray_frames.reshape(gray_frames.shape[0], -1)


# Fit the One-Class SVM
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(gray_frames)
print("step3 - fit of one-class svm done")
# Predict anomaly on each frame
anomalies = []
for i, frame in enumerate(gray_frames):
    pred = clf.predict([frame])
    if pred[0] == -1:
        anomalies.append(i)
print("Predictions anomalies ",anomalies)
# Visualize the results
mailSent = False
count = 1
for i, frame in enumerate(frames):
    if i in anomalies:
        print("Detected")
        cv2.putText(frame, "Anomaly", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        count = count + 1
        if (count == 5): 
          email_alert()
        cv2_imshow(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
