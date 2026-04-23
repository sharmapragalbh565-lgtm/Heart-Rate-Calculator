import cv2
import numpy as np
import time
from scipy.signal import butter, lfilter
from scipy.fftpack import fft


BUFFER_SIZE = 300  
FPS = 30  
LOW_CUT = 0.8  
HIGH_CUT = 3.0  

data_buffer = []
times = []

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

print("Starting Heart Rate Detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = frame[y:y+h//2, x:x+w] 
        green_channel = np.mean(roi[:, :, 1])  

        data_buffer.append(green_channel)
        times.append(time.time())

        if len(data_buffer) > BUFFER_SIZE:
            data_buffer.pop(0)
            times.pop(0)

        if len(data_buffer) == BUFFER_SIZE:
            filtered = bandpass_filter(
                np.array(data_buffer),
                LOW_CUT,
                HIGH_CUT,
                FPS
            )

            fft_data = np.abs(fft(filtered))
            freqs = np.fft.fftfreq(len(filtered), 1.0/FPS)

            idx = np.where((freqs > LOW_CUT) & (freqs < HIGH_CUT))
            peak_freq = freqs[idx][np.argmax(fft_data[idx])]
            bpm = peak_freq * 60

            cv2.putText(frame, f"BPM: {int(bpm)}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Heart Rate Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
