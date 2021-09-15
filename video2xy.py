import numpy as np
import cv2
import math

%matplotlib inline
import matplotlib.pyplot as plt

from scipy.io import wavfile

from tqdm.notebook import tqdm

def convert_image(img, contour_size_threshold, sample_rate, fps):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,100,200)

    ret, thresh = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
    
    frame_samples = math.floor(sample_rate / fps)

    data_frame = np.zeros((frame_samples,2), np.float32)

    points = np.array([], dtype=np.int32)
    
    for i in range(0, len(contours)):
        if contours[i].shape[0] < contour_size_threshold:
            continue
        points=np.append(points, contours[i][:,0,:])

    points = points.reshape((-1,2))

    w = np.max(img.shape)

    osc_image = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    if points.shape[0] > 0:
        stride = math.ceil(points.shape[0] / frame_samples)
        for i in range(0, points.shape[0], stride):
            idx = int(i / stride)
            osc_image[points[i,1],points[i,0],0] = 255
            data_frame[idx,0] = points[i,0] / w * 1.2 - 0.6
            data_frame[idx,1] = points[i,1] / w * (-1.2) + 0.6
    
    plt.imshow(osc_image, cmap='Greys', interpolation='bilinear')
    
    if (points.shape[0] < frame_samples):
        data_frame = np.tile(data_frame[:points.shape[0]], int(frame_samples / points.shape[0]) + 1)[:frame_samples]
    
    return data_frame

def process_photo(filename, outfile, contour_size_threshold=10, sample_rate=96000, fps=30):
    img = cv2.imread(filename)
    data = convert_image(img, contour_size_threshold=contour_size_threshold, sample_rate=sample_rate, fps=fps)
    wavfile.write(outfile, sample_rate, data)
    
def process_video(filename, outfile, frame_limit=50, contour_size_threshold=10, sample_rate=96000, fps=30):
    data = np.zeros((1,2), np.float32)
    cap = cv2.VideoCapture(filename)
    pbar = tqdm()
    i = 0
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret:
            pbar.update(1)
        #     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = convert_image(img, contour_size_threshold=60, sample_rate=96000, fps=30)
            data = np.concatenate([data, frame])
        else:
            break

        i += 1
        if (frame_limit > 0) and (i > frame_limit):
            break

    wavfile.write(outfile, sample_rate, data)

def process(filename, outfile, contour_size_threshold=10, sample_rate=96000, fps=30, frame_limit=50):
    if ('.mp4' in filename) or ('.mov' in filename):
        process_video(filename, outfile, frame_limit, contour_size_threshold, sample_rate, fps)
    if ('.png' in filename) or ('.jpg' in filename) or ('.jpeg' in filename):
        process_photo(filename, outfile, contour_size_threshold, sample_rate, fps)


process('image.jpeg', 'audio.wav', frame_limit=0, contour_size_threshold=100)