import os
import cv2
import shutil
import numpy as np
from skimage import data, filters
 
    
def get_video_median(path, root, count):
    cap = cv2.VideoCapture(path)
    # Randomly select 100 frames
    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=100)
    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frames.append(frame)
    # Calculate the median along the time axis
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8) 
    # Save the median image in the same folder as video
    img_save = '{}/median_100.jpg'.format(root)
    cv2.imwrite(img_save, medianFrame)
    shutil.copy(img_save, './median_result/{}.jpg'.format(str(count)))
    
    
def video_list(path):
    count = 0
    for root, _, vids in os.walk(path):
        for f in vids:
            if f[-4:] == '.avi':
                video_path = os.path.join(root, f)
                get_video_median(video_path, root, count)
                count += 1        
    print(count)
      

# video path
path = "./aicity23/data/"
video_list(path)