import cv2
import numpy as np


def make_video(frames, i_):
    size = (456, 248)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    out = cv2.VideoWriter(f'videos/harvest.webm', fourcc, 24, size)

    for i in range(len(frames)):
        # rgb_img = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        out.write(frames[i].astype(np.uint8))

    out.release()
