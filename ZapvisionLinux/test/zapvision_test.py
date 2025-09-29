from zapvision_py import ZapvisionTracker
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(
    description='Parse config.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image', type=str, default='Image.png',
                    help='image file')
arg = parser.parse_args()


img = cv2.imread(arg.image)
height, width, channels = img.shape

if channels == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

tracker = ZapvisionTracker()
tracker.process(img, width, height, width)
count = tracker.result_count()
print(f"{count} markers detected")

for i in range(count):
    type = tracker.result_type(i)
    value = tracker.result_qr_code(i)
    landmarks = tracker.result_landmarks(i)

    print(f"\ndetection {i}: \n\ttype: {type}\n\tvalue: {value}\n\tlandmarks: {landmarks}")

