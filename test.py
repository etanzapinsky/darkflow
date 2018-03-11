from darkflow.net.build import TFNet
import numpy as np
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT = 'PUT AWAY PHONE'

OPTIONS = {
    "model": "cfg/yolo.cfg",
    "load": "bin/yolo.weights",
    "threshold": 0.1
}


def main():
    tfnet = TFNet(OPTIONS)

    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # flip image horizontally to be mirror image
        disp = cv2.flip(frame, 1)
        height, width, channels = disp.shape

        result = tfnet.return_predict(disp)

        labels = [r['label'] for r in result]
        print(labels)

        # description
        '''
        If you take out your phone to take a picture, this version tells you to put away your
        phone.
        '''

        blank = np.zeros((height, width, channels), np.uint8)

        blank = cv2.putText(
            blank,
            TEXT,
            (int(width / 2), int(height / 2)),
            FONT,
            1,
            (255, 255, 255),  # white
            thickness=2,
            lineType=cv2.LINE_AA)

        if 'cell phone' in labels:
            cv2.imshow('frame', blank)
        else:
            cv2.imshow('frame', disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
