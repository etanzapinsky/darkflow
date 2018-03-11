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

        print(result)

        # description
        '''
        If you take out your phone to take a picture, this version blacks out everything
        except your phone.
        '''

        blank = np.zeros((height, width, channels), np.uint8)

        cells = [r for r in result if r['label'] == 'cell phone']

        for cell in cells:
            for y in range(cell['topleft']['y'], cell['bottomright']['y']):
                for x in range(cell['topleft']['x'], cell['bottomright']['x']):
                    blank[y][x] = disp[y][x]

        if cells:
            cv2.imshow('frame', blank)
        else:
            cv2.imshow('frame', disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
