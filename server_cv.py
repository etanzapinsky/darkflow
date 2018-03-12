from darkflow.net.build import TFNet
import subprocess as sp
import numpy as np
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT = 'PUT AWAY PHONE'

OPTIONS = {
    "model": "cfg/yolo.cfg",
    "load": "bin/yolo.weights",
    "threshold": 0.1
}

FFMPEG_BIN = 'ffmpeg'
command = [
    FFMPEG_BIN,
    '-i',
    'fifo',  # fifo is the named pipe
    '-pix_fmt',
    'bgr24',  # opencv requires bgr24 pixel format.
    '-vcodec',
    'rawvideo',
    '-an',
    '-sn',  # we want to disable audio processing (there is no audio)
    '-f',
    'image2pipe',
    '-'
]
pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)


def main():
    tfnet = TFNet(OPTIONS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('cv_out.avi', fourcc, 20.0, (640, 480))

    while (True):
        # Capture frame-by-frame
        raw_image = pipe.stdout.read(640 * 480 * 3)
        # transform the byte read into a numpy array
        frame = np.fromstring(raw_image, dtype='uint8')
        frame = frame.reshape(
            (480, 640,
             3))  # Notice how height is specified first and then width

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
            out.write(blank)
        else:
            out.write(disp)

        pipe.stdout.flush()

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
