import numpy as np
import cv2
from pathlib import Path
import click

FLIP_FRAME = True

@click.command()
@click.option('--dir', default='imgs', help='Destination directory', required=True)
@click.option('--flip', default=True, help='Flip frames', required=True)
@click.option('--capl', default=3, help='Video capture index (left)', required=True)
@click.option('--capr', default=1, help='Video capture index (right)', required=True)
def main(dir, flip, capl, capr):

    cap_left = cv2.VideoCapture(capl)
    cap_right = cv2.VideoCapture(capr)
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)

    Path(f'{str(dir)}/left').mkdir(parents=True, exist_ok=True)
    Path(f'{str(dir)}/right').mkdir(parents=True, exist_ok=True)

    counter = 0
    while(True):
        if not (cap_left.grab() and cap_right.grab()):
            break

        _, frame_left = cap_left.retrieve()
        _, frame_right = cap_right.retrieve()

        if FLIP_FRAME:
            frame_left = cv2.flip(frame_left, -1)
            frame_right = cv2.flip(frame_right, -1)
            
        cv2.imshow('frames', cv2.resize(np.concatenate([frame_left,frame_right], axis=1), None, fx=0.4, fy=0.4))
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == 32:
            cv2.imwrite(f"{str(dir)}/left/{counter:06d}.png", frame_left)
            cv2.imwrite(f"{str(dir)}/right/{counter:06d}.png", frame_right)

            counter += 1

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()