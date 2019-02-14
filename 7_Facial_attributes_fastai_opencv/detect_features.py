# USAGE
"""
python detect_features.py \
--input_path "../data/celeba/faces/" \
--output_path "./" \
--save_video False
"""

import click
import numpy as np
import cv2
import imutils
from pathlib import Path
from fastai.vision.data import ImageItemList
from fastai.vision.learner import create_cnn
from fastai.vision import models
from fastai.vision.image import pil2tensor,Image

@click.command()
@click.option(
    "--input_path",
    "-ip",
    default="./",
    required=True,
    help="Path to model and labels file",
)
@click.option(
    "--output_path", "-op", default="./", required=True, help="Path to the output video"
)
@click.option("--save_video", "-s", default=True, type=bool, help="Want to save video?")
def detect_facial_attributes(input_path, output_path, save_video):
    path = Path(input_path)

    # Creating a databunch
    imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data = (
        ImageItemList.from_csv(path, csv_name="labels.csv")
        .no_split()
        .label_from_df(label_delim=" ")
        .transform(None, size=128)
        .databunch(no_check=True)
        .normalize(imagenet_stats)
    )

    # Loading our model
    learn = create_cnn(data, models.resnet50, pretrained=False)
    learn.load("ff_stage-2-rn50")

    # Loading HAAR cascade
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    if save_video:
	    out = cv2.VideoWriter(output_path + "output.avi", -1, 20.0, (640, 480))

    while True:
        # Capture frame-by-frame
        _ , frame = cap.read()
        
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find faces using Haar cascade
        face_coord = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        ## Looping through each face
        for coords in face_coord:
            
            ## Finding co-ordinates of face
            X, Y, w, h = coords

            ## Finding frame size
            H, W, _ = frame.shape

            ## Computing larger face co-ordinates
            X_1, X_2 = (max(0, X - int(w * 0.35)), min(X + int(1.35 * w), W))
            Y_1, Y_2 = (max(0, Y - int(0.35 * h)), min(Y + int(1.35 * h), H))

            ## Cropping face and changing BGR To RGB
            img_cp = frame[Y_1:Y_2, X_1:X_2].copy()
            img_cp1 = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)

            ## Prediction of facial featues
            prediction = str(
                learn.predict(Image(pil2tensor(img_cp1, np.float32).div_(255)))[0]
            ).split(";")
            label = (
                " ".join(prediction)
                if "Male" in prediction
                else "Female " + " ".join(prediction)
            )
            label = (
                " ".join(prediction)
                if "No_Beard" in prediction
                else "Beard " + " ".join(prediction)
            )

            ## Drawing facial boundaries
            cv2.rectangle(
                img=frame,
                pt1=(X, Y),
                pt2=(X + w, Y + h),
                color=(128, 128, 0),
                thickness=2,
            )

            ## Drawing facial attributes identified
            label_list = label.split(" ")
            for idx in range(1, len(label_list) + 1):
                cv2.putText(
                    frame,
                    label_list[idx - 1],
                    (X, Y - 14 * idx),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 128, 0),
                    2,
                )

        # Display the resulting frame
        cv2.imshow("frame", frame)

        ## Save the resulting frame
        if save_video:
            out.write(frame)

        ## Escape keys
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_facial_attributes()
