# -*- coding: utf-8 -*-

import datetime
import cv2
from ultralytics import YOLO
from seged import video_irora_keszit

BEMENET = "1.mp4"
KIMENET = "eredmeny.mp4"
BIZALMI_SZINT = 0.5

def main():
    video = cv2.VideoCapture(BEMENET)
    if not video.isOpened():
        print(f"Nem tal�lom a vide�t: {BEMENET}")
        return

    iro = video_irora_keszit(video, KIMENET)
    modell = YOLO("yolov8n.pt")

    while True:
        kezd = datetime.datetime.now()

        ok, kep = video.read()
        if not ok:
            print("Vege a videonak.")
            break

        eredm = modell(kep)
        for e in eredm:
            dobozok = e.boxes
            if dobozok is None:
                continue

            for dob in dobozok:
                x1, y1, x2, y2 = map(int, dob.xyxy[0].tolist())
                bizalom = float(dob.conf[0])

                if bizalom < BIZALMI_SZINT:
                    continue

                cv2.rectangle(kep, (x1, y1), (x2, y2), (0,255,0), 2)

        veg = datetime.datetime.now()
        fps = 1.0 / (veg - kezd).total_seconds()

        cv2.putText(kep, f"FPS: {fps:.2f}", (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)

        cv2.imshow("Detektalas", kep)
        iro.write(kep)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    iro.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
