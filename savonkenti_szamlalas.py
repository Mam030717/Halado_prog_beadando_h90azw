# -*- coding: utf-8 -*-
"""
savonkenti_szamlalas.py

YOLOv8 + DeepSORT alapú járműkövetés,
sávonkénti számlálással.

- 3 vízszintes sáv (A, B, C) az út egyes részein
- csak a kamera felé (fentről lefelé) érkező járműveket számolja
- minden sávhoz külön számláló tartozik
"""

import os
import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from seged import video_irora_keszit

# DeepSORT importok
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.tools import generate_detections as gdet


# ========= BEÁLLÍTÁSOK =========
BEMENET_VIDEO = "1.mp4"
KIMENET_VIDEO = "savonkenti_szamlalas.mp4"

YOLO_MODELL = "yolov8n.pt"
BIZALMI_KUSZOB = 0.5
MAX_TAVOLSAG = 0.2

ENCODER_MODELL_UTVONAL = "deep_sort/resources/networks/mars-small128.pb"


def jarmu_dobozok_yolobol(eredmeny):
    """
    YOLO eredményobjektum -> lista:
    [ ((x1,y1,x2,y2), bizalom, osztaly_id), ... ]
    """
    detekciok = []

    dobozok = eredmeny.boxes
    if dobozok is None:
        return detekciok

    for dob in dobozok:
        x1, y1, x2, y2 = dob.xyxy[0].tolist()
        bizalom = float(dob.conf[0])
        osztaly_id = int(dob.cls[0])

        if bizalom < BIZALMI_KUSZOB:
            continue

        detekciok.append(((x1, y1, x2, y2), bizalom, osztaly_id))

    return detekciok


def fo():
    print("Encoder modell útvonal:", ENCODER_MODELL_UTVONAL)
    print("Létezik-e a fájl?", os.path.exists(ENCODER_MODELL_UTVONAL))
    if not os.path.exists(ENCODER_MODELL_UTVONAL):
        print("HIBA: nem találom a DeepSORT encoder modellt!")
        return

    # --- Videó megnyitása ---
    video = cv2.VideoCapture(BEMENET_VIDEO)
    if not video.isOpened():
        print(f"Nem tudom megnyitni a videót: {BEMENET_VIDEO}")
        return

    fps_video = video.get(cv2.CAP_PROP_FPS) or 30.0
    frame_szeles = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_mag = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video FPS: {fps_video:.2f}, méret: {frame_szeles}x{frame_mag}")

    # --- Videó író ---
    iro = video_irora_keszit(video, KIMENET_VIDEO)

    # --- YOLO modell ---
    print("YOLO modell betöltése...")
    yolo_modell = YOLO(YOLO_MODELL)

    # --- DeepSORT előkészítés ---
    print("DeepSORT előkészítése...")
    metrika = NearestNeighborDistanceMetric("cosine", MAX_TAVOLSAG, None)
    koveto = Tracker(metrika)

    encoder = gdet.create_box_encoder(
        ENCODER_MODELL_UTVONAL,
        batch_size=32
    )

    # ====== SÁVOK DEFINIÁLÁSA ======
    # Egy közös vízszintes vonal, ahol számlálunk (kb. az út közepe / alatt)
    vonal_y = int(frame_mag * 0.70)

    # Három sáv vízszintesen: A (bal), B (közép), C (jobb)
    # Az x koordinátákat a kép szélességéből számoljuk
    savok = [
        {
            "nev": "A",
            "x1": int(frame_szeles * 0.37),
            "x2": int(frame_szeles * 0.56),
            "szin": (0, 255, 0),    # zöld
            "db": 0
        },
        {
            "nev": "B",
            "x1": int(frame_szeles * 0.56),
            "x2": int(frame_szeles * 0.71),
            "szin": (255, 0, 0),    # kék (B-hez)
            "db": 0
        },
        {
            "nev": "C",
            "x1": int(frame_szeles * 0.71),
            "x2": int(frame_szeles * 0.84),
            "szin": (0, 0, 255),    # piros
            "db": 0
        },
    ]

    print("Számláló vonal Y:", vonal_y)
    for s in savok:
        print(f"Sáv {s['nev']}: x1={s['x1']}, x2={s['x2']}")

    # Nyomkövetés segédstruktúrák
    elozo_kozeppontok = {}          # track_id -> (x, y)
    mar_atlepett = set()            # (track_id, sav_nev) párok, hogy csak 1x számoljuk

    while True:
        kezd_ido = datetime.datetime.now()

        ok, kep = video.read()
        if not ok:
            print("Vége a videónak.")
            break

        eredmenyek = yolo_modell(kep)

        doboz_lista = []
        bizalom_lista = []
        osztaly_lista = []

        for eredm in eredmenyek:
            detekciok = jarmu_dobozok_yolobol(eredm)
            for (x1, y1, x2, y2), bizalom, osztaly in detekciok:
                doboz_lista.append([x1, y1, x2 - x1, y2 - y1])  # x, y, w, h
                bizalom_lista.append(bizalom)
                osztaly_lista.append(osztaly)

        if len(doboz_lista) > 0:
            doboz_np = np.array(doboz_lista, dtype=float)
            jellemzok = encoder(kep, doboz_np)

            detekcio_objektumok = [
                Detection(b, s, str(c), f)
                for b, s, c, f in zip(doboz_np, bizalom_lista, osztaly_lista, jellemzok)
            ]
        else:
            detekcio_objektumok = []

        koveto.predict()
        koveto.update(detekcio_objektumok)

        # ===== Kirajzolás + sávonkénti számlálás =====
        for nyom in koveto.tracks:
            if not nyom.is_confirmed() or nyom.time_since_update > 1:
                continue

            x, y, w, h = nyom.to_tlwh()
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            tid = nyom.track_id

            # aktuális középpont
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            elozo = elozo_kozeppontok.get(tid, None)
            if elozo is not None:
                elozo_cy = elozo[1]

                # csak akkor figyelünk, ha fentről lefelé keresztezi a vonalat
                if elozo_cy < vonal_y <= cy:
                    # megnézzük, hogy a középpont melyik sáv x-tartományába esik
                    for s in savok:
                        if s["x1"] <= cx < s["x2"]:
                            kulcs = (tid, s["nev"])
                            if kulcs not in mar_atlepett:
                                mar_atlepett.add(kulcs)
                                s["db"] += 1
                                print(f"Sáv {s['nev']} +1 (ID={tid})")
                            break

            elozo_kozeppontok[tid] = (cx, cy)

            # doboz kirajzolása
            cv2.rectangle(kep, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                kep,
                f"ID:{tid}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.circle(kep, (cx, cy), 4, (255, 0, 0), -1)

        # számláló vonal kirajzolása
        cv2.line(
            kep,
            (0, vonal_y),
            (frame_szeles, vonal_y),
            (0, 255, 255),
            2
        )

        # sávok kirajzolása a képen (alsó színes csíkok + felirat)
        sav_sav_vastagsag = 10
        sav_szoveg_y = vonal_y + 30

        for s in savok:
            # színes csík
            cv2.line(
                kep,
                (s["x1"], vonal_y),
                (s["x2"], vonal_y),
                s["szin"],
                sav_sav_vastagsag
            )
            # sáv neve + darabszám
            szoveg = f"{s['nev']}: {s['db']}"
            szoveg_x = int((s["x1"] + s["x2"]) / 2) - 30
            cv2.putText(
                kep,
                szoveg,
                (szoveg_x, sav_szoveg_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                s["szin"],
                2,
            )

        # FPS
        veg_ido = datetime.datetime.now()
        fps = 1.0 / (veg_ido - kezd_ido).total_seconds()
        cv2.putText(
            kep,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

        cv2.imshow("Savonkenti jarmu szamlalas", kep)
        iro.write(kep)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    iro.release()
    cv2.destroyAllWindows()

    print("\n=== Vegso savonkénti számlálás ===")
    for s in savok:
        print(f"Sáv {s['nev']}: {s['db']} jármű")


if __name__ == "__main__":
    fo()
