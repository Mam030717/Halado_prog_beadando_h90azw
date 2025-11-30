# -*- coding: utf-8 -*-
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

# Statisztika
from statisztika import AtlepesLogger

# ==== BEÁLLÍTÁSOK ====
BEMENET_VIDEO = "1.mp4"
KIMENET_VIDEO = "eredmeny_kovetes_szamlalas.mp4"

BIZALMI_KUSZOB = 0.5
MAX_TAVOLSAG = 0.2  # DeepSORT távolság threshold

YOLO_MODELL = "yolov8n.pt"

# DeepSORT feature extractor modell – mars-small128.pb
ENCODER_MODELL_UTVONAL = "deep_sort/resources/networks/mars-small128.pb"


def becsult_szin(bgr_kep: np.ndarray) -> str:
    """
    Egyszerű szín-becslés: átlag HSV alapján pár alapszínre kerekítünk.
    Nem tökéletes, de statisztikának jó.
    """
    if bgr_kep.size == 0:
        return "ismeretlen"

    hsv = cv2.cvtColor(bgr_kep, cv2.COLOR_BGR2HSV)
    h, s, v = np.mean(hsv.reshape(-1, 3), axis=0)

    if v < 60:
        return "fekete"
    if v > 190 and s < 60:
        return "fehér"
    if s < 60:
        return "szürke"

    if 0 <= h < 15 or 160 <= h <= 180:
        return "piros"
    if 15 <= h < 35:
        return "sárga"
    if 35 <= h < 85:
        return "zöld"
    if 85 <= h < 130:
        return "kék"
    if 130 <= h < 160:
        return "lila"

    return "ismeretlen"


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
        osztaly_id = int(dob.cls[0])  # most csak eltároljuk, de később NEM használjuk

        if bizalom < BIZALMI_KUSZOB:
            continue

        detekciok.append(((x1, y1, x2, y2), bizalom, osztaly_id))

    return detekciok


def fo():
    print("Encoder modell útvonal:", ENCODER_MODELL_UTVONAL)
    print("Létezik-e a fájl?", os.path.exists(ENCODER_MODELL_UTVONAL))

    if os.path.exists(ENCODER_MODELL_UTVONAL):
        print("Fájlméret (byte):", os.path.getsize(ENCODER_MODELL_UTVONAL))
    else:
        print("HIBA: nem találom a mars-small128.pb modellt!")
        return

    # --- Videó megnyitása ---
    video = cv2.VideoCapture(BEMENET_VIDEO)
    if not video.isOpened():
        print(f"Nem tudom megnyitni a videót: {BEMENET_VIDEO}")
        return

    # csak információ, most nem használjuk
    fps_video = video.get(cv2.CAP_PROP_FPS) or 30.0

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

    # Számláló vonal – kb. a kép 72%-ánál (lent, közel a kamerához)
    frame_mag = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vonal_y = int(frame_mag * 0.72)
    print("Számláló vonal Y:", vonal_y)

    elozo_kozeppontok = {}
    mar_atlepett_azonositok = set()
    atlepes_szamlalo = 0

    # Statisztika logger
    logger = AtlepesLogger()

    while True:
        kezd_ido = datetime.datetime.now()

        ok, kep = video.read()
        if not ok:
            print("Vége a videónak.")
            break

        eredmenyek = yolo_modell(kep)

        doboz_lista = []
        bizalom_lista = []
        osztaly_lista = []  # most nem fogjuk felhasználni, de bent hagyhatjuk

        for eredm in eredmenyek:
            detekciok = jarmu_dobozok_yolobol(eredm)
            for (x1, y1, x2, y2), bizalom, osztaly in detekciok:
                doboz_lista.append([x1, y1, x2 - x1, y2 - y1])  # x,y,w,h
                bizalom_lista.append(bizalom)
                osztaly_lista.append(osztaly)

        if len(doboz_lista) > 0:
            doboz_np = np.array(doboz_lista, dtype=float)
            jellemzok = encoder(kep, doboz_np)

            # Detection: bbox, score, class_name(str), feature
            # class_name-be most is az osztály_id megy stringként, de NEM használjuk később
            detekcio_objektumok = [
                Detection(b, s, str(c), f)
                for b, s, c, f in zip(doboz_np, bizalom_lista, osztaly_lista, jellemzok)
            ]
        else:
            detekcio_objektumok = []

        # DeepSORT lépései
        koveto.predict()
        koveto.update(detekcio_objektumok)

        # --- Kirajzolás + számlálás ---
        for nyom in koveto.tracks:
            if not nyom.is_confirmed() or nyom.time_since_update > 1:
                continue

            x, y, w, h = nyom.to_tlwh()
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            azon = nyom.track_id

            # Aktuális középpont
            kozep_x = int(x + w / 2)
            kozep_y = int(y + h / 2)

            # Előző középpont lekérdezése
            elozo_kozep = elozo_kozeppontok.get(azon, None)
            if elozo_kozep is not None:
                elozo_y = elozo_kozep[1]

                # Csak a szembejövő (fentről lefelé haladó) járműveket számoljuk
                if elozo_y < vonal_y <= kozep_y:
                    if azon not in mar_atlepett_azonositok:
                        mar_atlepett_azonositok.add(azon)
                        atlepes_szamlalo += 1

                        # --- jármű részlet kivágása színhez ---
                        jarmu_kep = kep[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].copy()
                        szin = becsult_szin(jarmu_kep)

                        # --- statisztika loggolása ---
                        # minden ismeretlen/vegyes típus: egyszerűen "jarmu"
                        logger.log("jarmu", szin)

            # Frissítjük az aktuális középpontot
            elozo_kozeppontok[azon] = (kozep_x, kozep_y)

            # Doboz kirajzolása
            cv2.rectangle(
                kep,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )
            cv2.putText(
                kep,
                f"ID: {azon}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Középpont jelölése
            cv2.circle(kep, (kozep_x, kozep_y), 4, (255, 0, 0), -1)

        # Vonal kirajzolása
        cv2.line(
            kep,
            (0, vonal_y),
            (kep.shape[1], vonal_y),
            (0, 0, 255),
            2
        )

        # FPS kiszámítása
        veg_ido = datetime.datetime.now()
        fps = 1.0 / (veg_ido - kezd_ido).total_seconds()

        # Szövegek a képen
        cv2.putText(
            kep,
            f"FPS: {fps:.2f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            kep,
            f"Szembe jovo atlepesek: {atlepes_szamlalo}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Jarmu kovetes es szamlalas", kep)
        iro.write(kep)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    iro.release()
    cv2.destroyAllWindows()

    # Statisztika kiírása a végén
    logger.kiir_osszefoglalas()

   

if __name__ == "__main__":
    fo()
