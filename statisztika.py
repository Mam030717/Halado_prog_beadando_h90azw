# -*- coding: utf-8 -*-
"""
statisztika.py

Egyszerű statisztika-logger:
- minden átlépésnél beír egy sort a statisztika.csv fájlba
"""

import csv
import os
from datetime import datetime


class AtlepesLogger:
    """
    Minden átlépést egy sorban rögzít:
    - időpont
    - track_id
    - irány (pl. 'lefelé')
    """

    def __init__(self, fajlnev: str = "statisztika.csv"):
        self.fajlnev = fajlnev

        # ha még nincs fájl, létrehozunk egy fejlécet
        if not os.path.exists(self.fajlnev):
            with open(self.fajlnev, "w", newline="", encoding="utf-8") as f:
                iro = csv.writer(f, delimiter=";")
                iro.writerow(["időpont", "track_id", "irány"])

    def log(self, track_id: int, irany: str):
        """Új esemény rögzítése a CSV-ben."""
        ido = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.fajlnev, "a", newline="", encoding="utf-8") as f:
            iro = csv.writer(f, delimiter=";")
            iro.writerow([ido, track_id, irany])

        print(f"[STAT] átlépés logolva: track_id={track_id}, irány={irany}")
