# YLO Labeling Tool

[![English](https://img.shields.io/badge/English-007BFF?style=for-the-badge&logo=google-chrome)](README.md)
[![Русский](https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-2E7D32?style=for-the-badge&logo=google-chrome)](README-ru.md)
[![Français](https://img.shields.io/badge/Fran%C3%A7ais-0055A4?style=for-the-badge&logo=google-chrome)](README-fr.md)
[![Deutsch](https://img.shields.io/badge/Deutsch-000000?style=for-the-badge&logo=google-chrome)](README-de.md)
[![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-BC002D?style=for-the-badge&logo=google-chrome)](README-ja.md)
[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-007A33?style=for-the-badge&logo=google-chrome)](README-zh.md)

Leistungsstarkes Werkzeug zur Annotation von Bildern im YOLO-Format mit Unterstützung für automatische Kennzeichnung durch neuronale Netze.

## Übersicht

Diese Anwendung bietet einen vollständigen Annotation-Workflow:
- Automatische Generierung von Begrenzungsrahmen und Polygonen mithilfe vortrainierter YOLO-Modelle (detect/OBB/segment)
- Manuelle Bearbeitung von Annotationen (hinzufügen, löschen, verschieben, skalieren)
- Echtzeit-Bewertung gegenüber Ground-Truth-Daten
- Projektmanagement und Export in Trainingsdatensätze

**Key Features:**
- ✅ Zwei Annotation-Modi: Box-Modus und Polygon-Modus
- ✅ Modellbasierte Auto-Kennzeichnung (YOLO detect/OBB/segment)
- ✅ IoU-basierte Bewertungsmodus mit Visualisierung von FP/FN
- ✅ Farbcodierung der Klassen und benutzerdefinierter Klassenmanager
- ✅ Vollständige Tastaturnavigation und Hotkeys
- ✅ Konfigurierbare UI-Sprache (Mehrsprachig unterstützt)

> 💡 **Tipp**: Verwenden Sie `H`, um Annotationen ein-/auszublenden, `N`, um in das Trainingsverzeichnis zu speichern, `E`, um den Bewertungsmodus zu aktivieren.

---

## Schnellstart

### Voraussetzungen
- Python 3.8+
- Windows / Linux / macOS

### Installation
```bash
git clone https://github.com/V171/yolo-labeling-tool.git
cd yolo-labeling-tool
python -m venv venv
source venv/bin/activate  # Für Windows: venv\Scripts\activate
pip install ultralytics opencv-python pyqt5 numpy
```

### Konfiguration
Bearbeiten Sie `config.json`:
```json
{
  "model_path": "yolov8n.pt",
  "images_dir": "images",
  "train_dir": "TrainDataset",
  "language": "de",
  "annotation_mode": "box",
  "iou_threshold": 0.5
}
```
- `model_path`: Pfad zu Ihrem YOLO-Modell (.pt).
- `images_dir`: Verzeichnis mit Rohbildern zur Annotation.
- `train_dir`: Verzeichnis zum Speichern annotierter Bilder als Ground Truth (kann beliebig sein).
- `language`: UI-Sprache (`en`, `ru`, `fr`, `de`, `ja`, `zh`). Standard ist `en`.
- `annotation_mode`: Standardmodus (`box` oder `poly`).
- `iou_threshold`: Mindest-IoU für Bewertungsübereinstimmung.

> 📌 Die Verzeichnisse `images_dir` und `train_dir` sind **externe Verzeichnisse** und können außerhalb des Toolordners liegen. Sie können sie auf beliebige Orte setzen.

### Ausführen
```bash
python labeler.py
```

> ⚠️ **Leistungshinweis**: Der erste Inferenzschritt nach dem Start kann langsam sein, da das Modell geladen wird. Bei großen Bilderverzeichnissen empfiehlt sich die Aufteilung in Unterordner, um die Bildwechselzeiten zu reduzieren.

---

## Navigation

| Funktion | Beschreibung |
|--------|-------------|
| **Bildliste** | Öffnen über Menü: `Ansicht > Bildliste anzeigen/ausblenden`. Klicken Sie auf einen Eintrag, um dorthin zu springen. Hintergrundfarben zeigen den Status an: Weiß (unbearbeitet), Gelb (annotiert), Grün (in Train gespeichert). |
| **Klassenliste** | Öffnen über Menü: `Ansicht > Klassen anzeigen/ausblenden`. Klicken Sie auf eine Klasse, um sie der ausgewählten Annotation zuzuweisen. |
| **Auswertungszusammenfassung** | Öffnen über Menü: `Ansicht > Auswertungszusammenfassung anzeigen/ausblenden`. Zeigt Statistiken für das aktuelle Bild an. |

---

## Annotation-Modi

| Modus | Shortcut | Beschreibung |
|------|----------|-------------|
| **Box-Modus** | `B` | Rechteckige Begrenzungsrahmen zeichnen |
| **Polygon-Modus** | `P` | Freie Polygone mit Steuerung der Eckpunkte zeichnen |

> 📌 Im **Polygon-Modus** beenden Sie das Zeichnen mit einem Rechtsklick, nachdem mindestens zwei Punkte platziert wurden.

---

## Hotkeys

| Taste | Aktion |
|-----|--------|
| `←` / `→` | Vorheriges / Nextes Bild |
| `Z` | Zufälliges Bild |
| `N` | Aktuelles Bild + Annotationen in Train-Verzeichnis speichern (Vertrauenswerte werden entfernt) |
| `R` | Annotationen zurücksetzen (Modell erneut auf aktuelles Bild ausführen) |
| `H` | Sichtbarkeit der Annotationen umschalten |
| `V` | Ansicht zurücksetzen (zentrieren + Maßstab 1x) |
| `Leertaste` | Annotationen temporär ausblenden |
| `Delete` | Ausgewählte Annotation löschen |
| `0-9` | Klassen-ID (0-9) für ausgewählten Rahmen festlegen |
| `Strg+←/→/↑/↓` | Feineinstellung der Größe des ausgewählten Rahmens |
| `E` | Auswertungsmodus umschalten |
| `C` | Anzeige des Klassennamens umschalten (ID vs Name) |
| `B` | In Box-Modus wechseln |
| `P` | In Polygon-Modus wechseln |

> 🔍 **Hinweis**: `N` speichert Annotationen in `train_dir` und **entfernt Vertrauenswerte** — macht sie trainingsgeeignet. Annotationen bleiben im Tool sichtbar gemäß Threshold-Einstellung.

---

## Dateiformate

### YOLO Box-Format
Jede `.txt`-Datei entspricht ihrem Bildnamen:
```
<class_id> <x_center> <y_center> <width> <height> [<score>]
```

### Polygone (für OBB und Segmentierung)
Für Polygonannotationen lautet das Format:
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn> [<score>]
```

- Koordinaten sind normiert `[0,1]` entsprechend Bildbreite und -höhe
- **Vertrauenswert wird bei Speicherung in `train_dir` weggelassen** (Ground-Truth-Format)

---

## Auswertungsmodus

Bei Aktivierung (Taste `E`):
- Vergleicht aktuelle Annotationen mit Ground Truth in `train_dir`
- Markiert:
  - 🔴 **Falsch positive (FP)**: Erkannt, aber nicht in GT
  - 🔵 **Falsch negative (FN)**: In GT, aber nicht erkannt
- **Wahre positive (TP) werden NICHT visualisiert** — sie gelten als korrekt.
- Um die **Fehlerliste** zu öffnen:
  1. Gehen Sie zum Panel „Auswertungszusammenfassung“ (`Ansicht > Auswertungszusammenfassung anzeigen/ausblenden`).
  2. **Doppelklicken** Sie auf eine Zelle in der Spalte **FP** oder **FN** für eine bestimmte Klasse.
  3. Die Dock-Leiste „Fehlerliste“ öffnet sich und zeigt nur Fehler dieses Typs und dieser Klasse.
- Um zu einem Fehler zu navigieren:
  - **Doppelklicken** Sie auf einen Eintrag in der **Fehlerliste**. Die Anwendung lädt das entsprechende Bild.

> 📊 **Verhalten der Fehlerliste**:
> - Nur Fehler der **ausgewählten Klasse** erscheinen.
> - Nur Fehler des **geklickten Typs (FP/FN)** erscheinen.
> - Doppelklick auf eine Zeile öffnet das entsprechende Bild im Viewer.

---

## Klassenverwaltung

Verwenden Sie **Ansicht > Klassen anzeigen/ausblenden**, um die Klassenliste zu öffnen.
Klicken Sie auf eine Klasse, um sie der aktuellen Annotation zuzuweisen.

Um Klassen zu verwalten:
- **Menü > Aktion > Klassen verwalten...**
- Fügen Sie hinzu, umbenennen, entfernen oder setzen Sie Klassen vom Modell zurück
- Weisen Sie benutzerdefinierte Farben pro Klasse zu

---

## Projektorganisation

``` 
your-project/
├── config.json
├── labeler.py
├── images/                 # Ihre Rohbilder (in config.json eingestellt)
│   ├── img1.jpg
│   ├── img1.txt            # Automatisch generierte oder manuelle Annotationen
│   └── ...
├── TrainDataset/           # Exportierte Trainingsdaten (in config.json eingestellt)
│   ├── img1.jpg
│   └── img1.txt
└── README.md               # Diese Datei
```

> 📌 Annotationen werden in `images_dir` gespeichert. Nutzen Sie `N`, um Bild + Annotationen in `train_dir` mit entfernten Vertrauenswerten zu kopieren.

---

## Fehlerbehebung

| Problem | Lösung |
|-------|----------|
| Kein Modell geladen | Prüfen Sie `model_path` in `config.json`; stellen Sie sicher, dass die `.pt`-Datei existiert |
| Langsame erste Inferenz | Normal — Modell wird beim ersten Gebrauch geladen |
| Keine Annotationen sichtbar | Drücken Sie `H`, um die Sichtbarkeit umzuschalten; prüfen Sie den Schwellenwert |
| Falsche Klassennamen | Modell neu laden oder „Aus Modell zurücksetzen“ im Klassenmanager verwenden |
| Fehlerliste öffnet sich nicht | Doppelklicken Sie auf eine Zelle in der Spalte **FP** oder **FN** in der Auswertungszusammenfassung — **nicht** auf die Kopfzeile |
| Bilder laden nicht | Stellen Sie sicher, dass `images_dir` in `config.json` korrekt ist und gültige Bilddateien enthält |

---

## Dank

- [Ultralytics](https://github.com/ultralytics/ultralytics) für YOLO
- [OpenCV](https://opencv.org/) für Computer Vision
- [PyQt5](https://pypi.org/project/PyQt5/) für das GUI-Framework
- [Qwen](https://chat.qwen.ai/) für AI-gestützte Entwicklung

© 2025 MIT License
