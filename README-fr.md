# YOLO Labeling Tool

[![English](https://img.shields.io/badge/English-007BFF?style=for-the-badge&logo=google-chrome)](README.md)
[![Русский](https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-2E7D32?style=for-the-badge&logo=google-chrome)](README-ru.md)
[![Français](https://img.shields.io/badge/Fran%C3%A7ais-0055A4?style=for-the-badge&logo=google-chrome)](README-fr.md)
[![Deutsch](https://img.shields.io/badge/Deutsch-000000?style=for-the-badge&logo=google-chrome)](README-de.md)
[![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-BC002D?style=for-the-badge&logo=google-chrome)](README-ja.md)
[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-007A33?style=for-the-badge&logo=google-chrome)](README-zh.md)

Outil d'étiquetage puissant pour annoter des images au format YOLO avec prise en charge de l'étiquetage automatique via des réseaux neuronaux.

## Aperçu

Cette application fournit un flux de travail d'annotation complet :
- Génération automatique de boîtes englobantes et de polygones via des modèles YOLO pré-entraînés (detect/OBB/segment)
- Édition manuelle des annotations (ajouter, supprimer, déplacer, redimensionner)
- Évaluation en temps réel par rapport aux données de référence
- Gestion de projet et export vers des jeux de données d'entraînement

**Fonctionnalités clés :**
- ✅ Deux modes d'annotation : Mode Boîte et Mode Polygone
- ✅ Étiquetage automatisé basé sur modèle (YOLO detect/OBB/segment)
- ✅ Mode d'évaluation basé sur IoU avec visualisation des faux positifs (FP) et faux négatifs (FN)
- ✅ Codage couleur des classes et gestion personnalisée des classes
- ✅ Navigation complète par clavier et raccourcis
- ✅ Langue de l'interface configurable (multilingue)

> 💡 **Astuce** : Utilisez `H` pour afficher/masquer les annotations, `N` pour enregistrer dans le jeu d'entraînement, `E` pour activer le mode d'évaluation.

---

## Démarrage rapide

### Prérequis
- Python 3.8+
- Windows / Linux / macOS

### Installation
```bash
git clone https://github.com/V171/yolo-labeling-tool.git
cd yolo-labeling-tool
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
pip install ultralytics opencv-python pyqt5 numpy
```

### Configuration
Modifiez `config.json` :
```json
{
  "model_path": "yolov8n.pt",
  "images_dir": "images",
  "train_dir": "TrainDataset",
  "language": "fr",
  "annotation_mode": "box",
  "iou_threshold": 0.5
}
```
- `model_path` : Chemin vers votre modèle YOLO (.pt).
- `images_dir` : Répertoire contenant les images brutes à annoter.
- `train_dir` : Répertoire où sauvegarder les images annotées comme données de référence (peut être n'importe quel chemin).
- `language` : Langue de l'interface (`en`, `ru`, `fr`, `de`, `ja`, `zh`). Valeur par défaut : `en`.
- `annotation_mode` : Mode par défaut (`box` ou `poly`).
- `iou_threshold` : Seuil minimal IoU pour la correspondance lors de l'évaluation.

> 📌 Les répertoires `images_dir` et `train_dir` sont **externes** et peuvent se trouver hors du dossier de l'outil. Vous pouvez les définir sur n'importe quel emplacement.

### Lancement
```bash
python labeler.py
```

> ⚠️ **Note de performance** : La première inférence après le lancement peut être lente en raison du chargement du modèle. Pour de grands dossiers d'images, envisagez de les diviser en sous-dossiers pour réduire les temps de changement d'image.

---

## Navigation

| Fonction | Description |
|--------|-------------|
| **Liste des images** | Ouvrez via le menu : `Affichage > Afficher/Masquer la liste d'images`. Cliquez sur un élément pour y accéder directement. Les couleurs de fond indiquent l'état : blanc (non traité), jaune (annoté), vert (enregistré dans Train). |
| **Liste des classes** | Ouvrez via le menu : `Affichage > Afficher/Masquer les classes`. Cliquez sur une classe pour l'assigner à l'annotation sélectionnée. |
| **Résumé d'évaluation** | Ouvrez via le menu : `Affichage > Afficher/Masquer le résumé d'évaluation`. Affiche les statistiques pour l'image courante. |

---

## Modes d'annotation

| Mode | Raccourci | Description |
|------|----------|-------------|
| **Mode Boîte** | `B` | Dessiner des boîtes englobantes rectangulaires |
| **Mode Polygone** | `P` | Dessiner des polygones libres avec contrôle des sommets |

> 📌 En mode **Mode Polygone**, faites un clic droit pour terminer le tracé après avoir placé au moins deux points.

---

## Raccourcis clavier

| Touche | Action |
|-----|--------|
| `←` / `→` | Image précédente / suivante |
| `Z` | Image aléatoire |
| `N` | Enregistrer l'image courante et ses annotations dans le répertoire Train (supprime les scores de confiance) |
| `R` | Réinitialiser les annotations (relancer le modèle sur l'image courante) |
| `H` | Basculer la visibilité des annotations |
| `V` | Réinitialiser la vue (centrer + mise à l'échelle 1x) |
| `Espace` | Masquer temporairement les annotations |
| `Suppr` | Supprimer l'annotation sélectionnée |
| `0-9` | Définir l'ID de classe (0-9) pour la boîte sélectionnée |
| `Ctrl+←/→/↑/↓` | Ajuster finement la taille de la boîte sélectionnée |
| `E` | Basculer le mode d'évaluation |
| `C` | Basculer l'affichage du nom de classe (ID vs Nom) |
| `B` | Passer en mode Mode Boîte |
| `P` | Passer en mode Mode Polygone |

> 🔍 **Remarque** : `N` enregistre les annotations dans `train_dir` et **supprime les scores de confiance** — ce qui les rend adaptées à l'entraînement. Les annotations restent visibles dans l'outil selon le seuil configuré.

---

## Formats de fichiers

### Format YOLO box
Chaque fichier `.txt` correspond à son image :
```
<class_id> <x_center> <y_center> <width> <height> [<score>]
```

### Polygones (pour OBB et segmentation)
Pour les annotations polygonales, le format est :
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn> [<score>]
```

- Les coordonnées sont normalisées `[0,1]` selon la largeur et la hauteur de l'image
- **Le score de confiance est omis** lors de l'enregistrement dans `train_dir` (format données de référence)

---

## Mode d'évaluation

Lorsqu'il est activé (touche `E`) :
- Compare les annotations actuelles aux données de référence dans `train_dir`
- Met en évidence :
  - 🔴 **Faux positifs (FP)** : détectés mais absents des données de référence
  - 🔵 **Faux négatifs (FN)** : présents dans les données de référence mais non détectés
- **Les vrais positifs (TP) ne sont pas visualisés** — ils sont considérés comme corrects.
- Pour ouvrir la **Liste d'erreurs** :
  1. Allez dans le panneau **Résumé d'évaluation** (`Affichage > Afficher/Masquer le résumé d'évaluation`).
  2. **Double-cliquez** sur une cellule dans la colonne **FP** ou **FN** pour une classe spécifique.
  3. Le panneau latéral **Liste d'erreurs** s'ouvre, affichant uniquement les erreurs de ce type et cette classe.
- Pour naviguer vers une erreur :
  - **Double-cliquez** sur un élément dans la **Liste d'erreurs**. L'application chargera l'image correspondante.

> 📊 **Comportement de la Liste d'erreurs** :
> - Seules les erreurs du **classe sélectionnée** sont affichées.
> - Seules les erreurs du **type cliqué (FP/FN)** sont affichées.
> - Un double-clic sur une ligne ouvre l'image correspondante dans le visionneur.

---

## Gestion des classes

Utilisez **Affichage > Afficher/Masquer les classes** pour ouvrir la liste des classes.
Cliquez sur une classe pour l'assigner à l'annotation sélectionnée.

Pour gérer les classes :
- **Menu > Action > Gérer les classes...**
- Ajoutez, renommez, supprimez ou réinitialisez les classes à partir du modèle
- Attribuez des couleurs personnalisées à chaque classe

---

## Organisation du projet

```
your-project/
├── config.json
├── labeler.py
├── images/                 # Vos images brutes (définies dans config.json)
│   ├── img1.jpg
│   ├── img1.txt            # Annotations générées automatiquement ou manuellement
│   └── ...
├── TrainDataset/           # Données d'entraînement exportées (définies dans config.json)
│   ├── img1.jpg
│   └── img1.txt
└── README.md               # Ce fichier
```

> 📌 Les annotations sont sauvegardées dans `images_dir`. Utilisez `N` pour copier l'image et ses annotations dans `train_dir` avec suppression des scores de confiance.

---

## Dépannage

| Problème | Solution |
|---------|----------|
| Modèle non chargé | Vérifiez `model_path` dans `config.json` ; assurez-vous que le fichier `.pt` existe |
| Première inférence lente | Normal — le modèle se charge à la première utilisation |
| Annotations non visibles | Appuyez sur `H` pour basculer la visibilité ; vérifiez le curseur de seuil |
| Noms de classes incorrects | Rechargez le modèle ou utilisez « Réinitialiser à partir du modèle » dans le gestionnaire de classes |
| Liste d'erreurs non ouverte | Double-cliquez sur une cellule dans la colonne **FP** ou **FN** du résumé d'évaluation — **pas** sur l'en-tête de ligne |
| Images non chargées | Assurez-vous que `images_dir` dans `config.json` est correct et contient des fichiers d'image valides |

---

## Remerciements

- [Ultralytics](https://github.com/ultralytics/ultralytics) pour YOLO
- [OpenCV](https://opencv.org/) pour la vision par ordinateur
- [PyQt5](https://pypi.org/project/PyQt5/) pour l'interface graphique
- [Qwen](https://chat.qwen.ai/) pour le développement assisté par IA

© 2025 MIT License
