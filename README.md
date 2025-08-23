# YOLO Labeling Tool

[[English](#english)] [[Русский](#Русский)] [[Français](#français)] [[Deutsch](#deutsch)] [[日本語](#日本語)] [[中文](#中文)]

---

## <a name="english"></a> English

A tool for annotating images in YOLO format with support for automatic labeling using neural networks.

### Features

- Automatic image labeling using a YOLO model.
- Manual annotation editing (add, delete, resize, move).
- Support for multiple classes with color coding.
- Image navigation and project management.
- Export annotations in YOLO format.
- Hotkeys for faster operation.
- State saving between sessions.
- **Multi-language UI (English, Russian, French, German, Japanese, Chinese). Language can be changed via the 'Language' menu. English is the default.**

### Installation

#### Requirements

- Python 3.7 or higher
- Windows, Linux, or macOS

#### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/V171/yolo-labeling-tool.git
   cd yolo-labeling-tool
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   **Linux/macOS:**
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install ultralytics opencv-python pyqt5 numpy
   ```

5. Download a pre-trained YOLO model (e.g., `yolov8n.pt`) from [official Ultralytics releases](https://github.com/ultralytics/assets/releases) or use your own model.

### Configuration

1. Open the `config.json` file in the project's root directory:
   ```json
   {
       "model_path": "path/to/your/model.pt",
       "images_dir": "path/to/your/images",
       "train_dir": "TrainDataset",
       "language": "en"
   }
   ```

2. Adjust the settings:
   - `model_path`: Path to your YOLO model file (.pt).
   - `images_dir`: Directory containing images for annotation.
   - `train_dir`: Directory to save annotated images.
   - `language`: UI language code (`en`, `ru`, `fr`, `de`, `ja`, `zh`). Default is `en`.

### Usage

#### Running the Application

```bash
python labeler.py
```

#### Key Functions

##### Navigation
- **←/→**: Previous/Next image.
- **Z**: Random image.
- **Toolbar buttons**: Open folder, navigate, etc.

##### Working with Annotations
- **Right-click → "Add Box"**: Create a new bounding box.
- **Left-click on a box**: Select the box.
- **Drag a box**: Move it.
- **Drag box corners**: Resize it.
- **Delete**: Remove the selected box.
- **0-9**: Quickly change the class of the selected box.
- **Ctrl+←/→/↑/↓**: Resize the selected box.

##### Display Management
- **H**: Hide/Show annotations.
- **Space (hold)**: Temporarily hide annotations.
- **V**: Reset image view.
- **Mouse wheel**: Zoom.
- **Drag with left mouse button**: Pan the image.

##### Class Management
- **Menu "Show Classes"**: Open the class list.
- **Click on a class in the list**: Assign it to the selected box.
- **"Numbers/Names" button**: Toggle class name display.

##### Saving and Exporting
- **N**: Save the image and annotations to the train directory.
- **R**: Reset annotations (recreate using the model).

#### Hotkeys

| Key | Action |
|-----|--------|
| ←/→ | Previous/Next image |
| Z | Random image |
| N | Save to Train |
| R | Reset annotations |
| H | Hide/Show annotations |
| V | Reset view |
| Space | Temporarily hide annotations |
| Delete | Delete selected box |
| 0-9 | Quick class change |
| Ctrl+←/→/↑/↓ | Resize box |

#### Working with Lists

##### Image List
- Opened via the "Show/Hide List" menu.
- Click an image to jump to it.
- Background color:
  - White: Not processed.
  - Yellow: Has annotations.
  - Green: Saved to Train.

##### Class List
- Opened via the "Show Classes" menu.
- Click a class to assign it to the selected box.
- Background color matches the box color on the image.

### File Formats

#### Configuration File (`config.json`)
```json
{
    "model_path": "path/to/model.pt",
    "images_dir": "path/to/images",
    "train_dir": "TrainDataset",
    "language": "en"
}
```

#### Annotation Format (YOLO)
Each annotation is saved in a text file with the same name as the image:
```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

Where:
- `class_id`: Class identifier.
- `x_center`, `y_center`: Normalized center coordinates (0-1).
- `width`, `height`: Normalized size (0-1).
- `confidence`: Model confidence (for automatic annotations).

### Example Workflow

1. **Setup**: Specify the model and image paths in `config.json`.
2. **Run**: Launch the application - the specified folder will open automatically.
3. **Annotate**:
   - Review automatic annotations.
   - Correct errors: delete extra boxes, add missing ones.
   - Change classes of misidentified objects.
4. **Save**: Press "N" to save the image with annotations.
5. **Continue**: Move to the next image.

### Tips for Efficient Work

1. Use hotkeys to speed up your work.
2. For multiple objects of the same class, use keys 0-9 for quick class switching.
3. Hold Space to temporarily view the original image.
4. Use the image and class lists for quick access.
5. Regularly save results using the "N" key.

### Troubleshooting

#### Startup Issues
- Ensure all dependencies are installed.
- Check the paths in `config.json` are correct.
- Make sure the model file exists.

#### Display Issues
- If the image goes off-screen, use the "Reset View" button (V).
- If annotations are not displayed, check the "Hide Annotations" button (H).

#### Annotation Issues
- If classes are not displayed, check the model file.
- If automatic annotations are incorrect, try changing the model.

### License

MIT License - see the `LICENSE` file for details.

### Contributing

Pull requests and bug reports are welcome. For major changes, please open an issue first to discuss.

### Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO.
- [OpenCV](https://opencv.org/) for computer vision.
- [PyQt5](https://pypi.org/project/PyQt5/) for the GUI framework.
- [Qwen](https://chat.qwen.ai/) for significant assistance in developing this tool.

---

## <a name="Русский"></a> Русский

Инструмент для аннотирования изображений в формате YOLO с поддержкой автоматической разметки с помощью нейросетей.

### Возможности

- Автоматическая разметка изображений с помощью YOLO модели.
- Ручное редактирование аннотаций (добавление, удаление, изменение размера, перемещение).
- Поддержка множества классов с цветовой кодировкой.
- Навигация по изображениям и управление проектом.
- Экспорт аннотаций в формате YOLO.
- Горячие клавиши для ускорения работы.
- Сохранение состояния между сессиями.
- **Многоязычный интерфейс (Английский, Русский, Французский, Немецкий, Японский, Китайский). Язык можно изменить через меню 'Язык'. По умолчанию Английский.**

### Установка

#### Требования

- Python 3.7 или выше.
- Windows, Linux или macOS.

#### Шаги установки

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/V171/yolo-labeling-tool.git
   cd yolo-labeling-tool
   ```

2. Создайте виртуальное окружение:
   ```bash
   python -m venv venv
   ```

3. Активируйте виртуальное окружение:

   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   **Linux/macOS:**
   ```bash
   source venv/bin/activate
   ```

4. Установите зависимости:
   ```bash
   pip install ultralytics opencv-python pyqt5 numpy
   ```

5. Скачайте предварительно обученную модель YOLO (например, `yolov8n.pt`) с [официальных релизов Ultralytics](https://github.com/ultralytics/assets/releases) или используйте свою модель.

### Настройка

1. Откройте файл `config.json` в корневой директории проекта:
   ```json
   {
       "model_path": "путь/к/вашей/модели.pt",
       "images_dir": "путь/к/директории/с/изображениями",
       "train_dir": "TrainDataset",
       "language": "ru"
   }
   ```

2. Настройте параметры:
   - `model_path`: Путь к файлу вашей YOLO модели (.pt).
   - `images_dir`: Директория с изображениями для аннотирования.
   - `train_dir`: Директория для сохранения аннотированных изображений.
   - `language`: Код языка интерфейса (`en`, `ru`, `fr`, `de`, `ja`, `zh`). По умолчанию `ru`.

### Использование

#### Запуск приложения

```bash
python labeler.py
```

#### Основные функции

##### Навигация
- **←/→**: Предыдущее/следующее изображение.
- **Z**: Случайное изображение.
- **Кнопки на панели инструментов**: Открыть папку, навигация и т.д.

##### Работа с аннотациями
- **Правый клик → "Добавить бокс"**: Создать новый bounding box.
- **Левый клик по боксу**: Выбрать бокс.
- **Перетаскивание бокса**: Перемещение.
- **Перетаскивание углов бокса**: Изменение размера.
- **Delete**: Удалить выбранный бокс.
- **0-9**: Быстрая смена класса выбранного бокса.
- **Ctrl+←/→/↑/↓**: Изменение размера выбранного бокса.

##### Управление отображением
- **H**: Скрыть/показать аннотации.
- **Пробел (удерживание)**: Временно скрыть аннотации.
- **V**: Сброс вида изображения.
- **Колесо мыши**: Масштабирование.
- **Перетаскивание левой кнопкой мыши**: Перемещение изображения.

##### Работа с классами
- **Меню "Показать классы"**: Открыть список классов.
- **Клик по классу в списке**: Назначить класс выбранному боксу.
- **Кнопка "Номера классов/Имена классов"**: Переключение отображения названий.

##### Сохранение и экспорт
- **N**: Сохранить изображение и аннотации в train директорию.
- **R**: Сбросить аннотации (пересоздать с помощью модели).

#### Горячие клавиши

| Клавиша | Действие |
|---------|----------|
| ←/→ | Предыдущее/следующее изображение |
| Z | Случайное изображение |
| N | Сохранить в Train |
| R | Сбросить разметку |
| H | Скрыть/показать аннотации |
| V | Сброс вида |
| Пробел | Временно скрыть аннотации |
| Delete | Удалить выбранный бокс |
| 0-9 | Быстрая смена класса |
| Ctrl+←/→/↑/↓ | Изменение размера бокса |

#### Работа со списками

##### Список изображений
- Открывается через меню "Показать/скрыть список".
- Клик по изображению для перехода к нему.
- Цвет фона:
  - Белый: Не обработано.
  - Желтый: Есть аннотации.
  - Зеленый: Сохранено в Train.

##### Список классов
- Открывается через меню "Показать классы".
- Клик по классу для назначения его выбранному боксу.
- Цвет фона соответствует цвету бокса на изображении.

### Формат файлов

#### Конфигурационный файл (`config.json`)
```json
{
    "model_path": "путь/к/модели.pt",
    "images_dir": "путь/к/изображениям",
    "train_dir": "TrainDataset",
    "language": "ru"
}
```

#### Формат аннотаций (YOLO)
Каждая аннотация сохраняется в текстовом файле с тем же именем, что и изображение:
```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

Где:
- `class_id`: Идентификатор класса.
- `x_center`, `y_center`: Нормализованные координаты центра (0-1).
- `width`, `height`: Нормализованный размер (0-1).
- `confidence`: Уверенность модели (для автоматических аннотаций).

### Пример рабочего процесса

1. **Настройка**: Укажите путь к модели и изображениям в `config.json`.
2. **Запуск**: Запустите приложение - автоматически откроется указанная папка.
3. **Аннотирование**:
   - Просмотрите автоматические аннотации.
   - Исправьте ошибки: удалите лишние боксы, добавьте недостающие.
   - Измените классы неправильно распознанных объектов.
4. **Сохранение**: Нажмите "N" для сохранения изображения с аннотациями.
5. **Продолжение**: Перейдите к следующему изображению.

### Советы по эффективной работе

1. Используйте горячие клавиши для ускорения работы.
2. При множественных объектах одного класса используйте клавиши 0-9 для быстрой смены класса.
3. Удерживайте пробел для временного просмотра оригинального изображения.
4. Используйте списки изображений и классов для быстрого доступа.
5. Регулярно сохраняйте результаты клавишей "N".

### Устранение неполадок

#### Проблемы с запуском
- Убедитесь, что все зависимости установлены.
- Проверьте правильность путей в `config.json`.
- Убедитесь, что файл модели существует.

#### Проблемы с отображением
- Если изображение "ушло" за границы экрана, используйте кнопку "Сброс вида" (V).
- Если аннотации не отображаются, проверьте кнопку "Скрыть разметку" (H).

#### Проблемы с аннотациями
- Если классы не отображаются, проверьте файл модели.
- Если автоматические аннотации некорректны, попробуйте изменить модель.

### Лицензия

MIT License - см. файл `LICENSE` для подробностей.

### Вклад в развитие

Приветствуются pull requests и сообщения об ошибках. Для крупных изменений сначала создайте issue для обсуждения.

### Благодарности

- [Ultralytics](https://github.com/ultralytics/ultralytics) за YOLO.
- [OpenCV](https://opencv.org/) за компьютерное зрение.
- [PyQt5](https://pypi.org/project/PyQt5/) за GUI фреймворк.
- [Qwen](https://chat.qwen.ai/) за значительную помощь в разработке этого инструмента.

---

## <a name="français"></a> Français

Un outil pour annoter des images au format YOLO avec prise en charge de l'étiquetage automatique à l'aide de réseaux de neurones.

### Fonctionnalités

- Étiquetage automatique des images à l'aide d'un modèle YOLO.
- Édition manuelle des annotations (ajouter, supprimer, redimensionner, déplacer).
- Prise en charge de plusieurs classes avec codage couleur.
- Navigation dans les images et gestion de projet.
- Exportation des annotations au format YOLO.
- Raccourcis clavier pour un fonctionnement plus rapide.
- Sauvegarde de l'état entre les sessions.
- **Interface utilisateur multilingue (anglais, russe, français, allemand, japonais, chinois). La langue peut être modifiée via le menu « Langue ». L'anglais est la valeur par défaut.**

### Installation

#### Exigences

- Python 3.7 ou supérieur
- Windows, Linux ou macOS

#### Étapes d'installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/V171/yolo-labeling-tool.git
   cd yolo-labeling-tool
   ```

2. Créer un environnement virtuel :
   ```bash
   python -m venv venv
   ```

3. Activer l'environnement virtuel :

   **Windows :**
   ```bash
   venv\Scripts\activate
   ```
   **Linux/macOS :**
   ```bash
   source venv/bin/activate
   ```

4. Installer les dépendances :
   ```bash
   pip install ultralytics opencv-python pyqt5 numpy
   ```

5. Télécharger un modèle YOLO pré-entraîné (par exemple, `yolov8n.pt`) à partir des [versions officielles d'Ultralytics](https://github.com/ultralytics/assets/releases) ou utiliser votre propre modèle.

### Configuration

1. Ouvrir le fichier `config.json` dans le répertoire racine du projet :
   ```json
   {
       "model_path": "chemin/vers/votre/modèle.pt",
       "images_dir": "chemin/vers/vos/images",
       "train_dir": "TrainDataset",
       "language": "fr"
   }
   ```

2. Ajuster les paramètres :
   - `model_path` : Chemin vers votre fichier de modèle YOLO (.pt).
   - `images_dir` : Répertoire contenant les images à annoter.
   - `train_dir` : Répertoire pour enregistrer les images annotées.
   - `language` : Code de langue de l'interface utilisateur (`en`, `ru`, `fr`, `de`, `ja`, `zh`). La valeur par défaut est `fr`.

### Utilisation

#### Exécution de l'application

```bash
python labeler.py
```

#### Fonctions principales

##### Navigation
- **←/→** : Image précédente/suivante.
- **Z** : Image aléatoire.
- **Boutons de la barre d'outils** : Ouvrir le dossier, naviguer, etc.

##### Travailler avec les annotations
- **Clic droit → "Ajouter une boîte"** : Créer une nouvelle zone de délimitation.
- **Clic gauche sur une boîte** : Sélectionner la boîte.
- **Faire glisser une boîte** : La déplacer.
- **Faire glisser les coins de la boîte** : La redimensionner.
- **Supprimer** : Supprimer la boîte sélectionnée.
- **0-9** : Changer rapidement la classe de la boîte sélectionnée.
- **Ctrl+←/→/↑/↓** : Redimensionner la boîte sélectionnée.

##### Gestion de l'affichage
- **H** : Masquer/Afficher les annotations.
- **Espace (maintenir)** : Masquer temporairement les annotations.
- **V** : Réinitialiser la vue de l'image.
- **Molette de la souris** : Zoom.
- **Glisser avec le bouton gauche de la souris** : Déplacer l'image.

##### Gestion des classes
- **Menu "Afficher les classes"** : Ouvrir la liste des classes.
- **Cliquer sur une classe dans la liste** : L'attribuer à la boîte sélectionnée.
- **Bouton "Nombres/Noms"** : Basculer l'affichage des noms de classe.

##### Sauvegarde et exportation
- **N** : Enregistrer l'image et les annotations dans le répertoire d'entraînement.
- **R** : Réinitialiser les annotations (recréer à l'aide du modèle).

#### Raccourcis clavier

| Touche | Action |
|--------|--------|
| ←/→ | Image précédente/suivante |
| Z | Image aléatoire |
| N | Enregistrer dans Train |
| R | Réinitialiser les annotations |
| H | Masquer/Afficher les annotations |
| V | Réinitialiser la vue |
| Espace | Masquer temporairement les annotations |
| Supprimer | Supprimer la boîte sélectionnée |
| 0-9 | Changement rapide de classe |
| Ctrl+←/→/↑/↓ | Redimensionner la boîte |

#### Travailler avec les listes

##### Liste des images
- Ouvert via le menu "Afficher/Masquer la liste".
- Cliquer sur une image pour y accéder.
- Couleur d'arrière-plan :
  - Blanc : Non traité.
  - Jaune : Possède des annotations.
  - Vert : Enregistré dans Train.

##### Liste des classes
- Ouvert via le menu "Afficher les classes".
- Cliquer sur une classe pour l'attribuer à la boîte sélectionnée.
- La couleur d'arrière-plan correspond à la couleur de la boîte sur l'image.

### Formats de fichiers

#### Fichier de configuration (`config.json`)
```json
{
    "model_path": "chemin/vers/le/modèle.pt",
    "images_dir": "chemin/vers/les/images",
    "train_dir": "TrainDataset",
    "language": "fr"
}
```

#### Format d'annotation (YOLO)
Chaque annotation est enregistrée dans un fichier texte portant le même nom que l'image :
```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

Où :
- `class_id` : Identifiant de classe.
- `x_center`, `y_center` : Coordonnées du centre normalisées (0-1).
- `width`, `height` : Taille normalisée (0-1).
- `confidence` : Confiance du modèle (pour les annotations automatiques).

### Exemple de flux de travail

1. **Configuration** : Spécifier les chemins du modèle et des images dans `config.json`.
2. **Exécution** : Lancer l'application - le dossier spécifié s'ouvrira automatiquement.
3. **Annotation** :
   - Passer en revue les annotations automatiques.
   - Corriger les erreurs : supprimer les boîtes supplémentaires, en ajouter celles qui manquent.
   - Modifier les classes des objets mal identifiés.
4. **Sauvegarde** : Appuyer sur "N" pour enregistrer l'image avec les annotations.
5. **Continuer** : Passer à l'image suivante.

### Conseils pour un travail efficace

1. Utiliser les raccourcis clavier pour accélérer le travail.
2. Pour plusieurs objets de la même classe, utiliser les touches 0-9 pour un changement rapide de classe.
3. Maintenir la barre d'espace pour afficher temporairement l'image originale.
4. Utiliser les listes d'images et de classes pour un accès rapide.
5. Enregistrer régulièrement les résultats à l'aide de la touche "N".

### Dépannage

#### Problèmes de démarrage
- S'assurer que toutes les dépendances sont installées.
- Vérifier que les chemins dans `config.json` sont corrects.
- S'assurer que le fichier modèle existe.

#### Problèmes d'affichage
- Si l'image sort de l'écran, utiliser le bouton "Réinitialiser la vue" (V).
- Si les annotations ne s'affichent pas, vérifier le bouton "Masquer les annotations" (H).

#### Problèmes d'annotation
- Si les classes ne s'affichent pas, vérifier le fichier modèle.
- Si les annotations automatiques sont incorrectes, essayer de changer le modèle.

### Licence

Licence MIT - voir le fichier `LICENSE` pour plus de détails.

### Contribution

Les demandes d'extraction et les rapports de bogues sont les bienvenus. Pour les changements majeurs, veuillez d'abord ouvrir une issue pour en discuter.

### Remerciements

- [Ultralytics](https://github.com/ultralytics/ultralytics) pour YOLO.
- [OpenCV](https://opencv.org/) pour la vision par ordinateur.
- [PyQt5](https://pypi.org/project/PyQt5/) pour le framework GUI.
- [Qwen](https://chat.qwen.ai/) pour l'aide significative apportée au développement de cet outil.

---

## <a name="deutsch"></a> Deutsch

Ein Tool zur Annotation von Bildern im YOLO-Format mit Unterstützung für die automatische Kennzeichnung mithilfe neuronaler Netze.

### Funktionen

- Automatische Bildkennzeichnung mit einem YOLO-Modell.
- Manuelles Bearbeiten von Annotationen (Hinzufügen, Löschen, Größenänderung, Verschieben).
- Unterstützung für mehrere Klassen mit Farbcodierung.
- Bildnavigation und Projektmanagement.
- Export von Annotationen im YOLO-Format.
- Tastenkürzel für einen schnelleren Betrieb.
- Zustandsspeicherung zwischen Sitzungen.
- **Mehrsprachige Benutzeroberfläche (Englisch, Russisch, Französisch, Deutsch, Japanisch, Chinesisch). Die Sprache kann über das Menü „Sprache“ geändert werden. Englisch ist die Standardeinstellung.**

### Installation

#### Anforderungen

- Python 3.7 oder höher
- Windows, Linux oder macOS

#### Installationsschritte

1. Repository klonen:
   ```bash
   git clone https://github.com/V171/yolo-labeling-tool.git
   cd yolo-labeling-tool
   ```

2. Virtuelle Umgebung erstellen:
   ```bash
   python -m venv venv
   ```

3. Virtuelle Umgebung aktivieren:

   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   **Linux/macOS:**
   ```bash
   source venv/bin/activate
   ```

4. Abhängigkeiten installieren:
   ```bash
   pip install ultralytics opencv-python pyqt5 numpy
   ```

5. Laden Sie ein vortrainiertes YOLO-Modell (z. B. `yolov8n.pt`) von den [offiziellen Veröffentlichungen von Ultralytics](https://github.com/ultralytics/assets/releases) herunter oder verwenden Sie Ihr eigenes Modell.

### Konfiguration

1. Öffnen Sie die Datei `config.json` im Stammverzeichnis des Projekts:
   ```json
   {
       "model_path": "pfad/zu/ihrer/modell.pt",
       "images_dir": "pfad/zu/ihren/bildern",
       "train_dir": "TrainDataset",
       "language": "de"
   }
   ```

2. Passen Sie die Einstellungen an:
   - `model_path`: Pfad zu Ihrer YOLO-Modelldatei (.pt).
   - `images_dir`: Verzeichnis mit Bildern für die Annotation.
   - `train_dir`: Verzeichnis zum Speichern annotierter Bilder.
   - `language`: Sprachcode der Benutzeroberfläche (`en`, `ru`, `fr`, `de`, `ja`, `zh`). Standard ist `de`.

### Verwendung

#### Ausführen der Anwendung

```bash
python labeler.py
```

#### Hauptfunktionen

##### Navigation
- **←/→**: Vorheriges/Nächstes Bild.
- **Z**: Zufälliges Bild.
- **Symbolleisten-Schaltflächen**: Ordner öffnen, navigieren usw.

##### Arbeiten mit Annotationen
- **Rechtsklick → „Box hinzufügen“**: Erstellen Sie ein neues Begrenzungsrahmen.
- **Linksklick auf eine Box**: Wählen Sie die Box aus.
- **Ziehen einer Box**: Verschieben Sie sie.
- **Ziehen an den Ecken der Box**: Ändern Sie die Größe.
- **Entf**: Entfernen Sie die ausgewählte Box.
- **0-9**: Ändern Sie schnell die Klasse der ausgewählten Box.
- **Strg+←/→/↑/↓**: Ändern Sie die Größe der ausgewählten Box.

##### Anzeigeverwaltung
- **H**: Annotationen ausblenden/einblenden.
- **Leertaste (halten)**: Annotationen vorübergehend ausblenden.
- **V**: Bildansicht zurücksetzen.
- **Mausrad**: Zoomen.
- **Mit der linken Maustaste ziehen**: Bild verschieben.

##### Klassenverwaltung
- **Menü „Klassen anzeigen“**: Öffnen Sie die Klassenliste.
- **Klicken Sie auf eine Klasse in der Liste**: Weisen Sie sie der ausgewählten Box zu.
- **Schaltfläche „Zahlen/Namen“**: Umschalten der Klassennamenanzeige.

##### Speichern und Exportieren
- **N**: Speichern Sie das Bild und die Annotationen im Trainingsverzeichnis.
- **R**: Annotationen zurücksetzen (mit dem Modell neu erstellen).

#### Tastenkürzel

| Taste | Aktion |
|-------|--------|
| ←/→ | Vorheriges/Nächstes Bild |
| Z | Zufälliges Bild |
| N | In Train speichern |
| R | Annotationen zurücksetzen |
| H | Annotationen ausblenden/einblenden |
| V | Ansicht zurücksetzen |
| Leertaste | Annotationen vorübergehend ausblenden |
| Entf | Ausgewählte Box löschen |
| 0-9 | Schneller Klassenwechsel |
| Strg+←/→/↑/↓ | Boxgröße ändern |

#### Arbeiten mit Listen

##### Bildliste
- Geöffnet über das Menü „Liste ein-/ausblenden“.
- Klicken Sie auf ein Bild, um dorthin zu springen.
- Hintergrundfarbe:
  - Weiß: Nicht verarbeitet.
  - Gelb: Hat Annotationen.
  - Grün: In Train gespeichert.

##### Klassenliste
- Geöffnet über das Menü „Klassen anzeigen“.
- Klicken Sie auf eine Klasse, um sie der ausgewählten Box zuzuweisen.
- Die Hintergrundfarbe entspricht der Farbe der Box im Bild.

### Dateiformate

#### Konfigurationsdatei (`config.json`)
```json
{
    "model_path": "pfad/zum/modell.pt",
    "images_dir": "pfad/zu/den/bildern",
    "train_dir": "TrainDataset",
    "language": "de"
}
```

#### Annotationsformat (YOLO)
Jede Annotation wird in einer Textdatei mit demselben Namen wie das Bild gespeichert:
```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

Dabei gilt:
- `class_id`: Klassenbezeichner.
- `x_center`, `y_center`: Normalisierte Mittelpunktkoordinaten (0-1).
- `width`, `height`: Normalisierte Größe (0-1).
- `confidence`: Modellvertrauen (für automatische Annotationen).

### Beispielworkflow

1. **Einrichtung**: Geben Sie die Modell- und Bildpfade in `config.json` an.
2. **Ausführen**: Starten Sie die Anwendung - der angegebene Ordner wird automatisch geöffnet.
3. **Annotation**:
   - Überprüfen Sie die automatischen Annotationen.
   - Korrigieren Sie Fehler: Löschen Sie zusätzliche Boxen, fügen Sie fehlende hinzu.
   - Ändern Sie die Klassen von falsch identifizierten Objekten.
4. **Speichern**: Drücken Sie „N“, um das Bild mit den Annotationen zu speichern.
5. **Weiter**: Gehen Sie zum nächsten Bild.

### Tipps für effizientes Arbeiten

1. Verwenden Sie Tastenkürzel, um Ihre Arbeit zu beschleunigen.
2. Verwenden Sie bei mehreren Objekten derselben Klasse die Tasten 0-9 für einen schnellen Klassenwechsel.
3. Halten Sie die Leertaste gedrückt, um das Originalbild vorübergehend anzuzeigen.
4. Verwenden Sie die Bild- und Klassenlisten für schnellen Zugriff.
5. Speichern Sie die Ergebnisse regelmäßig mit der Taste „N“.

### Fehlerbehebung

#### Startprobleme
- Stellen Sie sicher, dass alle Abhängigkeiten installiert sind.
- Überprüfen Sie, ob die Pfade in `config.json` korrekt sind.
- Stellen Sie sicher, dass die Modelldatei vorhanden ist.

#### Anzeigeprobleme
- Wenn das Bild den Bildschirm verlässt, verwenden Sie die Schaltfläche „Ansicht zurücksetzen“ (V).
- Wenn Annotationen nicht angezeigt werden, überprüfen Sie die Schaltfläche „Annotationen ausblenden“ (H).

#### Annotationsprobleme
- Wenn Klassen nicht angezeigt werden, überprüfen Sie die Modelldatei.
- Wenn automatische Annotationen falsch sind, versuchen Sie, das Modell zu ändern.

### Lizenz

MIT-Lizenz - siehe die Datei `LICENSE` für Details.

### Mitwirken

Pull-Requests und Fehlerberichte sind willkommen. Öffnen Sie für größere Änderungen zunächst ein Issue zur Diskussion.

### Danksagungen

- [Ultralytics](https://github.com/ultralytics/ultralytics) für YOLO.
- [OpenCV](https://opencv.org/) für Computer Vision.
- [PyQt5](https://pypi.org/project/PyQt5/) für das GUI-Framework.
- [Qwen](https://chat.qwen.ai/) für die wesentliche Unterstützung bei der Entwicklung dieses Tools.

---

## <a name="日本語"></a> 日本語

ニューラルネットワークを使用した自動ラベリングをサポートする、YOLO形式の画像アノテーションツールです。

### 特徴

- YOLOモデルを使用した自動画像ラベリング。
- 手動によるアノテーション編集（追加、削除、サイズ変更、移動）。
- 複数のクラスを色分けしてサポート。
- 画像ナビゲーションとプロジェクト管理。
- YOLO形式でのアノテーションのエクスポート。
- より速い操作のためのホットキー。
- セッション間の状態保存。
- **多言語UI（英語、ロシア語、フランス語、ドイツ語、日本語、中国語）。言語は「言語」メニューから変更できます。デフォルトは英語です。**

### インストール

#### 要件

- Python 3.7 以上
- Windows、Linux、または macOS

#### インストール手順

1. リポジトリをクローンします：
   ```bash
   git clone https://github.com/V171/yolo-labeling-tool.git
   cd yolo-labeling-tool
   ```

2. 仮想環境を作成します：
   ```bash
   python -m venv venv
   ```

3. 仮想環境をアクティブにします：

   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   **Linux/macOS:**
   ```bash
   source venv/bin/activate
   ```

4. 依存関係をインストールします：
   ```bash
   pip install ultralytics opencv-python pyqt5 numpy
   ```

5. 事前学習済みのYOLOモデル（例：`yolov8n.pt`）を[Ultralyticsの公式リリース](https://github.com/ultralytics/assets/releases)からダウンロードするか、独自のモデルを使用します。

### 設定

1. プロジェクトのルートディレクトリにある `config.json` ファイルを開きます：
   ```json
   {
       "model_path": "あなたのモデルへのパス.pt",
       "images_dir": "あなたの画像へのパス",
       "train_dir": "TrainDataset",
       "language": "ja"
   }
   ```

2. 設定を調整します：
   - `model_path`：YOLOモデルファイル（.pt）へのパス。
   - `images_dir`：アノテーションする画像を含むディレクトリ。
   - `train_dir`：アノテーションされた画像を保存するディレクトリ。
   - `language`：UI言語コード（`en`, `ru`, `fr`, `de`, `ja`, `zh`）。デフォルトは `ja` です。

### 使用方法

#### アプリケーションの実行

```bash
python labeler.py
```

#### 主な機能

##### ナビゲーション
- **←/→**：前の画像/次の画像。
- **Z**：ランダム画像。
- **ツールバーのボタン**：フォルダを開く、ナビゲートなど。

##### アノテーションの操作
- **右クリック → 「ボックスを追加」**：新しいバウンディングボックスを作成します。
- **ボックスを左クリック**：ボックスを選択します。
- **ボックスをドラッグ**：移動します。
- **ボックスの角をドラッグ**：サイズを変更します。
- **Delete**：選択したボックスを削除します。
- **0-9**：選択したボックスのクラスを素早く変更します。
- **Ctrl+←/→/↑/↓**：選択したボックスのサイズを変更します。

##### 表示管理
- **H**：アノテーションを非表示/表示。
- **スペースキー（押し続け）**：アノテーションを一時的に非表示にします。
- **V**：画像ビューをリセットします。
- **マウスホイール**：ズーム。
- **左マウスボタンでドラッグ**：画像をパンします。

##### クラス管理
- **メニュー「クラスを表示」**：クラスリストを開きます。
- **リスト内のクラスをクリック**：選択したボックスに割り当てます。
- **「番号/名前」ボタン**：クラス名表示を切り替えます。

##### 保存とエクスポート
- **N**：画像とアノテーションをトレインディレクトリに保存します。
- **R**：アノテーションをリセットします（モデルを使用して再作成）。

#### ホットキー

| キー | アクション |
|------|----------|
| ←/→ | 前の画像/次の画像 |
| Z | ランダム画像 |
| N | Trainに保存 |
| R | アノテーションをリセット |
| H | アノテーションを非表示/表示 |
| V | ビューをリセット |
| スペース | アノテーションを一時的に非表示 |
| Delete | 選択したボックスを削除 |
| 0-9 | クイッククラス変更 |
| Ctrl+←/→/↑/↓ | ボックスのサイズ変更 |

#### リストの操作

##### 画像リスト
- 「リストを表示/非表示」メニューから開きます。
- 画像をクリックしてジャンプします。
- 背景色：
  - 白：未処理。
  - 黄色：アノテーションあり。
  - 緑：Trainに保存済み。

##### クラスリスト
- 「クラスを表示」メニューから開きます。
- クラスをクリックして、選択したボックスに割り当てます。
- 背景色は画像上のボックスの色と一致します。

### ファイル形式

#### 設定ファイル (`config.json`)
```json
{
    "model_path": "モデルへのパス.pt",
    "images_dir": "画像へのパス",
    "train_dir": "TrainDataset",
    "language": "ja"
}
```

#### アノテーション形式 (YOLO)
各アノテーションは、画像と同じ名前のテキストファイルに保存されます：
```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

ここで：
- `class_id`：クラス識別子。
- `x_center`, `y_center`：正規化された中心座標（0-1）。
- `width`, `height`：正規化されたサイズ（0-1）。
- `confidence`：モデルの信頼度（自動アノテーションの場合）。

### ワークフロー例

1. **セットアップ**：`config.json`でモデルと画像のパスを指定します。
2. **実行**：アプリケーションを起動します - 指定されたフォルダが自動的に開きます。
3. **アノテーション**：
   - 自動アノテーションを確認します。
   - エラーを修正します：余分なボックスを削除し、不足しているボックスを追加します。
   - 誤って識別されたオブジェクトのクラスを変更します。
4. **保存**：「N」を押して、アノテーション付きの画像を保存します。
5. **続行**：次の画像に移動します。

### 効率的な作業のためのヒント

1. ホットキーを使用して作業をスピードアップします。
2. 同じクラスの複数のオブジェクトの場合、キー0-9を使用してクラスを素早く切り替えます。
3. スペースキーを押し続けると、元の画像を一時的に表示できます。
4. 画像リストとクラスリストを使用して素早くアクセスします。
5. 「N」キーを使用して結果を定期的に保存します。

### トラブルシューティング

#### 起動時の問題
- すべての依存関係がインストールされていることを確認します。
- `config.json`のパスが正しいことを確認します。
- モデルファイルが存在することを確認します。

#### 表示の問題
- 画像が画面外に出た場合は、「ビューをリセット」ボタン（V）を使用します。
- アノテーションが表示されない場合は、「アノテーションを非表示」ボタン（H）を確認します。

#### アノテーションの問題
- クラスが表示されない場合は、モデルファイルを確認します。
- 自動アノテーションが正しくない場合は、モデルを変更してみてください。

### ライセンス

MITライセンス - 詳細については `LICENSE` ファイルを参照してください。

### 貢献

プルリクエストとバグレポートを歓迎します。大きな変更については、最初に議論するためにIssueを開いてください。

### 謝辞

- YOLOの[Ultralytics](https://github.com/ultralytics/ultralytics)。
- コンピュータビジョンの[OpenCV](https://opencv.org/)。
- GUIフレームワークの[PyQt5](https://pypi.org/project/PyQt5/)。
- このツールの開発に多大な支援をいただいた[Qwen](https://chat.qwen.ai/)。

---

## <a name="中文"></a> 中文

一个用于以 YOLO 格式标注图像的工具，支持使用神经网络进行自动标注。

### 功能

- 使用 YOLO 模型自动标注图像。
- 手动编辑标注（添加、删除、调整大小、移动）。
- 支持多种颜色编码的类别。
- 图像导航和项目管理。
- 以 YOLO 格式导出标注。
- 快捷键加快操作。
- 会话间保存状态。
- **多语言用户界面（英语、俄语、法语、德语、日语、中文）。可以通过“语言”菜单更改语言。默认为英语。**

### 安装

#### 要求

- Python 3.7 或更高版本
- Windows、Linux 或 macOS

#### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/V171/yolo-labeling-tool.git
   cd yolo-labeling-tool
   ```

2. 创建虚拟环境：
   ```bash
   python -m venv venv
   ```

3. 激活虚拟环境：

   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   **Linux/macOS:**
   ```bash
   source venv/bin/activate
   ```

4. 安装依赖项：
   ```bash
   pip install ultralytics opencv-python pyqt5 numpy
   ```

5. 从 [Ultralytics 官方发布页面](https://github.com/ultralytics/assets/releases) 下载预训练的 YOLO 模型（例如 `yolov8n.pt`），或使用您自己的模型。

### 配置

1. 打开项目根目录下的 `config.json` 文件：
   ```json
   {
       "model_path": "你的模型路径.pt",
       "images_dir": "你的图像目录",
       "train_dir": "TrainDataset",
       "language": "zh"
   }
   ```

2. 调整设置：
   - `model_path`：YOLO 模型文件 (.pt) 的路径。
   - `images_dir`：包含待标注图像的目录。
   - `train_dir`：保存标注图像的目录。
   - `language`：用户界面语言代码 (`en`, `ru`, `fr`, `de`, `ja`, `zh`)。默认为 `zh`。

### 使用方法

#### 运行应用程序

```bash
python labeler.py
```

#### 主要功能

##### 导航
- **←/→**：上一张/下一张图像。
- **Z**：随机图像。
- **工具栏按钮**：打开文件夹、导航等。

##### 处理标注
- **右键单击 → “添加框”**：创建新的边界框。
- **左键单击框**：选择该框。
- **拖动框**：移动它。
- **拖动框角**：调整大小。
- **Delete**：删除所选框。
- **0-9**：快速更改所选框的类别。
- **Ctrl+←/→/↑/↓**：调整所选框的大小。

##### 显示管理
- **H**：隐藏/显示标注。
- **空格键（按住）**：临时隐藏标注。
- **V**：重置图像视图。
- **鼠标滚轮**：缩放。
- **按住鼠标左键拖动**：平移图像。

##### 类别管理
- **菜单“显示类别”**：打开类别列表。
- **单击列表中的类别**：将其分配给所选框。
- **“编号/名称”按钮**：切换类别名称显示。

##### 保存和导出
- **N**：将图像和标注保存到训练目录。
- **R**：重置标注（使用模型重新创建）。

#### 快捷键

| 按键 | 操作 |
|------|------|
| ←/→ | 上一张/下一张图像 |
| Z | 随机图像 |
| N | 保存到训练集 |
| R | 重置标注 |
| H | 隐藏/显示标注 |
| V | 重置视图 |
| 空格 | 临时隐藏标注 |
| Delete | 删除所选框 |
| 0-9 | 快速类别切换 |
| Ctrl+←/→/↑/↓ | 调整框大小 |

#### 使用列表

##### 图像列表
- 通过“显示/隐藏列表”菜单打开。
- 单击图像可跳转到该图像。
- 背景颜色：
  - 白色：未处理。
  - 黄色：有标注。
  - 绿色：已保存到训练集。

##### 类别列表
- 通过“显示类别”菜单打开。
- 单击类别可将其分配给所选框。
- 背景颜色与图像上框的颜色匹配。

### 文件格式

#### 配置文件 (`config.json`)
```json
{
    "model_path": "模型路径.pt",
    "images_dir": "图像路径",
    "train_dir": "TrainDataset",
    "language": "zh"
}
```

#### 标注格式 (YOLO)
每个标注都保存在一个与图像同名的文本文件中：
```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

其中：
- `class_id`：类别标识符。
- `x_center`, `y_center`：归一化的中心坐标（0-1）。
- `width`, `height`：归一化的尺寸（0-1）。
- `confidence`：模型置信度（用于自动标注）。

### 示例工作流程

1. **设置**：在 `config.json` 中指定模型和图像路径。
2. **运行**：启动应用程序 - 指定的文件夹将自动打开。
3. **标注**：
   - 查看自动标注。
   - 纠正错误：删除多余的框，添加缺失的框。
   - 更改错误识别对象的类别。
4. **保存**：按 "N" 保存带有标注的图像。
5. **继续**：移动到下一张图像。

### 高效工作的技巧

1. 使用快捷键加快工作速度。
2. 对于同一类别的多个对象，使用 0-9 键快速切换类别。
3. 按住空格键临时查看原始图像。
4. 使用图像和类别列表快速访问。
5. 定期使用 "N" 键保存结果。

### 故障排除

#### 启动问题
- 确保所有依赖项都已安装。
- 检查 `config.json` 中的路径是否正确。
- 确保模型文件存在。

#### 显示问题
- 如果图像移出屏幕，请使用“重置视图”按钮 (V)。
- 如果未显示标注，请检查“隐藏标注”按钮 (H)。

#### 标注问题
- 如果未显示类别，请检查模型文件。
- 如果自动标注不正确，请尝试更换模型。

### 许可证

MIT 许可证 - 有关详细信息，请参见 `LICENSE` 文件。

### 贡献

欢迎提交拉取请求和错误报告。对于重大更改，请先打开一个议题进行讨论。

### 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) 提供 YOLO。
- [OpenCV](https://opencv.org/) 提供计算机视觉。
- [PyQt5](https://pypi.org/project/PyQt5/) 提供 GUI 框架。
- 感谢 [Qwen](https://chat.qwen.ai/) 在开发此工具时提供的重大帮助。
