# Commandes utiles
$PY = "C:\Users\nikko\Documents\GitHub\surgical-tool-tracking\venv\Scripts\python.exe"

gif:
ffmpeg -ss 18 -t 3 -i "outputs/tracking/v3_test1.mp4" `
  -vf "fps=10,scale=640:-1:flags=lanczos" `
  "outputs/tracking/v3_test1_18s_21s.gif"

Guide pratique pour utiliser le projet depuis PowerShell.

Tous les exemples utilisent l'interpreteur du venv :

```powershell
$PY = "venv\Scripts\python.exe"
```

Le projet principal suit ce flux :

```text
video -> YOLO -> ByteTrack -> video annotee + CSV de predictions
```

## 0. Preparation

Depuis la racine du projet :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
$PY = "venv\Scripts\python.exe"
```

Verifier que le modele existe :

```powershell
Test-Path "models\best.pt"
```

Voir toutes les options du pipeline :

```powershell
& $PY "main.py" --help
```

## 1. Inspecter une video

Exemple avec `video_originale2.mp4` :

```powershell
& $PY -c "import cv2; p='data/samples/video_originale2.mp4'; c=cv2.VideoCapture(p); print('readable=',c.isOpened()); print('frames=',int(c.get(cv2.CAP_PROP_FRAME_COUNT))); print('fps=',c.get(cv2.CAP_PROP_FPS)); print('size=',int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),'x',int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))); c.release()"
```

La duree est :

```text
duree = nombre_de_frames / FPS
```

## 2. Decouper une video

Exemple : decouper `video_originale2.mp4` de `04:40` a `04:50`.

`4:40 = 280 secondes` et `4:50 = 290 secondes`.

```powershell
New-Item -ItemType Directory -Force "outputs\clips" | Out-Null

& $PY "experiments\tools\cut_video.py" `
  --input "data\samples\video_originale2.mp4" `
  --output "outputs\clips\video_originale2_04m40_04m50.mp4" `
  --start 280 `
  --end 290
```

Verification du morceau :

```powershell
& $PY -c "import cv2; p='outputs/clips/video_originale2_04m40_04m50.mp4'; c=cv2.VideoCapture(p); print('readable=',c.isOpened()); print('frames=',int(c.get(cv2.CAP_PROP_FRAME_COUNT))); print('fps=',c.get(cv2.CAP_PROP_FPS)); print('duration=',int(c.get(cv2.CAP_PROP_FRAME_COUNT))/c.get(cv2.CAP_PROP_FPS)); c.release()"
```

Le morceau doit durer environ `10 secondes`.

## 3. Lancer YOLO + ByteTrack

```powershell
New-Item -ItemType Directory -Force "outputs\tracking" | Out-Null

& $PY "main.py" `
  --model "models\best.pt" `
  --source "outputs\clips\video_originale2_04m40_04m50.mp4" `
  --output "outputs\tracking\video_originale2_04m40_04m50_tracked.mp4" `
  --predictions "outputs\tracking\video_originale2_04m40_04m50_predictions.csv" `
  --confidence 0.25 `
  --imgsz 640 `
  --device cpu
```

Avec le premier GPU CUDA :

```powershell
& $PY "main.py" `
  --model "models\best.pt" `
  --source "outputs\clips\video_originale2_04m40_04m50.mp4" `
  --output "outputs\tracking\video_originale2_04m40_04m50_tracked.mp4" `
  --predictions "outputs\tracking\video_originale2_04m40_04m50_predictions.csv" `
  --confidence 0.25 `
  --imgsz 640 `
  --device 0
```

Pour tester une configuration ByteTrack experimentale :

```powershell
& $PY "main.py" `
  --model "models\best.pt" `
  --source "outputs\clips\video_originale2_04m40_04m50.mp4" `
  --output "outputs\tracking\video_originale2_04m40_04m50_bytetrack_exp.mp4" `
  --predictions "outputs\tracking\video_originale2_04m40_04m50_bytetrack_exp.csv" `
  --confidence 0.15 `
  --imgsz 640 `
  --device cpu `
  --tracker "experiments\configs\bytetrack_video41_experiment.yaml"
```

Pour une source qui est un dossier d'images :

```powershell
& $PY "main.py" `
  --model "models\best.pt" `
  --source "data\samples\vieos41-45\Images\video_41" `
  --output "outputs\tracking\video41_tracked.mp4" `
  --predictions "outputs\tracking\video41_predictions.csv" `
  --source-fps 10 `
  --device cpu
```

## 4. Mesurer le comportement runtime

Cette mesure ne donne pas une precision de tracking. Elle mesure le comportement
sur une video sans labels temporels : couverture, IDs observes, latence et FPS.

```powershell
New-Item -ItemType Directory -Force "outputs\reports\test_video2" | Out-Null

& $PY "experiments\tools\evaluate_tracking.py" `
  --model "models\best.pt" `
  --source "data\test_video2.mp4" `
  --report-dir "outputs\reports\test_video2" `
  --imgsz 640 `
  --device cpu
```

Fichiers generes :

```text
outputs/reports/test_video2/summary.md
outputs/reports/test_video2/summary.json
outputs/reports/test_video2/frame_metrics.csv
outputs/reports/test_video2/track_metrics.csv
```

Les metriques utiles a presenter sont :

- FPS source ;
- FPS d'inference ;
- latence p50 et p95 ;
- couverture de detection ;
- nombre moyen de detections par frame ;
- nombre d'IDs observes ;
- longueur du plus long track ;
- gaps de reapparition ;
- classes detectees.

Ne pas appeler ces valeurs precision ou recall de tracking sans ground truth temporelle.

## 5. Comparer avec des targets annotees

A utiliser uniquement avec un CSV target et un CSV prediction alignes sur les
memes frames. Les targets videos 41-45 sont des references de projet reconstruites,
pas une ground truth officielle.

```powershell
& $PY "experiments\tools\evaluate_tracking_reference.py" `
  --gt-dir "outputs\targets" `
  --pred-dir "outputs\predictions" `
  --output "outputs\reference_metrics.csv" `
  --iou-threshold 0.5
```

## 6. Creer une video demo pour GitHub

La video trackee complete est deja produite par `main.py`. Pour creer un GIF de
10 secondes, 10 FPS, largeur 640 pixels :

```powershell
New-Item -ItemType Directory -Force "experiments\docs\assets" | Out-Null

& ffmpeg -y `
  -ss 0 `
  -t 10 `
  -i "outputs\tracking\video_originale2_04m40_04m50_tracked.mp4" `
  -vf "fps=10,scale=640:-1:flags=lanczos" `
  -loop 0 `
  "experiments\docs\assets\video_originale2_demo.gif"
```

Verifier la taille :

```powershell
Get-Item "experiments\docs\assets\video_originale2_demo.gif" | Select-Object Name,Length
```

Pour GitHub, un GIF court et leger est plus pratique qu'un MP4 lourd. Si le GIF
depasse environ 10 MB, reduire la duree ou la largeur :

```powershell
& ffmpeg -y -ss 0 -t 6 `
  -i "outputs\tracking\video_originale2_04m40_04m50_tracked.mp4" `
  -vf "fps=8,scale=480:-1:flags=lanczos" `
  -loop 0 `
  "experiments\docs\assets\video_originale2_demo_small.gif"
```

Ajouter ensuite dans `README.md` :

```markdown
![YOLO + ByteTrack demo](experiments/docs/assets/video_originale2_demo.gif)
```

## 7. Choisir une image representative

Extraire une frame a 5 secondes du morceau :

```powershell
& ffmpeg -y `
  -ss 5 `
  -i "outputs\tracking\video_originale2_04m40_04m50_tracked.mp4" `
  -frames:v 1 `
  -q:v 2 `
  "experiments\docs\assets\video_originale2_frame_005s.jpg"
```

Extraire automatiquement trois images :

```powershell
& ffmpeg -y -ss 2 -i "outputs\tracking\video_originale2_04m40_04m50_tracked.mp4" -frames:v 1 -q:v 2 "experiments\docs\assets\demo_01.jpg"
& ffmpeg -y -ss 5 -i "outputs\tracking\video_originale2_04m40_04m50_tracked.mp4" -frames:v 1 -q:v 2 "experiments\docs\assets\demo_02.jpg"
& ffmpeg -y -ss 8 -i "outputs\tracking\video_originale2_04m40_04m50_tracked.mp4" -frames:v 1 -q:v 2 "experiments\docs\assets\demo_03.jpg"
```

Choisir l'image qui montre le mieux : plusieurs outils, IDs lisibles et une
situation representative. Eviter une frame vide ou une frame trop chargee.

## 8. Utiliser une vraie video sans labels

Pour une video externe comme YouTube :

```powershell
& $PY "main.py" `
  --model "models\best.pt" `
  --source "data\samples\video_originale2.mp4" `
  --output "outputs\tracking\video_originale2_tracked.mp4" `
  --predictions "outputs\tracking\video_originale2_predictions.csv" `
  --device cpu
```

Sans labels temporels, utiliser cette video pour une demonstration qualitative et
une mesure runtime, pas pour annoncer une precision de tracking.

## 9. Nettoyer les resultats generes

Attention : cette commande supprime les resultats locaux generes.

```powershell
if (Test-Path "outputs") { Get-ChildItem "outputs" -Force | Remove-Item -Recurse -Force }
New-Item -ItemType Directory -Force "outputs" | Out-Null
```

## 10. Validation rapide du code

```powershell
& $PY -m py_compile "main.py" "src\tracker.py"
& $PY "main.py" --help
```

## 11. Ce qu'il faut mettre sur GitHub

Publier :

- `README.md` ;
- `main.py` et `src/tracker.py` ;
- `requirements.txt` ;
- une courte demo GIF si la source est redistribuable ;
- une image representative ;
- les resultats detection documentes avec leur protocole.

Ne pas presenter comme ground truth officielle :

- les IDs reconstruits automatiquement ;
- MOTA, IDF1 ou HOTA calcules sur ces IDs reconstruits ;
- un MP4 encode a 30 FPS comme preuve de 30 FPS d'inference.
