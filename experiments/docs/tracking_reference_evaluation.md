# Évaluation de tracking avec références cibles

## But

Les fichiers de [outputs/tracking_gt_by_video](../outputs/tracking_gt_by_video) servent de références cibles pour comparer les sorties d’un modèle de tracking. Ce sont les données de référence que tu prendras comme objectif d’évaluation.

Le script [src/evaluate_tracking_reference.py](../src/evaluate_tracking_reference.py) lit :
- un dossier de références (GT/reference)
- un dossier de prédictions du modèle

puis compare les deux frame par frame.

## Métriques calculées

Le script produit des métriques pratiques :
- précision,
- rappel,
- F1,
- MOTA,
- ID switches,
- IoU moyen sur les objets matchés.

Ces mesures sont utiles pour un benchmark de projet et pour un portfolio, surtout si l’on veut comparer plusieurs variantes de modèle.

## Format attendu

Les CSV doivent contenir au moins les colonnes suivantes :
- `frame_name`
- `track_id`
- `bbox_x`
- `bbox_y`
- `bbox_width`
- `bbox_height`

Les fichiers reconstruits du projet respectent ce format.

## Commande d’utilisation

```powershell
python src/evaluate_tracking_reference.py \
  --gt-dir outputs/tracking_gt_by_video \
  --pred-dir outputs/predictions_by_video \
  --output outputs/tracking_reference_metrics.csv \
  --iou-threshold 0.5
```

## Ce qu’il faut comparer

Pour chaque vidéo :
- la référence cible (GT reconstruite à partir des annotations)
- la prédiction du modèle (tracks obtenus par le tracker)

Ensuite on compare :
- combien d’objets sont correctement détectés,
- combien de faux positifs et faux négatifs,
- combien de changements d’identité sur des trajectoires cohérentes,
- la qualité globale du tracking par MOTA.

## Limite importante

Il s’agit d’un benchmark basé sur des références construites dans ce projet. Ce n’est pas une vérité terrain clinique officielle, mais c’est la bonne manière de comparer des modèles de manière reproductible et intelligible pour un portfolio.

C’est exactement le bon niveau pour :
- montrer un pipeline d’évaluation,
- comparer plusieurs modèles,
- présenter une analyse rigoureuse autour de tracking chirurgical.

## Utilisation pour ton projet

1. Génère les références par vidéo.
2. Exécute ton modèle sur les mêmes clips.
3. Exporte les prédictions dans un dossier `outputs/predictions_by_video`.
4. Lance le script d’évaluation.
5. Compare les métriques par vidéo et documente-les dans le README ou dans un rapport.

C’est cette logique qui permet d’avoir une vraie comparaison entre :
- cible de référence,
- sortie du modèle,
- performance observée.
