# Reconstruction des trajectoires à partir des annotations par frame

## Objectif

Le dossier [vieos41-45/ROI_Labels.csv](../vieos41-45/ROI_Labels.csv) contient des annotations de détection par frame :
- une boîte par outil,
- sur une image donnée,
- avec sa classe, sa position et sa taille.

Il ne contient pas de `TrackID` déjà défini pour chaque objet dans le temps. Pour faire du tracking, il faut reconstruire ces identités à partir des boîtes de frame en frame.

## Idée générale

On part de l'hypothèse que si un outil apparaît dans la frame `t` puis dans la frame `t+1` avec une boîte très proche, alors c'est le même objet.

On mesure cette proximité avec l'IoU (Intersection over Union) entre deux boîtes :

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

Si l'IoU est supérieur à un seuil, par exemple `0.10`, on considère que c'est le même outil et on lui attribue le même `track_id`.

## Pourquoi ce script est utile

Ce script transforme les annotations de détection en une sortie compatible avec un benchmark de tracking, sans modifier les données originales.

Il produit un CSV de type :
- `surgery_num`
- `frame_name`
- `frame_index`
- `track_id`
- `tool_name`
- `bbox_x`
- `bbox_y`
- `bbox_width`
- `bbox_height`

Cela permet ensuite :
- de comparer les prédictions du modèle à une référence stable,
- d'évaluer qualitativement les trajectoires,
- de préparer un mini benchmark MOT (tracking multi-objet).

## Ce que le script fait

1. Lit le CSV d'annotations.
2. Regroupe les lignes par vidéo et par frame.
3. Pour chaque frame, compare les boîtes à celles de la frame précédente.
4. Si une boîte a une proximité suffisante, elle reçoit le même `track_id`.
5. Sinon, un nouveau `track_id` est créé.
6. Écrit un CSV reconstruit dans `outputs/track_gt_reconstructed.csv`.

## Limite importante

Ce n'est pas une vérité terrain officielle de tracking.

C'est une reconstruction raisonnable à partir de détections, avec un seuil IoU simple. Cela reste utile pour :
- l'analyse de séquences,
- la démonstration d'un pipeline de benchmark,
- la préparation d'un portfolio technique.

Mais il manque encore une vraie vérité terrain de tracking si on veut des métriques scientifiques rigoureuses comme :
- MOTA,
- IDF1,
- HOTA,
- ID switches,
- track fragmentation.

## Utilisation

Le script produit désormais un CSV séparé par vidéo, dans un dossier dédié.

```powershell
python src/reconstruct_tracking_gt.py --labels vieos41-45/ROI_Labels.csv --output outputs/tracking_gt_by_video
```

ou simplement :

```powershell
python src/reconstruct_tracking_gt.py
```

Le script auto-découvre le dossier si `ROI_Labels.csv` est présent.

Les fichiers générés sont du type :
- `outputs/tracking_gt_by_video/video_41_tracking_gt.csv`
- `outputs/tracking_gt_by_video/video_42_tracking_gt.csv`
- etc.

Chaque fichier contient uniquement les annotations d’une vidéo.

## Exemple d'interprétation

Si un outil reste visible pendant 200 frames, son `track_id` reste le même pendant 200 frames. S'il sort puis réapparaît, il peut recevoir un nouveau `track_id` selon le seuil de matching.

Cela fait déjà une base exploitable pour montrer que tu sais :
- transformer des annotations de détection en référence de tracking,
- gérer la cohérence temporelle des objets,
- préparer un benchmark de suivi d’outils chirurgicaux.

## Conclusion

Les données de [vieos41-45/ROI_Labels.csv](../vieos41-45/ROI_Labels.csv) sont très utiles pour le projet, mais le tracking exige un petit travail de reconstruction des trajectoires. Ce script est le bon point de départ pour construire une vérité terrain de suivi à partir des annotations de détection déjà présentes.
