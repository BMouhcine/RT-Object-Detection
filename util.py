import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

# DÉFINIR LA FONCTION DE SIGMOID;
def _sigmoid(x):
	return 1. / (1. + np.exp(-x))

# DÉFINIR LA FONCTION decnet POUR TRAITER LES 3 SORTIES DU ConvNet.
def decnet(out, anchors, mask):
	"""
	PARAMS:
	-	out:	est une des 3 sorties du ConvNet de de forme: (1, lignes, colonnes, 255)
				la taille du dernier axe est 255 = 85 * 3. Car on a 3 Anchor boxes chacun occupe 85 lignes.

	-	anchors:	est une liste qui contient les anchors boxes.
	-	mask:	est une liste qui détermine les indices des anchors correspondants au Prediction Scale: 13, 26 ou 52.

	RETURNS:
	-	boxes (h, w, 3, 4): liste qui contient tous les boxes prédits.
	-	box_confidence (h, w, 3, 1): liste qui contient la probabilité de l'existence d'un objet dans un box prédit.
	-	box_class_probs (h, w, 3, 80): liste qui contient les probabilités qu'un objet appartient à une telle classe.
	"""

	"""DÉFINIR grid_h et grid_w RESPECTIVEMENT:
	LE # DE LIGNES ET LE # DE COLONNES DE LA MATRICE SORTIE out POUR DÉFINIR LA TAILLE DE GRILLE."""
	grid_h, grid_w = out.shape[1:3]
	# LE NOMBRE des ANCHORS est 3 (3 par prediction scale et y en a 3.)
	nb_box=3
	# RETAILLER out SOUS LA FORME: (lignes, colonnes, nb_box, 85)
	out = out.reshape((grid_h, grid_w, nb_box, -1))
	# SÉLECTIONNER SEULEMENT LES ANCHORS QU'ON A BESOIN: CE QUI CORRESPONDENT À CE Scale Prediction.
	anchors = [anchors[i] for i in mask]
	# ADAPTER LA FORME DE anchors À CELLE DU out: (height, width, nb_box, nbr_de_box_params)
	anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)
	#out = out[0]

	# APPLIQUER LA FONCTION SIGMOID SUR LES 2 1ERS PARAMS POUR CALCULER ET EXTRAIRE LE CENTRE (X, Y) DU BOX PRÉDIT.
	box_xy = _sigmoid(out[..., :2])
	# APPLIQUER SEULEMENT exp() SUR LES 2 SUIVANTS PARAMS POUR CALCULER ET EXTRAIRE LA LONGUEUR ET LA LARGEUR.
	box_wh = np.exp(out[..., 2:4])
	# MULTIPLIER LE RÉSULTAT PAR LES COORDONNÉES DES ANCHORS (COMME IL EST INDIQUÉ SUR YOLOv3 PAPER).
	box_wh = box_wh * anchors_tensor
	# ACCÉDER À LA PROBABILITÉ DE L'EXISTENCE D'UN OBJET DANS UN BOX PRÉDIT.(5ème paramètre)
	box_confidence = _sigmoid(out[..., 4])
	# RETAILLER box_confidence SOUS LA FORME: (lignes, colonnes, nb_box, 1).
	box_confidence = np.expand_dims(box_confidence, axis=-1)
	# APPLIQUER LA FONCTION SIGMOID SUR LE RESTE DES PARAMS POUR AVOIR LES PROBABILITÉS DES CLASSES.
	box_class_probs = _sigmoid(out[..., 5:])

	# PRÉPARER LES OFFSETS POUR RÉPARTIR LES BOXES TOUT AU LONG DE LA GRILLE.

	colonnes_2_repeat = np.arange(0, grid_w)
	lignes_2_repeat = np.arange(0, grid_h).reshape(-1, 1)

	colonnes = np.tile(colonnes_2_repeat, grid_w).reshape(-1, grid_w)
	lignes = np.tile(lignes_2_repeat, grid_h)

	colonnes = colonnes.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
	lignes = lignes.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
	# colonnes ET lignes ONT CHACUNE LA FORME: 	(13, 13, 3, 1).
	# grid A LA FORME IDENTIQUE À CELLES DU box_xy et box_wh: (13, 13, 3, 2).
	grid = np.concatenate((colonnes, lignes), axis=-1)
	# AJOUTER LES OFFSETS ASSOCIÉS À CHAQUE BOXES CORRESPONDENT À UNE CELLULE DE LA GRILLE.
	box_xy += grid
	# METTRE À L'ÉCHELLE LES (X, Y) QUI SONT LE CENTRE EVENTUEL D'UN BOX ENCADRANT UN OBJET AINSI QUE SES (W, H).
	box_xy /= (grid_w, grid_h)
	box_wh /= (416, 416)
	# TRANSFORMER LES CENTRE (X, Y) EN UN POINT HAUT-GAUCHE (start_point).
	box_xy -= (box_wh / 2.)
	# CONCATENER LE TOUT DANS boxes QUI AURA LA FORME: (h, w, nb_box, 4).
	boxes = np.concatenate((box_xy, box_wh), axis=-1)

	return boxes, box_confidence, box_class_probs


# DÉFINIR LA FONCTION filtrer_boxes POUR FILTRER LES BOXES PAR LE CRITÈRE DE CLASS_THRESH.
def filtrer_boxes(boxes, box_confidences, box_class_probs, class_threshold=.6):
	"""
	FONCTION QUI VA FILTRER, EN FONCTION DE class_threshold,  LES BOXES PRÉDITS BRUTS
	EN GARDANT SEULEMENT CEUX QUI ONT UN SCORE >= class_threshold.
	PARAMS:
	-	boxes (h, w, 3, 4): liste qui contient tous les boxes prédits.
	-	box_confidence (h, w, 3, 1): liste qui contient la probabilité de l'existence d'un objet dans un box prédit.
	-	box_class_probs (h, w, 3, 80): liste qui contient les probabilités qu'un objet appartient à une telle classe.
	RETURNS:
	-	boxes (number_boxes, 4): liste qui contient tous les boxes résultants après le filtrage.
	-	classes (number_boxes, ): liste qui contient tous les classes correspondantes aux boxes.
	-	scores (number_boxes, ): liste qui contient tous les scores correspondants aux classes et aux boxes.
	"""

	# CALCULER LES PROBABILITÉS X DE L'INTERSECTION DE L'ÉVÉNEMENT DE L'EXISTENCE D'UN OBJET ET L'APPARTENANCE À UNE CLASSE.
	# UNE SIMPLE MULTIPLICATION NOUS RENDRA LE SERVICE COMME SUIT:
	box_scores = box_confidences * box_class_probs # Forme résultante (h, w, 3, 80)
	# DÉVOILER LES INDICES ASSOCIÉS AUX CLASSES AYANT LA PLUS GRANDE PROBABILITÉ.
	box_classes = np.argmax(box_scores, axis=-1) # Forme: (h, w, 3)
	# DÉVOILER LA PLUS GRANDE VALEUR DE LA PROBABILITÉ X DANS box_scores.
	box_class_scores = np.max(box_scores, axis=-1) # Forme: (h, w, 3)
	# RÉCUPERER LES POSITIONS OÙ LA CONDITION DE FILTRAGE EST SATISFAITE.
	pos = np.where(box_class_scores >= class_threshold)
	# SÉLECTIONNER CES POSITIONS PARALLÈLEMENT POUR LES boxes LES classes ET LES scores.
	boxes = boxes[pos]
	classes = box_classes[pos]
	scores = box_class_scores[pos]
	#scores = scores

	return boxes, classes, scores



# DÉFINIR LA FONCTION nm_suppression POUR FILTRER LES BOXES VIA NMS EN UTILISANT IOU_THRESH.
def nm_suppression(boxes, scores, iou_threshold=.5):
	"""SUPPRIMER LES BOXES NON-SATISFAISANTS AUX NMS.
	PARAMS:
	-	boxes (number_boxes, 4): liste qui contient tous les boxes résultants après le filtrage.
	-	scores (number_boxes, ): liste qui contient tous les scores correspondants aux classes et aux boxes.
	RETURN:
	-	eff_boxes_ind: liste des indices des boxes effectives.
	"""

	# EXTRAIRE LES COORDONNÉES DES boxes.
	x = boxes[:, 0]
	y = boxes[:, 1]
	w = boxes[:, 2]
	h = boxes[:, 3]
	# DÉFINIR TOUS LES AIRES CORRESPONDANTS AUX boxes.
	areas = w * h
	# EXTRAIRE LES INDICES DES VALEURS DE scores TRIÉS ASC. PUIS INVERTIR POUR COMMENCER AVEC LE + GRAND SCORE.
	order = np.flip(scores.argsort())

	eff_boxes_ind = []
	while order.size > 0:
		i = order[0]
		eff_boxes_ind.append(i)

		xx1 = np.maximum(x[i], x[order[1:]])
		yy1 = np.maximum(y[i], y[order[1:]])
		xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
		yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

		w1 = np.maximum(0.0, xx2 - xx1 + 1)
		h1 = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w1 * h1
		# CALCUL DU TAUX INTERSECTION/UNION.
		ovr = inter / (areas[i] + areas[order[1:]] - inter)
		# ABANDONNER LES BOXES QUI ONT UN TAUX IOU > iou_threshold, ET EXTRAIRE LES INDICES POUR LE RESTE.
		indices_2_keep = np.where(ovr <= iou_threshold)[0]
		# SUPPRIMER LE PREMIER ÉLÉMENT DU ORDER PUISQU'IL EST EN COURS D'UTILISATION.
		order = np.delete(order,0)
		# ACTUALISER LA LISTE DES BOXES À TRAITER DANS L'ITÉRATION SUIVANTE.
		order = order[indices_2_keep]
		# order = order[indices_2_keep + 1]
	# TRANSOFRMER LA LISTE EN UN ARRAY DE NumPy.
	eff_boxes_ind = np.array(eff_boxes_ind)

	return eff_boxes_ind


# DÉFINIR UNE FONCTION QUI ENGLOBE TOUT LE PIPELINE DU TRAITEMENT POUR NE PAS RÉPÉTER L'ENCHAINEMENT À CHAQUE FOIS.
def treatnetout(yhat, shape, masks, anchors):
	# INITIALISATIOn DES PLACEHOLDERS DES boxes, classes ET scores
	boxes, classes, scores = [], [], []
	# ITÉRER PARALLÈLEMENT LA PRÉDICTION BRUTE yhat ET LES masks POUR DÉTÉRMINER LES INDICES DES ANCHORS CORRESPONDANTS.
	for out, mask in zip(yhat, masks):
		# DÉCODER LA SORTIE DU NETWORK out À L'AIDE DE LA FONCTION DÉFINIE EN HAUT: decnet
		b, c, s = decnet(out, anchors, mask)
		# FILTRER LES BOXES EN UTILISANT filtrer_boxes SELON LE class_tresh.
		b, c, s = filtrer_boxes(b, c, s)
		# AJOUTER LES boxes, classes ET scores FILTRÉS AUX PLACEHOLDERS DÉFINIS EN HAUT.
		boxes.append(b)
		classes.append(c)
		scores.append(s)

	# TRANSFORMER LES LISTES boxes, classes ET scores EN UN NumPy Array À L'AIDE D'UNE SIMPLE MONO CONCATÉNATION.
	boxes = np.concatenate(boxes)
	classes = np.concatenate(classes)
	scores = np.concatenate(scores)
	# ÉCHELONNER LES boxes RELATIVEMENT À LA TAILLE ORIGINALE DE L'IMAGE.
	width, height = shape[1], shape[0]
	# RÉPÉTER [width, height] 2x POUR ADAPTER LA FORME DE image_dims AVEC CELLE DE boxes POUR ÉVITER D'UTILISER UNE BOUCLE.
	image_dims = [width, height, width, height]
	# MAINTENANT EFFECTUER LE BROADCAST DES START_POINTS ET LES width & height DES boxes AVEC image_dims.
	boxes = boxes * image_dims
	# MAINTENANT boxes CONTIENT EVENTUELLEMENT DES ÉLÉMENTS BIEN ÉCHELONNÉS AVEC LA TAILLE ORIGINALE DE L'IMAGE.

	# CRÉER DE NOUVEAUX PLACEHOLDERS POUR REGROUPER LES boxes SELON LA CLASSE.
	nboxes, nclasses, nscores = [], [], []
	# PARCOURIR LES CLASSES SANS REDONDANCES: set(classes). POUR NE PAS TRAITER EVENTUELLEMENT UNE CLASSE PLUSIEURS FOIS.
	for c in set(classes):
		# EXTRAIRE EVENTUELLEMENT LES INDICES ASSOCIÉS À LA CLASSE COURANTE c.
		indices = np.where(classes == c)
		# RÉCUPÉRER LES boxes, classes ET LES scores CORRESPONDANTS.
		b = boxes[indices]
		c = classes[indices]
		s = scores[indices]


		# ÉFFECTUER LE NON-MAX SUPPRESSION ET RÉCUPÉRER LES ÉLÉMENTS RÉSULTANTS.
		#reste = tf.image.non_max_suppression(boxes, scores, 10, .5)
		reste = nm_suppression(b, s)

		nboxes.append(b[reste])
		nclasses.append(c[reste])
		nscores.append(s[reste])

	# UN box QUI N'A AUCUNE CLASSE NI SCORE SERA ABANDONNÉ.
	if not nclasses and not nscores:
		return None, None, None
	# RETRANSFORMER LES 3 RÉSULTATS EN NumPy Arrays ET LES RETOURNER.
	# boxes = np.add(nboxes)
	boxes = np.concatenate(nboxes)
	classes = np.concatenate(nclasses)
	scores = np.concatenate(nscores)


	return boxes, classes, scores

# CHARGER UNE IMAGE.
def load_image_pixels(filename, shape=(416, 416)):

	image = load_img(filename)
	width, height = image.size

	image = load_img(filename, target_size=shape)
	# CONVERTIR EN NumPy Array.
	image = img_to_array(image)

	image = image.astype('float32')
	image /= 255.0
	# RETAILLER POUR AVOIR (batch, lignes, colonnes, canales)
	image = expand_dims(image, 0)
	return image, width, height


# UNE FONCTION QUI LIT UNE IMAGE ET DÉSSINE LES BOXES EN UTILISANT PYPLOT.
def draw_plot_boxes(filename, boxes, labels_indices, scores, labels):
	# CHARGER L'IMAGE;
	data = plt.imread(filename)

	# FAIRE PLOT L'IMAGE.
	plt.imshow(data)
	# EXTRAIRE LE CONTEXTE POUR DÉSSINER SUR LE PLOT.
	ax = plt.gca()
	# PARCOURIR TOUS LES boxes ET FAIRE PLOT CHACUN.
	for i in range(len(boxes)):
		box = boxes[i]

		#y1, x1, y2, x2 = box[1], box[0], box[3], box[2]
		x, y, width, height = box[0], box[1], box[2], box[3]




		rect = Rectangle((x, y), width, height, fill=False, color='white')

		ax.add_patch(rect)

		label = "%s (%.3f)" % (labels[labels_indices[i]], scores[i])
		print(label)
		# ICI ON DOIT AUTOMATISER LE DESSIN DES BOXES EN FONCTION DU CONTRASTE DE L'IMAGE OU DE LA RÉGION CONCÉRNÉE.
		# POUR LE MOMENT, ON GARDE LE BLANC.
		plt.text(x, y, label, color='white')

	plt.show()



# DÉFINIR UNE FONCTION QUI PRÉDIT ET QUI PLOT LES RÉSULTATS.
def predict_and_plot(filename, model, masks, anchors, labels):
	img = cv2.imread(filename)
	image, _, _ = load_image_pixels(filename)
	yhat = model.predict(image, steps=1)
	pred_boxes, pred_classes, pred_scores = treatnetout(yhat, img.shape, masks, anchors)
	draw_plot_boxes(filename, pred_boxes, pred_classes, pred_scores, labels)
