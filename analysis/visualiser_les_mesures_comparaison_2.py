"""
Ce script analyse et évalue la qualité de la classification d'oxygène sur une image d'échantillon.
Il utilise une image en niveaux de gris (obtenue à partir d'une image couleur) et compare
les résultats avec des données de référence pour calculer différentes métriques de performance.
"""

import json
import os
from typing import Callable
from enum import Enum
from math import cos, sin, pi
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


class CLASSIFICATION(Enum):
    TRANSPARENT = -6
    FALSE_POSITIVE = -3
    FALSE_NEGATIVE = -2
    NEGATIVE = 0
    POSITIVE = 1
    TRUE_POSITIVE = 2
    TRUE_NEGATIVE = 3


class COULEURS(Enum):
    NEGATIVE = (49 / 255, 0, 174 / 255)  # Bleu foncé
    POSITIVE = (182 / 255, 1, 0)  # Vert citron
    TRUE_POSITIVE = "white"
    FALSE_NEGATIVE = (1, 0, 0)  # Rouge clair
    TRUE_NEGATIVE = "black"
    FALSE_POSITIVE = (139 / 255, 0, 0)  # Rouge foncé
    TRANSPARENT = (160 / 255, 160 / 255, 160 / 255)  # Gris


def classifier_oxygene_binaire(
    values: np.ndarray[tuple, np.dtype[np.float32]],
) -> np.ndarray:
    """
    Transforme les valeurs brutes d'oxygène en classification binaire.
    Applique une transformation linéaire suivie d'un seuillage pour déterminer
    les zones riches/pauvres en oxygène, avec deux conditions différentes
    selon la position dans le tableau.

    Args:
        values: Tableau des valeurs d'oxygène mesurées
    Returns:
        Tableau binaire (0 ou 1) représentant la classification
    """
    threshold = -7.99
    transformed = 1.09 * values + 13.01

    # Create mask for different conditions
    first_part = np.arange(len(values)) < 3174

    # Apply conditions using numpy where
    # Inversion de la condition pour la première partie
    result = np.where(
        first_part,
        transformed >= threshold,  # condition pour i < 3174
        transformed < threshold,  # condition pour i >= 3174
    )

    return result.astype(int)


def ouvrir_image(path):
    """
    Charge l'image de l'échantillon depuis le chemin spécifié.

    Args:
        path: Chemin vers le dossier contenant l'image
    Returns:
        Une copie de l'image chargée
    """
    image = plt.imread(os.path.join(path, "roche.png"))
    return np.copy(image)


def polar(metrics):
    """
    Génère un graphique radar pour visualiser 4 métriques de performance.
    Affiche Precision, Specificity, F1-score et Accuracy sur des axes différents
    pour une évaluation visuelle rapide de la qualité de la classification.

    Args:
        metrics: Liste des 4 métriques à afficher
    """
    labels = ["Precision", "Specificity", "F1", "Accuracy"]
    if len(metrics) != len(labels):
        return

    theta_offset = pi / 2
    angles = [2 * i * pi / len(metrics) + theta_offset for i in range(len(metrics))]
    x = [metrics[i] * cos(ang) for i, ang in enumerate(angles)]
    y = [metrics[i] * sin(ang) for i, ang in enumerate(angles)]
    x.append(x[0])
    y.append(y[0])

    plt.figure()

    # Axes radiaux
    for i in range(len(metrics)):
        ang = 2 * i * pi / len(metrics) + theta_offset
        plt.plot([0, cos(ang)], [0, sin(ang)], "-k", linewidth=1)

    # Cercles intermédiaires
    for radius in np.arange(0.1, 1.0, 0.1):
        cx = [radius * cos(ang) for ang in angles] + [radius * cos(angles[0])]
        cy = [radius * sin(ang) for ang in angles] + [radius * sin(angles[0])]
        plt.plot(cx, cy, "-", linewidth=1, color=(210 / 255, 210 / 255, 210 / 255))

    # Cercle principal
    cx = [cos(ang) for ang in angles] + [cos(angles[0])]
    cy = [sin(ang) for ang in angles] + [sin(angles[0])]
    plt.plot(cx, cy, "-", linewidth=3, color=(195 / 255, 195 / 255, 195 / 255))

    # Courbe des métriques
    plt.plot(x, y, "-or")
    plt.plot([0], [0], "ok")
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.axis("off")

    # Étiquettes
    positions = [(0, 1.05), (-1.35, 0), (1.05, 0), (-0.15, -1.05)]
    for label, pos in zip(labels, positions):
        plt.text(pos[0], pos[1], label)


def val_classification(image_gray: np.ndarray, threshold: int):
    """
    Évalue la qualité de la classification en comparant avec les données de référence.

    1. Sépare les coordonnées de points selon leur classe d'oxygène (haut/bas)
    2. Applique la classification sur l'image en niveaux de gris selon le seuil
    3. Calcule les métriques de performance (TP, TN, FP, FN)
    4. Renvoie un dictionnaire avec toutes les métriques et l'image classifiée

    Args:
        image_gray: Image convertie en niveaux de gris
        threshold: Seuil de classification
    Returns:
        Dictionnaire contenant les métriques de performance et l'image classifiée
    """
    # Séparation des points selon leur classe (0 ou 1)
    low_oxygen_coordinates, high_oxygen_coordinates = [], []
    for i, label in enumerate(OXYGEN_BINARY_LABELS):
        if label == CLASSIFICATION.NEGATIVE.value:
            low_oxygen_coordinates.append(l_x[i])
        else:
            high_oxygen_coordinates.append(l_x[i])

    img_classified = np.ones_like(image_gray, dtype=int)
    result = classify_image(
        image_gray,
        threshold,
        low_oxygen_coordinates,
        high_oxygen_coordinates,
        img_classified,
    )

    (
        false_negative,
        true_negative,
        false_positive,
        true_positive,
    ) = result

    real_positive = true_positive + false_negative
    real_negative = true_negative + false_positive

    total = real_positive + real_negative
    if total == 0:
        raise ValueError("Total number of samples is zero, cannot compute metrics.")

    computed_positive = true_positive + false_positive
    computed_negative = true_negative + false_negative

    true_positive_rate = true_positive / real_positive if real_positive > 0 else 0
    false_negative_rate = 1 - true_positive_rate

    false_positive_rate = false_positive / real_negative if real_negative > 0 else 0
    true_negative_rate = 1 - false_positive_rate

    f1_score = (
        2 * true_positive / (2 * (total - true_negative))
        if (total - true_negative) > 0
        else 0
    )

    accuracy = (true_positive + true_negative) / total

    positive_predictive_value = (  # Aussi appelé "précision"
        true_positive / computed_positive if computed_positive > 0 else 0
    )
    false_discovery_rate = 1 - positive_predictive_value

    negative_predictive_value = (
        true_negative / computed_negative if computed_negative > 0 else 0
    )
    false_ommission_rate = 1 - negative_predictive_value

    phi = float(
        np.sqrt(
            (
                true_positive_rate
                * true_negative_rate
                * positive_predictive_value
                * negative_predictive_value
            )
        )
        - np.sqrt(
            false_negative_rate
            * false_positive_rate
            * false_ommission_rate
            * false_discovery_rate
        )
    )

    # print("True Positive Rate (TPR): ", true_positive_rate)
    # print("True Negative Rate (TNR): ", true_negative_rate)
    # print("False Positive Rate (FPR): ", false_positive_rate)
    # print("False Negative Rate (FNR): ", false_negative_rate)
    # print("Specificity:", true_negative_rate)
    # print("F1: ", f1_score)
    # print("Accuracy (ACC): ", accuracy)
    # print("Phi Coefficient: ", phi)

    return {
        "img_classified": img_classified,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive_rate": true_positive_rate,
        "false_negative_rate": false_negative_rate,
        "false_positive_rate": false_positive_rate,
        "true_negative_rate": true_negative_rate,
        "specificity": true_negative_rate,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "phi": phi,
    }

    # polar([precision, specificity, f1_score, accuracy])


def colorier_image_classifiee(
    img_classified: np.ndarray,
    mask: np.ndarray,
):
    """
    Colorie une image classifiée en utilisant un code couleur pour visualiser
    les résultats de classification (vrais/faux positifs, vrais/faux négatifs).

    Args:
        img_classified: Image avec les valeurs de classification
        mask: Masque indiquant les zones transparentes
    Returns:
        Image colorée selon la classification
    """
    # Masquer les zones transparentes
    img_classified[mask] = CLASSIFICATION.TRANSPARENT.value

    # Affichage avec couleurs
    cmap = mcolors.ListedColormap(
        [
            COULEURS.TRANSPARENT.value,
            COULEURS.FALSE_POSITIVE.value,
            COULEURS.FALSE_NEGATIVE.value,
            COULEURS.NEGATIVE.value,
            COULEURS.POSITIVE.value,
            COULEURS.TRUE_POSITIVE.value,
            COULEURS.TRUE_NEGATIVE.value,
        ]
    )

    bounds = (
        np.array(
            [
                CLASSIFICATION.TRANSPARENT.value,
                CLASSIFICATION.FALSE_POSITIVE.value,
                CLASSIFICATION.FALSE_NEGATIVE.value,
                CLASSIFICATION.NEGATIVE.value,
                CLASSIFICATION.POSITIVE.value,
                CLASSIFICATION.TRUE_POSITIVE.value,
                CLASSIFICATION.TRUE_NEGATIVE.value,
            ]
        )
        - 0.5
    ).tolist() + [1000]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    return cmap(norm(img_classified))


def afficher_image_classifiee(
    # image_rgba: np.ndarray,
    mask: np.ndarray,
    img_classified: np.ndarray,
):
    """
    Affiche une image classifiée avec un code couleur.

    Args:
        mask: Masque indiquant les zones transparentes
        img_classified: Image avec les valeurs de classification
    """
    # mask = image_rgba[:, :, 3] < 0.02

    img_classified = colorier_image_classifiee(img_classified, mask)

    plt.figure()
    plt.imshow(img_classified)
    plt.axis("off")
    plt.tight_layout()


def classify_image(
    image_gray: np.ndarray,
    threshold: int,
    low_oxygen_coordinates: list[tuple[int, int]],
    high_oxygen_coordinates: list[tuple[int, int]],
    img_classified: np.ndarray,
):
    """
    Classifie une image selon un seuil et évalue les performances en comparant
    avec les coordonnées de référence des zones à haute et basse teneur en oxygène.

    Args:
        image_gray: Image en niveaux de gris à classifier
        threshold: Seuil de classification
        low_oxygen_coordinates: Liste des coordonnées des points à faible teneur en oxygène
        high_oxygen_coordinates: Liste des coordonnées des points à haute teneur en oxygène
        img_classified: Tableau pour stocker l'image classifiée
    Returns:
        Tuple contenant les compteurs (faux négatifs, vrais négatifs, faux positifs, vrais positifs)
    """
    if threshold > 0:
        img_classified[image_gray <= threshold] = 0
        img_classified[image_gray > threshold] = 1
    else:
        threshold = -threshold
        img_classified[image_gray <= threshold] = 1
        img_classified[image_gray > threshold] = 0

    # Évaluation
    count_true_positive = count_false_positive = count_true_negative = (
        count_false_negative
    ) = 0
    for x, y in high_oxygen_coordinates:
        if img_classified[y, x] == CLASSIFICATION.POSITIVE.value:
            count_true_positive += 1
            img_classified[y, x] = CLASSIFICATION.TRUE_POSITIVE.value
        else:
            count_false_negative += 1
            img_classified[y, x] = CLASSIFICATION.FALSE_NEGATIVE.value
    for x, y in low_oxygen_coordinates:
        if img_classified[y, x] == CLASSIFICATION.NEGATIVE.value:
            count_true_negative += 1
            img_classified[y, x] = CLASSIFICATION.TRUE_NEGATIVE.value
        else:
            count_false_positive += 1
            img_classified[y, x] = CLASSIFICATION.FALSE_POSITIVE.value
    return (
        count_false_negative,
        count_true_negative,
        count_false_positive,
        count_true_positive,
    )


def val_classifications(image_gray: np.ndarray):
    """
    Évalue la qualité de la classification pour différents seuils et renvoie les résultats.
    Teste tous les seuils possibles de -255 à 254 et calcule les métriques de performance
    pour chacun d'eux, en gardant une trace du meilleur seuil selon le coefficient phi.

    Args:
        image_gray: Image en niveaux de gris à classifier
    Returns:
        Dictionnaire contenant les métriques de performance pour chaque seuil testé
    """
    x_max = -255
    phi_max = -1.0

    x = np.arange(-255, 255).astype(int)
    tpr = []
    tnr = []
    f1_score = []
    accuracy = []
    phi = []
    img_classified = []
    for threshold in x:
        metrics = val_classification(image_gray, threshold)
        tpr.append(metrics["true_positive_rate"])
        tnr.append(metrics["true_negative_rate"])
        f1_score.append(metrics["f1_score"])
        accuracy.append(metrics["accuracy"])
        phi.append(metrics["phi"])
        img_classified.append(metrics["img_classified"])
        if metrics["phi"] > phi_max:
            phi_max = metrics["phi"]
            x_max = int(threshold)

        # print("")

    resultats = {
        "tpr": tpr,
        "tnr": tnr,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "phi": phi,
        "img_classified": img_classified,
        "x_max": x_max,
    }
    return resultats


def afficher_resultats(resultats):
    """
    Affiche les résultats de l'évaluation de la classification.

    Args:
        resultats: Dictionnaire contenant les métriques à afficher
    """
    x = np.arange(-255, 255)  # Create x-axis values from -255 to 254

    x_max = np.argmax(resultats["phi"])
    print("Meilleur seuil pour le coefficient phi :", x[x_max])

    plt.figure(figsize=(10, 6))
    plt.plot(x, resultats["tpr"], label="True Positive Rate (TPR)", color="blue")
    plt.plot(x, resultats["tnr"], label="True Negative Rate (TNR)", color="green")
    plt.plot(x, resultats["f1_score"], label="F1 Score", color="orange")
    plt.plot(x, resultats["accuracy"], label="Accuracy", color="red")
    plt.plot(x, resultats["phi"], label="Phi Coefficient", color="purple")
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")

    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Classification Metrics vs Threshold")
    plt.legend()
    plt.grid()

    polar(
        [
            resultats["tpr"][x_max],
            resultats["tnr"][x_max],
            resultats["f1_score"][x_max],
            resultats["accuracy"][x_max],
        ]
    )
    return x_max


def test_val_classification(methods: dict[str, Callable], image_rgb: np.ndarray):
    """
    Teste différentes méthodes de classification sur l'image fournie.

    Args:
        methods: Dictionnaire contenant les méthodes de classification à tester
        image_rgb: Image RGB à classifier
    Returns:
        Dictionnaire contenant les résultats de chaque méthode
    """
    shutil.rmtree("results", ignore_errors=True)
    os.makedirs("results")
    data = {}
    mask = image_rgb < 0.02
    for name, method in methods.items():
        print(f"Testing method: {name}")
        try:
            image_gray = method(image_rgb)

            results = val_classifications(image_gray)

            print(f"{name} : {results['phi'][results['x_max'] + 255]}")
            plt.imsave(
                os.path.join("results", f"{name}_classified.png"),
                colorier_image_classifiee(
                    results["img_classified"][results["x_max"] + 255], image_gray < 0.02
                ),
            )

            del results["img_classified"]

            data[name] = results
        except Exception as e:
            print(f"Error testing method {name}: {e}")
            raise e

    with open("results.json", "w") as f:
        json.dump(data, f, indent=4)


def main():
    # Chargement et préparation de l'image
    # image = ouvrir_image(base_dir) * 255.0

    image = plt.imread("analysis/tmp.png") * 255.0

    from color2gray.methods import AverageMethod, all_methods

    methods = {name: Method() for name, Method in all_methods.items()}
    # methods = {
    #     "Average": AverageMethod(),
    #     # "Luminance": LuminanceMethod(),
    #     # "Lightness": LightnessMethod(),
    #     # "Luma": LumaMethod(),
    #     # "HSV": HSVMethod(),
    #     # "YUV": YUVMethod(),
    # }
    test_val_classification(methods, image)

    return

    # Conversion en niveaux de gris (utilisation du canal rouge)
    image_grayscale = np.array(255 * image[:, :, 2], dtype=int)
    white_image = np.ones_like(image_grayscale, dtype=int) * 255
    image_grayscale = white_image

    # Évaluation de la qualité de la classification
    res = val_classifications(image_grayscale)

    image_colored = colorier_image_classifiee(
        res["img_classified"][res["x_max"] + 255], image_grayscale < 0.02
    )
    plt.imsave("image_classifiee.png", image_colored)

    afficher_resultats(res)
    afficher_image_classifiee(
        image_grayscale < 0.02,
        res["img_classified"][res["x_max"] + 255],
    )
    plt.show()

    while True:
        seuil = input(
            "Entrez un seuil de classification (0-255) ou 'q' pour quitter : "
        )
        if seuil.lower() == "q":
            break

        try:
            seuil = int(seuil)
        except ValueError:
            print("Entrée invalide, veuillez entrer un nombre entier ou 'q'.")
            continue

        if not -255 <= seuil <= 255:
            print("Seuil invalide, veuillez entrer un nombre entre 0 et 255.")
            continue

        metrics = val_classification(image_grayscale, seuil)
        afficher_image_classifiee(
            image_grayscale < 0.02,
            metrics["img_classified"],
        )


# Chargement des données de position
base_dir = os.path.abspath(os.path.dirname(__file__))
chemin = os.path.join(base_dir, "d_18_O_carto_full.json")
with open(chemin, "r", encoding="ascii") as f:
    data = json.load(f)
    l_x = np.array(data["x"], dtype=np.int32)
    OXYGEN_BINARY_LABELS = np.array(data["y"], dtype=np.float32)

OXYGEN_BINARY_LABELS = classifier_oxygene_binaire(OXYGEN_BINARY_LABELS)


if __name__ == "__main__":
    main()
