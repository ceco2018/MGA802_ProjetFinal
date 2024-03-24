import numpy as np


def calculer_recompense(count_episode, count_false, count_true, verifier_contrainte, vect_objectif_now,
                        vect_objectif_prev, liste_bonus_malus):
    """
    Calcule la récompense en fonction de différents paramètres.

    Args:
        count_episode (int): Le numéro de l'épisode actuel.
        count_false (int): Le nombre de violations de contraintes jusqu'à présent.
        count_true (int): Le nombre de respect des contraintes jusqu'à présent.
        verifier_contrainte (bool): Un indicateur indiquant si les contraintes sont respectées.
        vect_objectif_now (numpy.array): Le vecteur d'objectifs actuels.
        vect_objectif_prev (numpy.array): Le vecteur d'objectifs précédents.
        liste_bonus_malus (tupple): Une liste de bonus/malus pour différentes situations.

    Returns:
        tuple: Un tuple contenant la récompense calculée, le nouveau nombre de respect des contraintes,
               et le nouveau nombre de violations de contraintes.
    """
    reward = 0  # On initialise la récompense à chaque étape
    if (count_episode == 0) and (not verifier_contrainte):  # Contrainte violée lors du premier épisode
        count_false += 1
        reward = -1
    elif (count_episode == 0) and verifier_contrainte:  # Contrainte respectée lors du premier épisode
        count_true += 1
        reward = 0
    elif count_episode >= 1:
        # Calcul de la différence entre les objectifs actuels et précédents
        vect_delta_objectif = vect_objectif_now - vect_objectif_prev
        # Création du vecteur de récompenses pour chaque objectif
        vect_reward = np.where(vect_delta_objectif < 0, liste_bonus_malus[0],
                               np.where(vect_delta_objectif == 0, liste_bonus_malus[2], liste_bonus_malus[1]))
        if not verifier_contrainte:  # Contrainte violée
            count_false += 1
            reward = np.sum(vect_reward) + liste_bonus_malus[3]
        else:
            count_true += 1
            reward = np.sum(vect_reward)

    return reward, count_true, count_false
