import gym
from gym import spaces
import numpy as np
from tqdm import tqdm


class ProblemMinimizationEnv(gym.Env):
    """
    Un environnement pour un problème de minimisation avec des contraintes.

    Args:
        dict_objectifs (dict): Un dictionnaire des fonctions objectifs.
        dict_contraintes (dict): Un dictionnaire des équations de contraintes.
        bornes_variables (list): Les bornes des variables d'entrée.
        n_iterations (int): Le nombre total d'itérations.

    Attributes:
        bonus_objectif (int): Bonus pour atteindre l'objectif.
        malus_objectif (int): Malus pour ne pas atteindre l'objectif.
        malus_objectif_constant (int): Malus constant pour le non-respect de l'objectif.
        malus_contrainte (int): Malus pour la violation de la contrainte.
        liste_bonus_malus (tuple): Liste des valeurs de bonus/malus.
        historique_recompense (list): Historique des récompenses.
        historique_valeurs_objectif (list): Historique des valeurs d'objectif.
        historique_actions (list): Historique des actions.
        verifications_contrainte (list): Historique des vérifications de contraintes.
        episode (list): Liste des épisodes.
        valeurs_objectif_prev (numpy.array): Valeurs d'objectif précédentes.
        compteur_true (int): Compteur de respect de contraintes.
        compteur_false (int): Compteur de violation de contraintes.
        compteur_episode (int): Compteur d'épisodes.
        barre_progression (tqdm): Barre de progression.
    """
    bonus_objectif = 2              # l'agent recoit 2 points si son action diminue l'objectif
    malus_objectif = -1             # l'agent perd 1 points si son action augmente l'objectif
    malus_objectif_constant = 0     # l'agent ne perd ni ne gagne de points si son action ne change rien à l'objectif
    malus_contrainte = -5           # l'agent perd 5 points si son action viole une ou plusieurs contraintes
    # liste des paramètres de récompense de l'agent
    liste_bonus_malus = (bonus_objectif, malus_objectif, malus_objectif_constant, malus_contrainte)

    def __init__(self, dict_objectifs, dict_contraintes, bornes_variables, n_iterations):
        """
        Initialise l'environnement.

        Initialise l'environnement avec les paramètres donnés.

        Args:
            dict_objectifs (dict): Un dictionnaire des fonctions objectifs.
            dict_contraintes (dict): Un dictionnaire des équations de contraintes.
            bornes_variables (list): Les bornes des variables d'entrée.
            n_iterations (int): Le nombre total d'itérations.
        """
        super(ProblemMinimizationEnv, self).__init__()
        self.dict_objectifs = dict_objectifs        # Dictionnaire de fonctions objectif (nom objectif : expression)
        self.dict_contraintes = dict_contraintes    # Dictionnaire d'équations de contrainte (equation : expression)
        self.bornes_variables = bornes_variables    # Liste de bornes de variables d'optimisation
        self.n_iterations = n_iterations            # Nombre d'itérations de calcul souhaité
        self.observation_space = spaces.Box(low=np.array([bound[0] for bound in bornes_variables], dtype=np.float32),
                                            high=np.array([bound[1] for bound in bornes_variables], dtype=np.float32))
        self.action_space = spaces.Box(low=np.array([bound[0] for bound in bornes_variables], dtype=np.float32),
                                       high=np.array([bound[1] for bound in bornes_variables], dtype=np.float32))
        self.historique_recompense = []             # Historique des récompenses
        self.historique_valeurs_objectif = []       # Historique des valeurs de fonctions objectif
        self.historique_actions = []                # Historique des actions
        self.verifications_contrainte = []          # Historique de vérification du respect des contraintes
        self.episode = []                           # liste de comptage du nombre d'épisode
        self.valeurs_objectif_prev = np.zeros((len(dict_objectifs.values()), 1))

        # Initialisation des compteurs
        self.compteur_true = 0
        self.compteur_false = 0
        self.compteur_episode = 0

        # Initialiser la barre de progression
        self.barre_progression = tqdm(total=self.n_iterations, ncols=100, colour="#00ffff", desc="Progression: ")

    def reset(self, *args, **kwargs):
        """
        Réinitialise l'environnement.

        Réinitialise l'environnement à un état aléatoire.

        Returns:
            numpy.array: L'état initial.
        """
        return np.random.uniform(low=[bound[0] for bound in self.bornes_variables],
                                 high=[bound[1] for bound in self.bornes_variables])

    def step(self, action):
        """
        Effectue une étape de l'environnement.

        Effectue une étape de l'environnement en fonction de l'action donnée.

        Args:
            action (numpy.array): L'action à effectuer.

        Returns:
            tuple: Un tuple contenant l'état, la récompense, un indicateur de vérification de contrainte et
                   des informations supplémentaires.
        """
        # Action effectuée par l'agent
        action = np.clip(action, [bound[0] for bound in self.bornes_variables],
                         [bound[1] for bound in self.bornes_variables])

        # Evaluer les fonctions objectif pour l'action effectuée
        liste_valeur_objectif = self.evaluer_objectifs(action)

        # Déterminer si l'action effectuée respecte les contraintes
        bouleen_verifier_contrainte = self.verifier_contrainte(action)

        # sauvegarde des vauleurs des objectifs avant récompense
        valeurs_objectif_now = np.array(liste_valeur_objectif)

        # appelle de la constante de classe 'liste_bonus_malus'
        liste_bonus_malus = ProblemMinimizationEnv.liste_bonus_malus

        # Calcul de la récompense, et comptage des nombre de violation et de respect des contraintes
        (recompense,
         self.compteur_true,
         self.compteur_false) = calculer_recompense(self.compteur_episode, self.compteur_false, self.compteur_true,
                                                    bouleen_verifier_contrainte, valeurs_objectif_now,
                                                    self.valeurs_objectif_prev, liste_bonus_malus)

        # sauvegarde des vauleurs des objectifs après récompense
        self.valeurs_objectif_prev = np.array(liste_valeur_objectif)

        # Compter le nombre d'épisode
        self.compteur_episode += 1

        # Stockage des informations
        self.episode.append(self.compteur_episode)
        self.historique_actions.append(action)
        self.historique_valeurs_objectif.append(liste_valeur_objectif)
        self.historique_recompense.append(recompense)
        self.verifications_contrainte.append(int(bouleen_verifier_contrainte))

        # Mise à jour de la barre de progression
        self.barre_progression.update(1)

        info = {"compteur_episode": self.episode,
                "historique_actions": self.historique_actions,
                "historique_valeurs_objectif": self.historique_valeurs_objectif,
                "historique_recompense": self.historique_recompense,
                "verifications_contrainte": self.verifications_contrainte}

        return action, recompense, bouleen_verifier_contrainte, info

    def evaluer_objectifs(self, x):
        """
        Évalue les fonctions objectifs pour un vecteur donné.

        Args:
            x (numpy.array): Le vecteur d'entrée.

        Returns:
            list: Une liste des valeurs d'objectif.
        """
        liste_valeur_objectif = []  # Initialisation de la liste
        for objectif in self.dict_objectifs.values():
            objectif_fonc = eval(objectif)
            objectif_valeur = objectif_fonc(x)
            liste_valeur_objectif.append(objectif_valeur)
        return liste_valeur_objectif

    def verifier_contrainte(self, x):
        """
        Vérifie si les contraintes sont respectées pour un vecteur donné.

        Args:
            x (numpy.array): Le vecteur d'entrée.

        Returns:
            bool: True si les contraintes sont respectées, False sinon.
        """
        for equation in self.dict_contraintes.values():
            equation_contrainte = eval(equation)
            if not equation_contrainte(x):
                return False
        return True

    def close(self):
        """
        Ferme l'environnement.

        Ferme la barre de progression.
        """
        self.barre_progression.close()


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
