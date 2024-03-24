import numpy as np
import optimiseur_rl
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import warnings

# Ignorer les avertissements liés à l'utilisation des environnements OpenAI Gym dans Stable-Baselines3
warnings.filterwarnings("ignore", category=UserWarning, message="You provided an OpenAI Gym environment.*")

# Affiche un message d'introduction
print("\nMESSAGE D'INTRODUCTION --------------------------------------------------------------------------------------")
print("Ce programme permet la résolution de problèmes d'optimisation mono & multi-objectifs avec ou sans contraintes,\n"
      "à l'aide des algorithmes d'apprentissage par renforcement (A2C, DDPG, PPO, SAC et TD3)")

# Instructions pour fournir les données du problème dans un fichier YAML
print("\033[93m \nVOUS DEVEZ FOURNIR LES DONNEES DU PROBLEME !!!-----------!!!----------!!!-----------!!!-----------!!!"
      "--------\033[0m")
print("\033[93m  Veuillez compléter les données du fichier YAML 'deck.yaml' avec vos paramètres d’entrées en suivant\n"
      "  rigoureusement les instructions consignées dans ledit fichier, tout en respectant la syntaxe /-save-/\033[0m")

# Attend que l'utilisateur entre "GO" en lettres capitales pour indiquer que les données sont prêtes
while True:
    executer_programme = input("\033[95m \nVeuillez saisir 'GO' en lettres capitales si votre fichier de données YAML "
                               "est prêt : \033[0m")
    if executer_programme == 'GO':
        break  # Sortir de la boucle si les données sont valides
    else:
        print("\nLa saisie n'est pas valide. Veuillez entrer 'GO' en lettres capitales.")


def main():
    """
    Cette fonction est la fonction principale du programme, qui est appelée lorsque l'on exécute le programme.
    """
    # On crée un objet YAML au sein duquel on charge une instance de LecteurYAML qui lit le fichier "deck.yamL"
    parser = optimiseur_rl.LecteurYAML('deck.yaml')

    # On exécute la fonction read_yaml() de notre objet LecteurYAML
    parsed_data = parser.importer_donnees_yaml()

    # Initialisation des données d'entrées pour la formulation du probleme d'optimisation
    dict_objectifs = {}
    dict_contraintes = {}
    bornes_variables = []

    # Récupération des données du fichier YAML
    type_execution = parsed_data['type_execution']
    if type_execution == "entrainement":
        dict_objectifs = parsed_data['dict_objectif_training']
        dict_contraintes = parsed_data['dict_contrainte_training']
        n_variables = parsed_data['n_variables_training']
        bornes_variables = parsed_data['bornes_variable_training'] * n_variables
    elif type_execution == "exploitation":
        dict_objectifs = parsed_data['dict_objectif_exploitation']
        dict_contraintes = parsed_data['dict_contrainte_exploitation']
        bornes_variables = parsed_data['bornes_variable_exploitation']

    n_iterations = parsed_data['n_episodes']
    choix_algorithme = parsed_data['choix_algorithme']
    print(f"\nAlgorithme de Reinforcement Learning : {choix_algorithme}\n")

    # Sélection de l'algorithme RL en fonction des données du fichier YAML
    algo_rl = globals()[choix_algorithme]

    nombre_objectifs = len(dict_objectifs)
    nombre_variables = len(bornes_variables)
    nombre_contraintes = len(dict_contraintes)

    # Création de l'environnement pour la résolution du problème d'optimisation
    env = optimiseur_rl.ProblemMinimizationEnv(dict_objectifs, dict_contraintes, bornes_variables, n_iterations)

    # Sélection des hyperparamètres en fonction de l'algorithme choisi
    hyperparametres = optimiseur_rl.selectionner_hyperparametres(choix_algorithme)

    # Création de l'environnement vectorisé
    vec_env = DummyVecEnv([lambda: env])

    # Création et entraînement du modèle RL
    model = algo_rl("MlpPolicy", vec_env, verbose=1, **hyperparametres)
    model.learn(total_timesteps=n_iterations)

    # Conversion des résultats en tableaux numpy
    episode_array = np.array(env.episode)
    actions_history_array = np.array(env.historique_actions)
    objective_values_history_array = np.array(env.historique_valeurs_objectif)
    reward_history_array = np.array(env.historique_recompense)
    constraint_verifications_array = np.array(env.verifications_contrainte)

    # Calcul de la somme cumulée des récompenses
    recompense_cumulee = np.cumsum(reward_history_array)

    # Construction du tableau de synthèse des résultats bruts
    resultats_bruts = np.column_stack((episode_array, actions_history_array, objective_values_history_array,
                                       reward_history_array, constraint_verifications_array))

    # Construction du tableau de synthèse des solutions faisables (celles qui respectent les contraintes) :
    # Extraction des lignes de resultats_bruts où le bouléen de respect contrainte vau 1 (dernière colonne)
    solutions_faisable = resultats_bruts[resultats_bruts[:, -1] == 1]

    # Construction du tableau de synthèse des solutions faisables avec sommation algébrique des objectifs
    # Somme ligne par ligne des éléments des colonnes de la matrice des objectifs
    somme_colonnes_objectif = np.sum(solutions_faisable[:, (-2-nombre_objectifs):-2], axis=1)
    # Construction de la matrice
    solutions_faisable_reduit = np.column_stack((solutions_faisable[:, :(1+nombre_variables)], somme_colonnes_objectif,
                                                 solutions_faisable[:, -2:]))

    # Recherche des solution optimales iso-pondérées
    min_somme_objectif = np.min(solutions_faisable_reduit[:, -3])
    solutions_optimales_iso = solutions_faisable_reduit[solutions_faisable_reduit[:, -3] == min_somme_objectif]

    # Affichage des résultats et des graphiques
    print("\033[93m \n---------------- SYNTHESE DES RESULTATS DE L'OPTIMISATION -----------------\033[0m")
    # Définition du format pour l'affichage
    formatter = {'float_kind': lambda x: "{:.4g}".format(x), 'int': lambda x: '%4d' % x}
    # Définir le séparateur avec plus d'espaces
    separator = '  '

    print("\nTABLEAU DES RESULTATS BRUTS (extrait de cinq lignes)--------------------")
    print("| Etape (1) | Variables | Ojectif(s) | Recompense (1) | Respect Contrainte (1) |")
    print(np.array2string(resultats_bruts[:5], formatter=formatter, separator=separator))
    print("\nTABLEAU DES SOLUTIONS FAISABLES (extrait de cinq lignes) ---------------")
    print("| Etape | Variables | Ojectif(s) | Recompense | Respect Contrainte |")
    print(np.array2string(solutions_faisable[:5], formatter=formatter, separator=separator))
    print("\033[95m \nSOLUTION(s) OPTIMALE(s) ISO PONDEREE ------------------------------------\033[0m")
    print("| Etape | Variables | Σ Ojectif | Recompense | Respect Contrainte |")
    print(np.array2string(solutions_optimales_iso, formatter=formatter, separator=separator))

    print("\n---------------- VOIR GRAPHIQUES DES RESULTATS D'OPTIMISATION -----------\n")

    # Affichage de la courbe de l'historique des recompenses
    courbe_hist_recompense = optimiseur_rl.AffichageGraphique(episode_array, reward_history_array,
                                                              "Historique des Récompenses", "épisodes",
                                                              "récompenses cumulées")
    courbe_hist_recompense.tracer_plot()

    # Affichage de la courbe des recompenses cumulée
    courbe_recompense_cumulee = optimiseur_rl.AffichageGraphique(episode_array, recompense_cumulee,
                                                                 "Cumule des Récompenses", "épisodes",
                                                                 "récompenses cumulées")
    courbe_recompense_cumulee.tracer_plot()

    # SubPlot des objectif(s) et et des variables
    tableau_var_obj = np.column_stack((actions_history_array, objective_values_history_array))
    if (nombre_objectifs+nombre_variables) <= 5:
        graphe_hist_variables_objectifs = optimiseur_rl.AffichageGraphique(episode_array, tableau_var_obj)
        graphe_hist_variables_objectifs.tracer_subplot(nombre_variables+nombre_objectifs, nombre_variables)

    # SubPlot des historiques de variables si moins que six
    if nombre_variables <= 5:
        graphe_hist_variables = optimiseur_rl.AffichageGraphique(episode_array, actions_history_array)
        graphe_hist_variables.tracer_subplot(nombre_variables, nombre_variables)

    # Courbe de l'historique de l'objectif
    if nombre_objectifs == 1:
        graphe_hist_mono_objectif = optimiseur_rl.AffichageGraphique(episode_array, objective_values_history_array,
                                                                     "Historique de l'objectif", "episodes",
                                                                     "objectif")
        graphe_hist_mono_objectif.tracer_plot()

    # SubPlot des historiques des objectifs
    if nombre_objectifs >= 2:
        graphe_hist_multi_objectifs = optimiseur_rl.AffichageGraphique(episode_array, objective_values_history_array)
        graphe_hist_multi_objectifs.tracer_subplot(nombre_objectifs, 0)

    # Front de Pareto bidimensionnel 2D
    if nombre_objectifs == 2:
        # Front de Pareto 2D sans contraintes
        front_pareto2d = optimiseur_rl.AffichageGraphique(objective_values_history_array[:, 0],
                                                          objective_values_history_array[:, 1], "Front de Pareto 2D",
                                                          "objectif 1", "objectif 2")
        front_pareto2d.tracer_scatter(constraint_verifications_array)

        # Front de Pareto 2D avec respect des contraintes
        if nombre_contraintes != 0:
            front_pareto2d_contraint = optimiseur_rl.AffichageGraphique(solutions_faisable[:, -4],
                                                                        solutions_faisable[:, -3], "Front de Pareto 2D",
                                                                        "objectif 1", "objectif 2")
            front_pareto2d_contraint.tracer_scatter(solutions_faisable[:, -1])

    # Front de Pareto tridimensionnel 3D
    elif nombre_objectifs == 3:
        front_pareto3d = optimiseur_rl.AffichageGraphique(objective_values_history_array[:, :2],
                                                          objective_values_history_array[:, 2], "Front de Pareto 3D",
                                                          "objectif 1", "objectif 2")
        front_pareto3d.tracer_scatter(constraint_verifications_array)

        # Front de Pareto 3D avec respect des contraintes
        if nombre_contraintes != 0:
            front_pareto3d_contraint = optimiseur_rl.AffichageGraphique(solutions_faisable[:, -5:-3],
                                                                        solutions_faisable[:, -3], "Front de Pareto 3D",
                                                                        "objectif 1", "objectif 2")
            front_pareto3d_contraint.tracer_scatter(solutions_faisable[:, -1])

    plt.show()


if __name__ == "__main__":
    main()
