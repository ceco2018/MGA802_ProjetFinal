from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3


def selectionner_hyperparametres(choix_algorithme):
    """
    Sélectionne et retourne les hyperparamètres appropriés pour un algorithme donné.

    Args:
        choix_algorithme (type): L'algorithme de RL pour lequel les hyperparamètres doivent être sélectionnés.

    Returns:
        dict: Un dictionnaire contenant les hyperparamètres pour l'algorithme choisi.
    """
    hyperparametres = {}
    if choix_algorithme == A2C:
        hyperparametres = dict(
            gamma=0.999,            # (float) Le facteur d'escompte
            n_steps=int(2 ** 6),    # (int) Le nombre d'étapes à exécuter pour chaque environnement par mise à jour
            ent_coef=0.01,          # (float) Coefficient d'entropie pour le calcul de la perte
            max_grad_norm=0.5,      # (float) Valeur maximale pour le clipping de gradient
            learning_rate=0.0007,   # (float) Taux d'apprentissage
            alpha=0.99,             # (float) Paramètre de décroissance RMSProp (par défaut : 0.99)
            gae_lambda=0.88,        # (float) Paramètre lambda pour GAE
        )

    elif choix_algorithme == DDPG:
        hyperparametres = dict(
            gamma=0.99,              # (float) Le facteur d'escompte
            nb_train_steps=50,       # (int) Le nombre d'étapes d'entraînement
            tau=0.001,      # (float) Le coefficient de mise à jour souple (garder les anciennes valeurs, entre 0 et 1)
            batch_size=128,          # (int) La taille du lot pour l'apprentissage de la politique
            actor_lr=0.0001,         # (float) Le taux d'apprentissage de l'acteur
            critic_lr=0.001,         # (float) Le taux d'apprentissage du critique
        )

    elif choix_algorithme == PPO:
        hyperparametres = dict(
            gamma=0.999,            # (float) Le facteur d'escompte
            n_steps=int(2 ** 6),    # (int) Le nombre d'étapes à exécuter pour chaque environnement par mise à jour
            ent_coef=0.01,          # (float) Coefficient d'entropie pour le calcul de la perte
            learning_rate=0.00025,  # (float) Taux d'apprentissage
            vf_coef=0.5,            # (float) Coefficient de la fonction de valeur pour le calcul de la perte
            max_grad_norm=0.5,      # (float) Valeur maximale pour le clipping de gradient
            lam=0.95,               # (float) Facteur d'échange de biais par rapport à la variance pour GAE
            nminibatches=4,         # (int) Nombre de minibatchs d'entraînement par mise à jour.
            noptepochs=4,           # (int) Nombre d'époques lors de l'optimisation du surrogate
            cliprange=0.2,          # (float ou callable) Paramètre de clipping, peut être une fonction
        )

    elif choix_algorithme == SAC:
        hyperparametres = dict(
            gamma=0.999,            # (float) Le facteur d'escompte
            learning_rate=0.0003,   # (float) Taux d'apprentissage
            buffer_size=50000,      # (int) Taille du buffer de replay
            learning_starts=100,    # (int) Combien d'étapes du modèle collecter des transitions avant de commencer
            train_freq=1,           # (int) Mettre à jour le modèle tous les train_freq pas.
            batch_size=64,          # (int) Taille du minibatch pour chaque mise à jour de gradient
            tau=0.005,              # (float) Le coefficient de mise à jour souple (“mise à jour polyak”, entre 0 et 1)
            verbose=0,              # (int) verbose : 0 aucun, 1 information d'entraînement, 2 débogage tensorflow
        )

    elif choix_algorithme == TD3:
        hyperparametres = dict(
            gamma=0.999,            # (float) Le facteur d'escompte
            learning_rate=0.0003,   # (float) Taux d'apprentissage
            buffer_size=50000,      # (int) Taille du buffer de replay
            learning_starts=100,    # (int) Combien d'étapes du modèle collecter des transitions avant de commencer
            train_freq=100,         # (int) Mettre à jour le modèle tous les train_freq pas.
            gradient_steps=100,     # (int) Combien de mises à jour de gradient après chaque étape
            batch_size=128,         # (int) Taille du minibatch pour chaque mise à jour de gradient
            tau=0.005,              # (float) Le coefficient de mise à jour souple (“mise à jour polyak”, entre 0 et 1)
            verbose=0,              # (int) verbose : 0 aucun, 1 information d'entraînement, 2 débogage tensorflow
        )
    return hyperparametres
