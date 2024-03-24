import matplotlib.pyplot as plt
from functools import lru_cache


class AffichageGraphique:
    """
    Cette classe fournit des méthodes pour l'affichage de graphiques.
    """
    def __init__(self, x, y, titre="", xlabel="", ylabel=""):
        """
        Initialise une instance de la classe AffichageGraphique.

        Args:
            x (array-like): Les données de l'axe des abscisses.
            y (array-like): Les données de l'axe des ordonnées.
            titre (str, optional): Le titre du graphique. Par défaut, "".
            xlabel (str, optional): L'étiquette de l'axe des abscisses. Par défaut, "".
            ylabel (str, optional): L'étiquette de l'axe des ordonnées. Par défaut, "".
        """
        self.x = x
        self.y = y
        self.titre = titre
        self.xlabel = xlabel
        self.ylabel = ylabel

    @lru_cache(maxsize=None)
    def tracer_plot(self):
        """
        Affiche un graphique simple.
        """
        plt.figure()
        plt.plot(self.x, self.y, marker='o', markersize=1, linestyle='-')
        plt.title(self.titre)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.grid(True)

    @lru_cache(maxsize=None)
    def tracer_subplot(self, n_rows, n_actions, n_cols=1):
        """
        Affiche plusieurs sous-graphiques.

        Args:
            n_rows (int): Le nombre de lignes de sous-graphiques.
            n_actions (int): Le nombre d'actions.
            n_cols (int, optional): Le nombre de colonnes de sous-graphiques. Par défaut, 1.
        """
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 6))
        for i in range(n_rows):
            axs[i].plot(self.x, self.y[:, i], marker='o', markersize=2, linestyle='')
            if i < n_actions:
                axs[i].set_xlabel("Étape", fontsize=6, fontweight='bold')
                axs[i].set_ylabel(f"Action {i + 1}", fontsize=6, fontweight='bold')
                axs[i].set_title(f"Historique de l'action {i + 1}", fontsize=8, fontweight='bold')
            elif i >= n_actions:
                axs[i].set_xlabel("Étape", fontsize=6, fontweight='bold')
                axs[i].set_ylabel(f"Objectif {i - n_actions + 1}", fontsize=6, fontweight='bold')
                axs[i].set_title(f"Historique de l'objectif {i - n_actions + 1}", fontsize=8, fontweight='bold')
            axs[i].grid(True)
        plt.tight_layout()

    def tracer_scatter(self, vecteur_classification, zlabel="objectif 3"):
        """
        Affiche un graphique en nuage de points.

        Args:
            vecteur_classification (array-like): Vecteur de classification.
            zlabel (str, optional): L'étiquette de l'axe z pour les graphiques 3D. Par défaut, "objectif 3".
        """
        if self.x.ndim == 1:
            plt.figure()
            scatter = plt.scatter(self.x, self.y, c=vecteur_classification, alpha=0.75, s=25)
            plt.title(self.titre)
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            plt.grid(True)
            plt.legend(*scatter.legend_elements(), title="Respect Contrainte")
        elif self.x.ndim == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(self.x[:, 0], self.x[:, 1], self.y, c=vecteur_classification)
            ax.set_title(self.titre)
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)
            ax.set_zlabel(zlabel)
            ax.legend(*scatter.legend_elements(), title="Respect Contrainte")
