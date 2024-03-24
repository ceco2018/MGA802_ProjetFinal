import yaml


class LecteurYAML:
    """
    Classe pour lire des données à partir d'un fichier YAML.

    Attributes:
        file_path (str): Le chemin du fichier YAML.
    """

    def __init__(self, file_path):
        """
        Initialise le lecteur YAML avec le chemin du fichier.

        Args:
            file_path (str): Le chemin du fichier YAML.
        """
        self.file_path = file_path

    def importer_donnees_yaml(self):
        """
        Importe les données YAML à partir du fichier.

        Returns:
            dict: Les données YAML sous forme de dictionnaire.
        """
        with open(self.file_path, 'r') as file:
            try:
                data = yaml.safe_load(file)
                return data
            except yaml.YAMLError as e:
                print(f"Erreur de lecture du fichier YAML : {e}")
