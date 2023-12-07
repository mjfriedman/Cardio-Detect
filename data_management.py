import pandas as pd


class DataManager:
    def __init__(self, x_train, y_train, x_val, y_val, num_classes, data_path):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.data_path = data_path

    def load_data(self):
        """
        Charge les données à partir du fichier spécifié.

        Input : Chemin du fichier de données.
        Output : DataFrame des données.
        """

    def split_data(self, test_size=0.2, random_state=42):
        """
        Divise les données en ensembles d'entraînement et de test.

        Input : Taille du jeu de test, seed pour la reproductibilité.

        Output : X_train, X_test, y_train, y_test.

        """
        # À FAIRE : Utilisez scikit learn pour effectuer la division.

    def get_train_data(self):
        """
        Retourne les données d'entraînement.

        Output : X_train, y_train.

        """
        #  À FAIRE : Retournez les données d'entraînement.

    def get_test_data(self):
        """
        Retourne les données de test

        Output : X_test, y_test.

        """
        #  À FAIRE : Retourne les données de test.
