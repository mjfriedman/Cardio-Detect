from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
import pandas as pd


class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def handle_missing_values(self, df, strategie='moyenne'):
        """
        Gère les valeurs manquantes (NaN) dans le jeu de données en utilisant une stratégie spécifiée.

        Inputs :
        - df : Le DataFrame des données.
        - strategie : La stratégie de gestion des valeurs nulles ('moyenne', 'mediane', 'mode', 'supprimer').

        Output :
        - data_processed : Le DataFrame après gestion des valeurs manquantes.
        """
        data_processed = df
        # Avec des "if", gérer les différentes stratégies
        return data_processed

    def scale_features(self, data, scaler='robust'):
        """
        Met à l'échelle les variables du jeu de données.

        Inputs :
        - data : Le DataFrame des données.
        - scaler : Le type de mise à l'échelle ('standard', 'minmax', robust).

        Outputs :
        - data_scaled : Le DataFrame avec les fonctionnalités mises à l'échelle.
        """
        if scaler == 'standard':
            scaler = StandardScaler()
        elif scaler == 'minmax':
            scaler = MinMaxScaler()
        elif scaler == 'robust':
            scaler = RobustScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        return data_scaled

    def handle_categorical_variables(self, data, nominal_variables=None, ordinal_variables=None):
        """
        Gère les variables catégorielles dans le jeu de données.

        Inputs :
        - data : Le DataFrame des données.
        - nominal_variables : Liste des noms des variables catégorielles nominales.
        - ordinal_variables : Liste des noms des variables catégorielles ordinales.

        Outputs :
        - data_processed : Le DataFrame après gestion des variables catégorielles.
        """
        data_processed = data
        #  À FAIRE : Gérer les deux cas
        return data_processed
