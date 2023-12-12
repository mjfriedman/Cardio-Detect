from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.cible = "output"

    def handle_missing_values(self, df, strategie='moyenne'):
        """
        Gère les valeurs manquantes (NaN) dans le jeu de données en utilisant une stratégie spécifiée.

        Inputs :
        - df : Le DataFrame des données.
        - strategie : La stratégie de gestion des valeurs nulles ('moyenne', 'mediane', 'mode', 'supprimer').

        Output :
        - data_processed : Le DataFrame après gestion des valeurs manquantes.
        """
        data_processed = df.copy()

        if strategie == 'mean':
            data_processed = data_processed.fillna(data_processed.mean())
        elif strategie == 'median':
            data_processed = data_processed.fillna(data_processed.median())
        elif strategie == 'mode':
            data_processed = data_processed.fillna(data_processed.mode().iloc[0])  # Remplir avec le mode (peut y avoir plusieurs modes)
        elif strategie == 'drop':
            data_processed = data_processed.dropna()
        else:
            raise ValueError("Stratégie non valide. Choisissez parmi 'mean', 'median', 'mode', 'drop'.")

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
        elif scaler == None:
            return data
        data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        return data_scaled

    def prepare_data(self, data, scaler):
        # Mise à l'échelle des variables numériques
        scaled_data = self.scale_features(data, scaler=scaler)

        # Séparation des données en ensembles d'entraînement et de test (70%, 30%)
        X = scaled_data.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
        y = data.iloc[:, -1]   # Dernière colonne

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        return X_train, X_test, y_train, y_test
