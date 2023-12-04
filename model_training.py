# Imports pour les modèles
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score


class ModelTrainer:
    def __init__(self, model_type='svm', params=None):
        if model_type == 'svm':
            self.model = SVC(**params) if params else SVC()
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(**params) if params else RandomForestClassifier()
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(**params) if params else LogisticRegression()
        # ... ajoutez d'autres modèles ici

    def train_model(self, x_train, y_train):
        """
        Entraîne le modèle sur les données d'entraînement.
        Implémente la validation croisée avec la fonction cross_validation()

        Input : X_train, y_train.
        Output : Modèle entraîné.

        """
        # À FAIRE : Utilisez la méthode fit du modèle.

    def cross_validation(self, x, y, cv=10):
        """
        Effectue la Validation croisée k-fold stratifiée pour évaluer les performances d'un modèle.


        Paramètres :
        - x : Les caractéristiques du jeu de données.
        - y : Les étiquettes du jeu de données.
        - cv : Le nombre de plis pour la validation croisée (par défaut, 10).

        Output :
        - scores : Dictionnaire des scores de performance pour chaque pli.
        """
        scores = dict()
        #  À FAIRE : Implémenter la validation croisée stratifiée et retourner les résultats
        #  pour certaines métriques qu'on aura à choisir. Utilise train_model() pour entrainer
        return scores

    def evaluate_model(self, model, x_test, y_test):
        """
        Évalue les performances du modèle sur les données de test.

        Input : Modèle entraîné, X_test, y_test.

        Output : Métriques de performance.

        """
        # À FAIRE : Utilisez des métriques comme la précision, le rappel, l'AUC, etc.

    def save_model(self, model, filename):
        """
        Sauvegarde le modèle entraîné.

        Input : Modèle entraîné, nom du fichier.

        Output : Modèle sauvegardé.

        """
        # À FAIRE : Utilisez la méthode save ou pickle.
