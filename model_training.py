# Import
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
import pickle
import pandas as pd


class ModelTrainer:
    def __init__(self, X_train, y_train, model_type='svm', params=None):
        self.X_train = X_train
        self.y_train = y_train
        self.model_type = model_type
        self.params = params
        self.model = self.initialize_model()
        self.model_accuracy = None
    

    def initialize_model(self):
        if self.model_type == 'svm':
            return SVC(**self.params) if self.params else SVC()
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(**self.params) if self.params else RandomForestClassifier()
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(**self.params) if self.params else LogisticRegression()
        elif self.model_type == 'ada_boost':
            return AdaBoostClassifier(**self.params) if self.params else AdaBoostClassifier()
        elif self.model_type == 'mlp':
            return MLPClassifier(**self.params) if self.params else MLPClassifier()
        elif self.model_type == 'lda':
            return LinearDiscriminantAnalysis()
        else:
            raise ValueError("Modèle non pris en charge. Choisissez parmi 'svm', 'random_forest', 'logistic_regression', 'ada_boost', 'mlp', 'lda'.")


    def train_model(self):
        """
        Entraîne le modèle sur les données d'entraînement.

        Output : Modèle entraîné.
        """

        if self.model_type == 'mlp':
            # Simulate train / test / validation sets
            X_train_new, X_valid, y_train_new, y_valid = train_test_split(self.X_train, self.y_train, train_size=0.8)

            # Initialize
            batch_size, train_loss_, valid_loss_, train_accuracy_, valid_accuracy_ = 50, [], [], [], []

            # Training Loop
            for _ in range(self.model.max_iter):
                for b in range(batch_size, len(y_train_new), batch_size):
                    X_batch, y_batch = X_train_new[b-batch_size:b], y_train_new[b-batch_size:b]
                    self.model.partial_fit(X_batch, y_batch, classes=np.unique(y_batch))
                    train_loss_.append(self.model.loss_)
                    valid_loss_.append(log_loss(y_valid, self.model.predict_proba(X_valid)))

            # Calculate accuracy during each epoch
            train_accuracy_.append(accuracy_score(y_train_new, self.model.predict(X_train_new)))
            valid_accuracy_.append(accuracy_score(y_valid, self.model.predict(X_valid)))
            

            # Affichage de la courbe de la perte
            plt.figure(figsize=(20, 8))
            plt.plot(train_loss_, 'blue', label='Training')
            plt.plot(valid_loss_, 'red', label='Validation')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.show()

            # Affichage de la courbe de l'accuracy
            plt.figure(figsize=(20, 8))
            plt.plot(train_accuracy_, 'blue', label='Training ')
            plt.plot(valid_accuracy_, 'red', label='Validation ')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.show()
        else:
            # Entrainement pour les autres modèles 
            self.model.fit(self.X_train, self.y_train)

        return self.model
    


    def cross_validation(self, cv=5, scoring='accuracy'):
        """
        Effectue la Validation croisée k-fold stratifiée pour évaluer les performances d'un modèle.

        Paramètres :
        - cv : Le nombre de plis pour la validation croisée (par défaut, 10).
        - scoring : La métrique de performance à évaluer (par défaut, 'accuracy').

        Output :
        - scores : Liste des scores de performance pour chaque pli.
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, self.X_train, self.y_train, cv=skf, scoring=scoring)
        return scores


    def grid_search(self, param_grid, cv=5, scoring='accuracy'):
        """
        Effectue une recherche d'hyperparamètres à l'aide de la validation croisée.

        Paramètres :
        - param_grid : Dictionnaire des hyperparamètres à tester.
        - cv : Le nombre de plis pour la validation croisée (par défaut, 5).
        - scoring : La métrique de performance à évaluer (par défaut, 'accuracy').

        Output :
        - best_params : Les meilleurs hyperparamètres trouvés.
        """
        if self.model_type == 'lda':
            self.train_model()
            best_score = accuracy_score(self.y_train, self.model.predict(self.X_train))  
            print(f"Score '{scoring}' : {best_score*100} %")
            
        else : 
            print("Grid Search Hyperparameters : ")
            grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring=scoring)
            grid_search.fit(self.X_train, self.y_train)

            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            print(f"Meilleurs Hyperparametres trouvés: {best_params}")
            print(f"Score '{scoring}' : {best_score*100} %")

            # Mise à jour du modèle avec les meilleurs paramètres trouvés
            self.model = grid_search.best_estimator_


    def evaluate_model(self, X_test, y_test):
        """
        Évalue le modèle sur l'ensemble de test.

        Parameters:
        - X_test, y_test: Données de test.
        """
        y_pred = self.model.predict(X_test)
     
        # Enregistre l'accuracy du model 
        self.model_accuracy = accuracy_score(y_test, y_pred)*100

        # Affiche les métriques
        print(f"Test Accuracy: {self.model_accuracy} %")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        # Listes pour stocker l'accuracy à chaque epoch
        train_accuracy = []
        test_accuracy = []

        # Si le modèle est MLP, affiche la courbe de perte et d'accuracy
        if self.model_type == 'mlp':
            if hasattr(self.model, 'loss_curve_'):
                train_loss = self.model.loss_curve_
                test_loss = []

                # Calcul de la loss sur l'ensemble de test
                for epoch in range(len(train_loss)):
                    
                    self.model.partial_fit(X_test, y_test, classes=np.unique(y_test))
                    # self.model.partial_fit(self.X_train, self.y_train, classes=np.unique(self.y_train))

                    # Prédiction sur l'ensemble de test
                    y_test_pred_proba = self.model.predict_proba(X_test)
                    
                    # Calcul de la perte sur l'ensemble de validation
                    test_loss.append(log_loss(y_test, y_test_pred_proba))

                    # Calcul de l'accuracy sur l'ensemble d'entraînement et de test
                    #self.model.score(self.X_train, self.y_train)
                    train_accuracy.append(accuracy_score(self.y_train, self.model.predict(self.X_train)))
                    test_accuracy.append(accuracy_score(y_test, self.model.predict(X_test)))

                # Affichage de la courbe de perte
                plt.figure()

                plt.plot(train_loss, label="train")
                plt.plot(test_loss, label='test')
                plt.title('Courbe de la "loss"')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()
                

                plt.figure()
                plt.plot(train_accuracy, label="train")
                plt.plot(test_accuracy, label='test')
                plt.title('Courbe de l\'accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()

        # Affiche la matrice de confusion pour tous les modèles
        self.plot_confusion_matrix(y_test, y_pred, classes=np.unique(y_test))


    def plot_confusion_matrix(self, y_true, y_pred, classes, normalize=False, cmap='Blues'):
        """
        Affiche la matrice de confusion.

        Parameters:
        - y_true: Les vraies étiquettes.
        - y_pred: Les étiquettes prédites par le modèle.
        - classes: La liste des classes (étiquettes).
        - normalize: Si True, normalise les valeurs par ligne.
        - cmap: Colormap pour la visualisation (par défaut, 'Blues').
        """
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else 'd', cmap=cmap,
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()


    def save_model(self, filename):
        """
        Sauvegarde le modèle entraîné.

        Input : Nom du fichier.

        Output : Modèle sauvegardé.

        """
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)