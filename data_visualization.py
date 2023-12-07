import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import auc
import itertools

class Visualization:
    def __init__(self, data):
        self.data = data
        self.cible = "output"

class Visualization:
    def __init__(self, data):
        self.data = data
        self.cible = "HeartDisease"

    def plot_distributions(self, is_numeric=True, variables=None):
        """
        Affiche les distributions des variables spécifiées dans un subplot.

        Input :
        - is_numeric : True si vous voulez visualiser les variables numériques, False pour les variables catégorielles.
        - variables : Liste d'indices des variables à visualiser.

        Output : Graphiques de distribution.
        """
        num_cols = len(variables) if variables else len(self.data.columns)
        num_rows = (num_cols // 3) + (num_cols % 3)  # Calcul du nombre de lignes en fonction du nombre de colonnes
        plt.figure(figsize=(15, 5 * num_rows))


        if variables:
            columns_to_plot = [self.data.columns[i] for i in variables]
        else:
            columns_to_plot = self.data.columns

        palette = sns.color_palette("Set1", n_colors=len(variables))

        for i, col in enumerate(columns_to_plot, 1):
            if is_numeric :
                plt.subplot(num_rows, 3, i)
                sns.histplot(self.data[col], color = palette[i-1], stat="percent", kde=True)
                plt.xlabel(col)
                plt.ylabel("Pourcentage")
                plt.title(f"Distribution de {col}")
                plt.tight_layout()
            else :
                plt.subplot(num_rows, 3, i)
                sns.countplot(data=self.data, x=col, color = palette[i-1], stat="percent")
                plt.title(f"Distribution de {col}")
                plt.xlabel(col)
                plt.ylabel("Pourcentage")
                plt.tight_layout()

        # Afficher le graphique
        plt.show()


    def plot_correlation(self):
        """
        Affiche la matrice de corrélation.

        Output : Heatmap de la matrice de corrélation.

        """
        # Matrice de corrélation
        corr_matrix = self.data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)

        # Ajouter un titre à la figure
        plt.title('Matrice de corrélation')

        # Afficher le graphique
        plt.show()

    def plot_roc_curve(self, model, x_test, y_test):
        """
        Affiche la courbe ROC pour évaluer la performance du modèle.

        Input : Modèle entraîné, X_test, y_test.

        Output : Courbe ROC.

        """
        # Prédiction des probabilités
        y_pred_prob = model.predict_proba(x_test)[:, 1]

        # Calcul des taux de faux positifs et vrais positifs
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

        # Calcul de l'aire sous la courbe ROC (AUC)
        roc_auc = auc(fpr, tpr)

        # Tracé de la courbe ROC
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title('Courbe ROC')
        plt.legend(loc="lower right")
        plt.show()


    def plot_confusion_matrix(self, model, x_test, y_test):
        """
        Affiche la matrice de confusion pour évaluer la performance du modèle.

        Input : Modèle entraîné, X_test, y_test.

        Output : Matrice de confusion.

        """
        # Prédiction des classes
        y_pred = model.predict(x_test)

        # Calcul de la matrice de confusion
        cm = confusion_matrix(y_test, y_pred)

        # Plot de la matrice de confusion
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5, cbar=False)
        plt.xlabel('Prédiction')
        plt.ylabel('Vraie valeur')
        plt.title('Matrice de confusion')
        plt.show()
    

    def plot_distribution_hue(self, variable, hue, is_numeric):
        """
        Affiche la distribution d'une variable en fonction par rapport à une autre (hue).

        Input :
        - variable : Nom de la variable.
        - hue : Nom de la variable à utiliser comme teinte.
        - is_numeric : Type de la variable (True si numérique False si catégorique).

        Output : Graphique de distribution.
        """
        if is_numeric :
            # Variable numérique
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.data, x=variable, hue=hue, multiple="stack", stat="percent", kde=False)
            plt.title(f"Distribution de {variable} par rapport {hue}")
            plt.xlabel(variable)
            plt.ylabel("Pourcentage")
            plt.show()
        else:
            # Variable catégorique
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.data, x=variable, hue=hue, stat="percent")
            plt.title(f"Distribution de {variable} par rapport {hue}")
            plt.xlabel(variable)
            plt.ylabel("Pourcentage")
            plt.show()


    def plot_boxplot(self, variables, y):
       """
        Affiche les boxplots des variables spécifiées dans un subplot.

        Input :
        - variables : Liste d'indices des variables à visualiser.
        - y : nom de la variable en abscisse (variable par rapport à laquelle on visualise le boxplot)

        Output : Boxplots des variables spécifiées.
        """
       num_cols = len(variables) if variables else len(self.data.columns)
       num_rows = (num_cols // 3) + (num_cols % 3)  # Calcul du nombre de lignes en fonction du nombre de colonnes
       plt.figure(figsize=(15, 5 * num_rows))
       if variables:
            columns_to_plot = [self.data.columns[i] for i in variables]
       else:
        columns_to_plot = self.data.columns

       for i, col in enumerate(columns_to_plot, 1):
            plt.subplot(num_rows, 3, i)
            sns.boxplot(data=self.data, x=y, y=col, hue=y)
            plt.title(f"Boxplot de {col}")
            plt.xlabel(y)
            plt.ylabel(col)
            plt.tight_layout()

       # Afficher le graphique
       plt.show()
       


       
        