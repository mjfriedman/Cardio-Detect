import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


class Visualization:
    def __init__(self, data):
        self.data = data

    def plot_distribution(self, data, feature):
        """
        Affiche la distribution d'une caractéristique.

        Input : DataFrame des données, nom de la caractéristique.

        Output : Graphique de distribution.

        """
        #  À FAIRE : Utilisez des histogrammes avec seaborn.

    def plot_correlation(self, data):
        """
        Affiche la matrice de corrélation.

        Input : DataFrame des données.

        Output : Heatmap de la matrice de corrélation.

        """
        #  À FAIRE : Utilisez seaborn

    def plot_roc_curve(self, model, x_test, y_test):
        """
        Affiche la courbe ROC pour évaluer la performance du modèle.

        Input : Modèle entraîné, X_test, y_test.

        Output : Courbe ROC.

        """
        # À FAIRE : Utilisez scikit learn pour calculer et tracer la courbe ROC.

    def plot_confusion_matrix(self, model, x_test, y_test):
        """
        Affiche la matrice de confusion pour évaluer la performance du modèle.

        Input : Modèle entraîné, X_test, y_test.

        Output : Matrice de confusion.

        """
        # À FAIRE : Utilisez scikit learn pour calculer et tracer la matrice de confusion.

    def distribution_par_var_cible(self, df):
        """
        Affiche la distribution d'une variable par rapport à la cible.

        Input : dataframe df, variable var

        Output : distribution la variable var par rapport à la cible.

        """
        plt.show()
        #  À FAIRE : Utiliser seaborn

    def distribution_var(self, df, var):
        """
        Affiche la distribution d'une variable "var" du jeu de données.

        Input : dataframe df, variable var

        Output : distribution (histogramme) d'une variable du jeu de données

        """
        plt.show()
        #  À FAIRE : Utiliser seaborn

    def boxplot(self, df, var):
        """
        Affiche un boxplot pour une variable

        Input : dataframe df, variable var

        Output : boxplot d'une variable du jeu de données
        """
        plt.show()
        #  À FAIRE : Utiliser seaborn
