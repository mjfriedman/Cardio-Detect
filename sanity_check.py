from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def test_svm_classifier(X_train, y_train, X_test, y_test):
    model = SVC()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accu = accuracy_score(y_test, pred)
    return accu


def mlp_sanity_check(X_train, y_train, num_classes):
    """
    Verifie si une initialisation aléaotoire des poids dans un modèle MLPClassifier donne une loss maximale.
    """
    # Génération du modèle avec des poids W aléatoires
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, random_state=42)

    # Entraînement itératif pour une seule époque
    mlp_model.fit(X_train, y_train)

    # Prédiction sur l'ensemble d'entraînement
    y_pred_proba = mlp_model.predict_proba(X_train)

    # Calcul de la perte
    loss = log_loss(y_train, y_pred_proba)

    # Comparaison avec le résultat attendu
    loss_expected = -np.log(1.0 / num_classes)  # Résultat aléatoire attendu

    # Affichage des résultats
    print(f'Loss après initialisation aléatoire: {loss:.5f}')
    print(f'Loss attendue: {loss_expected:.5f}')

    # Vérification de la validité du résultat
    if abs(loss - loss_expected) > 0.05:
        print('ERREUR: la sortie de la fonction est incorrecte.')
    else:
        print('SUCCÈS')



def overfitting_sanity_check(model_name, X_train, y_train, n_check=5):
    """
    Teste si un modèle sklearn peut overfitter sur un petit jeu de données.

    Parameters:
    - model_name: Nom du modèle à tester (ex: 'svm', 'random_forest', ...).
    - X_train, y_train: Données d'entraînement.
    - X_val, y_val: Données de validation.
    - n_check: Nombre d'échantillons à utiliser pour le test (par défaut, 5).
    """
    # Sélection du modèle
    if model_name == 'svm':
        model = SVC()
    elif model_name == 'rf':
        model = RandomForestClassifier()
    elif model_name == 'lr':
        model = LogisticRegression()
    elif model_name == 'adaboost':
        model = AdaBoostClassifier()
    elif model_name == 'mlp':
        model= MLPClassifier()
    elif model_name == 'lda':
        model = LinearDiscriminantAnalysis()

    # Sélection des échantillons pour le test
    X_check = X_train[:n_check]
    y_check = y_train[:n_check]

    # Entraînement du modèle sur un petit jeu de données
  
    model.fit(X_check, y_check)

    # Calcul de l'accuracy
    accu_train_finale = accuracy_score(y_check, model.predict(X_check))

    # Affichage des résultats
    print(f'Accuracy d\'entraînement, devrait être 1.0: {accu_train_finale:.3f}')
    
    if accu_train_finale < 0.9999:
        print('ATTENTION: L\'accuracy d\'entraînement n\'est pas 100%.')
    else:
        print('SUCCÈS')


