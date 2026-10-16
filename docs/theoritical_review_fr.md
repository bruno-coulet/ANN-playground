# Perceptron Multicouches (PMC)

## 1. Intoduction

Le Perceptron Multicouches (PMC), également appelé Multilayer Perceptron (MLP), est un réseau de neurones artificiels composé de plusieurs couches de neurones interconnectés. Il constitue l’un des fondements historiques du Deep Learning moderne.

Contrairement au perceptron simple, limité aux problèmes linéairement séparables, le PMC est capable d’apprendre des relations complexes et non linéaires grâce à l’utilisation de plusieurs couches cachées et de fonctions d’activation.

Les PMC sont utilisés pour des tâches de classification, de régression, de reconnaissance de motifs et de prédiction.

## 2. Architecture du PMC

Un Perceptron Multicouches est composé de plusieurs couches de neurones organisées de manière séquentielle.

Entrée → Couche cachée(s) → Sortie

Chaque neurone d'une couche est généralement connecté à tous les neurones de la couche suivante.

### 2.1 Couche d'entrée

La couche d'entrée reçoit les caractéristiques (features) du jeu de données.

Exemple :

    * âge
    * moyenne générale
    * nombre d'absences

Chaque caractéristique correspond à un neurone d'entrée.

### 2.3 Couches cachées

Les couches cachées réalisent les transformations intermédiaires permettant au réseau d'apprendre des relations complexes.

Chaque neurone calcule :

z = Wx + b

puis applique une fonction d'activation.

### 2.4 Couche de sortie

La couche de sortie produit la prédiction finale.

Classification binaire : 1 neurone + Sigmoid

Classification multiclasse : N neurones + Softmax

Régression : 1 neurone linéaire

## 3. PMC pour la calssification et la régression 

Le choix de l'architecture dépend du type de problème.

### 3.1 Classification 

La classification consiste à prédire une catégorie. 

Exemple : 

    * Spam / Non Spam
    * Résussite / Échec
    / Chat / Chien 

Fonctions généralement utilisées : 

    * ReLU dans les couches cachées
    * Softmax ou Sigmoid en sortie

Loss : 

    * Binary Cross Entropy
    * Categorical Cross Entropy 

### 3.2 Régression 

La régression consiste à prédure une valeure numérique continue. 

Exemples : 

    * Prix immobilier
    * Température
    * Chiffre d'affaires

Fonction de sortie : 
    
    * Linéraire

Loss : 

    * Mean Squared Error

## 4. Fonciton d'activation 

Les fonctions d'activation permettent d'introduire de la non-linéarité dans le réseau de neurones.
Sans elles, un PMC composé de plusieurs couches serait équivalent à une simple régression linéaire.

Elles déterminent comment les informations sont transformées et propagées à travers le réseau.

### 4.1 Sigmoid

La fonction Sigmoid transforme une valeur réelle en une valeur comprise entre 0 et 1. 

f(x) = 1 / (1 +e^(-x))

Avantages : 

    * Facile à interpréter comme une probabilité.
    * Favorise la phénomène de vanishing gradient. 

### 4.2 Tanh

La foncton Tangente Hyperbolique retourne une valeur comprise entre -1 et 1.

Avantages : 

    * Sorties centrées autour de zéro.
    * Convergence souvent meilleur que Sigmoid.

Incovénients : 

    * Toujours senseible aux vanishig gradient. 

### 4.3 ReLU

Rectified Linear Unit : 

f(x) = max(0,x)

Avantages :

    * Trs raoude à calculer.
    * Réduit le vanishing gradient. 
    * Fcontion la plus utilisée aujourdh'ui.

Inconvénients : 

    * Risque de neurones morts lorsque les sorties restent constamments négatives. 

### 4.4 Leaky ReLU

Variante de ReLU permettant une faible pente pour les valeurs négatives afin de réduire le phénomène de neurones morts.

### 4.5 Softmax

La fonction Sostmax converti un vectur de valeurs en probabilité.

La somme de sprobabilités est égale à 1.

ELle est principalement utilisée dans les problèmes de calssification multicalsse.

## 5. Propagation avant
La propagation avant consiste à faire circuler les données d'entrées à travers les différentes couches du réseau afind 'obtenir une prédiction.

Pour cahque neurone : 

z= W.x + b

où :

    * W représente les poids
    * x représentes les entrées
    * b représente le biais

Une fonction d'actiaition est ensuite appliquée : 

a = f(z)

Le résultat est transmis à la couche suivante jusqu'a la couche de sortie. 

Exemple : 

Entrée : 

    * âge
    * absences
    * moyenne

    ↓

    Couche cachée

    ↓ 

    Couche cachée

    ↓ 

    Prédiction : Dropout / Graduate / Enrolled

## 6. Rétropropagation

La rétropropagation (Backpropagation) est l'algorithme permettant d'ajuster les points du réseau afin de minimiser l'erreur. 

Le princie est le suivant : 

    * 1. Calcul de la prédiciton 
    * 2. Calcul de l'erreur
    * 3. Propagation de cette erreur vers les couches précédentes
    * 4. Mise à jour des poids

Le calcul utilise la règle de dérivaiton en châine (Chain Rule) du calcul différentiel. 

L'objectif étant de déterminer la contribution de cahque poids à l'erreur finale.

Sans rétropropagation, le réseau serait incapable d'apprendre à partir des données.

Elle constiture le mécanisme fondamental d'apprentissage des réseaux de neurones modernes.

## 7. Fonction de perte

La fonction de perte mesure l'écart entre la prédiction du modèle et la valeur réelle.

### 7.1 Binary Cross Entropy

Utilisée pour la calssification binaire. 

Exemple : 

Réussi / Échec

### 7.2 Categorical Cross Entropy

Utilisée pour la classification multiclasse.

Exemple : 

    * Dropout
    * Enrolled
    * Graduate

### 7.3 Mean Squared Error (MSE)

Utilisée pour la régression, elle calcule la moyenne des carrés.

## 8. Descente de gradient
La descente de gradient est l'algorithme d'optimisation permettant de minimiser la fonction de perte. 

Après chaque itération, les poids sont modifiés dans la direction opposée au gradient. 

Nouveau poids : 

W = W - η * gradient

où η représente le learning rate. 

### 8.1 Impacte du leraning rate

Un learning rate trop faible : 

    * Aprentissage très lent

Un learning rate trop élevé : 

    * Oscillations
    * Divergence possible

Valeur classique : 

    * 0.1
    * 0.01
    * 0.001

## 9. Vanishing Gradient

Le phénomène de vanishing gradient apparâit lorsque les gradients deviennent extêmement faibles lors de la rétropropagation. 

Les couches proches de l'entrée reçoivent alors de smises à jour quasi nulles. 

Conséquences : 

    * Apprentissage très lent
    * Blocage de l'entrainement 
    * Difficultés à entrainer des réseaux profonds

Causes : 

Les princiales causes sont :

    * Fonction Sigmoid
    * Fonction Tanh
    * Réseaux 

Solutions : 

Les approches modernes utilisent : 

    * ReLU
    * Leaky ReLU
    * Bacth Normalization
    * Initialisation adaptée des poids

## 10. Hyperparamètres

Les hyperparamètres sont des paramètres défnisis avant l'entrainement.

### 10.1 Nombre de couches cachées

Le nombre de couches cachées à un impact non négligeable : 

    * Plus de couches = plus de capacité d'apprentissage
    * Trop de couches = risque d'overfitting.

Bonnes pratique : 

1 à 3 couches

### 10.1 Nombre de neurones

...


### 10.2 Leanrning Rate

...

### 10.3 Bacth Size

...

### 10.4 Nombre d'Epochs

...

### 10.5 Fonction d'activation 

...

### 10.6 Dropout

...

### 10.7 Optimisateur
...

### ..... autres hyperparametres ? ...



## 11. Bonnes pratiques

## 12. Conclusion 