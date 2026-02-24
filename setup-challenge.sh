#!/bin/bash

# Nom du dossier racine
ROOT_DIR="CHALLENGE 30 DAYS ML"
mkdir -p $ROOT_DIR
cd $ROOT_DIR

# Liste des thèmes par jour
days=(
"Linear-Algebra-Foundations" "Calculus-for-ML" "Simple-Linear-Regression" "Multiple-Linear-Regression" "Regularization-Ridge-Lasso"
"Logistic-Regression-Binary" "Softmax-Regression-Multiclass" "Decision-Trees-ID3-CART" "Tree-Pruning-Optimization" "Random-Forest-Bagging"
"AdaBoost-From-Scratch" "Gradient-Boosting-Machine" "XGBoost-LightGBM-HandsOn" "Feature-Importance-SHAP" "k-Nearest-Neighbors"
"SVM-Linear-Classification" "SVM-Kernel-Trick" "Naive-Bayes-Classifier" "Gaussian-Mixture-Models" "Linear-Discriminant-Analysis"
"Model-Evaluation-Metrics" "K-Means-Clustering" "Hierarchical-Clustering" "Principal-Component-Analysis" "t-SNE-UMAP-Visualization"
"The-Perceptron-Model" "Multi-Layer-Perceptron-MLP" "Backpropagation-Chain-Rule" "Optimizers-Adam-RMSProp" "Deployment-FastAPI-Summary"
)

echo "🚀 Création de l'architecture du challenge..."

# Boucle pour créer les dossiers et les fichiers
for i in "${!days[@]}"; do
    day_num=$(printf "%02d" $((i+1)))
    dir_name="Day-${day_num}-${days[$i]}"
    
    mkdir -p "$dir_name/data" "$dir_name/images"
    
    # Création d'un notebook vide avec un titre et une structure de base
    cat <<EOF > "$dir_name/notebook.ipynb"
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Day ${day_num} : ${days[$i]}\\n",
    "## 🎯 Objectif du jour\\n",
    "Expliquer les mathématiques derrière **${days[$i]}** et implémenter un projet pratique.\\n",
    "\\n",
    "--- \\n",
    "### 1. Théorie Mathématique\\n",
    "*(Équations en LaTeX ici)*\\n",
    "\\n",
    "### 2. Implémentation From Scratch (Numpy)\\n",
    "\\n",
    "### 3. Application avec Scikit-Learn\\n",
    "\\n",
    "### 4. Conclusion & Visualisation"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

    echo "✅ Créé : $dir_name"
done

# Création du README global
touch README.md
echo "# 30 Days ML Math Challenge" >> README.md
echo "Mon dépôt pour le challenge de 30 jours sur les mathématiques du Machine Learning." >> README.md

echo "🏁 Terminé ! Ton environnement est prêt."