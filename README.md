Projet GitHub : https://github.com/Arthur94R/m1-nlp-word2vec

# 🎬 TP2 — Construction embeddings avec GloVe & comparaison avec Word2Vec (TP1)

Projet universitaire — Master 1 IA & Big Data, Université Paris 8

## 📋 Description

Comparaison de deux méthodes d'embeddings sur le dataset de films :
- **Word2Vec** : Approche prédictive (fenêtre glissante, Skip-gram)
- **GloVe** : Approche statistique (matrice de co-occurrence globale)

L'objectif est d'analyser les différences entre ces deux techniques et de visualiser comment elles capturent le sens sémantique des mots.

## 🎯 Objectifs

- Entraîner Word2Vec et GloVe sur le même corpus
- Comparer les mots similaires trouvés par chaque méthode
- Visualiser les embeddings avec t-SNE
- Mesurer la corrélation entre les deux espaces vectoriels

## 🔍 Différences clés

| Aspect | Word2Vec | GloVe |
|--------|----------|-------|
| **Approche** | Prédictive (locale) | Statistique (globale) |
| **Principe** | Prédit le contexte à partir d'un mot | Factorisation de la matrice de co-occurrence |
| **Focus** | Fenêtre glissante (contexte immédiat) | Statistiques globales du corpus entier |
| **Méthode** | Réseau de neurones (Skip-gram) | Optimisation : `vecteur1 · vecteur2 ≈ log(co_occ)` |

## 🛠️ Stack technique

- **Python 3.13** — Langage principal
- **Gensim** — Word2Vec
- **TensorFlow** — Entraînement GloVe
- **NLTK** — Preprocessing
- **Scikit-learn** — t-SNE
- **Pandas / NumPy** — Traitement des données
- **Matplotlib** — Visualisations

## 📁 Structure

```
TP2_Word2Vec_vs_GloVe/
├── data/
│   ├── movies_metadata.csv              → Dataset films (à télécharger)
│   ├── word2vec_films.bin               → Modèle W2V entraîné
│   ├── glove_embeddings_films.npy       → Vecteurs GloVe
│   └── glove_vocab_films.npy            → Vocabulaire GloVe
├── src/
│   ├── glove_films.py                   → Entraînement GloVe
│   └── compare_embeddings.py            → Comparaison W2V vs GloVe
└── results/
    ├── glove_tsne.png                   → Visualisation GloVe
    ├── w2v_vs_glove_comparison.png      → Comparaison t-SNE
    └── similarity_correlation.png       → Corrélation des similarités
```

## 📥 Données

**Dataset à télécharger :**
- `movies_metadata.csv` → [Kaggle - The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Placer dans le dossier `data/`.

## 🚀 Installation et lancement

### Installation
```bash
# Installer les dépendances
pip install pandas numpy matplotlib scikit-learn gensim nltk tensorflow

# Télécharger ressources NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Lancement

**Étape 1 : GloVe**
```bash
python src/glove_films.py
# Durée : ~5-10 minutes
# Génère : glove_embeddings_films.npy, glove_vocab_films.npy
```

**Étape 2 : Comparaison**
```bash
python src/compare_embeddings.py
# Génère les graphiques de comparaison
```

## 📊 Résultats attendus

### 1. Mots similaires

**Exemple pour "love" :**

```
Word2Vec :
  madly           : 0.675
  headoverheels   : 0.647
  nandini         : 0.639
  blossom         : 0.631
  alraune         : 0.629

GloVe :
  finds           : 0.995
  find            : 0.995
  family          : 0.994
  world           : 0.993
  must            : 0.991
```

**Observation :** Certains voisins sont communs, d'autres diffèrent.

### 2. Visualisation t-SNE

Deux graphiques côte à côte montrant l'organisation des mots dans chaque espace vectoriel.

**Différences possibles :**
- Clusters différents
- Distances relatives modifiées
- Certains mots mieux séparés dans un modèle

### 3. Corrélation

```
Corrélation W2V vs GloVe : 0.334
```

**Interprétation :**
- **r > 0.7** → Modèles assez corrélés (capturent des infos similaires)
- **r ~ 0.5** → Différences notables
- **r < 0.3** → Très différents

## 🎓 Concepts clés

### Word2Vec (Skip-gram)
- **Principe** : Prédit les mots du contexte à partir d'un mot central
- **Apprentissage** : Réseau de neurones avec negative sampling
- **Avantage** : Rapide, capture bien le contexte local

### GloVe (Global Vectors)
- **Principe** : Factorise la matrice de co-occurrence globale
- **Objectif** : `vecteur(mot1) · vecteur(mot2) ≈ log(co_occurrence)`
- **Avantage** : Capture les statistiques globales du corpus

### Matrice de co-occurrence
Compte combien de fois deux mots apparaissent ensemble dans une fenêtre :
```
"I love romantic comedy films"

Co-occurrence (window=5) :
  (love, romantic) : 1.0
  (love, comedy)   : 0.5
  (love, I)        : 1.0
  ...
```

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
Réduit les dimensions (100D → 2D) en préservant les distances relatives pour visualisation.