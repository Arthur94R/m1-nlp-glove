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
│   ├── main.py                          → Word2Vec (TP1)
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
  romance         : 0.735
  affection       : 0.741
  madly           : 0.730

GloVe :
  romance         : 0.820
  passion         : 0.798
  heart           : 0.765
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
Corrélation W2V vs GloVe : 0.72
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

## 📈 Analyse comparative

### Points communs
- Les deux capturent la similarité sémantique
- Mots similaires ont des vecteurs proches
- Corrélation généralement > 0.6

### Différences
- **Word2Vec** : Meilleur sur le contexte immédiat et syntaxe
- **GloVe** : Meilleur sur les relations sémantiques globales et analogies
- **W2V** : Plus rapide à entraîner
- **GloVe** : Plus stable (déterministe)

## 📝 Livrables

- ✅ Code source (Word2Vec, GloVe, comparaison)
- ✅ Embeddings entraînés
- ✅ Visualisations comparatives
- ✅ Analyse des corrélations
- ✅ README

## 🔗 Lien avec TP1

Ce TP2 étend le TP1 en :
- Ajoutant une deuxième méthode d'embeddings (GloVe)
- Comparant systématiquement les résultats
- Analysant les forces/faiblesses de chaque approche

## 📚 Références

- **Word2Vec** : Mikolov et al. (2013) - "Efficient Estimation of Word Representations in Vector Space"
- **GloVe** : Pennington et al. (2014) - "GloVe: Global Vectors for Word Representation"
- **t-SNE** : van der Maaten & Hinton (2008) - "Visualizing Data using t-SNE"

## 💡 Observations typiques

**Corrélation élevée (r > 0.7) :**
- Les deux modèles capturent des informations similaires
- Différences subtiles dans l'organisation de l'espace

**Corrélation moyenne (r ~ 0.5) :**
- Approches complémentaires
- GloVe capture mieux certaines relations globales
- W2V capture mieux le contexte local

**Cas d'usage :**
- **Word2Vec** : Classification de texte, analyse de sentiment
- **GloVe** : Analogies, relations sémantiques complexes