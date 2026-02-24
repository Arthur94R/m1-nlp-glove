import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')

print("="*60)
print("COMPARAISON WORD2VEC vs GLOVE")
print("="*60 + "\n")

# ===========================
# 1. CHARGEMENT DES MODÈLES
# ===========================
print("1. Chargement des embeddings...\n")

# Word2Vec
w2v_model = Word2Vec.load(os.path.join(DATA_DIR, 'word2vec_films.bin'))
print(f"   Word2Vec : {len(w2v_model.wv)} mots, {w2v_model.wv.vector_size} dimensions")

# GloVe
glove_embeddings = np.load(os.path.join(DATA_DIR, 'glove_embeddings_films.npy'))
glove_vocab = np.load(os.path.join(DATA_DIR, 'glove_vocab_films.npy'), allow_pickle=True)
glove_vocab_to_ix = {word: ix for ix, word in enumerate(glove_vocab)}
print(f"   GloVe    : {len(glove_vocab)} mots, {glove_embeddings.shape[1]} dimensions\n")

# ===========================
# 2. FONCTION SIMILARITÉ GLOVE
# ===========================
def most_similar_glove(word, top_n=5):
    if word not in glove_vocab_to_ix:
        return []
    
    word_ix = glove_vocab_to_ix[word]
    word_vec = glove_embeddings[word_ix]
    
    # Normaliser
    word_vec_norm = word_vec / np.linalg.norm(word_vec)
    all_vecs_norm = glove_embeddings / np.linalg.norm(glove_embeddings, axis=1, keepdims=True)
    
    # Similarité cosinus
    similarities = np.dot(all_vecs_norm, word_vec_norm)
    
    # Top N (exclure le mot lui-même)
    most_similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    
    results = [(glove_vocab[ix], similarities[ix]) for ix in most_similar_indices]
    return results

# ===========================
# 3. COMPARAISON MOTS SIMILAIRES
# ===========================
print("2. Comparaison des mots similaires\n")

test_words = ['love', 'action', 'hero', 'family', 'world']

for word in test_words:
    print(f"=== '{word.upper()}' ===\n")
    
    # Word2Vec
    if word in w2v_model.wv:
        w2v_similar = w2v_model.wv.most_similar(word, topn=5)
        print("   Word2Vec :")
        for sim_word, score in w2v_similar:
            print(f"      {sim_word:15s} : {score:.3f}")
    else:
        print("   Word2Vec : mot non trouvé")
    
    print()
    
    # GloVe
    glove_similar = most_similar_glove(word, top_n=5)
    if glove_similar:
        print("   GloVe :")
        for sim_word, score in glove_similar:
            print(f"      {sim_word:15s} : {score:.3f}")
    else:
        print("   GloVe : mot non trouvé")
    
    print("\n" + "-"*60 + "\n")

# ===========================
# 4. VISUALISATION COMPARATIVE
# ===========================
print("3. Visualisation comparative t-SNE...\n")

# Sélectionner des mots communs aux deux modèles
common_words = []
for word in test_words + ['story', 'life', 'find', 'young', 'city', 'night', 
                           'death', 'power', 'war', 'evil', 'kill', 'fight']:
    if word in w2v_model.wv and word in glove_vocab_to_ix:
        common_words.append(word)

print(f"   Mots communs pour visualisation : {len(common_words)}")

# Récupérer les vecteurs
w2v_vectors = np.array([w2v_model.wv[word] for word in common_words])
glove_vectors = np.array([glove_embeddings[glove_vocab_to_ix[word]] for word in common_words])

# t-SNE
print("   Application t-SNE sur Word2Vec...")
w2v_tsne = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(w2v_vectors)

print("   Application t-SNE sur GloVe...")
glove_tsne = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(glove_vectors)

# Plot côte à côte
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Word2Vec
for i, word in enumerate(common_words):
    x, y = w2v_tsne[i]
    axes[0].scatter(x, y, s=100, alpha=0.7, c='blue')
    axes[0].text(x + 0.5, y + 0.5, word, fontsize=11)

axes[0].set_title('Word2Vec Embeddings (t-SNE)', fontweight='bold', fontsize=14)
axes[0].set_xlabel('Dimension 1')
axes[0].set_ylabel('Dimension 2')
axes[0].grid(True, alpha=0.3)

# GloVe
for i, word in enumerate(common_words):
    x, y = glove_tsne[i]
    axes[1].scatter(x, y, s=100, alpha=0.7, c='green')
    axes[1].text(x + 0.5, y + 0.5, word, fontsize=11)

axes[1].set_title('GloVe Embeddings (t-SNE)', fontweight='bold', fontsize=14)
axes[1].set_xlabel('Dimension 1')
axes[1].set_ylabel('Dimension 2')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'w2v_vs_glove_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Graphique sauvegardé : results/w2v_vs_glove_comparison.png\n")

# ===========================
# 5. DISTANCE ENTRE EMBEDDINGS
# ===========================
print("4. Analyse des distances vectorielles\n")

# Calculer la corrélation entre les espaces vectoriels
# Pour chaque paire de mots, comparer les similarités W2V vs GloVe
similarities_w2v = []
similarities_glove = []

for i, word1 in enumerate(common_words[:10]):
    for word2 in common_words[:10]:
        if word1 != word2:
            # Word2Vec
            sim_w2v = w2v_model.wv.similarity(word1, word2)
            similarities_w2v.append(sim_w2v)
            
            # GloVe
            vec1 = glove_embeddings[glove_vocab_to_ix[word1]]
            vec2 = glove_embeddings[glove_vocab_to_ix[word2]]
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            sim_glove = np.dot(vec1_norm, vec2_norm)
            similarities_glove.append(sim_glove)

# Corrélation
correlation = np.corrcoef(similarities_w2v, similarities_glove)[0, 1]
print(f"   Corrélation entre W2V et GloVe : {correlation:.3f}")
print(f"   (1.0 = parfaitement corrélés, 0.0 = pas de corrélation)\n")

# Scatter plot des similarités
plt.figure(figsize=(8, 8))
plt.scatter(similarities_w2v, similarities_glove, alpha=0.6, s=30)
plt.plot([0, 1], [0, 1], 'r--', label='Parfaite corrélation', linewidth=2)
plt.xlabel('Similarité Word2Vec', fontsize=12)
plt.ylabel('Similarité GloVe', fontsize=12)
plt.title(f'Corrélation W2V vs GloVe (r={correlation:.3f})', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'similarity_correlation.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Graphique corrélation sauvegardé : results/similarity_correlation.png\n")

# ===========================
# 6. RÉSUMÉ
# ===========================
print("="*60)
print("RÉSUMÉ DE LA COMPARAISON")
print("="*60 + "\n")

print("OBSERVATIONS :")
print("  • Word2Vec et GloVe capturent des informations sémantiques similaires")
print("  • Certains mots sont mieux représentés par l'un ou l'autre")
print(f"  • Corrélation : {correlation:.3f} (assez corrélés mais pas identiques)")
print("\nDIFFÉRENCES :")
print("  • Word2Vec : apprentissage local (fenêtre glissante)")
print("  • GloVe     : statistiques globales (co-occurrences)")
print("\nFICHIERS GÉNÉRÉS :")
print("  • results/w2v_vs_glove_comparison.png")
print("  • results/similarity_correlation.png")

print("\n" + "="*60)
print("COMPARAISON TERMINÉE")
print("="*60)
