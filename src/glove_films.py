import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.manifold import TSNE
import tensorflow as tf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*60)
print("GLOVE - DATASET FILMS")
print("="*60 + "\n")

# ===========================
# 1. PREPROCESSING
# ===========================
print("1. Preprocessing...")
df = pd.read_csv(os.path.join(DATA_DIR, 'movies_metadata.csv'), low_memory=False)
df = df[df['overview'].notna()].copy()

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    tokens = word_tokenize(text)
    return [w for w in tokens if w not in stop_words and len(w) > 2]

# Échantillon
df_sample = df.head(5000)
df_sample['tokens'] = df_sample['overview'].apply(preprocess)

# Vocabulaire
all_tokens = [t for tokens in df_sample['tokens'] for t in tokens]
token_counts = Counter(all_tokens)
vocab = [w for w, c in token_counts.items() if c >= 5]
vocab_to_ix = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print(f"   Vocabulaire : {vocab_size:,} mots\n")

# ===========================
# 2. MATRICE CO-OCCURRENCE
# ===========================
print("2. Matrice de co-occurrence...")

WINDOW_SIZE = 5
co_occurrence = defaultdict(lambda: defaultdict(int))

for tokens in tqdm(df_sample['tokens'], desc="   Processing"):
    valid_tokens = [t for t in tokens if t in vocab_to_ix]
    for i, center in enumerate(valid_tokens):
        start = max(0, i - WINDOW_SIZE)
        end = min(len(valid_tokens), i + WINDOW_SIZE + 1)
        for j in range(start, end):
            if i != j:
                context = valid_tokens[j]
                co_occurrence[center][context] += 1.0 / abs(i - j)

# Format pour TensorFlow
co_data = []
for center, contexts in co_occurrence.items():
    for context, count in contexts.items():
        co_data.append((vocab_to_ix[center], vocab_to_ix[context], count))

center_ix = np.array([x[0] for x in co_data], dtype=np.int32)
context_ix = np.array([x[1] for x in co_data], dtype=np.int32)
counts = np.array([x[2] for x in co_data], dtype=np.float32)

print(f"   {len(co_data):,} paires\n")

# ===========================
# 3. ENTRAÎNEMENT GLOVE
# ===========================
print("3. Entraînement GloVe...")

EMBEDDING_SIZE = 100
center_vec = tf.Variable(np.random.randn(vocab_size, EMBEDDING_SIZE).astype(np.float32) * 0.01)
context_vec = tf.Variable(np.random.randn(vocab_size, EMBEDDING_SIZE).astype(np.float32) * 0.01)
center_bias = tf.Variable(np.zeros(vocab_size, dtype=np.float32))
context_bias = tf.Variable(np.zeros(vocab_size, dtype=np.float32))

optimizer = tf.optimizers.Adam(0.05)

@tf.function
def train_step(c_ix, ctx_ix, co_counts):
    with tf.GradientTape() as tape:
        c_vec = tf.gather(center_vec, c_ix)
        ctx_vec = tf.gather(context_vec, ctx_ix)
        c_b = tf.gather(center_bias, c_ix)
        ctx_b = tf.gather(context_bias, ctx_ix)
        
        pred = tf.reduce_sum(c_vec * ctx_vec, axis=1) + c_b + ctx_b
        target = tf.math.log(co_counts + 1.0)
        loss = tf.reduce_mean(tf.square(pred - target))
    
    grads = tape.gradient(loss, [center_vec, context_vec, center_bias, context_bias])
    optimizer.apply_gradients(zip(grads, [center_vec, context_vec, center_bias, context_bias]))
    return loss

# Entraînement
for epoch in range(30):
    indices = np.random.permutation(len(center_ix))
    epoch_loss = train_step(center_ix[indices], context_ix[indices], counts[indices])
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1}/30 - Loss: {epoch_loss:.4f}")

print(f"\n   ✓ Entraîné\n")

# ===========================
# 4. MOTS SIMILAIRES
# ===========================
print("4. Mots similaires\n")

def most_similar(word, top_n=5):
    word_ix = vocab_to_ix[word]
    word_vec = center_vec[word_ix].numpy()
    
    all_vecs = center_vec.numpy()
    word_norm = word_vec / np.linalg.norm(word_vec)
    all_norm = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    
    sims = np.dot(all_norm, word_norm)
    top_ix = np.argsort(sims)[::-1][1:top_n+1]
    
    return [(vocab[i], sims[i]) for i in top_ix]

for word in ['love', 'action', 'hero', 'world']:
    if word in vocab_to_ix:
        similar = most_similar(word)
        print(f"   '{word}' :")
        for w, s in similar:
            print(f"      {w:15s} : {s:.3f}")
        print()

# ===========================
# 5. VISUALISATION
# ===========================
print("5. Visualisation t-SNE...")

top_words = [w for w, _ in token_counts.most_common(40) if w in vocab_to_ix]
top_ix = [vocab_to_ix[w] for w in top_words]
vecs = center_vec.numpy()[top_ix]

tsne = TSNE(n_components=2, random_state=42)
embedded = tsne.fit_transform(vecs)

plt.figure(figsize=(12, 8))
for i, word in enumerate(top_words):
    x, y = embedded[i]
    plt.scatter(x, y, s=50, alpha=0.7)
    plt.text(x + 0.5, y + 0.5, word, fontsize=9)

plt.title('GloVe t-SNE', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'glove_tsne.png'), dpi=300)
plt.close()

print("   ✓ Graphique sauvegardé\n")

# ===========================
# 6. SAUVEGARDE
# ===========================
np.save(os.path.join(DATA_DIR, 'glove_embeddings_films.npy'), center_vec.numpy())
np.save(os.path.join(DATA_DIR, 'glove_vocab_films.npy'), vocab)
print("   ✓ Embeddings sauvegardés\n")

print("="*60)
print("TERMINÉ")
print("="*60)