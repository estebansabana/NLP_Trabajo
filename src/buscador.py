"""Buscador de comentarios por aspecto, desde la terminal.

Uso (desde la raíz del repositorio, después de ejecutar el notebook una vez para llenar cache/):
    python -m src.buscador "piscina"
    python -m src.buscador "slow wifi" -k 5 --sistema bm25

Sistemas: bm25, bm25_exp (expansión con Word2Vec), densa (Sentence-BERT multilingüe),
hibrida (RRF de bm25_exp y densa) y auto (híbrida si todas las palabras están en el vocabulario inglés; si no, densa).
"""
import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .preprocesamiento import CUSTOM_STOP_WORDS, cargar_corpus, tokens_ir

ROOT = Path(__file__).resolve().parents[1]


class Buscador:
    def __init__(self, root=ROOT, k1=1.2, b=0.75):
        import torch
        from gensim.models import Word2Vec
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

        self.df = cargar_corpus(root / 'data' / 'LuxuryHotels.csv')
        cv = CountVectorizer(tokenizer=tokens_ir, lowercase=False, token_pattern=None)
        tf = cv.fit_transform(self.df['Review']).tocoo()
        self.voc, n = cv.vocabulary_, tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        dft = np.bincount(tf.col, minlength=tf.shape[1])
        idf = np.log((n - dft + 0.5) / (dft + 0.5) + 1.0)
        peso = idf[tf.col] * tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / dl.mean()))
        self.w = sp.csc_matrix((peso.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        self.df_ratio = dict(zip(cv.get_feature_names_out(), dft / n))

        cache = root / 'cache'
        self.wv = Word2Vec.load(str(cache / 'w2v_hoteles.model')).wv
        local = root / 'modelos' / 'paraphrase-multilingual-MiniLM-L12-v2'
        self.sbert = SentenceTransformer(str(local) if (local / 'modules.json').exists()
                                         else 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cpu',
                                         model_kwargs={'torch_dtype': torch.float32})
        huella = hashlib.md5('\n'.join(self.df['Review']).encode()).hexdigest()
        f_emb, f_md5 = cache / 'emb_resenas.npy', cache / 'emb_resenas.md5'
        if f_emb.exists() and f_md5.exists() and f_md5.read_text() == huella:
            self.emb = np.load(f_emb)
        else:
            self.emb = self.sbert.encode(self.df['Review'].tolist(), batch_size=64, normalize_embeddings=True)

    def _bm25(self, pesos):
        s = np.zeros(self.w.shape[0], dtype=np.float32)
        for t, p in pesos.items():
            if t in self.voc:
                s += p * self.w[:, self.voc[t]].toarray().ravel()
        return s

    @staticmethod
    def _top(s, k):
        if s.max() <= 0:
            return np.array([], dtype=int)
        k = min(k, int((s > 0).sum()))
        idx = np.argpartition(-s, k - 1)[:k]
        return idx[np.lexsort((idx, -s[idx]))]

    def expandir(self, q, k=3, umbral=0.70, peso_exp=0.5, max_df=0.25):
        toks = tokens_ir(q)
        pesos = {t: 1.0 for t in toks}
        for t in toks:
            if t not in self.wv.key_to_index or self.df_ratio.get(t, 0) > max_df:
                continue
            n_add = 0
            for vecino, sim in self.wv.most_similar(t, topn=15):
                if n_add == k or sim < umbral:
                    break
                if vecino in self.voc and vecino not in pesos and vecino not in CUSTOM_STOP_WORDS and not vecino.isdigit():
                    pesos[vecino] = peso_exp
                    n_add += 1
        return pesos

    def bm25(self, q, k=100):
        return self._top(self._bm25({t: 1.0 for t in tokens_ir(q)}), k)

    def bm25_exp(self, q, k=100):
        return self._top(self._bm25(self.expandir(q)), k)

    def densa(self, q, k=100):
        s = self.emb @ self.sbert.encode([q], normalize_embeddings=True)[0]
        idx = np.argpartition(-s, k - 1)[:k]
        return idx[np.lexsort((idx, -s[idx]))]

    def hibrida(self, q, k=100, c=60):
        fus = {}
        for ranking in (self.bm25_exp(q), self.densa(q)):
            for pos, d in enumerate(ranking, 1):
                fus[d] = fus.get(d, 0.0) + 1.0 / (c + pos)
        return np.array(sorted(fus, key=lambda d: (-fus[d], d))[:k], dtype=int)

    def auto(self, q, k=100):
        toks = tokens_ir(q)
        ingles = len(toks) > 0 and all(t in self.wv.key_to_index for t in toks)
        return self.hibrida(q, k) if ingles else self.densa(q, k)

    def buscar(self, q, k=10, sistema='auto'):
        idx = getattr(self, sistema)(q)[:k]
        q_emb = self.sbert.encode([q], normalize_embeddings=True)[0]
        filas = []
        for d in idx:
            texto = self.df['Review'].iat[d]
            frases = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', texto) if len(s.split()) >= 3] or [texto]
            frase = frases[int(np.argmax(self.sbert.encode(frases, normalize_embeddings=True) @ q_emb))]
            filas.append((int(self.df['doc_id'].iat[d]), int(self.df['Rating'].iat[d]), frase))
        return filas


def main():
    ap = argparse.ArgumentParser(description='Busca reseñas de hoteles por aspecto, en cualquier idioma.')
    ap.add_argument('consulta')
    ap.add_argument('-k', type=int, default=10)
    ap.add_argument('--sistema', default='auto', choices=['auto', 'bm25', 'bm25_exp', 'densa', 'hibrida'])
    args = ap.parse_args()
    for i, (doc_id, estrellas, frase) in enumerate(Buscador().buscar(args.consulta, args.k, args.sistema), 1):
        print(f'{i:2d}. [{estrellas}★ · reseña {doc_id}] {frase[:200]}')


if __name__ == '__main__':
    main()
