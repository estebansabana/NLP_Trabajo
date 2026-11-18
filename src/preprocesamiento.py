"""Limpieza y tokenización de reseñas (las mismas funciones de las secciones 1, 3 y 9 del notebook)."""
import re
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

NEGATIONS_TO_KEEP = {'not', 'no', 'nor', 'never', 'none', 'nothing', 'nobody', 'neither', 'nowhere', 'cannot'}
CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS_TO_KEEP

CONTRACCIONES = {'dont': 'do not', 'didnt': 'did not', 'doesnt': 'does not', 'wasnt': 'was not', 'isnt': 'is not',
                 'werent': 'were not', 'arent': 'are not', 'couldnt': 'could not', 'wouldnt': 'would not',
                 'shouldnt': 'should not', 'havent': 'have not', 'hasnt': 'has not', 'hadnt': 'had not',
                 'cant': 'can not', 'wont': 'will not', 'aint': 'is not'}
RE_CONTR = re.compile(r'\b(' + '|'.join(CONTRACCIONES) + r')\b')
RE_HTML = re.compile(r'<[^>]+>')
RE_URL = re.compile(r'https?://\S+|www\.\S+')
RE_WORD = re.compile(r'[a-z]+')
TOK_IR = re.compile(r'[a-z0-9]+')


def cargar_corpus(ruta_csv):
    """Lee el CSV, elimina textos repetidos y el texto con calificaciones contradictorias (#NAME?)."""
    raw = pd.read_csv(ruta_csv)
    conf = raw.groupby('Review')['Rating'].nunique()
    conflicto = set(conf[conf > 1].index)
    df = raw[~raw['Review'].isin(conflicto)].drop_duplicates('Review').drop(columns='Bubbles').reset_index(drop=True)
    df['doc_id'] = range(len(df))
    return df


def normalizar(texto):
    """Minúsculas, HTML, URLs, truncamiento y negaciones contraídas (con y sin apóstrofo)."""
    t = str(texto).lower().replace('’', "'").replace('…', ' ')
    t = RE_HTML.sub(' ', t)
    t = RE_URL.sub(' ', t)
    t = re.sub(r"\bwon't\b", 'will not', t)
    t = re.sub(r"\bcan't\b", 'can not', t)
    t = re.sub(r"n't\b", ' not', t)
    return RE_CONTR.sub(lambda m: CONTRACCIONES[m.group(1)], t)


def clean_text(texto):
    """Texto para BoW / TF-IDF: solo letras, sin stop words, conservando negaciones."""
    return ' '.join(w for w in RE_WORD.findall(normalizar(texto)) if w not in CUSTOM_STOP_WORDS and len(w) > 1)


def tokens_w2v(texto):
    """Tokens para Word2Vec: limpieza mínima, con stop words."""
    return RE_WORD.findall(normalizar(texto))


def tokens_ir(texto):
    """Tokenización de la sesión de IR ([a-z0-9]+, sin stop words), sin tildes."""
    t = unicodedata.normalize('NFKD', str(texto).lower().replace('…', ' ')).encode('ascii', 'ignore').decode()
    return [w for w in TOK_IR.findall(t) if w not in CUSTOM_STOP_WORDS]
