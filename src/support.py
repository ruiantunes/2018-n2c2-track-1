#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Copyright 2018, 2019, 2020 Rui Antunes
                           João Figueira Silva
                           Arnaldo Pereira
                           Sérgio Matos

https://github.com/ruiantunes/2018-n2c2-track-1


Text mining support utilities.

Due to performance and readability purposes, there are weak or none
consistency validations (duck typing).


Routines listing
----------------
load_stopwords
clean_medical_documents

"""


# third-party modules
from gensim.utils import deaccent
import re


def load_stopwords(filepath):
    r"""
    Load stopwords from a specific file.

    Parameters
    ----------
    filepath : str
        Stopwords list filepath.

    Returns
    -------
    stopwords : set
        A `set` with the stopwords.

    Example
    -------
    >>> import os
    >>> from __init__ import REPO
    >>> from support import load_stopwords
    >>> filepath = os.path.join(REPO, 'data', 'wrd_stop.txt')
    >>> stopwords = load_stopwords(filepath)
    >>> len(stopwords)
    313
    >>> 'not' in stopwords
    True
    >>>

    """
    stopwords = set()
    with open(filepath) as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords


def clean_medical_documents(docs):
    r"""
    Clean medical reports.

    This function makes a simple pre-processing of medical texts. The
    steps are:
    1. String is deaccented.
    2. Sequences of at least 2 letters are extracted (numbers and other
       characters are ignored).
    3. Tokens with all uppercase letters are kept. Other tokens are
       converted to lowercase.
    4. The tokens 'pt', and 'pts' are replaced by 'patient'.

    Parameters
    ----------
    docs : list
        `list` of documents. Each document is a `str`.

    Returns
    -------
    docs : list
        `list` of documents. Each document is a `list` of lines. Each
        line is a `list` of tokens.

    Example
    -------
    >>> from support import clean_medical_documents
    >>> docs = [
    ...     'The pt. appears awake.',
    ...     'Cardiologist:  Dr. C. Núttèr',
    ... ]
    >>> clean_docs = clean_medical_documents(docs)
    >>> print(clean_docs)
    ['the patient appears awake', 'cardiologist dr nutter']
    >>>

    """
    # to not modify the input parameter
    docs = list(docs)
    regex = re.compile(pattern=r'[a-zA-Z]{2,}')
    for i, doc in enumerate(docs):
        doc = deaccent(doc)
        # keep uppercase words
        tokens = [
            token.lower() if not token.isupper() else token
            for token in regex.findall(doc)
        ]
        # replace tokens
        for j, token in enumerate(tokens):
            if token in ('pt', 'pts'):
                tokens[j] = 'patient'
        # clean document
        docs[i] = ' '.join(tokens)
    return docs
