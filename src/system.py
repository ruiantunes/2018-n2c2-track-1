#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Copyright 2018, 2019, 2020 Rui Antunes
                           João Figueira Silva
                           Arnaldo Pereira
                           Sérgio Matos

https://github.com/ruiantunes/2018-n2c2-track-1


Tentative of classification for the n2c2 challenge. This script can be
run in different running modes:
- `cross-validation`: cross-validation is applied in the train dataset.
- `predict`: models are trained using the train set. After training,
they are used for predicting the test dataset.

Train and test directories are hard-coded in the script.

"""


# third-party modules
from datetime import datetime
import numpy as np
import os
import random as rn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# own modules
from __init__ import REPO
from evaluator import evaluate2str
from reader import TAGS
from reader import N_TAGS
from reader import MONTHS
from reader import TAG2MONTHS
from reader import Corpus
from rules import ImprovedRuleBasedClassifier
from rules import RuleBasedClassifier
from support import load_stopwords
from utils import str2line
from utils import Printer


# input arguments (change them here)

# running mode
MODES = ('cross-validation', 'predict')
MODE = MODES[0]

# it is to use the rule-based system?
RULES = True

# rule-based classifiers
RULE_CLASSIFIERS = [
    RuleBasedClassifier(),
    ImprovedRuleBasedClassifier(),
]
RULE_CLASSIFIER = RULE_CLASSIFIERS[0]

# seed number
SEED = 0

# seed
np.random.seed(SEED)
rn.seed(SEED)

# directory and file paths

# n2c2 train dataset
TRAIN_DPATH = os.path.join(REPO, 'data', 'n2c2', 'train')

# n2c2 test dataset
TEST_DPATH = os.path.join(REPO, 'data', 'n2c2', 'test')

# stopwords filepath
STOPWORDS_FPATH = os.path.join(REPO, 'data', 'wrd_stop.txt')

# logs file
FN = datetime.now().strftime('%Y-%m-%d-%H%M%S-%f')
LOGS_FPATH = os.path.join(REPO, 'logs', FN + '-logs.txt')

# printer (logging)
PRINTER = Printer(filepath=LOGS_FPATH)
D_ = PRINTER.date
P_ = PRINTER.print

# stopwords
STOPWORDS = load_stopwords(STOPWORDS_FPATH)

# cross-validation
SKF = StratifiedKFold(
    n_splits=3,
    shuffle=False,
    random_state=None,
)

RULES_TAGS = [
    'ADVANCED-CAD',
    'ALCOHOL-ABUSE',
    'ASP-FOR-MI',
    'CREATININE',
    'DIETSUPP-2MOS',
    'DRUG-ABUSE',
    'ENGLISH',
    'HBA1C',
    'KETO-1YR',
    'MAKES-DECISIONS',
    'MI-6MOS',
]

# vectorizer
VECTORIZER = (
    'TfidfVectorizer',
    TfidfVectorizer(
        lowercase=False,
        stop_words=STOPWORDS,
        ngram_range=(1, 1),
        sublinear_tf=True,
    ),
)

# classifiers
CLASSIFIERS = [
    (
        'AdaBoostClassifier',
        AdaBoostClassifier(
            random_state=SEED,
        ),
    ),
    (
        'BaggingClassifier',
        BaggingClassifier(
            random_state=SEED,
        ),
    ),
    (
        'GradientBoostingClassifier',
        GradientBoostingClassifier(
            random_state=SEED,
        ),
    ),
    (
        'DecisionTreeClassifier',
        DecisionTreeClassifier(
            random_state=SEED,
        ),
    ),
    (
        'XGBClassifier',
        XGBClassifier(
            random_state=SEED,
        ),
    ),
]


def cross_validation_mode():
    # load train corpus
    TRUE_CORPUS = Corpus(dpath=TRAIN_DPATH)
    # load the same corpus but for prediction purposes
    PRED_CORPUS = Corpus(dpath=TRAIN_DPATH)
    # total number of patients
    n = len(TRUE_CORPUS.get_patients())
    # raw documents (for rule-based)
    raw_docs = {
        months: TRUE_CORPUS.get_documents(months=months, clean=False)
        for months in MONTHS
    }
    # clean documents (for machine learning)
    clean_docs = {
        months: TRUE_CORPUS.get_documents(months=months, clean=True)
        for months in MONTHS
    }
    # labels
    labels = {
        tag: TRUE_CORPUS.get_labels(tag=tag)
        for tag in TAGS
    }
    for clf in CLASSIFIERS:
        # print classifier
        P_('{}'.format(str2line(clf)))
        pipe = Pipeline([VECTORIZER, clf])
        for i, tag in enumerate(TAGS):
            months = TAG2MONTHS[tag]
            if (RULES and (tag in RULES_TAGS)):
                X = raw_docs[months]
                y_pred = RULE_CLASSIFIER.predict(tag=tag, X=X)
                PRED_CORPUS.set_labels(tag=tag, labels=y_pred)
            else:
                X = clean_docs[months]
                y = labels[tag]
                y_pred = np.zeros(n)
                # two distinct labels should exist for classification!
                if len(set(y)) > 1:
                    for train_index, test_index in SKF.split(X, y):
                        X_train = [X[i] for i in train_index]
                        y_train = [y[i] for i in train_index]
                        X_test = [X[i] for i in test_index]
                        # train pipeline
                        _ = pipe.fit(X=X_train, y=y_train)
                        # predict test samples
                        y_pred[test_index] = pipe.predict(X_test)
                    PRED_CORPUS.set_labels(tag=tag, labels=y_pred)
        table = evaluate2str(TRUE_CORPUS, PRED_CORPUS)
        P_(table)


def predict_mode():
    r"""
    Prediction of the test dataset.

    """
    # train dataset
    TRAIN_CORPUS = Corpus(dpath=TRAIN_DPATH)
    train_raw_docs = {
        months: TRAIN_CORPUS.get_documents(months=months, clean=False)
        for months in MONTHS
    }
    train_clean_docs = {
        months: TRAIN_CORPUS.get_documents(months=months, clean=True)
        for months in MONTHS
    }
    train_labels = {
        tag: TRAIN_CORPUS.get_labels(tag=tag)
        for tag in TAGS
    }
    # test dataset
    TEST_CORPUS = Corpus(dpath=TEST_DPATH)
    GS_TEST_CORPUS = Corpus(dpath=TEST_DPATH)
    test_raw_docs = {
        months: TEST_CORPUS.get_documents(months=months, clean=False)
        for months in MONTHS
    }
    test_clean_docs = {
        months: TEST_CORPUS.get_documents(months=months, clean=True)
        for months in MONTHS
    }
    # select best classifiers (based on preliminary results)
    # (`None` where rules will be used, since it is expected that they
    # provide better results)
    BEST_CLASSIFIERS = {
        'ABDOMINAL': CLASSIFIERS[2],
        'ADVANCED-CAD': None,
        'ALCOHOL-ABUSE': None,
        'ASP-FOR-MI': None,
        'CREATININE': None,
        'DIETSUPP-2MOS': None,
        'DRUG-ABUSE': None,
        'ENGLISH': None,
        'HBA1C': None,
        'KETO-1YR': None,
        'MAJOR-DIABETES': CLASSIFIERS[0],
        'MAKES-DECISIONS': None,
        'MI-6MOS': None,
    }
    for tag in TAGS:
        months = TAG2MONTHS[tag]
        if RULES and (tag in RULES_TAGS):
            X = test_raw_docs[months]
            y_pred = RULE_CLASSIFIER.predict(tag=tag, X=X)
        else:
            pipe = Pipeline([VECTORIZER, BEST_CLASSIFIERS[tag]])
            X_train = list(train_clean_docs[months])
            y_train = list(train_labels[tag])
            # train pipeline
            _ = pipe.fit(X=X_train, y=y_train)
            # predict test samples
            X = test_clean_docs[months]
            y_pred = pipe.predict(X=X)
        # set predicted labels
        TEST_CORPUS.set_labels(tag=tag, labels=y_pred)
    table = evaluate2str(GS_TEST_CORPUS, TEST_CORPUS)
    P_(table)
    # at the end write the test corpus
    dpath = 'test-prediction'
    TEST_CORPUS.write(dpath=dpath)


# main
def main():
    # print constants
    D_('START')
    P_(
        '-----------------------------------------------------\n'
        '--- National NLP Clinical Challenges (n2c2) ---------\n'
        '--- Track 1: Cohort selection for clinical trials ---\n'
        '-----------------------------------------------------\n'
    )
    P_('input arguments\n')
    P_(
        '\tMODE\n'
        '\t\t{}\n'.format(MODE)
    )
    P_(
        '\tRULES\n'
        '\t\t{}\n'.format(RULES)
    )
    P_(
        '\tRULE_CLASSIFIER\n'
        '\t\t{}\n'.format(str2line(RULE_CLASSIFIER))
    )
    P_(
        '\tSEED\n'
        '\t\t{}\n'.format(SEED)
    )
    P_('directory and file paths\n')
    P_(
        '\tTRAIN_DPATH\n'
        '\t\t{}\n'.format(TRAIN_DPATH)
    )
    P_(
        '\tTEST_DPATH\n'
        '\t\t{}\n'.format(TEST_DPATH)
    )
    P_(
        '\tSTOPWORDS_FPATH\n'
        '\t\t{}\n'.format(STOPWORDS_FPATH)
    )
    P_(
        '\tLOGS_FPATH\n'
        '\t\t{}\n'.format(LOGS_FPATH)
    )
    P_('other constants\n')
    P_(
        '\tSTOPWORDS\n'
        '\t\t{}\n'.format(STOPWORDS)
    )
    P_(
        '\tSKF\n'
        '\t\t{}\n'.format(SKF)
    )
    P_(
        '\tRULES_TAGS\n'
        '\t\t{}\n'.format(RULES_TAGS)
    )
    P_(
        '\tVECTORIZER\n'
        '\t\t{}\n'.format(str2line(VECTORIZER))
    )
    P_(
        '\tCLASSIFIERS'
    )
    for c in CLASSIFIERS:
        P_(
            '\t\t{}'.format(str2line(c))
        )
    P_()
    # select running mode
    if MODE == 'cross-validation':
        cross_validation_mode()
    elif MODE == 'predict':
        predict_mode()
    # end
    D_('END')


if __name__ == '__main__':
    main()
