#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Copyright 2018, 2019, 2020 Rui Antunes
                           João Figueira Silva
                           Arnaldo Pereira
                           Sérgio Matos

https://github.com/ruiantunes/2018-n2c2-track-1


Evaluator for the n2c2 challenge.

Two corpus are compared: true (gold) and pred (system). The official
evaluation for the Track 1 is the micro-averaged F1-score. The goal of
this is script is to provide functions for mimicking the official
evaluation script `track1_eval.py`.


Routines listing
----------------
tp_tn_fp_fn
tpr
tnr
fpr
fnr
ppv
npv
acc
f1
mcc
dor
evaluate
metrics2str
evaluate2str

"""


# own modules
from reader import Corpus


def tp_tn_fp_fn(true, pred):
    r"""
    It calculates the true/false positives/negatives.

    It is expected to be a binary classification problem. That is the
    labels are 0 or 1.

    True Positive (TP) when:
        (true == 1) and (pred == 1)
    True Negative (TN) when:
        (true == 0) and (pred == 0)
    False Positive (FP) when:
        (true == 0) and (pred == 1)
    False Negative (FN) when:
        (true == 1) and (pred == 0)

    Parameters
    ----------
    true : list
        True labels.
    pred : list
        Predicted labels.

    Returns
    -------
    tp : int
        Number of true positives.
    tn : int
        Number of true negatives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    """
    tp = tn = fp = fn = 0
    for t, p in zip(true, pred):
        assert (t == 0) or (t == 1)
        assert (p == 0) or (p == 1)
        if (t == 1) and (p == 1):
            tp += 1
        elif (t == 0) and (p == 0):
            tn += 1
        elif (t == 0) and (p == 1):
            fp += 1
        elif (t == 1) and (p == 0):
            fn += 1
    return (tp, tn, fp, fn)


def tpr(tp, tn, fp, fn):
    r"""
    It calculates the True Positive Rate (TPR).

    TPR is the same as recall and sensitivity.

    Parameters
    ----------
    tp : int
        Number of true positives.
    tn : int
        Number of true negatives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    tpr : float
        True positive rate.

    """
    try:
        tpr = tp / (tp + fn)
    except ZeroDivisionError:
        tpr = 0.0
    return tpr


def tnr(tp, tn, fp, fn):
    r"""
    It calculates the True Negative Rate (TNR).

    TNR is the same as specificity.

    Parameters
    ----------
    tp : int
        Number of true positives.
    tn : int
        Number of true negatives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    tnr : float
        True negative rate.

    """
    try:
        tnr = tn / (tn + fp)
    except ZeroDivisionError:
        tnr = 0.0
    return tnr


def fpr(tp, tn, fp, fn):
    r"""
    It calculates the False Positive Rate (FPR).

    FPR is the same as fall-out and probability of false alarm.

    Parameters
    ----------
    tp : int
        Number of true positives.
    tn : int
        Number of true negatives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    fpr : float
        False positive rate.

    """
    try:
        fpr = fp / (fp + tn)
    except ZeroDivisionError:
        fpr = 0.0
    return fpr


def fnr(tp, tn, fp, fn):
    r"""
    It calculates the False Negative Rate (FNR).

    FNR is the same as miss rate.

    Parameters
    ----------
    tp : int
        Number of true positives.
    tn : int
        Number of true negatives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    fnr : float
        False negative rate.

    """
    try:
        fnr = fn / (fn + tp)
    except ZeroDivisionError:
        fnr = 0.0
    return fnr


def ppv(tp, tn, fp, fn):
    r"""
    It calculates the Positive Predictive Value (PPV).

    PPV is the same as precision.

    Parameters
    ----------
    tp : int
        Number of true positives.
    tn : int
        Number of true negatives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    ppv : float
        Positive predicitive value.

    """
    try:
        ppv = tp / (tp + fp)
    except ZeroDivisionError:
        ppv = 0.0
    return ppv


def npv(tp, tn, fp, fn):
    r"""
    It calculates the Negative Predictive Value (NPV).

    Parameters
    ----------
    tp : int
        Number of true positives.
    tn : int
        Number of true negatives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    npv : float
        Negative predicitive value.

    """
    try:
        npv = tn / (tn + fn)
    except ZeroDivisionError:
        npv = 0.0
    return npv


def acc(tp, tn, fp, fn):
    r"""
    It calculates the accuracy.

    Parameters
    ----------
    tp : int
        Number of true positives.
    tn : int
        Number of true negatives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    acc : float
        Accuracy.

    """
    try:
        acc = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        acc = 0.0
    return acc


def f1(tp, tn, fp, fn):
    r"""
    It calculates the F1-score.

    Parameters
    ----------
    tp : int
        Number of true positives.
    tn : int
        Number of true negatives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    f1 : float
        F1-score.

    """
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0.0
    return f1


def mcc(tp, tn, fp, fn):
    r"""
    It calculates the Matthews Correlation Coefficient (MCC).

    Parameters
    ----------
    tp : int
        Number of true positives.
    tn : int
        Number of true negatives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    mcc : float
        Matthews correlation coefficient.

    """
    try:
        num = (tp * tn) - (fp * fn)
        den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = num / den
    except ZeroDivisionError:
        mcc = 0.0
    return mcc


def dor(tp, tn, fp, fn):
    r"""
    It calculates the Diagnostic Odds Ratio (DOR).

    Parameters
    ----------
    tp : int
        Number of true positives.
    tn : int
        Number of true negatives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    dor : float
        Diagnostic odds ratio.

    """
    try:
        dor = (tp * tn) / (fp * fn)
    except ZeroDivisionError:
        dor = 0.0
    return dor


def evaluate(true, pred):
    r"""
    Calculates metrics between two corpus (true and pred).

    Several metrics are calculated between the true/gold `Corpus` and
    the predicted `Corpus`. It is mandatory the both corpus to have the
    same patients.

    Parameters
    ----------
    true : Corpus
        True corpus. Corpus with the true/gold labels.
    pred : Corpus
        Predicted corpus. Corpus with the predicted labels.

    Returns
    -------
    metrics : dict
        The metrics `dict` contain the number of files/patients found,
        the TP, TN, FP, FN, precision, recall, and f1-score for each of
        the two classes: "met" and "not met". Also, overall, micro- and
        macro-averaged scores are calculated.

    Example
    -------
    >>> import os
    >>> from __init__ import REPO
    >>> from evaluator import evaluate
    >>> from reader import Corpus
    >>> true = Corpus(os.path.join(REPO, 'data', 'n2c2', 'train'))
    >>> pred = Corpus(os.path.join(REPO, 'data', 'n2c2', 'train'))
    >>> metrics = evaluate(true, pred)
    >>> print(metrics['micro']['overall']['F1'])
    1.0
    >>>

    """
    # consistency check
    assert isinstance(true, Corpus) and isinstance(pred, Corpus)
    # the two corpus have the same patients? hope so!
    assert true.get_patients() == pred.get_patients()
    # auxiliar functions
    def aux_dict():
        return {
            'met': dict(),
            'not met': dict(),
            'overall': dict(),
        }
    def invert(elems):
         return [int(not elem) for elem in elems]
    # tags
    tags = [
        'ABDOMINAL',
        'ADVANCED-CAD',
        'ALCOHOL-ABUSE',
        'ASP-FOR-MI',
        'CREATININE',
        'DIETSUPP-2MOS',
        'DRUG-ABUSE',
        'ENGLISH',
        'HBA1C',
        'KETO-1YR',
        'MAJOR-DIABETES',
        'MAKES-DECISIONS',
        'MI-6MOS',
    ]
    n_tags = len(tags)
    # how many patients are?
    patients = true.get_patients()
    n_patients = len(patients)
    # initialize `metrics`
    metrics = {
        'patients': n_patients,
        'tags': {tag: aux_dict() for tag in tags},
        'micro': aux_dict(),
        'macro': aux_dict(),
    }
    # tags
    # go through for each tag
    for tag in tags:
        # --- met ---
        # get true/pred labels
        true_labels = true.get_labels(tag=tag)
        pred_labels = pred.get_labels(tag=tag)
        tp, tn, fp, fn = tp_tn_fp_fn(true_labels, pred_labels)
        metrics['tags'][tag]['met']['TP'] = tp
        metrics['tags'][tag]['met']['TN'] = tn
        metrics['tags'][tag]['met']['FP'] = fp
        metrics['tags'][tag]['met']['FN'] = fn
        metrics['tags'][tag]['met']['PPV'] = ppv(tp, tn, fp, fn)
        metrics['tags'][tag]['met']['TPR'] = tpr(tp, tn, fp, fn)
        metrics['tags'][tag]['met']['F1'] = f1(tp, tn, fp, fn)
        # --- not met ---
        # get true/pred labels
        true_labels = invert(true.get_labels(tag=tag))
        pred_labels = invert(pred.get_labels(tag=tag))
        tp, tn, fp, fn = tp_tn_fp_fn(true_labels, pred_labels)
        metrics['tags'][tag]['not met']['TP'] = tp
        metrics['tags'][tag]['not met']['TN'] = tn
        metrics['tags'][tag]['not met']['FP'] = fp
        metrics['tags'][tag]['not met']['FN'] = fn
        metrics['tags'][tag]['not met']['PPV'] = ppv(tp, tn, fp, fn)
        metrics['tags'][tag]['not met']['TPR'] = tpr(tp, tn, fp, fn)
        metrics['tags'][tag]['not met']['F1'] = f1(tp, tn, fp, fn)
        # --- overall ---
        metrics['tags'][tag]['overall']['F1'] = (
            metrics['tags'][tag]['met']['F1'] +
            metrics['tags'][tag]['not met']['F1']
        ) / 2
    # === micro-averaged ===
    # --- met ---
    tp = tn = fp = fn = 0
    for tag in tags:
        tp += metrics['tags'][tag]['met']['TP']
        tn += metrics['tags'][tag]['met']['TN']
        fp += metrics['tags'][tag]['met']['FP']
        fn += metrics['tags'][tag]['met']['FN']
    metrics['micro']['met']['TP'] = tp
    metrics['micro']['met']['TN'] = tn
    metrics['micro']['met']['FP'] = fp
    metrics['micro']['met']['FN'] = fn
    metrics['micro']['met']['PPV'] = ppv(tp, tn, fp, fn)
    metrics['micro']['met']['TPR'] = tpr(tp, tn, fp, fn)
    metrics['micro']['met']['F1'] = f1(tp, tn, fp, fn)
    # --- not met ---
    tp = tn = fp = fn = 0
    for tag in tags:
        tp += metrics['tags'][tag]['not met']['TP']
        tn += metrics['tags'][tag]['not met']['TN']
        fp += metrics['tags'][tag]['not met']['FP']
        fn += metrics['tags'][tag]['not met']['FN']
    metrics['micro']['not met']['TP'] = tp
    metrics['micro']['not met']['TN'] = tn
    metrics['micro']['not met']['FP'] = fp
    metrics['micro']['not met']['FN'] = fn
    metrics['micro']['not met']['PPV'] = ppv(tp, tn, fp, fn)
    metrics['micro']['not met']['TPR'] = tpr(tp, tn, fp, fn)
    metrics['micro']['not met']['F1'] = f1(tp, tn, fp, fn)
    # --- overall ---
    metrics['micro']['overall']['F1'] = (
        metrics['micro']['met']['F1'] +
        metrics['micro']['not met']['F1']
    ) / 2
    # === macro-averaged ===
    # --- met ---
    ppv_ = 0.0
    tpr_ = 0.0
    f1_ = 0.0
    for tag in tags:
        ppv_ += metrics['tags'][tag]['met']['PPV']
        tpr_ += metrics['tags'][tag]['met']['TPR']
        f1_ += metrics['tags'][tag]['met']['F1']
    metrics['macro']['met']['PPV'] = ppv_ / n_tags
    metrics['macro']['met']['TPR'] = tpr_ / n_tags
    metrics['macro']['met']['F1'] = f1_ / n_tags
    # --- not met ---
    ppv_ = 0.0
    tpr_ = 0.0
    f1_ = 0.0
    for tag in tags:
        ppv_ += metrics['tags'][tag]['not met']['PPV']
        tpr_ += metrics['tags'][tag]['not met']['TPR']
        f1_ += metrics['tags'][tag]['not met']['F1']
    metrics['macro']['not met']['PPV'] = ppv_ / n_tags
    metrics['macro']['not met']['TPR'] = tpr_ / n_tags
    metrics['macro']['not met']['F1'] = f1_ / n_tags
    # --- overall ---
    metrics['macro']['overall']['F1'] = (
        metrics['macro']['met']['F1'] +
        metrics['macro']['not met']['F1']
    ) / 2
    # finally
    return metrics


def metrics2str(metrics):
    r"""
    It puts all metrics into a pretty organized string (table).

    Parameters
    ----------
    metrics : dict
        A `dict` containing the metrics.

    Returns
    -------
    table : str
        All metrics pretty organized.

    """
    draw = (
        '------------------------------------------------------------------------------------------------------------------\n'
        '999999 patients  ----------------- met ------------------  --------------- not met ----------------  -- overall --\n'
        '                   TP   TN   FP   FN    PPV    TPR     F1    TP   TN   FP   FN    PPV    TPR     F1             F1\n'
        '---------------  ----------------------------------------  ----------------------------------------  -------------\n'
        '      ABDOMINAL  9999 9999 9999 9999 0.9999 0.9999 0.9999  9999 9999 9999 9999 0.9999 0.9999 0.9999         0.9999\n'
        '   ADVANCED-CAD  9999 9999 9999 9999 0.9999 0.9999 0.9999  9999 9999 9999 9999 0.9999 0.9999 0.9999         0.9999\n'
        '  ALCOHOL-ABUSE  9999 9999 9999 9999 0.9999 0.9999 0.9999  9999 9999 9999 9999 0.9999 0.9999 0.9999         0.9999\n'
        '     ASP-FOR-MI  9999 9999 9999 9999 0.9999 0.9999 0.9999  9999 9999 9999 9999 0.9999 0.9999 0.9999         0.9999\n'
        '     CREATININE  9999 9999 9999 9999 0.9999 0.9999 0.9999  9999 9999 9999 9999 0.9999 0.9999 0.9999         0.9999\n'
        '  DIETSUPP-2MOS  9999 9999 9999 9999 0.9999 0.9999 0.9999  9999 9999 9999 9999 0.9999 0.9999 0.9999         0.9999\n'
        '     DRUG-ABUSE  9999 9999 9999 9999 0.9999 0.9999 0.9999  9999 9999 9999 9999 0.9999 0.9999 0.9999         0.9999\n'
        '        ENGLISH  9999 9999 9999 9999 0.9999 0.9999 0.9999  9999 9999 9999 9999 0.9999 0.9999 0.9999         0.9999\n'
        '          HBA1C  9999 9999 9999 9999 0.9999 0.9999 0.9999  9999 9999 9999 9999 0.9999 0.9999 0.9999         0.9999\n'
        '       KETO-1YR  9999 9999 9999 9999 0.9999 0.9999 0.9999  9999 9999 9999 9999 0.9999 0.9999 0.9999         0.9999\n'
        ' MAJOR-DIABETES  9999 9999 9999 9999 0.9999 0.9999 0.9999  9999 9999 9999 9999 0.9999 0.9999 0.9999         0.9999\n'
        'MAKES-DECISIONS  9999 9999 9999 9999 0.9999 0.9999 0.9999  9999 9999 9999 9999 0.9999 0.9999 0.9999         0.9999\n'
        '        MI-6MOS  9999 9999 9999 9999 0.9999 0.9999 0.9999  9999 9999 9999 9999 0.9999 0.9999 0.9999         0.9999\n'
        '---------------  ----------------------------------------  ----------------------------------------  -------------\n'
        ' micro-averaged  9999 9999 9999 9999 0.9999 0.9999 0.9999  9999 9999 9999 9999 0.9999 0.9999 0.9999         0.9999\n'
        ' macro-averaged                      0.9999 0.9999 0.9999                      0.9999 0.9999 0.9999         0.9999\n'
        '------------------------------------------------------------------------------------------------------------------\n'
    )
    draw = draw.replace('999999', '{:6d}')
    draw = draw.replace('0.9999', '{:6.4f}')
    draw = draw.replace('9999', '{:4d}')
    # put all the values into a unique list
    values = list()
    # patients
    values.append(metrics['patients'])
    # tags
    for label2results in metrics['tags'].values():
        for results in label2results.values():
            values.extend(results.values())
    # micro
    for results in metrics['micro'].values():
        values.extend(results.values())
    # macro
    for results in metrics['macro'].values():
        values.extend(results.values())
    # final table
    return draw.format(*values)


def evaluate2str(true, pred):
    return metrics2str(evaluate(true, pred))
