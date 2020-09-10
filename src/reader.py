#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Copyright 2018, 2019, 2020 Rui Antunes
                           João Figueira Silva
                           Arnaldo Pereira
                           Sérgio Matos

https://github.com/ruiantunes/2018-n2c2-track-1


Corpus reader/writer for the n2c2 challenge.

A dataset is expected to be a folder containing several XML files. Each
XML file is a sample of the dataset. A complete sample contains two
fields: <TEXT> and <TAGS> (but both are optional). The <TEXT> value is a
string containing a set of longitudinal records, separated by lines of
asterisks, of a single patient. The <TAGS> field represents the values
of the 13 selection criteria (of the respective patient). The tags are
the following:
    - ABDOMINAL
    - ADVANCED-CAD
    - ALCOHOL-ABUSE
    - ASP-FOR-MI
    - CREATININE
    - DIETSUPP-2MOS
    - DRUG-ABUSE
    - ENGLISH
    - HBA1C
    - KETO-1YR
    - MAJOR-DIABETES
    - MAKES-DECISIONS
    - MI-6MOS

If the two fields (<TEXT> and <TAGS>) are not present, default values
are assumed. By default, <TEXT> value is an empty string, <TAGS> labels
are equal to 'not met'.


Routines listing
----------------
str2date
months_difference


Classes listing
---------------
Patient
Corpus
Extractor

"""


# third-party modules
from datetime import date
from lxml import etree
import os
import re

# own modules
from support import clean_medical_documents
from utils import create_directory


# tags
TAGS = [
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

# number of tags
N_TAGS = len(TAGS)

# unique values of months
MONTHS = [None, 6, 12]

# required months of each respective tag
TAG2MONTHS = {
    'ABDOMINAL': None,
    'ADVANCED-CAD': None,
    'ALCOHOL-ABUSE': None,
    'ASP-FOR-MI': None,
    'CREATININE': None,
    'DIETSUPP-2MOS': None,
    'DRUG-ABUSE': None,
    'ENGLISH': None,
    'HBA1C': None,
    'KETO-1YR': 12,
    'MAJOR-DIABETES': None,
    'MAKES-DECISIONS': None,
    'MI-6MOS': 6,
}


def str2date(s):
    r"""
    Converts a `str` date to a `datetime.date` date.

    Parameters
    ----------
    s : str
        The expected format is 'yyyy-mm-dd', e.g.: '2018-04-04'.

    Returns
    -------
    d : datetime.date
        The converted date.

    Example
    -------
    >>> from reader import str2date
    >>> s = '2018-04-04'
    >>> d = str2date(s)
    >>> type(d)
    <class 'datetime.date'>
    >>> (d.year, d.month, d.day)
    (2018, 4, 4)
    >>> print(d)
    2018-04-04
    >>>

    """
    return date(*[int(e) for e in s.split('-')])


def months_difference(d1, d2):
    r"""
    This function calculates the months difference between two dates.

    The difference `d2 - d1` will be considered for saying the number of
    months that they distance themselves. Only the years and months of
    the dates are considered (the days are ignored).

    Parameters
    ----------
    d1 : datetime.date
        The first date.
    d2 : datetime.date
        The second date.

    Returns
    -------
    m : int
        The number of months (difference between the two dates).

    Example
    -------
    >>> from reader import months_difference
    >>> d1 = str2date('2017-02-28')
    >>> d2 = str2date('2018-04-04')
    >>> months_difference(d1, d2)
    14
    >>>

    """
    return (d2.year * 12 + d2.month) - (d1.year * 12 + d1.month)


class Patient:
    r"""
    This class implements a patient.

    A patient information is obtained from loading a XML file. A
    complete file has two fields: <TEXT> and <TAGS>. The <TEXT> has a
    CDATA section [1]_ containing a set of 2-5 longitudinal records with
    their dates and texts. The <TAGS> field represent the values of the
    13 labels (selection criteria).
    Also note that, the months difference for each visit to the hospital
    is calculated (for each patient record) considering the current date
    equal to the date of the most recent visit.
    When reading the patient XML file both fields (<TEXT> and <TAGS>)
    are optional. However, if they are not present, default values are
    assumed. By default, <TEXT> value is an empty string, and <TAGS>
    labels are set to 'not met'.
    For the XML parsing it is used the `lxml` toolkit [2]_. The
    `ElementTree` API [3]_ has no ability to handle CDATA sections.

    References
    ----------
    ..[1] http://lxml.de/api.html#cdata
    ..[2] http://lxml.de/tutorial.html
    ..[3] https://docs.python.org/3/library/xml.etree.elementtree.html

    Class attributes
    ----------------
    _tags
    _tag2int
    _int2tag
    _label2int
    _int2label
    _regex_text

    Instance attributes
    -------------------
    _fpath
    _dpath
    _fname
    _text
    _now
    _labels
    _records

    Methods listing
    ---------------
    __init__
    get_patient
    get_label
    get_labels
    get_records
    get_document
    set_label
    set_labels
    to_xml
    write

    Example
    -------
    >>> import os
    >>> from __init__ import REPO
    >>> from reader import Patient
    >>> fpath = os.path.join(REPO, 'data', 'n2c2', 'train', '180.xml')
    >>> p = Patient(fpath)
    >>> p.get_labels()
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0]
    >>> p.set_labels(['not met', 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1])
    >>> p.get_labels()
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1]
    >>> records = p.get_records()
    >>> len(records)
    5
    >>> records_mi6mos = p.get_records(months=6)
    >>> len(records_mi6mos)
    1
    >>> document = p.get_document()
    >>> len(document)
    13761
    >>> document_mi6mos = p.get_document(months=6)
    >>> len(document_mi6mos)
    2532
    >>> print(p.to_xml()[:10])
    <?xml vers
    >>>

    """
    #
    _tags = [
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
    _tag2int = {t: i for i, t in enumerate(_tags)}
    _int2tag = {i: t for i, t in enumerate(_tags)}
    _label2int = {'not met': 0, 'met': 1}
    _int2label = {0: 'not met', 1: 'met'}
    _regex_text = re.compile(
        pattern=(
            r'.*?Record date: ([0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9])'
            r'\s*(.*?)\s*\*{100,}'
        ),
        flags=re.DOTALL,
    )
    #
    def __init__(self, fpath):
        r"""
        Constructor method. The file is read.

        Parameters
        ----------
        fpath : str
            A filepath containing a full record of a patient. The
            `fpath` is expected to point to a `.xml` input file.

        """
        fpath = os.path.abspath(fpath)
        e = 'Invalid filepath!'
        assert os.path.isfile(fpath) and fpath.endswith('.xml'), e
        self._fpath = fpath
        self._dpath, self._fname = os.path.split(fpath)
        # initialize default values
        self._text = ''
        self._now = None
        self._labels = {t: 0 for t in self._tags}
        # tree is initialized with the contents of the given XML file
        tree = etree.ElementTree(file=fpath)
        # find <TEXT> tag
        m = tree.find('TEXT')
        if m is not None:
            # get text inside the <TEXT> tag
            text = m.text
            if text is not None:
                self._text = text
                # separate records from text (using a regular
                # expression) - each record is a tuple (date, text)
                m = self._regex_text.findall(text)
                # if there are matches of records
                if len(m) > 0:
                    m = [(str2date(d), t) for d, t in m]
                    # most recent date (current date)
                    self._now = max([d for d, t in m])
                    # list of records (each record is a dict)
                    self._records = list()
                    for date, raw_text in m:
                        self._records.append(
                            {
                                'raw_text': raw_text,
                                'clean_text':
                                    clean_medical_documents([raw_text])[0],
                                'months': months_difference(date, self._now),
                            }
                        )
        # get the labels from the tags (if they exist)
        for tag in self._tags:
            m = tree.find('TAGS/' + tag)
            if m is not None:
                self._labels[tag] = self._label2int[m.get('met')]
    #
    def get_patient(self):
        r"""
        Returns the filename of the patient.

        Returns
        -------
        patient : str
            The filename of the patient.

        """
        return self._fname
    #
    def get_label(self, tag, text=False):
        r"""
        Returns the value of a specific label.

        Parameters
        ----------
        tag : str, optional
            Specifies the desired tag to return its respective label.
        text : bool, optional
            If `True` the label is converted to `str` type ('not met',
            'met'), otherwise it is returned in its `int` type (0, 1).
            Default: `False`.

        Returns
        -------
        label : int, str
            The respective label. It can be `int` or `str` depending on
            the `text` input argument.

        """
        label = self._labels[tag]
        if text:
            label = self._int2label[label]
        return label
    #
    def get_labels(self, text=False):
        r"""
        Returns the 13 labels of the patient.

        Parameters
        ----------
        text : bool, optional
            If `True` the labels are converted to `str` type ('not met',
            'met'), otherwise they are returned in their `int` type (0,
            1). Default: `False`.

        Returns
        -------
        labels : list
            A list containing the values of the 13 tags of the patient.

        """
        labels = list(self._labels.values())
        if text:
            return [self._int2label[l] for l in labels]
        else:
            return labels
    #
    def get_records(self, months=None, clean=False):
        r"""
        Return the list of records given a specific number of months.

        Parameters
        ----------
        months : int
            Number of past months to be considered. Past records may be
            ignored. If `None`, all records will be returned. Default:
            `None`.
        clean : bool, optional
            If `True` the records are clean (only letters are
            considered, etc.). Default: `False`.

        Returns
        -------
        records : list
            A list of the records of the patient. Each record is of
            `str` type. Some older records are not returned given a
            specific number of months.

        """
        if months is None:
            if clean:
                records = [r['clean_text'] for r in self._records]
            else:
                records = [r['raw_text'] for r in self._records]
        else:
            if clean:
                records = [
                    r['clean_text'] for r in self._records
                    if r['months'] <= months
                ]
            else:
                records = [
                    r['raw_text'] for r in self._records
                    if r['months'] <= months
                ]
        return records
    #
    def get_document(self, months=None, clean=False):
        r"""
        Return the full text given a specific number of months.

        The individual records are concatenated into a unique string.

        Parameters
        ----------
        months : int
            Number of past months to be considered. Past records may be
            ignored. If `None`, all records will be returned. Default:
            `None`.
        clean : bool, optional
            If `True` the document is clean (only letters are
            considered, etc.). Default: `False`.

        Returns
        -------
        text : str
            The full text of a patient given a specific number of past
            months.

        """
        document = '\n'.join(self.get_records(months=months, clean=clean))
        return document
    #
    def set_label(self, tag, label):
        r"""
        Change the label value of a specific tag.

        Parameters
        ----------
        tag : str
            The tag that will have its value changed.
        label : int, str
            The tag value will be changed to this value. It can be `str`
            ('not met', 'met') or `int` (0, 1).

        """
        if not isinstance(label, str):
            # convert label to `str`
            label = self._int2label[int(label)]
        assert isinstance(label, str)
        # convert label to `int`
        self._labels[tag] = self._label2int[label]
    #
    def set_labels(self, labels):
        r"""
        Change the value of the 13 tags of the patient.

        Parameters
        ----------
        labels : list
            A list containing 13 values (one per tag) following the
            respective (alphabetic) order.

        """
        assert len(labels) == len(self._tags)
        for tag, label in zip(self._tags, labels):
            self.set_label(tag, label)
    #
    def to_xml(self, use_text=True, use_tags=True):
        r"""
        Returns the respective `.xml` string.

        The <TEXT> and <TAGS> fields are optional.

        Parameters
        ----------
        use_text : bool, optional
            If `True` the generated XML contains the <TEXT> field,
            otherwise the <TEXT> field is omitted. Default: `True`.
        use_tags : bool, optional
            If `True` the generated XML contains the <TAGS> field,
            otherwise the <TAGS> field is omitted. Default: `True`.

        """
        root = etree.Element('PatientMatching')
        if use_text:
            text = etree.Element('TEXT')
            text.text = etree.CDATA(self._text)
            root.append(text)
        if use_tags:
            tags = etree.Element('TAGS')
            for tag in self._tags:
                elem = etree.Element(tag)
                elem.attrib['met'] = self.get_label(tag, text=True)
                tags.append(elem)
            root.append(tags)
        return etree.tostring(
            element_or_tree=root,
            encoding='utf-8',
            xml_declaration=True,
            pretty_print=True,
        ).decode('utf-8')
    #
    def write(self, dpath, use_text=True, use_tags=True, overwrite=False):
        r"""
        The patient XML tree is written to a file.

        Only the directory path needs to be specified, since the
        filename is already defined. The <TEXT> and <TAGS> fields can be
        omitted.

        Parameters
        ----------
        dpath : str
            The output directory path. If necessary, directories are
            created. The filename is the same (as defined).
        use_text : bool, optional
            If `True` the generated XML contains the <TEXT> field,
            otherwise the <TEXT> field is omitted. Default: `True`.
        use_tags : bool, optional
            If `True` the generated XML contains the <TAGS> field,
            otherwise the <TAGS> field is omitted. Default: `True`.
        overwrite: bool, optional
            If `True` overwritten is allowed, otherwise it is not.
            Default: `False`.

        Raises
        ------
        AssertionError
            The output directory path cannot be the same as the input
            directory path.

        """
        dpath = os.path.abspath(dpath)
        # create directory (if necessary)
        create_directory(dpath)
        e = 'Writing to the same directory is not allowed!'
        assert not os.path.samefile(self._dpath, dpath), e
        fpath = os.path.join(dpath, self._fname)
        if os.path.exists(fpath):
            e = 'Filepath already exists and it is not a file!'
            assert os.path.isfile(fpath), e
            assert overwrite, 'Overwitten is not allowed!'
        with open(fpath, mode='w', encoding='utf-8') as f:
            _ = f.write(self.to_xml(use_text=use_text, use_tags=use_tags))


class Corpus:
    r"""
    This class implements a corpus (set of patients).

    Attributes
    ----------
    _dpath
    _patients

    Methods listing
    ---------------
    __init__
    get_patients
    get_documents
    get_labels
    set_labels
    write

    Example
    -------
    >>> import os
    >>> from __init__ import REPO
    >>> from reader import Corpus
    >>> dpath = os.path.join(REPO, 'data', 'n2c2', 'train')
    >>> c = Corpus(dpath)
    >>> docs = c.get_documents()
    >>> len(docs)
    202
    >>> tag = 'ABDOMINAL'
    >>> labels = c.get_labels(tag=tag)
    >>> labels[:5]
    [0, 0, 1, 0, 0]
    >>> len(labels)
    202
    >>> # how to set labels
    >>> c.set_labels(tag=tag, labels=labels)
    >>> # how to persist the new corpus to disk
    >>> c.write(dpath='test')
    >>>

    """
    #
    def __init__(self, dpath):
        r"""
        Constructor method. It initializes the object.

        Parameters
        ----------
        dpath : str
            The `dpath` is expected to be a directory containing several
            `.xml` files. Each file represents a patient.

        """
        self._dpath = os.path.abspath(dpath)
        self._patients = [
            Patient(os.path.join(dpath, fname))
            for fname in sorted(os.listdir(dpath))
            if fname.endswith('.xml')
            and os.path.isfile(os.path.join(dpath, fname))
        ]
    #
    def get_patients(self):
        r"""
        Returns the list of patients (filenames).

        Returns
        -------
        patients : list
            A list of the filenames of the patients.

        """
        return [p.get_patient() for p in self._patients]
    #
    def get_documents(self, months=None, clean=False):
        r"""
        Returns the raw documents given a specific tag.

        Parameters
        ----------
        months : int
            Number of past months to be considered. Past records may be
            ignored. If `None`, all records will be returned. Default:
            `None`.
        clean : bool, optional
            If `True` the documents are clean (only letters are
            considered, etc.). Default: `False`.

        Returns
        -------
        documents : list
            A `list` of the raw/clean longitudinal records. Each entry
            says respect to a unique patient.

        """
        return [
            p.get_document(months=months, clean=clean)
            for p in self._patients
        ]
    #
    def get_labels(self, tag, text=False):
        r"""
        Returns the labels given a specific tag.

        Parameters
        ----------
        tag : str
            Specifies the desired tag to return its respective labels.
        text : bool, optional
            If `True` the labels are converted to `str` type ('not met',
            'met'), otherwise they are returned in their `int` type (0,
            1). Default: `False`.

        Returns
        -------
        labels : list
            This is a `list` of the labels (of a specific tag). Each
            entry says respect to a unique patient.

        """
        return [
            p.get_label(tag=tag, text=text)
            for p in self._patients
        ]
    #
    def set_labels(self, tag, labels):
        r"""
        Modifies the labels given a specific tag.

        Parameters
        ----------
        tag : str
            The tag that will have its value changed.
        labels : list
            This is a `list` of the labels (of a specific tag). Each
            entry says respect to a unique patient.

        """
        assert len(labels) == len(self._patients)
        for p, l in zip(self._patients, labels):
            p.set_label(tag=tag, label=l)
    #
    def write(self, dpath, use_text=True, use_tags=True, overwrite=False):
        r"""
        The patient XML files are written into a specific directory.

        Only the directory path needs to be specified, since the patient
        filenames are already defined. The <TEXT> and <TAGS> fields can
        be omitted.

        Parameters
        ----------
        dpath : str
            The output directory path. If necessary, directories are
            created.
        use_text : bool, optional
            If `True` the generated XML contains the <TEXT> field,
            otherwise the <TEXT> field is omitted. Default: `True`.
        use_tags : bool, optional
            If `True` the generated XML contains the <TAGS> field,
            otherwise the <TAGS> field is omitted. Default: `True`.
        overwrite: bool, optional
            If `True` overwritten is allowed, otherwise it is not.
            Default: `False`.

        Raises
        ------
        AssertionError
            The output directory path cannot be the same as the input
            directory path.

        """
        dpath = os.path.abspath(dpath)
        # create directory (if necessary)
        create_directory(dpath)
        e = 'Writing to the same directory is not allowed!'
        assert not os.path.samefile(self._dpath, dpath), e
        for p in self._patients:
            p.write(
                dpath=dpath,
                use_text=use_text,
                use_tags=use_tags,
                overwrite=overwrite,
            )


class Extractor:
    #
    REGEX_CURRENT_LINE = re.compile(r'[  \t]+')
    REGEX_ROW = re.compile(r'\w+[ ]*[(]?\w*[ ]*\w*[)]?[ \t]+[<]?\d+(([ ]*[A-Za-z])|([.]?\d*[LH]?))[ \t]+[(]*\d+\.*\d*-\d+\.*\d*[)]*[ \t]+', re.IGNORECASE)
    REGEX_KEY = re.compile(r'[ \t]+[<]?\d+')
    REGEX_CURRENT_VALUE = re.compile(r'\d+[.]?\d*', re.IGNORECASE)
    REGEX_MIN_MAX = re.compile(r'\d+[.]?\d*-\d+[.]?\d*', re.IGNORECASE)
    #
    def __extract(self, raw_text):
        extracted_table = {}
        lines = raw_text.splitlines()
        extracted_text = ''
        previous_line = ''
        current_line = ''
        for line in lines:
            if not line.strip():
                continue
            else:
                current_line = self.REGEX_CURRENT_LINE.sub(' ', line.strip())
                if self.REGEX_ROW.search(current_line):
                    current_line = current_line
                    current_line = current_line.replace('Absolute', 'Abs')
                    current_line = current_line.replace('Isoenzymes', 'Isoenz')
                    current_line = current_line.replace('Carbon Dioxide', 'CO2')
                    current_line = current_line.replace('Bilirubin(Total)', 'Total Bilirubin')
                    current_line = current_line.replace('Bilirubin(Direct', 'Direct Bilirubin')
                    current_line = current_line.replace('Bilirubin(Direct)', 'Direct Bilirubin')
                    current_line = current_line.replace(' (Stat Lab)', '')
                    current_line = current_line.replace('Plasma ', '')
                    current_line = current_line.replace('Blood Urea Nitro', 'Urea Nitro')
                    current_line = current_line.replace('UREA N', 'Urea Nitro')
                    current_line = current_line.replace('Neutrophils - Au', 'Neutrophils')
                    current_line = current_line.replace('Neutrophils - Ma', 'Neutrophils')
                    current_line = current_line.replace('Lymphocytes - Au', 'Lymphocytes')
                    current_line = current_line.replace('Lymphocytes - Ma', 'Lymphocytes')
                    current_line = current_line.replace('Monocytes - Manu', 'Monocytes')
                    current_line = current_line.replace('Monocytes - Auto', 'Monocytes')
                    current_line = current_line.replace('Eosinophils - Ma', 'Eosinophils')
                    current_line = current_line.replace('Eosinophils - Au', 'Eosinophils')
                    current_line = current_line.replace('Basophils - Manu', 'Basophils')
                    current_line = current_line.replace('Basophils - Auto', 'Basophils')
                    key = self.REGEX_KEY.split(current_line)[0]
                    extracted_table[key] = [float(self.REGEX_CURRENT_VALUE.search(current_line).group(0)), float(self.REGEX_MIN_MAX.search(current_line).group(0).split('-')[0]), float(self.REGEX_MIN_MAX.search(current_line).group(0).split('-')[1])]
                else:
                    extracted_text = extracted_text + '\n' + current_line
                previous_line = current_line
        return extracted_text, extracted_table
    #
    def extract(self, raw_texts):
        extracted_texts = list()
        extracted_tables = list()
        for raw_text in raw_texts:
            text, table = self.__extract(raw_text)
            extracted_texts.append(text)
            extracted_tables.append(table)
        extracted_texts = clean_medical_documents(extracted_texts)
        return extracted_texts, extracted_tables
