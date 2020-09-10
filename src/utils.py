#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Copyright 2018, 2019, 2020 Rui Antunes
                           João Figueira Silva
                           Arnaldo Pereira
                           Sérgio Matos

https://github.com/ruiantunes/2018-n2c2-track-1


Utilities.

Due to performance and readability purposes, there are weak or none
consistency validations (duck typing).


Routines listing
----------------
create_directory
str2line


Classes listing
---------------
Printer

"""


# third-party modules
import datetime
import os


def create_directory(path):
    r"""
    Constructs (recursively) a directory.

    This function creates a specific directory. If the directory already
    exists as a directory nothing is made. Otherwise, if the path
    specified exists and is not a directory an exception is raised.

    Parameters
    ----------
    path : str
        Directory to create. Consecutive folders are allowed, e.g.
        'path/to/new/directory'. If `path` already exists but is not a
        directory, an `AssertionError` is raised.

    Raises
    ------
    AssertionError
        If `path` already exists and is not a directory.

    Example
    -------
    >>> import os
    >>> from utils import create_directory
    >>> create_directory(os.path.join('insert', 'here', 'the', 'path'))
    >>>

    """
    if os.path.exists(path):
        e = '\'{}\' already exists and is not a directory!'.format(path)
        assert os.path.isdir(path), e
    else:
        os.makedirs(path)


def str2line(s):
    r"""
    Converts white-space characters to spaces.

    Parameters
    ----------
    s : str
        String to be converted to a string without lines. The string is
        stripped.

    Returns
    -------
    s : str
        String without newlines.

    Example
    -------
    >>> from utils import str2line
    >>> s = 'This is\tsome\nstring.'
    >>> str2line(s)
    'This is some string.'
    >>>

    """
    return ' '.join(str(s).split())


class Printer:
    r"""
    Robust and flush-forced printing (date information can be added).

    This class handles printing. The purpose of this class is to print
    the information to the console (to one can see the progress
    information of a script in real-time) and simulteanously to save it
    to a file. This is useful to store outputs/results of a program.
    Notes:
    1. flush is `True` in the `print` function internal usage, so new
       prints in a same line will be instantaneously printed into the
       console.
    2. Only writing text is accepted (binary mode is not accepted).
    3. UTF-8 encoding is used.

    Attributes
    ----------
    _filepath
    _overwrite
    _sep
    _time_format

    Methods listing
    ---------------
    __init__
    print
    date

    Example
    -------
    >>> from utils import Printer
    >>> p = Printer()
    >>> p.date('Print: console.')
    12:29:04.863692 2018/Mar/17	Print: console.
    >>> p = Printer('test.log')
    >>> p.date('Print: console and file.')
    12:29:07.434040 2018/Mar/17	Print: console and file.
    >>> p.date('The 1st line.\nThe 2nd line.')
    12:29:09.735111 2018/Mar/17	The 1st line.
    The 2nd line.
    >>> p.date('A new line is (implicitly) added.')
    12:29:11.728569 2018/Mar/17	A new line is (implicitly) added.
    >>> p.date('A new line is (explicitly) added.\n', end='')
    12:29:13.282992 2018/Mar/17	A new line is (explicitly) added.
    >>>

    """
    #
    def __init__(self, filepath=None, overwrite=False, sep='\t'):
        r"""
        Constructor method. It initializes the object.

        Parameters
        ----------
        filepath : str, optional
            The filepath to save the prints. If `filepath` is considered
            `False`, then the prints are not saved to any file. Default:
            `None`.
        overwrite : bool, optional
            If `True` and the `filepath` specifies a file, that file is
            overwritten. If the file exists but `overwrite` is `False`,
            then an `AssertionError` is raised. Default: `False`.
        sep : str, optional
            If it is the case, this separator is used to separate the
            date and the message. Default: '\t' (tab).

        Raises
        ------
        AssertionError
            If `filepath` exists but is not a file. Or if `filepath` is
            already a file but `overwrite` is `False`.

        """
        if filepath:
            if os.path.exists(filepath):
                e = '{} is not a file!'.format(repr(filepath))
                assert os.path.isfile(filepath), e
                e = '{} cannot be overwritten!'.format(repr(filepath))
                assert overwrite, e
            else:
                dpath = os.path.split(filepath)[0]
                if dpath:
                    create_directory(dpath)
            with open(filepath, mode='w', encoding='utf-8') as f:
                _ = f.write('')
        self._filepath = filepath
        self._overwrite = overwrite
        self._sep = sep
        self._time_format = '%H:%M:%S.%f %Y/%b/%d'
    #
    def print(self, s='', end='\n', date=False):
        r"""
        Prints the message `s` (date can be added).

        Parameters
        ----------
        s : str, optional
            String to print. Default: `''`.
        end : str, optional
            String to append to the end of the message. Default: '\n'.
        date : bool, optional
            If `True` the current date is printed followed by the
            separator `sep` and the message `s`. Default: `False`.

        """
        if date:
            now = datetime.datetime.now().strftime(self._time_format)
            s = now + self._sep + s
        s += end
        print(s, end='', flush=True)
        if self._filepath:
            with open(self._filepath, mode='a', encoding='utf-8') as f:
                _ = f.write(s)
    #
    def date(self, s='', end='\n'):
        r"""
        Prints the current date and the message `s`.

        Parameters
        ----------
        s : str, optional
            String to print. Default: `''`.
        end : str, optional
            String to append to the end of the message. Default: '\n'.

        """
        self.print(s, end=end, date=True)
