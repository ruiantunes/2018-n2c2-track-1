# n2c2 2018 — Track 1: cohort selection for clinical trials

This repository contains source code from our participation in the
[n2c2 2018 shared-task (Track 1)](https://portal.dbmi.hms.harvard.edu/projects/n2c2-2018-t1/).


## Requirements

Ubuntu, Python 3.6.4. Install the required packages:
```
$ pip install -r requirements.txt
```


## Usage

1. Place the n2c2 dataset into the [data/n2c2/](data/n2c2/) directory.

1. Input parameters can be modified in the [src/system.py](src/system.py) script.

1. In the [src/](src) directory run:
    ```
    $ python3 system.py
    ```


## Reference

If you use this code in your work, please cite our
[publication](https://doi.org/10.5220/0007349300590067):

```
@inproceedings{Antunes2019a,
  address   = {{Prague, Czech Republic}},
  author    = {Antunes, Rui and Silva, Jo{\~a}o Figueira and Pereira, Arnaldo and Matos, S{\'e}rgio},
  booktitle = {12th {{International Joint Conference}} on {{Biomedical Engineering Systems}} and {{Technologies}}},
  month     = feb,
  pages     = {59--67},
  publisher = {{SciTePress}},
  title     = {Rule-based and machine learning hybrid system for patient cohort selection},
  url       = {https://doi.org/10.5220/0007349300590067},
  year      = {2019},
}
```
