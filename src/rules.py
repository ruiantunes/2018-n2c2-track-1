#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Copyright 2018, 2019, 2020 Rui Antunes
                           João Figueira Silva
                           Arnaldo Pereira
                           Sérgio Matos

https://github.com/ruiantunes/2018-n2c2-track-1


Rules for classifying medical reports.


Classes listing
---------------
RuleBasedClassifier
ImprovedRuleBasedClassifier
BaselineClassifier

"""


import os
import re
from __init__ import REPO
from reader import TAGS


class RuleBasedClassifier:
    r"""
    Rules for classifying medical reports.

    Attributes
    ----------
    _tags

    Methods listing
    ---------------
    __init__
    predict

    Example
    -------
    >>> from rules import RuleBasedClassifier
    >>> clf = RuleBasedClassifier()
    >>> X = ['Creatinine 2.2', 'CRE 1.0', 'Cr. 1.53']
    >>> clf.predict(tag='CREATININE', X=X)
    [1, 0, 1]
    >>>

    """
    #
    _tags = [
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
    #
    def __init__(self):
        r"""
        Constructor method. Nothing is done.

        """
        pass
    #
    def predict(self, tag, X):
        r"""
        Predict method.

        Predicts a list of documents given a specific tag. Each tag
        represents a different classification problem.

        Parameters
        ----------
        tag : str
            Tag to be considered.
        X : list
            List of documents to be predicted.

        """
        assert tag in self._tags
        y = []
        if tag == 'ADVANCED-CAD':
            with open(os.path.join(REPO, 'data', 'CAD_drugs_clean.txt')) as fin:
                cad_drugs = '|'.join(fin.read().splitlines())
            regex = re.compile(r"\b(%s)\b" %cad_drugs, re.IGNORECASE)
            regex_mi = re.compile(r"(.{0,40})\b(myocardial infarction|MI|IMI|AMI|ASMI|HMI|NQWMI|NSTEMI|OASMI|SEMI|STEMI|TIMI)\b(.{0,20})", re.IGNORECASE)
            regex_angina = re.compile(r"(.{0,40})\bangina\b(.{0,20})", re.IGNORECASE)
            regex_ischemia = re.compile(r"(.{0,40})\bischemia\b(.{0,20})", re.IGNORECASE)
            regex_neg = re.compile(r"\b(rule-out|rule out|ruled out|ruling out|r\\?o|no|not|negative|free|unlikely|any|absence|absent|father|mother|dad|mom|grandfather|grandmother|brother|sister|son|daughter|family|fh)\b", re.IGNORECASE)
            for i, x in enumerate(X):
                p = 0
                complications = 0
                if len(list(set([m.group(0) for m in regex.finditer(x)]))) >= 2:
                    complications +=1
                for m in regex_mi.finditer(x):
                    if not regex_neg.search(m.group(1)) and not regex_neg.search(m.group(3)):
                        complications += 1
                        break
                for m in regex_angina.finditer(x):
                    if not regex_neg.search(m.group(1)) and not regex_neg.search(m.group(2)):
                        complications += 1
                        break
                for m in regex_ischemia.finditer(x):
                    if not regex_neg.search(m.group(1)) and not regex_neg.search(m.group(2)):
                        complications += 1
                        break
                if complications >= 2:
                    p = 1
                y.append(p)
        elif tag == 'ALCOHOL-ABUSE':
            # not met
            denies = re.compile(
                pattern=r'(?:deni|deny).{,20}?(?:alcohol|drink|etoh)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            no_abuse_drink = re.compile(
                pattern=r'(?:ago|no|past|prev|prior|history|h/o).{,20}?(?:abuse|dependence|heavy).{,20}?(?:alcohol|drink|etoh)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            no_drink_abuse = re.compile(
                pattern=r'(?:ago|no|past|prev|prior|history|h/o).{,20}?(?:alcohol|drink|etoh).{,20}?(?:abuse|dependence|heavy)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            drink_no_abuse = re.compile(
                pattern=r'(?:alcohol|drink|etoh).{,20}?(?:ago|no|past|prev|prior|history|h/o).{,20}?(?:abuse|dependence|heavy)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            abuse_drink_no = re.compile(
                pattern=r'(?:abuse|dependence|heavy).{,20}?(?:alcohol|drink|etoh).{,20}?(?:ago|no|past|prev|prior|history|h/o)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            # met
            limit = re.compile(
                pattern=r'limit.{,20}?(?:alcohol|drink|etoh)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            amount = re.compile(
                pattern=r'amount.{,20}?(?:alcohol|etoh).{,20}?drink',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            therapy = re.compile(
                pattern=r'therapy.{,20}?(?:alcohol|drink|etoh)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            drink_abuse = re.compile(
                pattern=r'(?:alcohol|drink|etoh).{,20}?(?:abuse|dependence|heavy)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            abuse_drink = re.compile(
                pattern=r'(?:abuse|dependence|heavy).{,20}?(?:alcohol|drink|etoh)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            for x in X:
                if denies.search(x) or no_abuse_drink.search(x) or no_drink_abuse.search(x) or drink_no_abuse.search(x) or abuse_drink_no.search(x):
                    y.append(0)
                    continue
                if limit.search(x) or amount.search(x) or therapy.search(x) or drink_abuse.search(x) or abuse_drink.search(x):
                    y.append(1)
                    continue
                y.append(0)
        elif tag == 'ASP-FOR-MI':
            regex = re.compile(r"\b(hypertension|myocardial infarction|MI|IMI|AMI|ASMI|HMI|NQWMI|NSTEMI|OASMI|SEMI|STEMI|TIMI)\b", re.IGNORECASE)
            regex2 = re.compile(r"(.{0,40})\b(aspirin|asa|acetylsalicylic)\b(.{0,40})", re.IGNORECASE)
            regex_neg = re.compile(r"(avoid|stop|causes|rash|ulcer|allerg)", re.IGNORECASE)
            for x in X:
                p = 0
                for m in regex2.finditer(x):
                    if not regex_neg.search(m.group(1)) and not regex_neg.search(m.group(3)):
                        p = 1
                y.append(p)
        elif tag == 'CREATININE':
            cre = re.compile(
                pattern=r'cre?\.?(?:atinine)?(?:\s+of)?\s+(\d+\.\d+)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            creatinine = re.compile(
                pattern=r'creatinine.{,30}?(\d+\.\d+)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            for x in X:
                p = 0
                for value in cre.findall(x) + creatinine.findall(x):
                    if float(value) > 1.5:
                        p = 1
                y.append(p)
        elif tag == 'DIETSUPP-2MOS':
            regex = re.compile(r"(.{0,40})\b(calcium|copper|cyanocobalamin|epogen|ferrous gluconate|ferrous sulfate|fish oil|folate|k-dur|klor-con|minerals|nephrocaps|niferex|procrit|tocopherol|tums|ascorbic acid|folic acid|calcium|chromium|iron|magnesium|potassium|selenium|zinc|vitamin B[-\s]?1|vitamin B[-\s]?2|vitamin B[-\s]?6|vitamin B[-\s]?12|vitamin B[-\s]?100|vitamin C|vitamin E|vitamin G|vitamin H|vitamin M|vitamin suppl|mineral suppl|Betaxin|niacin|m\.?v\.?i\.?|thiamine)\b(.{0,10})", re.IGNORECASE)
            regex_left_neg = re.compile(r"(elevated|high|low|normal|check|past|previous|was|recommend|counsel)", re.IGNORECASE)
            regex_right_neg = re.compile(r"(\s{3,}|[\s\n]*(is|was|were|of)?[\s\n]*\d+\.\d|[\s\n]*(is|was|were)|[\s\n]*(is|was)?[\s(]*(elevated|high|low|deficien|normal|channel|studies|study|stat|lab))", re.IGNORECASE)
            for i, x in enumerate(X):
                p = 0
                for m in regex.finditer(x):
                    if not regex_left_neg.search(m.group(1)) and not regex_right_neg.search(m.group(3)):
                        p = 1
                        break
                y.append(p)
        elif tag == 'DRUG-ABUSE':
            # met
            hist_drug_use = re.compile(
                pattern=r'(?:ago|past|prev|prior|history|h/o).{,20}?(?:cocaine|drug|heroin|illicit|substance).{,20}?(?:abuse|dependence|heavy|smok|use)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            hist_use_drug = re.compile(
                pattern=r'(?:ago|past|prev|prior|history|h/o).{,20}?(?:abuse|dependence|heavy|smok|use).{,20}?(?:cocaine|drug|heroin|illicit|substance)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            use_drug_hist = re.compile(
                pattern=r'(?:abuse|dependence|heavy|smok|use).{,20}?(?:cocaine|drug|heroin|illicit|substance).{,20}?(?:ago|past|prev|prior|history|h/o)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            for x in X:
                if hist_drug_use.search(x) or hist_use_drug.search(x) or use_drug_hist.search(x):
                    y.append(1)
                    continue
                y.append(0)
        elif tag=='ENGLISH':
            regex1 = re.compile(
                '(?:arabic|aramaic|armenian|bulgarian|burmese|cambodian|cantanese|cantonese|catonese|chinese|creole|croele|ethiopian|farsi|farsti|french|greek|gujarati|haitan|hindi|indonesian|infant|italian|japanese|korean|laotian|latvian|loatian|mandarin|nonenglish|persian|polish|portugese|portuguese|romanian|rusian|russian|somali|spainish|spanish|thai|tiawanese|urdu|vietmanese|vietnamese|yiddish)[\s-]+(?:speaking)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            regex2 = re.compile(
                "(?:male|woman|lady|patient|pt)[\s]+from[\s]+(the[\s]+)?(afghanistan|albania|algeria|andorra|angola|antigua|antigua and barbuda|argentina|armenia|australia|austria|azerbaijan|bahamas|bahrain|bangladesh|barbados|belarus|belgium|belize|benin|bhutan|bolivia|bosnia|bosnia and herzegovina|botswana|brazil|brunei|bulgaria|burkina|burkina faso|burundi|cabo verde|cape verde|cape vert|cambodia|cambodja|cameroon|canada|central african republic|chad|chile|china|colombia|comoros|congo|costa rica|croatia|cuba|cyprus|czechia|côte d'ivoire|ivory coast|korea|democratic republic of congo|republic of congo|denmark|djibouti|dominica|dominican republic|ecuador|egypt|el salvador|equatorial guinea|eritrea|estonia|ethiopia|faroe islands|fiji|finland|france|gabon|gambia|georgia|germany|ghana|greece|grenada|guatemala|guinea|guinea-bissau|guyana|haiti|honduras|hungary|iceland|india|indonesia|iran|iraq|ireland|israel|italy|jamaica|japan|jordan|kazakhstan|kenya|kiribati|kuwait|kyrgyzstan|laos|latvia|lebanon|lesotho|liberia|libya|lithuania|luxembourg|madagascar|malawi|malaysia|maldives|mali|malta|mauritania|mauritius|mexico|monaco|mongolia|montenegro|morocco|mozambique|myanmar|namibia|nauru|nepal|netherlands|new zealand|nicaragua|niger|nigeria|niue|norway|oman|pakistan|palau|panama|papua new guinea|papua|new guinea|paraguay|peru|philippines|poland|portugal|qatar|south korea|north korea|moldova|romania|russia|rwanda|st kitts|saint kitts|saint kitts and nevis|st lucia|saint lucia|st vincent|saint vincent|saint vincent and the grenadines|samoa|san marino|sao tome|saudi arabia|senegal|serbia|seychelles|sierra leone|singapore|slovakia|slovenia|solomon islands|somalia|south africa|south sudan|spain|sri lanka|sudan|suriname|swaziland|sweden|switzerland|syria|tajikistan|thailand|macedonia|timor|timor-leste|togo|tonga|trinidad|trinidad and tobago|tunisia|turkey|turkmenistan|tuvalu|uganda|ukraine|uae|united arab emirates|uk|united kingdom|tanzania|uruguay|uzbekistan|vanuatu|venezuela|vietnam|viet nam|yemen|zambia|zimbabwe)",
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            for x in X:
                p = 1
                if regex1.search(x) or regex2.search(x):
                    p = 0
                y.append(p)
        elif tag == 'HBA1C':
            a1c = re.compile(
                pattern=r'a1c.{,30}?(\d+\.\d+)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            for x in X:
                p = 0
                for value in a1c.findall(x):
                    f = float(value)
                    if (f >= 6.5) and (f <= 9.5):
                        p = 1
                y.append(p)
        elif tag == 'KETO-1YR':
            # not met
            no_keto = re.compile(
                pattern=r'no.{,30}?(?:dka|ketones|ketoacidosis)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            # met
            keto = re.compile(
                pattern=r'(?:ketones\s+pos)|(?:ketoacidosis)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            for x in X:
                if no_keto.search(x):
                    y.append(0)
                    continue
                if keto.search(x):
                    y.append(1)
                    continue
                y.append(0)
        elif tag == 'MAKES-DECISIONS':
            # not met
            regex1 = re.compile(
                pattern=r'(?:daughter|wife|husband|family|niece|father|mother|son|brother|sister|sibling).{,20}?make.{,20}?decision',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            regex2 = re.compile(
                pattern=r'(?:pt|patient).{,20}?no.{,20}?make.{,20}?decision',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            regex3 = re.compile(
                pattern=r'mental.{,20}?retardation',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            regex4 = re.compile(
                pattern=r'(?:confus|depress|altered|wors|bad).{,20}?mental.{,20}?status',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            regex5 = re.compile(
                pattern=r'(?:pt|patient).{,20}?intubat',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            for x in X:
                if regex1.search(x) or regex2.search(x) or regex3.search(x) or regex4.search(x) or regex5.search(x):
                    y.append(0)
                    continue
                y.append(1)
        elif tag == 'MI-6MOS':
            regex_mi = re.compile(r"(.{0,40})\b(myocardial infarction|MI|IMI|AMI|ASMI|HMI|NQWMI|NSTEMI|OASMI|SEMI|STEMI|TIMI)\b(.{0,20})", re.IGNORECASE)
            regex_neg = re.compile(r"\b(rule-out|rule out|ruled out|ruling out|r\\?o|old|past|prior|post|s\\?p|s/?p|no|not|negative|free|unlikely|any|absence|absent|had|father|mother|dad|mom|grandfather|grandmother|brother|sister|son|daughter|family|fh|history)\b", re.IGNORECASE)
            for i, x in enumerate(X):
                p = 0
                for m in regex_mi.finditer(x):
                    if not regex_neg.search(m.group(1)) and not regex_neg.search(m.group(3)):
                        p = 1
                        break
                y.append(p)
        assert len(y) == len(X)
        return y

# to improve rules
HTML_REGEX = re.compile(
    pattern=r'&#\d+;',
    flags=re.IGNORECASE,
)


def remove_html_chars(doc):
    return HTML_REGEX.sub('', doc)


DATE1_REGEX = re.compile(
    pattern=r'\d{4,4}/\d{1,2}/\d{1,2}|\d{1,2}/\d{1,2}/\d{4,4}|\d{1,2}/\d{1,2}/\d{1,2}|\d{4,4}/\d{1,2}|\d{1,2}/\d{4,4}|\d{1,2}/\d{1,2}|\d{4,4}',
    flags=re.IGNORECASE,
)


DATE2_REGEX = re.compile(
    pattern=r'\d{1,2}\s+(?:day|week|month|year)s?',
    flags=re.IGNORECASE,
)


DATE3_REGEX = re.compile(
    pattern=r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
    flags=re.IGNORECASE,
)


def remove_dates(doc):
    doc = DATE1_REGEX.sub('', doc)
    doc = DATE2_REGEX.sub('', doc)
    doc = DATE3_REGEX.sub('', doc)
    return doc


def get_lines(doc):
    lines = list()
    for line in doc.splitlines():
        line = line.strip()
        if line:
            line = ' '.join(line.split())
            lines.append(line)
    return lines


class ImprovedRuleBasedClassifier:
    #
    def predict(self, tag, X):
        r"""
        Predict method.

        Predicts a list of documents given a specific tag. Each tag
        represents a different classification problem.

        Parameters
        ----------
        tag : str
            Tag to be considered.
        X : list
            List of documents to be predicted.

        """
        assert tag in TAGS
        y = []
        if tag == 'ABDOMINAL':
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/iSU-Sk_00D0
            # "includes small bowel obstruction, but not large bowel obstruction"
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/qP5rABEYUYE
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/vleeLtXSy6A
            assert False
        elif tag == 'ADVANCED-CAD':
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/Yub-jzN6w4M
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/w8fgwqU7W8g
            with open(os.path.join(REPO, 'data', 'CAD_drugs_clean.txt')) as fin:
                cad_drugs = '|'.join(fin.read().splitlines())
            regex = re.compile(r"\b(%s)\b" %cad_drugs, re.IGNORECASE)
            regex_mi = re.compile(r"(.{0,40})\b(myocardial infarction|MI|IMI|AMI|ASMI|HMI|NQWMI|NSTEMI|OASMI|SEMI|STEMI|TIMI)\b(.{0,20})", re.IGNORECASE)
            regex_angina = re.compile(r"(.{0,40})\bangina\b(.{0,20})", re.IGNORECASE)
            regex_ischemia = re.compile(r"(.{0,40})\bischemia\b(.{0,20})", re.IGNORECASE)
            regex_neg = re.compile(r"\b(rule-out|rule out|ruled out|ruling out|r\\?o|r/o|no|not|negative|free|unlikely|any|absence|absent|father|mother|dad|mom|grandfather|grandmother|brother|sister|son|daughter|family|fh)\b", re.IGNORECASE)
            for i, x in enumerate(X):
                p = 0
                complications = 0
                if len(list(set([m.group(0) for m in regex.finditer(x)]))) >= 2:
                    complications +=1
                for m in regex_mi.finditer(x):
                    if not regex_neg.search(m.group(1)) and not regex_neg.search(m.group(3)):
                        complications += 1
                        break
                for m in regex_angina.finditer(x):
                    if not regex_neg.search(m.group(1)) and not regex_neg.search(m.group(2)):
                        complications += 1
                        break
                for m in regex_ischemia.finditer(x):
                    if not regex_neg.search(m.group(1)) and not regex_neg.search(m.group(2)):
                        complications += 1
                        break
                if complications >= 2:
                    p = 1
                y.append(p)
        elif tag == 'ALCOHOL-ABUSE':
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/7y3NFBDkFUg
            # not met
            denies = re.compile(
                pattern=r'\b(?:denies|deny)\b[^.,]{,20}?\b(?:alcohol|drink|drinking|etoh)\b',
                flags=re.DOTALL|re.IGNORECASE,
            )
            no_abuse_drink = re.compile(
                pattern=r'\b(?:ago|no|none|past|prev|previous|prior|history|h/o|hx)\b[^.,]{,20}?\b(?:abuse|dependence|heavy|ingestion)\b[^.,]{,20}?\b(?:alcohol|drink|drinking|etoh)\b',
                flags=re.DOTALL|re.IGNORECASE,
            )
            no_drink_abuse = re.compile(
                pattern=r'\b(?:ago|no|none|past|prev|previous|prior|history|h/o|hx)\b[^.,]{,20}?\b(?:alcohol|drink|drinking|etoh)\b[^.,]{,20}?\b(?:abuse|dependence|heavy|ingestion)\b',
                flags=re.DOTALL|re.IGNORECASE,
            )
            drink_no_abuse = re.compile(
                pattern=r'\b(?:alcohol|drink|drinking|etoh)\b[^.,]{,20}?\b(?:ago|no|none|past|prev|previous|prior|history|h/o|hx)\b[^.,]{,20}?\b(?:abuse|dependence|heavy|ingestion)\b',
                flags=re.DOTALL|re.IGNORECASE,
            )
            abuse_drink_no = re.compile(
                pattern=r'\b(?:abuse|binge|dependence|heavy|ingestion)\b[^.,]{,20}?\b(?:alcohol|drink|drinking|etoh)\b[^.,]{,20}?\b(?:ago|no|none|past|prev|previous|prior|history|h/o|hx)\b',
                flags=re.DOTALL|re.IGNORECASE,
            )
            # met
            limit = re.compile(
                pattern=r'\blimit\b[^.,]{,20}?\b(?:alcohol|drink|drinking|etoh)\b',
                flags=re.DOTALL|re.IGNORECASE,
            )
            amount = re.compile(
                pattern=r'\bamount\b[^.,]{,20}?\b(?:alcohol|etoh)\b[^.,]{,20}?\b(?:drink|drinking)\b',
                flags=re.DOTALL|re.IGNORECASE,
            )
            therapy = re.compile(
                pattern=r'\btherapy\b[^.,]{,20}?\b(?:alcohol|drink|drinking|etoh)\b',
                flags=re.DOTALL|re.IGNORECASE,
            )
            drink_abuse = re.compile(
                pattern=r'\b(?:alcohol|drink|drinking|etoh)\b[^.,]{,20}?\b(?:abuse|dependence|heavy|ingestion)\b',
                flags=re.DOTALL|re.IGNORECASE,
            )
            abuse_drink = re.compile(
                pattern=r'\b(?:abuse|binge|dependence|heavy|ingestion)\b[^.,]{,20}?\b(?:alcohol|drink|drinking|etoh)\b',
                flags=re.DOTALL|re.IGNORECASE,
            )
            for x in X:
                if denies.search(x) or no_abuse_drink.search(x) or no_drink_abuse.search(x) or drink_no_abuse.search(x) or abuse_drink_no.search(x):
                    y.append(0)
                    continue
                if limit.search(x) or amount.search(x) or therapy.search(x) or drink_abuse.search(x) or abuse_drink.search(x):
                    y.append(1)
                    continue
                y.append(0)
        elif tag == 'ASP-FOR-MI':
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/q3MEcmuhDVo
            regex = re.compile(r"(.{0,40})\b(aspirin|asa|acetylsalicylic)\b(.{0,40})", re.DOTALL|re.IGNORECASE)
            regex_neg = re.compile(r"(avoid|stop|causes|rash|ulcer|allerg|consider|other\sday|none|should)", re.DOTALL|re.IGNORECASE)
            for x in X:
                x = re.sub('asa physical status', '', x, flags=re.IGNORECASE)
                p = 0
                for m in regex.finditer(x):
                    if not regex_neg.search(m.group(1)) and not regex_neg.search(m.group(3)):
                        p = 1
                y.append(p)
        elif tag == 'CREATININE':
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/buhaysCxZN4
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/VLxv-yTkSnY
            # "Any value over normal at any time."
            text = re.compile(
                pattern=r'\b(?:creatinine was elevated to|creatinine stable|creatinine \(stat lab\)|cr|cr\.|cre|creatinine)[\s:]([^,;]{1,10})',
                flags=re.IGNORECASE,
            )
            num = re.compile(
                pattern=r'\b\d\.\d{1,2}\b',
                flags=re.IGNORECASE,
            )
            elevated_creatinine = re.compile(
                pattern=r'(?:elevated|rising serum)\b[^.,;:]{1,20}\b(?:creatinine)\b',
                flags=re.IGNORECASE,
            )
            for x in X:
                p = 0
                x = remove_html_chars(x)
                x = remove_dates(x)
                x = ' '.join(x.split())
                values = list()
                matches = text.findall(x)
                for m in matches:
                    m = num.findall(m)
                    if m:
                        values.append(float(m[0]))
                for v in values:
                    # see: 144.xml, 148.xml
                    # I considered this value, but the records are very inconsistent (I already saw thresholds of 1.2, 1.5, etc.)
                    if v > 1.4:
                        p = 1
                if elevated_creatinine.search(x):
                    p = 1
                y.append(p)
        elif tag == 'DIETSUPP-2MOS':
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/tcnTV2WIWls
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/gODs5SI8w3g
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/vleeLtXSy6A
            # "intravenous are not excluded"
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/NJjP_GQV2kI
            # "the assumption was that the meds carry over from the previous note"
            regex = re.compile(r"(.{0,40})\b(calcium|copper|cyanocobalamin|epogen|ferrous gluconate|ferrous sulfate|fish oil|folate|k-dur|klor-con|minerals|nephrocaps|niferex|procrit|tocopherol|tums|ascorbic acid|folic acid|calcium|chromium|iron|magnesium|potassium|selenium|zinc|vitamin B[-\s]?1|vitamin B[-\s]?2|vitamin B[-\s]?6|vitamin B[-\s]?12|vitamin B[-\s]?100|vitamin C|vitamin E|vitamin G|vitamin H|vitamin M|vitamin suppl|mineral suppl|Betaxin|niacin|m\.?v\.?i\.?|thiamine)\b(.{0,10})", re.IGNORECASE)
            regex_left_neg = re.compile(r"(elevated|high|low|normal|check|past|previous|was|recommend|counsel)", re.IGNORECASE)
            regex_right_neg = re.compile(r"(\s{3,}|[\s\n]*(is|was|were|of)?[\s\n]*\d+\.\d|[\s\n]*(is|was|were)|[\s\n]*(is|was)?[\s(]*(elevated|high|low|deficien|normal|channel|studies|study|stat|lab))", re.IGNORECASE)
            for i, x in enumerate(X):
                p = 0
                for m in regex.finditer(x):
                    if not regex_left_neg.search(m.group(1)) and not regex_right_neg.search(m.group(3)):
                        p = 1
                        break
                y.append(p)
        elif tag == 'DRUG-ABUSE':
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/7y3NFBDkFUg
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/ocnTOFlbv0c
            # "marijuana use does not constitute drug abuse"
            # not met
            denies_hist_drug_use = re.compile(
                pattern=r'\b(?:denies|deny|no)\b[^.,;:\n]{,25}\b(?:ago|past|prev|previous|previously|prior|history|h/o|hx|h/x)\b[^.,;:\n]{,25}\b(?:crack|cocaine|drug|heroin|illicit|substance)\b[^.,;:\n]{,25}\b(?:abuse|abused|dependence|heavy|smoke|smoked|smoking|use|used)\b',
                flags=re.IGNORECASE,
            )
            denies_hist_use_drug = re.compile(
                pattern=r'\b(?:denies|deny|no)\b[^.,;:\n]{,25}\b(?:ago|past|prev|previous|previously|prior|history|h/o|hx|h/x)\b[^.,;:\n]{,25}\b(?:abuse|abused|dependence|heavy|smoke|smoked|smoking|use|used)\b[^.,;:\n]{,25}\b(?:crack|cocaine|drug|heroin|illicit|substance)\b',
                flags=re.IGNORECASE,
            )
            # met
            hist_drug_use = re.compile(
                pattern=r'\b(?:ago|past|prev|previous|previously|prior|history|h/o|hx|h/x)\b[^.,;:\n]{,25}\b(?:crack|cocaine|drug|heroin|illicit|substance)\b[^.,;:\n]{,25}\b(?:abuse|abused|dependence|heavy|smoke|smoked|smoking|use|used)\b',
                flags=re.IGNORECASE,
            )
            hist_use_drug = re.compile(
                pattern=r'\b(?:ago|past|prev|previous|previously|prior|history|h/o|hx|h/x)\b[^.,;:\n]{,25}\b(?:abuse|abused|dependence|heavy|smoke|smoked|smoking|use|used)\b[^.,;:\n]{,25}\b(?:crack|cocaine|drug|heroin|illicit|substance)\b',
                flags=re.IGNORECASE,
            )
            use_drug_hist = re.compile(
                pattern=r'\b(?:abuse|abused|dependence|heavy|smoke|smoked|smoking|use|used)\b[^.,;:\n]{,25}\b(?:crack|cocaine|drug|heroin|illicit|substance)\b[^.,;:\n]{,25}\b(?:ago|past|prev|previous|previously|prior|history|h/o|hx|h/x)\b',
                flags=re.IGNORECASE,
            )
            for x in X:
                if denies_hist_drug_use.search(x) or denies_hist_use_drug.search(x):
                    y.append(0)
                    continue
                if hist_drug_use.search(x) or hist_use_drug.search(x) or use_drug_hist.search(x):
                    y.append(1)
                    continue
                y.append(0)
        elif tag=='ENGLISH':
            regex1 = re.compile(
                '(?:arabic|aramaic|armenian|bulgarian|burmese|cambodian|cantanese|cantonese|catonese|chinese|creole|croele|ethiopian|farsi|farsti|french|greek|gujarati|haitan|hindi|indonesian|infant|italian|japanese|korean|laotian|latvian|loatian|mandarin|nonenglish|persian|polish|portugese|portuguese|romanian|rusian|russian|somali|spainish|spanish|thai|tiawanese|urdu|vietmanese|vietnamese|yiddish)[\s-]+(?:speaking)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            #regex2 = re.compile(
            #    "(?:male|woman|lady|patient|pt)[\s]+from[\s]+(the[\s]+)?(afghanistan|albania|algeria|andorra|angola|antigua|antigua and barbuda|argentina|armenia|australia|austria|azerbaijan|bahamas|bahrain|bangladesh|barbados|belarus|belgium|belize|benin|bhutan|bolivia|bosnia|bosnia and herzegovina|botswana|brazil|brunei|bulgaria|burkina|burkina faso|burundi|cabo verde|cape verde|cape vert|cambodia|cambodja|cameroon|canada|central african republic|chad|chile|china|colombia|comoros|congo|costa rica|croatia|cuba|cyprus|czechia|côte d'ivoire|ivory coast|korea|democratic republic of congo|republic of congo|denmark|djibouti|dominica|dominican republic|ecuador|egypt|el salvador|equatorial guinea|eritrea|estonia|ethiopia|faroe islands|fiji|finland|france|gabon|gambia|georgia|germany|ghana|greece|grenada|guatemala|guinea|guinea-bissau|guyana|haiti|honduras|hungary|iceland|india|indonesia|iran|iraq|ireland|israel|italy|jamaica|japan|jordan|kazakhstan|kenya|kiribati|kuwait|kyrgyzstan|laos|latvia|lebanon|lesotho|liberia|libya|lithuania|luxembourg|madagascar|malawi|malaysia|maldives|mali|malta|mauritania|mauritius|mexico|monaco|mongolia|montenegro|morocco|mozambique|myanmar|namibia|nauru|nepal|netherlands|new zealand|nicaragua|niger|nigeria|niue|norway|oman|pakistan|palau|panama|papua new guinea|papua|new guinea|paraguay|peru|philippines|poland|portugal|qatar|south korea|north korea|moldova|romania|russia|rwanda|st kitts|saint kitts|saint kitts and nevis|st lucia|saint lucia|st vincent|saint vincent|saint vincent and the grenadines|samoa|san marino|sao tome|saudi arabia|senegal|serbia|seychelles|sierra leone|singapore|slovakia|slovenia|solomon islands|somalia|south africa|south sudan|spain|sri lanka|sudan|suriname|swaziland|sweden|switzerland|syria|tajikistan|thailand|macedonia|timor|timor-leste|togo|tonga|trinidad|trinidad and tobago|tunisia|turkey|turkmenistan|tuvalu|uganda|ukraine|uae|united arab emirates|uk|united kingdom|tanzania|uruguay|uzbekistan|vanuatu|venezuela|vietnam|viet nam|yemen|zambia|zimbabwe)",
            #    flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            #)
            regex2 = re.compile(
                r'\b(?:member|members|family)\b[^.,;]{,20}\b(?:interpret|translate|interpreting|translating)\b',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            regex3 = re.compile(
                r'\b(?:interpreter|translator)\b[^.,;]{,20}\b(?:present|required|necessary)\b',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            for x in X:
                p = 1
                if regex1.search(x) or regex2.search(x) or regex3.search(x):
                    p = 0
                y.append(p)
        elif tag == 'HBA1C':
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/VLxv-yTkSnY
            # we assumed: "at the most recent time"
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/jwogK3i-rDg
            # 6.5 <= HbA1c <= 9.5
            header = re.compile(
                pattern=r'^date.+(?:a1c|hgbaic|hbaic|hgaic)',
                flags=re.IGNORECASE,
            )
            text = re.compile(
                pattern=r'(?:a1c|hgbaic|hbaic|hgaic)(.{,50})',
                flags=re.IGNORECASE,
            )
            num = re.compile(
                pattern=r'(\d{1,2}(?:\.\d{1,2})?)',
                flags=re.IGNORECASE,
            )
            for x in X:
                x = remove_html_chars(x)
                x = remove_dates(x)
                lines = get_lines(x)
                values = list()
                previous_line_is_header = False
                for line in lines:
                    if previous_line_is_header:
                        m = num.findall(line)
                        if m:
                            values.append(float(m[0]))
                        previous_line_is_header = False
                        continue
                    if header.findall(line):
                        previous_line_is_header = True
                        continue
                    matches = text.findall(line)
                    for m in matches:
                        # until finds a comma or a semicolon
                        m = m.split(';')[0]
                        m = m.split(',')[0]
                        m = num.findall(m)
                        if m:
                            values.append(float(m[0]))
                p = 0
                if values:
                    last_value = values[-1]
                    if (last_value >= 6.5) and (last_value <= 9.5):
                        p = 1
                y.append(p)
        elif tag == 'KETO-1YR':
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/VLxv-yTkSnY
            # not met
            no_keto = re.compile(
                pattern=r'no.{,30}?(?:dka|ketones|ketoacidosis)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            # met
            keto = re.compile(
                pattern=r'(?:ketones\s+pos)|(?:ketoacidosis)',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            for x in X:
                if no_keto.search(x):
                    y.append(0)
                    continue
                if keto.search(x):
                    y.append(1)
                    continue
                y.append(0)
        elif tag == 'MAJOR-DIABETES':
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/ekZwPuroBvs
            # "So the annotators operated on the assumption that all patients were diabetic."
            # https://groups.google.com/a/simmons.edu/forum/#!topic/n2c2-2018-challenge-organizers-group/NQUeNxsrKJE
            assert False
        elif tag == 'MAKES-DECISIONS':
            # not met
            regex1 = re.compile(
                pattern=r'\b(?:daughter|wife|husband|family|niece|father|mother|son|brother|sister|sibling)\b[^.,;]{,20}(?:make|makes)\b[^.,;]{,20}\b(?:decision|decisions)\b',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            regex2 = re.compile(
                pattern=r'\b(?:pt|patient)\b[^.,;]{,20}\b(?:not)\b[^.,;]{,20}\b(?:make|makes)\b[^.,;]{,20}\b(?:decision|decisions)\b',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            regex3 = re.compile(
                pattern=r'\b(?:mental)\b[^.,;]{,20}\b(?:retardation)\b',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            regex4 = re.compile(
                pattern=r'\b(?:confusion|confused|depression|depressed|worst|worse|bad)\b[^.,;]{,20}\b(?:mental)[^.,;]{,20}\b(?:status)\b',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            regex5 = re.compile(
                pattern=r'\b(?:consult|appointment)\b[^.,;]{,20}\b(?:neuro|psych|psychiatric)[^.,;]{,20}\b(?:dementia|alzheimer)\b',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            regex6 = re.compile(
                pattern=r'\b(?:pt|patient)\b[^.,;]{,20}\b(?:diagnosed|dx)[^.,;]{,20}\b(?:dementia|alzheimer)\b',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            regex7 = re.compile(
                pattern=r'\b(?:severe)\b[^.,;]{,20}\b(?:dementia|alzheimer)\b',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            regex8 = re.compile(
                pattern=r'\b(?:unable|not able)\b[^.,;]{,20}\b(?:answer)\b[^.,;]{,20}\b(?:question|questions)\b',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            regex9 = re.compile(
                pattern=r'\b(?:pt|patient)\b[^.,;]{,20}\b(?:not)\b[^.,;]{,20}\b[^.,;]{,20}\b(?:acting|speaking|communicating)\b[^.,;]{,20}\b(?:himself|herself|self)\b',
                flags=re.DOTALL|re.IGNORECASE|re.MULTILINE,
            )
            for x in X:
                if regex1.search(x) or regex2.search(x) or regex3.search(x) or regex4.search(x) or regex5.search(x) or regex6.search(x) or regex7.search(x) or regex8.search(x) or regex9.search(x):
                    y.append(0)
                    continue
                y.append(1)
        elif tag == 'MI-6MOS':
            regex_mi = re.compile(r"(.{0,40})\b(myocardial infarction|MI|IMI|AMI|ASMI|HMI|NQWMI|NSTEMI|OASMI|SEMI|STEMI|TIMI)\b(.{0,20})", flags=re.IGNORECASE|re.DOTALL)
            regex_neg = re.compile(r"\b(rule-out|rule out|ruled out|ruling out|r\\?o|r/o|old|past|prior|post|s\\?p|s/p|no|not|negative|free|unlikely|any|absence|absent|had|father|mother|dad|mom|grandfather|grandmother|brother|sister|son|daughter|family|fh|\w{,2}story|\w{,2}hx|flow)\b", flags=re.IGNORECASE|re.DOTALL)
            for i, x in enumerate(X):
                p = 0
                for m in regex_mi.finditer(x):
                    if not regex_neg.search(m.group(1)) and not regex_neg.search(m.group(3)):
                        p = 1
                        break
                y.append(p)
        assert len(y) == len(X)
        return y


class BaselineClassifier:
    #
    def predict(self, tag, X):
        n = len(X)
        if tag in ('ABDOMINAL', 'ALCOHOL-ABUSE', 'CREATININE', 'DIETSUPP-2MOS', 'DRUG-ABUSE', 'HBA1C', 'KETO-1YR', 'MI-6MOS'):
            return [0 for i in range(n)]
        elif tag in ('ADVANCED-CAD', 'ASP-FOR-MI', 'ENGLISH', 'MAJOR-DIABETES', 'MAKES-DECISIONS'):
            return [1 for i in range(n)]
        else:
            assert False
