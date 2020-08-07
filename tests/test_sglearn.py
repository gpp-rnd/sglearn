#!/usr/bin/env python

"""Tests for `sglearn` package."""

from sglearn import featurization as ft
import pandas as pd


def test_guide_seq():
    # SpCas9
    assert ft.get_guide_sequence('CGTCCCCATCCACGGCCTTCACCCGGGCAG', 5, 20) == 'CCCATCCACGGCCTTCACCC'
    # AsCas12a
    assert ft.get_guide_sequence('CCAGTTTGAACTCTCGCCCATCACCTATCAGTGC', 9, 20) == 'AACTCTCGCCCATCACCTAT'


def test_context_order():
    cas12a = ft.get_context_order(34, 5, 4, 9, 23)
    assert len(cas12a) == 34
    assert cas12a == ['-4', '-3', '-2', '-1', 'P1', 'P2', 'P3', 'P4', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
                      '+1', '+2', '+3']
    cas9 = ft.get_context_order(30, 25, 3, 5, 20)
    assert len(cas9) == 30
    assert cas9 == ['-4', '-3', '-2', '-1', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                    'P1', 'P2', 'P3', '+1', '+2', '+3']


def test_featurization():
    features = ['Pos. Ind. 1mer',
                'Pos. Ind. 2mer',
                'Pos. Ind. 3mer',
                'Pos. Dep. 1mer',
                'Pos. Dep. 2mer',
                'Pos. Dep. 3mer',
                'GC content',
                'Tm']
    kmers = pd.Series(['ACTGGTGGG'])
    features = ft.featurize_guides(kmers, features, guide_start=3, guide_length=6, pam_start=2, pam_length=1).iloc[0]
    assert (features['1T'] == 1)
    assert (features['TGG'] == 0.5)
    assert (features['GC content'] == 2/3)
    assert (features['Tm, context'] != 0)
    assert (features['-1AC'] == 1)
    assert (features['5GGG'] == 1)
    assert (features['G'] == 2/3)
    assert (features['TG'] == 0.4)
    assert (features['-1T'] == 0)
