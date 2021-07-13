"""Tests for `sglearn` package."""

from sglearn import featurization as ft
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats

@pytest.fixture
def azimuth_training():
    df = pd.read_csv('https://raw.githubusercontent.com/MicrosoftResearch/Azimuth/master/azimuth/data/FC_plus_RES_withPredictions.csv',
                     index_col=0).reset_index(drop=True)
    return df


def test_guide_seq():
    # SpCas9
    assert ft.get_guide_sequence('CGTCCCCATCCACGGCCTTCACCCGGGCAG', 4, 20) == 'CCCATCCACGGCCTTCACCC'
    # AsCas12a
    assert ft.get_guide_sequence('CCAGTTTGAACTCTCGCCCATCACCTATCAGTGC', 8, 20) == 'AACTCTCGCCCATCACCTAT'


def test_context_order():
    cas12a = ft.get_context_order(34, 4, 4, 8, 23)
    assert len(cas12a) == 34
    assert cas12a == ['-4', '-3', '-2', '-1', 'P1', 'P2', 'P3', 'P4', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
                      '+1', '+2', '+3']
    cas9 = ft.get_context_order(30, 24, 3, 4, 20)
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
    kmers = pd.Series(['ACTGGTGGG', 'ACTGATGGG'])
    features = ft.featurize_guides(kmers, features, guide_start=2, guide_length=6, pam_start=1, pam_length=1,
                                   guide_sections=(3, 6)).iloc[0]
    assert (features['1T'] == 1)
    assert (features['TGG'] == 0.5)
    assert (features['GC content'] == 2/3)
    assert (features['Tm DD guide'] != 0)
    assert (features['DG RR guide'] == -20)
    assert (features['Tm RD 1 to 3'] != 0)
    assert (features['-1AC'] == 1)
    assert (features['5GGG'] == 1)
    assert (features['G'] == 2/3)
    assert (features['TG'] == 0.4)
    assert (features['-1T'] == 0)


def test_get_pam_interaction():
    feature_dict = {}
    ft.get_pam_interaction(feature_dict, 'AGGCT', ['A', 'C', 'T', 'G'], ['20', 'P1', 'P2', 'P3', '+1'], (0, 3))
    assert feature_dict['20A_P3C'] == 1
    assert feature_dict['20A_P3T'] == 0


def test_training(azimuth_training):
    featurized_guides = ft.featurize_guides(azimuth_training['30mer'], n_jobs=2)
    y = azimuth_training['score_drug_gene_rank']
    x = pd.concat([featurized_guides.reset_index(drop=True),
                   azimuth_training['Percent Peptide']], axis=1)
    model = GradientBoostingRegressor()
    model.fit(x, y)
    predictions = model.predict(x)
    assert stats.pearsonr(azimuth_training['predictions'], predictions)[0] > 0.93
    feature_importance = (pd.DataFrame({'feature': x.columns, 'importance': model.feature_importances_})
                          .sort_values('importance', ascending=False))
    assert '20G' in feature_importance['feature'].values[:10]



def test_polyn():
    curr_dict = {}
    ft.get_polyn(curr_dict, 'TAACCAAACCA', nts=['A', 'C', 'T', 'G'])
    assert curr_dict['PolyA'] == 3
    assert curr_dict['PolyT'] == 1
    assert curr_dict['PolyC'] == 2
    assert curr_dict['PolyG'] == 0
