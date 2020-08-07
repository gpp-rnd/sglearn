import pandas as pd
from Bio.SeqUtils import MeltingTemp


def get_frac_g_or_c(feature_dict, guide_sequence):
    """Get gc content

    Parameters
    ----------
    feature_dict: dict
        Feature dictionary
    guide_sequence: str
        Guide sequence

    Returns
    -------
    dict
        Feature dictionary
    """
    g_count = guide_sequence.count('G')
    c_count = guide_sequence.count('C')
    gc_frac = (g_count + c_count)/len(guide_sequence)
    feature_dict['GC content'] = gc_frac
    return feature_dict


def get_one_nt_frac(feature_dict, guide, nts):
    """Get fraction of single nt

    Parameters
    ----------
    feature_dict: dict
        Feature dictionary
    guide: str
        Guide sequence
    nts: list
        List of nucleotides

    Returns
    -------
    dict
        Feature dictionary
    """
    for nt in nts:
        nt_frac = guide.count(nt)/len(guide)
        feature_dict[nt] = nt_frac
    return feature_dict


def get_two_nt_frac(feature_dict, guide, nts):
    """Get fraction of two nts

    Parameters
    ----------
    feature_dict: dict
        Feature dictionary
    guide: str
        Guide sequence
    nts: list
        List of nucleotides

    Returns
    -------
    dict
        Feature dictionary
    """
    for nt1 in nts:
        for nt2 in nts:
            two_mer = nt1 + nt2
            nts_counts = guide.count(two_mer)
            nts_frac = nts_counts/(len(guide) - 1)
            feature_dict[nt1 + nt2] = nts_frac
    return feature_dict


def get_three_nt_counts(feature_dict, guide, nts):
    """Get fraction of three nts

    Parameters
    ----------
    feature_dict: dict
        Feature dictionary
    guide: str
        Guide sequence
    nts: list
        List of nucleotides

    Returns
    -------
    dict
        Feature dictionary
    """
    for nt1 in nts:
        for nt2 in nts:
            for nt3 in nts:
                k_mer = nt1 + nt2 + nt3
                nts_counts = guide.count(k_mer)
                nts_frac = nts_counts/(len(guide) - 2)
                feature_dict[nt1 + nt2 + nt3] = nts_frac
    return feature_dict


def get_one_nt_pos(feature_dict, context_sequence, nts, context_order):
    """One hot encode single nucleotide

    Parameters
    ----------
    feature_dict: dict
        Feature dictionary
    context_sequence: str
        Context sequence
    nts: list
        List of nucleotides
    context_order: list
        Position of context

    Returns
    -------
    dict
        Feature dictionary
    """
    for i in range(len(context_order)):
        curr_nt = context_sequence[i]
        for nt in nts:
            key = context_order[i] + nt
            if curr_nt == nt:
                feature_dict[key] = 1
            else:
                feature_dict[key] = 0
    return feature_dict


def get_two_nt_pos(feature_dict, context_sequence, nts, context_order):
    """One hot encode two nucleotides

        Parameters
        ----------
        feature_dict: dict
            Feature dictionary
        context_sequence: str
            Context sequence
        nts: list
            List of nucleotides
        context_order: list
            Position of context

        Returns
        -------
        dict
            Feature dictionary
    """
    for i in range(len(context_order) - 1):
        curr_nts = context_sequence[i:i+2]
        for nt1 in nts:
            for nt2 in nts:
                match_nts = nt1+nt2
                key = context_order[i] + match_nts
                if curr_nts == match_nts:
                    feature_dict[key] = 1
                else:
                    feature_dict[key] = 0
    return feature_dict


def get_three_nt_pos(feature_dict, context_sequence, nts, context_order):
    """One hot encode three nucleotides

        Parameters
        ----------
        feature_dict: dict
            Feature dictionary
        context_sequence: str
            Context sequence
        nts: list
            List of nucleotides
        context_order: list
            Position of context

        Returns
        -------
        dict
            Feature dictionary
    """
    for i in range(len(context_order) - 2):
        curr_nts = context_sequence[i:i+3]
        for nt1 in nts:
            for nt2 in nts:
                for nt3 in nts:
                    match_nts = nt1+nt2+nt3
                    key = context_order[i] + match_nts
                    if curr_nts == match_nts:
                        feature_dict[key] = 1
                    else:
                        feature_dict[key] = 0
    return feature_dict


def get_thermo(feature_dict, guide_sequence, context_sequence):
    """Use Biopython to get thermo info. from context and guides

            Parameters
            ----------
            feature_dict: dict
                Feature dictionary
            guide_sequence: str
                Guide sequence
            context_sequence: str
                Context sequence

            Returns
            -------
            dict
                Feature dictionary
        """
    #
    feature_dict['Tm, context'] = MeltingTemp.Tm_NN(context_sequence)
    third = len(guide_sequence)//3
    feature_dict['Tm, start'] = MeltingTemp.Tm_NN(guide_sequence[0:third])
    feature_dict['Tm, mid'] = MeltingTemp.Tm_NN(guide_sequence[third:2 * third])
    feature_dict['Tm, end'] = MeltingTemp.Tm_NN(guide_sequence[2 * third:])
    return feature_dict


def get_context_order(k, pam_start, pam_length, guide_start, guide_length):
    """Get named order of context sequence

    Parameters
    ----------
    k: int
       length of kmer
    pam_start: int
        Start of PAM, one-indexed
    pam_length: int
        PAM length
    guide_start: int
        Start of guide, one-indexed
    guide_length: int
        Length of guide

    Returns
    -------
    list
        Ordering for context sequence
    """
    pam_order = ['P' + str(x) for x in range(1, pam_length + 1)]
    guide_order = [str(x) for x in range(1, guide_length + 1)]
    if pam_start == min(pam_start, guide_start):
        second_ord = pam_order
        third_ord = guide_order
    else:
        second_ord = guide_order
        third_ord = pam_order
    first_ord = ['-' + str(x) for x in reversed(range(1, min(pam_start, guide_start)))]
    fourth_ord = ['+' + str(x) for x in range(1, k - (len(first_ord) + len(second_ord) + len(third_ord)) + 1)]
    context_order = first_ord + second_ord + third_ord + fourth_ord
    return context_order


def get_guide_sequence(context, guide_start, guide_length):
    return context[(guide_start - 1):(guide_start + guide_length - 1)]


def featurize_guides(kmers, features=None,
                     pam_start=25, pam_length=3,
                     guide_start=5, guide_length=20):
    """Featurize a list of guide sequences

    Parameters
    ----------
    kmers: list
        Context sequences
    features: list, optional
        List of features. Will default to rule set 2 features
    guide_start: int
        Position of guide start, one-indexed
    guide_length: int
        Length of guide
    pam_start: int
        Position of pam start, one-indexed
    pam_len: int
        Length of PAM

    Returns
    -------
    DataFrame
        Nucleotide features
    """
    if features is None:
        # RS2
        features = ['Pos. Ind. 1mer', 'Pos. Ind. 2mer',
                    'Pos. Dep. 1mer', 'Pos. Dep. 2mer',
                    'GC content', 'Tm']
    possible_feats = {'Pos. Ind. 1mer', 'Pos. Ind. 2mer', 'Pos. Ind. 3mer',
                      'Pos. Dep. 1mer', 'Pos. Dep. 2mer',
                      'Pos. Dep. 3mer', 'GC content', 'Tm'}
    if not set(features).issubset(possible_feats):
        diff = features - possible_feats
        assert ValueError(str(diff) + 'Are not currently supported as features')
    k = len(kmers[0])
    context_order = get_context_order(k, pam_start, pam_length, guide_start, guide_length)
    nts = ['A', 'C', 'T', 'G']
    feature_dict_list = []
    for i in range(len(kmers)):
        curr_dict = {}
        context = kmers[i]
        guide_sequence = get_guide_sequence(context, guide_start, guide_length)
        if 'GC content' in features:
            curr_dict = get_frac_g_or_c(curr_dict, guide_sequence)
        if 'Pos. Ind. 1mer' in features:
            curr_dict = get_one_nt_frac(curr_dict, guide_sequence, nts)
        if 'Pos. Ind. 2mer' in features:
            curr_dict = get_two_nt_frac(curr_dict, guide_sequence, nts)
        if 'Pos. Ind. 3mer' in features:
            curr_dict = get_three_nt_counts(curr_dict, guide_sequence, nts)
        if 'Pos. Dep. 1mer' in features:
            curr_dict = get_one_nt_pos(curr_dict, context, nts, context_order)
        if 'Pos. Dep. 2mer' in features:
            curr_dict = get_two_nt_pos(curr_dict, context, nts, context_order)
        if 'Pos. Dep. 3mer' in features:
            curr_dict = get_three_nt_pos(curr_dict, context, nts, context_order)
        if 'Tm' in features:
            curr_dict = get_thermo(curr_dict, guide_sequence, context)
        feature_dict_list.append(curr_dict)
    feature_matrix = pd.DataFrame(feature_dict_list)
    feature_matrix.index = kmers
    return feature_matrix
