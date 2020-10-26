from multiprocessing import Pool, cpu_count
from typing import List

import numpy as np
import pandas as pd
import spacy
from fuzzywuzzy import fuzz, process


def clean_data(input_file):
    wiki_place = pd.read_csv(
        input_file,
        header=None,
        names=["place", "type", "sentence", "wiki_title", "wiki_coords"],
    )

    wiki_place[['wiki_latitude', 'wiki_longitude']] = \
        wiki_place["wiki_coords"].str.split(" ", 1, expand=True)
    wiki_place.drop('wiki_coords', axis=1)
    return wiki_place


wiki_place = clean_data("./data_processing/data/results/predictions.csv")


nom_sum = wiki_place[wiki_place['type'] == 'PLACE_NOM']['place']\
    .value_counts()\
    .to_frame(name='count')\
    .reset_index()\
    .head(1000)
nom_places = wiki_place[(wiki_place['type'] == 'PLACE_NOM') &
                        (wiki_place['place'].isin(nom_sum['index']))]
nom = nom_places['place'].unique()


def get_lemmas(x):
    nlp = spacy.load("en_core_web_sm")
    return ' '.join([word.lemma_ for word in nlp(x)])


with Pool(cpu_count()) as p:
    nom_lemma = p.map(get_lemmas, nom)

nom = dict(zip(nom, nom_lemma))
nom_places['place'].apply(lambda x: nom[x])
nom_places['place'] = nom_places['place'].apply(lambda x: nom[x])

nom = nom_places['place'].unique()


def get_leven(nom: List[str], cutoff: int):
    # compare each string with all others
    score_sort = [
        (x,) + i
        for x in nom
        for i in process.extract(
            x,
            nom,
            scorer=fuzz.QRatio
        )
    ]
    # create a dataframe from the tuples
    similarity_sort = pd.DataFrame(
        score_sort,
        columns=['brand_sort', 'match_sort', 'score_sort']
    )
    # keep only matches above X similarity value
    # excluding strings that are identical
    similarity_sort['sorted_brand_sort'] = np.minimum(
        similarity_sort['brand_sort'], similarity_sort['match_sort']
    )
    high_score_sort = similarity_sort[
        (similarity_sort['score_sort'] >= cutoff) &
        (similarity_sort['brand_sort'] !=
         similarity_sort['match_sort']) &
        (similarity_sort['sorted_brand_sort'] !=
         similarity_sort['match_sort'])
    ]\
        .drop('sorted_brand_sort', axis=1)\
        .copy()\
        .groupby(['brand_sort', 'score_sort'])\
        .agg({'match_sort': ', '.join})\
        .sort_values(['score_sort'], ascending=False)\
        .reset_index()
    high_score_sort['chosen'] = high_score_sort\
        .apply(
        lambda x: x['brand_sort']
        if len(x['brand_sort']) <= len(x['match_sort'])
        else x['match_sort'], axis=1
    )

    nom_series = pd.Series(nom, name='nom')
    nom = pd.merge(nom_series,
                   high_score_sort[['match_sort', 'chosen']],
                   left_on='nom',
                   right_on='match_sort',
                   how='outer')

    nom['chosen'] = nom.apply(lambda x: x['chosen']
                              if x['chosen'] == x['chosen']
                              else x['nom'], axis=1)
    return {row['nom']: row['chosen'] for _, row in nom.iterrows()}


nom = get_leven(nom, 90)
nom_places['place'] = nom_places['place'].apply(lambda x: nom[x])

nom_places.to_csv("./data_processing/data/results/nom_places.csv", index=False)
nam_places = wiki_place[(wiki_place['type'] == 'PLACE_NAM')]
nam_places.to_csv("./data_processing/data/results/nam_places.csv", index=False)
