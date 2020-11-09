import pandas as pd


def read_wiki(input_file):
    wiki_place = pd.read_csv(
        input_file,
        header=None,
        names=["place", "type", "sentence", "wiki_title", "wiki_coords"],
    )

    wiki_place[['wiki_latitude', 'wiki_longitude']] = \
        wiki_place["wiki_coords"].str.split(" ", 1, expand=True)
    wiki_place.drop('wiki_coords', axis=1)
    wiki_place = wiki_place[wiki_place['type'] == 'PLACE_NAM']
    return wiki_place


wiki_place = read_wiki("./data_processing/data/results/predictions.csv")

wiki_place[wiki_place['place'] == 'Northern']['wiki_title']

geonames = pd.read_csv("./data_processing/data/raw/geonames/GB.txt", sep="\t",
                       header=None)\
    .iloc[:, [0, 1, 4, 5]]
geonames.columns = ['id', 'name', 'lat', 'lon']

geonames_unq = geonames['name'].unique()
wiki_place_unq = wiki_place['place'].unique()

new_places = set(wiki_place_unq) - set(geonames_unq)

new_places = wiki_place[wiki_place['place'].isin(new_places)]

new_counts = new_places['place'].value_counts().to_frame()
new_counts.to_csv("./data_processing/data/results/out_of_gazetteer.csv")
