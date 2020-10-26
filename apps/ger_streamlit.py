import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from allennlp.data.dataset_readers.dataset_utils import bioul_tags_to_spans
from allennlp.models.archival import load_archive
from ger_wiki.predictor import TextPredictor
from sklearn.cluster import dbscan


@st.cache(allow_output_mutation=True)
def load_predictor(archive_path):
    archive = load_archive(archive_path, cuda_device=0)
    return TextPredictor.from_archive(archive)


def run_model(predictor, passage):
    predictor.predict(passage)
    result = predictor.predict(passage)
    spans = bioul_tags_to_spans(result['tags'])

    passage_tokens = result["words"]

    tags = []
    start = []
    end = []
    for r in spans:
        tags.append(r[0])
        start.append(r[1][0])
        end.append(r[1][1])

    places = []
    for span in spans:
        start, end = span[1]

        if span[0] == 'PLACE_NAM':
            place = ' '.join(passage_tokens[start:end+1])
            places.append(place)

        passage_tokens[start:end+1] = [
            "**" + token + "**"
            for token in passage_tokens[start:end+1]
        ]
        passage_tokens[end] += " _<sub>" +\
            span[0].split("_")[1] +\
            "</sub>_"

    return passage_tokens, places


def load_coord_data(places):
    geonames_file = "./data_processing/data/raw/geonames/GB.txt"
    geoname_columns = ['geonameid', 'name', 'asciiname',
                       'alternatenames', 'latitude', 'longitude',
                       'feature class', 'feature code', 'country code', 'cc2',
                       'admin1 code', 'admin2 code', 'admin3 code',
                       'admin4 code', 'population', 'elevation', 'dem',
                       'timezone', 'modification date']
    geonames = pd.read_csv(geonames_file, sep='\t',
                           header=None, names=geoname_columns)
    geonames = geonames[['name', 'latitude', 'longitude']].dropna()

    wiki_file = "./data_processing/data/results/predictions.csv"
    wiki = pd.read_csv(
        wiki_file,
        names=["place", "type", "context", "wiki_page", "wiki_coords"],
        index_col=False
    )\
        .dropna()

    wiki[['latitude', 'longitude']] = wiki['wiki_coords'].str\
        .split(' ', expand=True).astype(float)
    wiki = wiki[['place', 'latitude', 'longitude']]
    wiki.columns = ['name', 'latitude', 'longitude']

    place_coords = pd.DataFrame()
    for place in places:
        geonames_coords = geonames[geonames['name'].str.contains(place)]
        geonames_coords['place'] = place
        wiki_coords = wiki[wiki['name'].str.contains(place)]
        wiki_coords['place'] = place

        place_coords = place_coords.append(geonames_coords)
        #place_coords = place_coords.append(wiki_coords)
    return place_coords


st.header("Geographic Entity Recognition.")
archive_path = "./models/archive_best_model/model.tar.gz"
passage = st.text_area(
    "sentence", "Headingley is a suburb of Leeds, West Yorkshire, England, approximately two miles out of the city centre, to the north west along the A660 road. Headingley is the location of the Beckett Park campus of Leeds Beckett University and Headingley Stadium."
)

click_model = st.button("Run Model")
if click_model:
    predictor = load_predictor(archive_path)
    passage_tokens, places = run_model(predictor, passage)
    st.markdown(" ".join(passage_tokens), unsafe_allow_html=True)

click_maps = st.button("Plot Coordinates")
use_dbscan = st.checkbox("Use DBSCAN")
if use_dbscan:
    eps = st.slider('eps', 100, 500, 100, 100)


def dbscan_places(place: str, eps_ratio=0.0001):
    eps = len(place)*eps_ratio
    _, lbls = dbscan(place[['longitude', 'latitude']],
                     eps=eps)
    lbls = pd.Series(lbls, index=place.index)
    place['db_labels'] = lbls
    return place


def plot_map():
    uk_base = gpd.read_file("./data_processing/data/raw/poly/GBR_adm0.shp")

    predictor = load_predictor(archive_path)
    _, places = run_model(predictor, passage)
    coords = load_coord_data(places)

    places_gdf = gpd.GeoDataFrame(
        coords,
        geometry=gpd.points_from_xy(
            coords['longitude'],
            coords['latitude']
        )
    )

    if use_dbscan:
        places_gdf = dbscan_places(places_gdf, eps_ratio=eps/500000)
        places_gdf = places_gdf[places_gdf['db_labels'] == 0]

    fig, ax = plt.subplots()
    zoom = 1
    ax.set_xlim(
        [min(places_gdf['longitude'])-zoom,
         max(places_gdf['longitude'])+zoom]
    )
    ax.set_ylim(
        [min(places_gdf['latitude'])-zoom,
         max(places_gdf['latitude'])+zoom]
    )

    # bounds = uk_base.geometry.total_bounds
    # ax.set_xlim([bounds[0], bounds[2]])
    # ax.set_ylim([bounds[1], bounds[3]])

    uk_base.plot(ax=ax, color='lightgrey')
    groups = places_gdf.groupby("place")
    for name, group in groups:
        plt.plot(group["longitude"], group["latitude"],
                 marker="o", linestyle="", label=name, markersize=1)

    st.map(places_gdf)
    st.dataframe(places_gdf.drop('geometry', axis=1))
    plt.legend()
    st.pyplot(fig)


if click_maps:
    plot_map()
