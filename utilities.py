import os
import pandas as pd
import scipy
from sentence_transformers import util

def read_text(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()


def read_from_folder(path):
    text = []
    filename = []
    count = 0
    for f in os.listdir(path):
        if f.startswith('.'):
            continue
        count += 1
        new_path = os.path.join(path, f)
        filename.append(f)
        text.append(read_text(new_path))
    return filename, text


def read_csv_to_df(f):
    return pd.read_csv(f)


def save_df_to_csv(df, f, index=False):
    df.to_csv(f, index=index)


def extract_df_by_keys(df, column_name, keys):
    return df[df[column_name].isin(keys)]


def df_column_to_list(df, column_name):
    return df[column_name].tolist()


def print_keywords(keywords_list):
    for i, keywords in enumerate(keywords_list):
        print(f"Keywords for document {i + 1}:")
        print(", ".join(keywords))
        print("\n")


def pivot_table(df, category1, category2):
    return pd.crosstab(df[category1], df[category2])


def doi_to_file(file_list, doi):
    df = extract_df_by_keys(file_list, 'doi', doi)
    return df['doc_name'].tolist()


def column_to_doi(metadata, column_name, keys):
    df = extract_df_by_keys(metadata, column_name, keys)
    return df_column_to_list(df, 'doi')


def keys_to_file(metadata, file_list, column_name, keys):
    doi = column_to_doi(metadata, column_name, keys)
    return doi_to_file(file_list, doi)


def keys_to_values(df, keys, key_column, value_column):
    values = []
    for k in keys:
        values.append(df.loc[df[key_column] == k, value_column].values[0])
    return values


def cosine_distance(query, corpus):
    print(query.shape)
    return scipy.spatial.distance.cdist([query], corpus, "cosine")[0]


def dot_score(query, corpus):
    scores = util.dot_score(query, corpus)[0].tolist()
    return scores


def relevance_calculation(query, corpus, mode='cosine'):
    if mode == 'cosine':
        distances = cosine_distance(query, corpus)
    elif mode == 'dot':
        distances = dot_score(query, corpus)
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])
    return zip(*results)


def idx_to_filename(idx, filename):
    return filename[idx]

def distance_to_similarity(distance):
    return 1-distance
