from utilities import *
from text_processing import *
from data_visualisation import plot_cross_table, plot_word_cloud
from ml_functions import covid_keywords_extraction


pAbstract = './abstracts'
fMetadata = 'metadata.csv'
fFile_list = 'file-list.csv'
keywords = ['covid', 'covid-19', 'sars-cov-2']
num_keywords = 10 # Set the number of keywords to extract for each document

filename, abstract = read_from_folder(pAbstract)
abstract = lowercase(abstract)
print(len(abstract))

# read metadata and file list
metadata = read_csv_to_df(fMetadata)
metadata = metadata.drop_duplicates(subset=['doi'], keep='last')
metadata['year'] = metadata['date'].apply(date_to_year)
# print(metadata.head())
file_list = read_csv_to_df(fFile_list)

# extract covid related paper
covid_paper = {}
for f, article in zip(filename, abstract):
    if string_with_keywords(article, keywords):
        covid_paper[f] = article

# plot cross table
covid_file = list(covid_paper.keys())
covid = extract_df_by_keys(file_list, 'doc_name', covid_file)
covid_doi = df_column_to_list(covid, 'doi')
covid = extract_df_by_keys(metadata, 'doi', covid_doi)
plot_cross_table(covid, 'category', 'year', ylabel='Number of Paper (log scale)')
# print(tabulate(year_category, headers='keys', tablefmt='pretty'))

# keyword extraction
documents = list(covid_paper.values())
print(len(documents))
keywords_list = covid_keywords_extraction(documents, num_keywords)
plot_word_cloud(keywords_list)

covid_20 = covid[covid['year'] =='2020']
print(len(covid_20['doi'].tolist()))
documents_20 = []
for doi in covid_20['doi'].tolist():
    f = file_list.loc[file_list['doi'] == doi, 'doc_name'].values[0]
    documents_20.append(covid_paper[f])
keywords_list_20 = covid_keywords_extraction(documents_20, num_keywords)
plot_word_cloud(keywords_list_20)

covid_21 = covid[covid['year'] =='2021']
print(len(covid_21['doi'].tolist()))
documents_21 = []
for doi in covid_21['doi'].tolist():
    f = file_list.loc[file_list['doi'] == doi, 'doc_name'].values[0]
    documents_21.append(covid_paper[f])
keywords_list_21 = covid_keywords_extraction(documents_21, num_keywords)
plot_word_cloud(keywords_list_21)