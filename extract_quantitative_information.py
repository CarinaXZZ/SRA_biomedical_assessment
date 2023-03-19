from text_processing import *
import pandas as pd
from utilities import read_from_folder, save_df_to_csv

pAbstract = './abstracts'
out_file = 'quantitative_information.csv'

filename, abstract = read_from_folder(pAbstract)
abstract = lowercase(abstract)
print(len(abstract))

has_quantitative = []
quantitative_information = []
for a in abstract:
    quantitative = sentence_with_number(a)
    if quantitative:
        has_quantitative.append(1)
        quantitative_information.append(' '.join(quantitative))
    else:
        has_quantitative.append(0)
        quantitative_information.append(None)


df = pd.DataFrame(zip(filename, has_quantitative, quantitative_information), columns=['doc_name','has_quantitative_information', 'quantitative sentences'])

save_df_to_csv(df, out_file)