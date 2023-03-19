from utilities import *
from text_processing import lowercase, split_sentence
from ml_functions import *
import pandas as pd
import os
import numpy as np


pAbstract = './abstracts/'
subjects = ['breast cancer']
fMetadata = 'metadata.csv'
fFile_list = 'file-list.csv'
# model_path = 'sentence-transformers/msmarco-distilbert-dot-v5'
model_path = 'stsb-distilbert-base'
max_sequence_length = 256

out_path = './relevance/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

logging.basicConfig(format='%(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                level=logging.INFO,
                handlers=[LoggingHandler()])

filename, abstract = read_from_folder(pAbstract)
abstract = lowercase(abstract)

# read metadata and file list
metadata = read_csv_to_df(fMetadata)
metadata = metadata.drop_duplicates(subset=['doi'], keep='last')
file_list = read_csv_to_df(fFile_list)

if __name__ == '__main__':
    abstract_embeddings = []
#     model = sentence_transformer_model(model_path)
    model = SentenceTransformer(model_path)
    print("Max Sequence Length:", model.max_seq_length)
    model.max_seq_length = max_sequence_length
    print("Max Sequence Length:", model.max_seq_length)
    pool = model.start_multi_process_pool()
    abstract_embeddings = model.encode_multi_process(abstract, pool)
#     for d in abstract[:10]:
#         sentences = split_sentence(d)
#         embeddings = model.encode_multi_process(sentences, pool)
# #         abstract_embeddings.append(embeddings)
#         abstract_embeddings.append(average_embedding(embeddings))
    model.stop_multi_process_pool(pool)
    print("Embeddings computed. Shape:", np.array(abstract_embeddings).shape)

    for subject in subjects:
        subject_embedding = sentence_embedding(subject, model_path)
        idx, cosine = relevance_calculation(subject_embedding, abstract_embeddings, mode='cosine')
        f = [idx_to_filename(i,filename) for i in idx]
        doi = keys_to_values(file_list, f, 'doc_name', 'doi')
        title = keys_to_values(metadata, doi, 'doi','title')
        cosine = map(distance_to_similarity, cosine)
        df = pd.DataFrame(zip(doi, f, title, cosine), columns=['doi', 'doc_name', 'title', 'cosine_similarity'])

        out_file = out_path+subject+'_relevance.csv'
        save_df_to_csv(df, out_file)

    

# data_tuples = list(zip(sentences,corpus_embeddings))
# d = {k:v for k,v in data_tuples}
# del data_tuples
# #     del question
# del corpus_embeddings

# print('Saving..')
# save_path = out_path + 'sentence_embedding/'
# os.makedirs(save_path, exist_ok=True)
# out_file = save_path + "s_e_nli_cls_rev.pickle"
# with open(out_file, 'wb') as f:
#     pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
# print('{} saved.'.format(out_file))
# idx_file = save_path + "i_s_nli_cls_rev.pickle"
# idx = [*range(len(d))]
# i_d = {k:v for k,v in zip(idx, sentences)}
# with open(idx_file, 'wb') as f:
#     pickle.dump(i_d, f, protocol=pickle.HIGHEST_PROTOCOL)
# print('{} saved.'.format(idx_file))
