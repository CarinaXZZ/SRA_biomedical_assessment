from gensim.summarization import keywords as text_rank_keywords
from text_processing import preprocess
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer, LoggingHandler, models
import logging
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
class AbstractDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])
#         return len(self.labels)


def extract_keywords(text_tokens, num_keywords=10):
    preprocessed_text = ' '.join(text_tokens)
    key_words = text_rank_keywords(preprocessed_text, words=num_keywords, split=True, scores=False, pos_filter=None, lemmatize=True)
    return key_words


def covid_keywords_extraction(documents, num_keywords):
    texts = [preprocess(document, custom_stopwords=['covid', 'cov', 'sars', 'study', 'pandemic']) for document in
             documents]
    # print(len(texts))

    keywords_list = [extract_keywords(text, num_keywords) for text in texts]
    # print_keywords(keywords_list)
    keywords_list = [item for sublist in keywords_list for item in sublist]
    return keywords_list


def train_val_test_split(dataX, dataY, train_ratio, test_ratio, val_ratio=None):
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio, random_state=55, shuffle=True)
    if val_ratio:
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + val_ratio), random_state=55, shuffle=True)
        return x_train,x_val, x_test, y_train, y_val, y_test
    else:
        return x_train, x_test, y_train, y_test


def tokenize_data(text, model="dmis-lab/biobert-base-cased-v1.1"):
    tokenizer = AutoTokenizer.from_pretrained(model)
#     return tokenizer(text, padding='max_length', truncation=True)
    return tokenizer(text,padding='max_length', max_length=512, truncation=True)


def compute_metrics(eval_pred):
    metric = evaluate.combine(["accuracy", "recall", "precision", "f1"])
    logits, labels = eval_pred
    # probabilities = tf.nn.softmax(logits)
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def training_model(num_label, model="dmis-lab/biobert-base-cased-v1.1"):
    return AutoModelForSequenceClassification.from_pretrained(model, num_labels=2)
#     return AutoModelForMaskedLM.from_pretrained(model, num_label=num_label)


def get_labels(prediction):
    pred_logits = prediction.predictions
    pred_logits = torch.from_numpy(pred_logits)
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(pred_logits)
    pred_labels = np.argmax(probabilities, axis=1)
    pred_labels = pred_labels.numpy()
    return pred_labels


def get_doi_from_predict(doi, labels):
    pos, neg = [], []
    for i in range(len(labels)):
        if labels[i] == 1:
            pos.append(doi[i])
    #         print(unrelated[i])
        else:
            neg.append(doi[i])
    return pos, neg

def sentence_transformer_model(model_path):
    word_embedding_model = models.Transformer(model_path, max_seq_length=65)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def average_embedding(embeddings):
    return np.mean(embeddings, axis=0)


# def multi_gpu_document_embedding(documents, model_path):
#     logging.basicConfig(format='%(asctime)s - %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     level=logging.INFO,
#                     handlers=[LoggingHandler()])
    
#     document_embeddings = []
#     if __name__ == '__main__':
#         model = sentence_transformer_model(model_path)
#     #     model = SentenceTransformer(model_path)
#         pool = model.start_multi_process_pool()
#         for d in documents:
#             sentences = split_sentence(d)
#             embeddings = model.encode_multi_process(sentences, pool)
#             document_embeddings.append(average_embedding(embeddings))
#         model.stop_multi_process_pool(pool)
#         return document_embeddings


def sentence_embedding(sentence, model_path):
#     model = sentence_transformer_model(model_path)
    model = SentenceTransformer(model_path)
    sentence_embedding = model.encode(sentence)
    return sentence_embedding
    