from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
import string
import regex


def split_sentence(text):
    # print(text)
    remove_newline = text.replace('\n', ' ')
    sentences = sent_tokenize(remove_newline)
    return sentences


def sentence_to_word(text):
    return word_tokenize(text)


def lowercase(text):
    lowercase_list = [t.lower() for t in text]
    return lowercase_list


def string_with_keywords(text, keywords):
    if any(w in text for w in keywords):
        return True
    else:
        return False


# def extract_content_by_keywords(text, keywords):
#     # sentences = split_sentence(text)
#     # # print(sentences)
#     content_with_key = []
#     content_without_key = []
#     for t in text:
#         if string_with_keywords(t, keywords):
#             content_with_key.append(t)
#         else:
#             content_without_key.append(t)
#
#     return content_with_key, content_without_key


def date_to_year(date):
    return date[:4]


def remove_punctuation(text):
    return ''.join(char for char in text if char not in string.punctuation)


def preprocess(text, custom_stopwords=None):
    stop_words = set(stopwords.words('english'))
    if custom_stopwords:
        custom_stopwords = set(custom_stopwords)
        stop_words = stop_words.union(custom_stopwords)
    lemmatizer = WordNetLemmatizer()
    # text = remove_punctuation(text)
    tokens = simple_preprocess(text, deacc=True)
    clean_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return clean_tokens


def find_number(text):
    pattern = r'(?<![cC][oO][vV][iI][dD]-|3[dD]|sars-cov-)\b\d+\b(?!\s*[cC][oO][vV][iI][dD]-)(?!\d{2}-\d{2}-\d{4})(?!\d{2}/\d{2}/\d{4})(?!\d{4}-\d{2}-\d{2})|(?<=\b\d+\s)[cC][oO][vV][iI][dD]-'
    numbers = regex.findall(pattern, text)
    numbers = [int(num) for num in numbers if num.isdigit()]
    return numbers


def sentence_with_number(document):
    quantitative = []
    sentences = split_sentence(document)
    for s in sentences:
        numbers = find_number(s)
        if numbers:
            quantitative.append(s)
    return quantitative



            


