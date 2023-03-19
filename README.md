# Biomedical Literature Analysis Project

This project focuses on analyzing a dataset of biomedical literature to answer a variety of research questions and extract valuable insights. Utilizing natural language processing (NLP) techniques and data analysis methods, the goal is to help researchers navigate the vast and rapidly-growing field of biomedical research more efficiently.


## Project Structure

**covid_analysis.py:** Script that runs the analysis on the progression of the COVID-19 pandemic on biomedical research\
**infectious_classifier.py:** Script that trains and evaluates the classifier for infectious and non-infectious disease.\
**relevance_abstract.py:** Script that runs the similarity search on relevant papers given a list of subjects.\
**extract_quantitative_information.py:** Script that runs the extraction of quantitative information in the literature.\
**utilities.py:** This script contains various utility functions and helper classes used throughout the project.\
**text_processing.py:** This script contains various functions for text processing.\
**data_visualisation.py:** This script contains various functions for vsualising the analysis results.\
**ml_functions.py:** This script contains various functions that related to NLP and machine learning.\
**metadata.csv:** metadata for the papers. (Excluded from this repository due to concerns regarding privacy.)\
**file-list.csv:** doi/file name reference. (Excluded from this repository due to concerns regarding privacy.)\
**abstracts:** Directory containing all the abstracts. (Excluded from this repository due to concerns regarding privacy.)\
**infectious_classifier:** Directory for the trained classifier for infectious diseases. (Excluded from this repository. Please download from the shared link.)\
**relevance:** Directory for storing the generated relevant papers for given subjects.\
**output:** Directory for storing the generated visualizations, data, and other output files.


## Dependencies

Python 3.7+ \
pandas\
numpy\
scikit-learn\
nltk\
gensim 3.8.0\
transformers\
matplotlib\
wordcloud\
regex\
string\
scipy\
sentence_transformers\
torch

## Hardware
GPUs were used while training the classifier and embedding the abstracts.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
