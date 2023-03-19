from utilities import *
from text_processing import lowercase
from ml_functions import *
from transformers import TrainingArguments, Trainer
import random
random.seed(10)

pAbstract = './abstracts/'
fMetadata = 'metadata.csv'
fFile_list = 'file-list.csv'
infectious_category = ['infectious diseases', 'hiv aids']
noninfectious_category = ['endocrinology', 'nephrology', 'neurology', 'oncology', 'otolaryngology',
                          'psychiatry and clinical psychology', 'rheumatology']
model_path = "dmis-lab/biobert-base-cased-v1.1"

# output directory
model_dir = './infectious_classifier/'
fPos = 'test_pos.csv'
fNeg = 'test_neg.csv' 


# read metadata and file list
metadata = read_csv_to_df(fMetadata)
metadata = metadata.drop_duplicates(subset=['doi'], keep='last')
# print(metadata.head())
file_list = read_csv_to_df(fFile_list)


infectious_doi = column_to_doi(metadata, 'category', infectious_category)
noninfectious_doi = column_to_doi(metadata, 'category', noninfectious_category)
print(len(infectious_doi), len(noninfectious_doi))

sampled_infectious_doi = random.sample(infectious_doi, len(noninfectious_doi))
data = sampled_infectious_doi + noninfectious_doi
label = [1] * len(sampled_infectious_doi) + [0] * len(noninfectious_doi)
x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(data, label, train_ratio=0.7, test_ratio=0.1, val_ratio=0.2)

abstract_train = lowercase([read_text(pAbstract+f) for f in keys_to_values(file_list, x_train, 'doi', 'doc_name')])
abstract_val = lowercase([read_text(pAbstract+f) for f in keys_to_values(file_list, x_val, 'doi', 'doc_name')])
abstract_test = lowercase([read_text(pAbstract+f) for f in keys_to_values(file_list, x_test, 'doi', 'doc_name')])


train_encoding = tokenize_data(abstract_train)
val_encoding = tokenize_data(abstract_val)
test_encoding = tokenize_data(abstract_test)

train_dataset = AbstractDataset(train_encoding, y_train)
val_dataset = AbstractDataset(val_encoding, y_val)
test_dataset = AbstractDataset(test_encoding, y_test)

training_args = TrainingArguments(
    output_dir=model_dir,
    logging_dir=model_dir+"logs/",
    logging_strategy='epoch',
    logging_steps=100,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-6,
    seed=42,
    save_strategy='epoch',
    save_steps=100,
    evaluation_strategy='epoch',
    load_best_model_at_end=True
)

model = training_model(2, model_path)

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=compute_metrics
)

trainer.train()
print(trainer.evaluate())

y_test_predict = trainer.predict(test_dataset)
print(y_test_predict)

# Save model
trainer.save_model(model_dir)

# Load model
model = training_model(2, model_dir)
print('Model loaded from '+ model_dir)


# prediction
test_args = TrainingArguments(
    output_dir = model_dir+'output/',
    do_train = False,
    do_predict = True,
    per_device_eval_batch_size = 8,   
    dataloader_drop_last = False    
)
trainer = Trainer(model=model, args = test_args, compute_metrics=compute_metrics)
prediction = trainer.predict(test_dataset)
print(prediction)

pred_labels = get_labels(prediction)
pos_doi, neg_doi = get_doi_from_predict(x_test, pred_labels)
print(len(pos_doi),len(neg_doi))

pos_meta = extract_df_by_keys(metadata, 'doi', pos_doi)
neg_meta = extract_df_by_keys(metadata, 'doi', neg_doi)
pos_meta['doc_name'] = doi_to_file(file_list, pos_doi)
neg_meta['doc_name'] = doi_to_file(file_list, neg_doi)

save_df_to_csv(pos_meta, fPos)
save_df_to_csv(pos_meta, fNeg)


