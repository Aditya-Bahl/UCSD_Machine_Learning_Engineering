# Model Inference

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
import time
import io

from google.cloud import storage
client = storage.Client.from_service_account_json(json_credentials_path='yourfile.json')
bucket = client.bucket('yourbucket1')

blob = bucket.blob('news_data.csv')
blob.download_to_filename('data.csv')
df = pd.read_csv('data.csv')

#load the XLNET model and pre-trained weights
model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=3)

model.load_state_dict(torch.load("model_with_retraining.ckpt", map_location=torch.device('cpu')))
# keep map_location
# model.cuda()

# prediction function to determine sentiment of news headlines
def predict_sentiment(text):
    review_text = text

    encoded_review = tokenizer.encode_plus(
    review_text,
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=False,
    return_attention_mask=True,
    return_tensors='pt',
    )

    input_ids = pad_sequences(encoded_review['input_ids'], maxlen=MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
    input_ids = input_ids.astype(dtype = 'int64')
    input_ids = torch.tensor(input_ids)
    attention_mask = pad_sequences(encoded_review['attention_mask'], maxlen=MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
    attention_mask = attention_mask.astype(dtype = 'int64')
    attention_mask = torch.tensor(attention_mask)

    input_ids = input_ids.reshape(1,128).to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    outputs = outputs[0][0].cpu().detach()

    probs = F.softmax(outputs, dim=-1).cpu().detach().numpy().tolist()
    _, prediction = torch.max(outputs, dim =-1)

    target_names = ['negative', 'neutral', 'positive']

    return probs, target_names[prediction]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
MAX_LEN = 128

probs_list = []
prediction_list = []
for sentence in df['headline']:
    probs, prediction = predict_sentiment(sentence)
    probs_list.append(probs)
    prediction_list.append(prediction)

probs_df = pd.DataFrame(probs_list)
probs_df.columns = ['negative', 'neutral', 'positive']

prediction_df = pd.DataFrame(prediction_list)
prediction_df.columns = ['Sentiment']

#classified news headlines
final_df = pd.concat([df,probs_df,prediction_df], axis=1)
final_df["datetime"] = pd.to_datetime(final_df["datetime"], unit='s').dt.strftime('%Y-%m-%d')
final_df = final_df.rename(columns={"datetime":"Date"})
final_df = final_df.rename(columns={"related":"name"})


# getting data from Google cloud
blob = bucket.blob('stock_data.csv')
blob.download_to_filename('stock1.csv')
hist = pd.read_csv('stock1.csv')

pd.set_option('display.max_columns', None)
complete_df = pd.merge(final_df, hist, how='left', on=['Date', 'name'])

# posting data to Google cloud
bucket = client.get_bucket('yourbucket2')
object_name_in_gcs_bucket = bucket.blob('complete_df_'+ time.strftime('%Y%m%d')+'.csv')
df = pd.DataFrame(data=complete_df).to_csv(encoding="UTF-8")
object_name_in_gcs_bucket.upload_from_string(data=df)
