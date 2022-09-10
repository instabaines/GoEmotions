import pandas as pd
import os
import numpy as np
import json

emotions =['admiration',
 'amusement',
 'anger',
 'annoyance',
 'approval',
 'caring',
 'confusion',
 'curiosity',
 'desire',
 'disappointment',
 'disapproval',
 'disgust',
 'embarrassment',
 'excitement',
 'fear',
 'gratitude',
 'grief',
 'joy',
 'love',
 'nervousness',
 'optimism',
 'pride',
 'realization',
 'relief',
 'remorse',
 'sadness',
 'surprise',
 'neutral']
with open('ekman_mapping.json') as file:
    ekman_mapping = json.load(file)

def load_file(path):
    data=pd.read_csv(path, sep='\t', header=None, names=['Text', 'Class', 'ID'])
    return data
def idx2class(idx_list):
    arr = []
    for i in idx_list:
        arr.append(emotions[int(i)])
    return arr


def EmotionMapping(emotion_list):
    map_list = []
    
    for i in emotion_list:
        if i in ekman_mapping['anger']:
            map_list.append('anger')
        if i in ekman_mapping['disgust']:
            map_list.append('disgust')
        if i in ekman_mapping['fear']:
            map_list.append('fear')
        if i in ekman_mapping['joy']:
            map_list.append('joy')
        if i in ekman_mapping['sadness']:
            map_list.append('sadness')
        if i in ekman_mapping['surprise']:
            map_list.append('surprise')
        if i == 'neutral':
            map_list.append('neutral')
            
    return map_list
def process(path): 
    df_train =load_file(os.path.join(path,'train.tsv'))
    df_dev =load_file(os.path.join(path,'dev.tsv'))
    df_train['List of classes'] = df_train['Class'].apply(lambda x: str(x).split(','))
    df_train['Len of classes'] = df_train['List of classes'].apply(lambda x: len(x))
    df_dev['List of classes'] = df_dev['Class'].apply(lambda x: x.split(','))
    df_dev['Len of classes'] = df_dev['List of classes'].apply(lambda x: len(x))
    df_train['Emotions'] = df_train['List of classes'].apply(idx2class)
    df_dev['Emotions'] = df_dev['List of classes'].apply(idx2class)
    df_train['Mapped Emotions'] = df_train['Emotions'].apply(EmotionMapping)
    df_dev['Mapped Emotions'] = df_dev['Emotions'].apply(EmotionMapping)
    df_train['anger'] = np.zeros((len(df_train),1))
    df_train['disgust'] = np.zeros((len(df_train),1))
    df_train['fear'] = np.zeros((len(df_train),1))
    df_train['joy'] = np.zeros((len(df_train),1))
    df_train['sadness'] = np.zeros((len(df_train),1))
    df_train['surprise'] = np.zeros((len(df_train),1))
    df_train['neutral'] = np.zeros((len(df_train),1))

    df_dev['anger'] = np.zeros((len(df_dev),1))
    df_dev['disgust'] = np.zeros((len(df_dev),1))
    df_dev['fear'] = np.zeros((len(df_dev),1))
    df_dev['joy'] = np.zeros((len(df_dev),1))
    df_dev['sadness'] = np.zeros((len(df_dev),1))
    df_dev['surprise'] = np.zeros((len(df_dev),1))
    df_dev['neutral'] = np.zeros((len(df_dev),1))
    for i in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise','neutral']:
        df_train[i] = df_train['Mapped Emotions'].apply(lambda x: 1 if i in x else 0)
        df_dev[i] = df_dev['Mapped Emotions'].apply(lambda x: 1 if i in x else 0)
    df_train.drop(['Class', 'List of classes', 'Len of classes', 'Emotions', 'Mapped Emotions', 'neutral'], axis=1, inplace=True)
    df_dev.drop(['Class','ID', 'List of classes', 'Len of classes', 'Emotions', 'Mapped Emotions', 'neutral',], axis=1, inplace=True)
    df_train = df_train.reset_index(drop=True)
    df_dev = df_dev.reset_index(drop=True)
    target_cols = [col for col in df_train.columns if col =='Text']
    return df_train,df_dev,target_cols


def main(path_to_files,path_to_store):
    df_train,df_dev,target_cols = process(path_to_files)
    df_train.to_csv(os.path.join(path_to_store,'train.csv'),index=False)
    df_dev.to_csv(os.path.join(path_to_store,'dev.csv'),index=False)
    json.dump(target_cols,open(os.path.join(path_to_store,'target_cols.json'),'w'))


if __name__=='__main__':
    main('path_to_files','path_to_store')
