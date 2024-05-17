import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import jieba
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.simplefilter('ignore')

parameters = {
    'vocab_dim': 200, # 词向量维数
    'hidden_dim': 128, # 隐藏层维数
    'output_dim': 12, # 输出层维数(12类别)
    'dropout': 0, 
    'batch_size': 32,
    'max_length': 28,
    'epoch': 15,
}


def load_data():
    data_dir = 'https://mirror.coggle.club/dataset/coggle-competition/'
    train_data = pd.read_csv(data_dir + 'intent-classify/train.csv', sep='\t', header=None)
    test_data = pd.read_csv(data_dir + 'intent-classify/test.csv', sep='\t', header=None)
    

    le = LabelEncoder()
    train_data[1] = le.fit_transform(train_data[1])

    train_data['text'] = train_data[0]
    train_data['label'] = train_data[1]
    train_data.drop(columns=[0, 1], inplace=True)

    test_data['text'] = test_data[0]
    test_data.drop(columns=[0], inplace=True)
    
    return train_data, test_data

def tokenize(corpus):
    cn_stopwords = pd.read_csv('https://mirror.coggle.club/stopwords/baidu_stopwords.txt', header=None)[0].values
    
    texts = []
    for i in range(len(corpus)):
        content = ''.join(corpus.iloc[i])
        words = jieba.lcut(content)
        words = [word for word in words if word not in cn_stopwords]
        texts.append(words)    

    # Count word frequencies
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
    return processed_corpus


def word_vector(corpus):
    word2vector_model = Word2Vec.load(r"D:\Major\Blog\Coggle-Assignments\Coggle_2024_5\word2vec_model.model")
    word2vector_model.train(corpus, total_examples=len(corpus), epochs=10)


def create_dictionaries(model=None, corpus=None):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.index_to_key, allow_update=True)
    w2indx = {v: k+1 for k, v in gensim_dict.items()}
    w2vec = {word: model.wv[word] for word in w2indx.keys()}
    
    def parse_dataset(corpus):
        data = []
        for sentence in corpus:
            new_txt = []
            for word in sentence:
                try:
                    new_txt.append(w2indx[word])
                except:
                    new_txt.append(0)
            new_txt = torch.tensor(new_txt)
            data.append(new_txt)
        return data
    
    corpus = parse_dataset(corpus)
    corpus = pad_sequence(corpus)
    
    return w2indx, w2vec, corpus

def word2vec_train(corpus, model):
    index_dict, word_vectors, conbinds = create_dictionaries(model=model, corpus=corpus)
    return index_dict, word_vectors, conbinds

def get_data(index_dict, word_vectors, corpus, y, vocab_dim):
    n_symbols = len(index_dict) + 1
    embedding_weight = np.zeros((n_symbols, vocab_dim))
    
    for word, index in index_dict.items():
        embedding_weight[index, :] = word_vectors[word]
    X_train, X_val, y_train, y_val = train_test_split(corpus, y, test_size=0.2)
    
    return n_symbols, embedding_weight, X_train, X_val, y_train, y_val

def convert_hidden_shape(hidden, batch_size):
    tensor_list = []

    for i in range(batch_size):
        ts = hidden[i,: , :].reshape(1, -1)
        tensor_list.append(ts)

    ts = torch.cat(tensor_list)
    return ts

class LSTMmodel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, dropout, output_size, num_layers, max_length):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * max_length, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = convert_hidden_shape(out, out.shape[0])
        out = self.fc(out)

        return out

def train():
    train_data, test_data = load_data()
    corpus = train_data['text']
    corpus = tokenize(corpus=corpus)
    
    word2vector_model = Word2Vec.load(r"D:\Major\Blog\Coggle-Assignments\Coggle_2024_5\word2vec_model.model")
    word2vector_model.train(corpus, total_examples=len(corpus), epochs=10)
    
    index_dict, word_vectors, conbinds = word2vec_train(corpus, model=word2vector_model)
    vocab_size, embedding_weight, X_train, X_val, y_train, y_val = get_data(index_dict, word_vectors, conbinds.T, train_data['label'], parameters['vocab_dim'])
    
    model = LSTMmodel(vocab_size, parameters['vocab_dim'], parameters['hidden_dim'], parameters['dropout'], parameters['output_dim'], num_layers=1, max_length=parameters['max_length'])
    model.embedding.weight.data.copy_(torch.from_numpy(embedding_weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  
    
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

    batch_size = parameters['batch_size']
    
    for epoch in range(parameters['epoch']):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train) - batch_size, batch_size):
            optimizer.zero_grad()
            x_batch = X_train[i:i+batch_size]
            y_batch_tensor = y_train_tensor[i:i+batch_size]
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / X_train.shape[0]}')

    # 验证模型
    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_val_tensor).sum().item() / len(y_val_tensor)
        print(f'Validation Accuracy: {accuracy}')


if __name__ == '__main__':
    train()



