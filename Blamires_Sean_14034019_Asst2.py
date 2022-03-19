#!/usr/bin/env python
# coding: utf-8

# Import numpy and pandas and upload the data into a pandas dataframe


import numpy as np
import pandas as pd

path = "C:\\Users\\z3066824\\OneDrive - UNSW\Documents\\"


file = "beer_reviews.csv"

df = pd.read_csv(path+file)


# View and clean the dataframe as needed

df

df.shape



df.info()



df.describe()



df_cleaned = df.copy()



df_cleaned.drop('brewery_id', axis=1, inplace=True)



df_cleaned.dropna(inplace=True)


df_cleaned.reset_index(drop=True, inplace=True)


df_cleaned


from sklearn.preprocessing import StandardScaler, OneHotEncoder


num_cols = ['review_overall', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'beer_abv', 'beer_beerid']


sc = StandardScaler()


df_cleaned[num_cols] = sc.fit_transform(df_cleaned[num_cols])


cat_cols = ['beer_style']


cat_cols


ohe = OneHotEncoder(sparse=False)


X_cat = pd.DataFrame(ohe.fit_transform(df_cleaned[cat_cols]))


X_cat.columns = ohe.get_feature_names(cat_cols)



df_cleaned.drop(cat_cols, axis=1, inplace=True)


# Assign X and y variables for modelling


X = pd.concat([df_cleaned, X_cat], axis=1)


X.shape



y = np.array(df['beer_beerid'])


y.shape


y.reshape(-1, 1)


# Import sk learn features and prepare testing and training datasets

from sklearn.model_selection import train_test_split


y_test = train_test_split(
    df['beer_beerid'], test_size=0.2, random_state=42)



X_train, X_test = train_test_split(df_cleaned, test_size=0.2, random_state=8)


from sklearn.dummy import DummyRegressor


dummy_regr = DummyRegressor(strategy="mean")


dummy_regr.fit(X, y[:1518478])


dummy_regr.predict(y)


dummy_regr.score(X, y[:1518478])


baseline_model = DummyRegressor()


# Develop a linear regression model fitting X and y


y_base = baseline_model.fit(X, y[:1518478])


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns


def fit(self, X, y=None):
        return self


def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.columns].values


# Create a pipeline for the model


simple_features = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
])



simple_model = Pipeline([
    ('simple', simple_features),
    ('linear', LinearRegression()),
])


num_transformer = Pipeline(
    steps=[
        ('scaler', StandardScaler())
    ]
)

cat_transformer = Pipeline(
    steps=[
        ('one_hot_encoder', OneHotEncoder(sparse=False, drop='first'))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ('num_cols', num_transformer, num_cols),
        ('cat_cols', cat_transformer, cat_cols)
    ]
)


simple_pipe = Pipeline(
    steps=[
        ('preprocessor', preprocessor)
    ]
)


simple_pipe.fit(df)


# Define a probability function



def positive_probability(baseline_model):
    def predict_proba(X):
        return model.predict_proba(X)[:, 1]
    return predictions



from random import randrange


from random import seed


seed(1)


train = X
test = y[:1518478]



def zero_rule_algorithm_regression(train, test):
    output_values = [row[-1] for row in train]
    prediction = sum(num_cols) / float(len(num_cols))
    predicted = [prediction for i in range(len(test))]
    return predicted


# train a custom neural networks model



conda create -n env_pytorch python=3.8



pip install torchvision


import torch
import torchvision


# Examine and enumerate the qualitative dataset



data = df['beer_style']



data[:10]


set(data)


vocab=set(data)


vocab_size = len(data)


vocab_size


word_to_index = {word: i for i, word in enumerate(vocab)}
word_to_index


data = [word_to_index[word] for word in data]


data[:10]


# Batch and define training data



batch_size = 5


train_data = [([data[i], data[i+1], data[i+2], data[i+3], data[i+4]], data[i+5]) for i in range (vocab_size - batch_size)]



train_data[:10]



embedding_dim = 5


# Build the neural Network


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


class Beers(nn.Module):
    def __init__(self, vocab_size, embedding_dim, batch_size):
        super(Beers, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(batch_size*embedding_dim, 128)
        self.linear2 = nn.Linear (128,512)
        self.linear3 = nn.Linear(512, vocab_size)


def forward(self, inputs):
    embeds = self.embeddings(inputs).view(1,-1)
    out = F. relu(self.linear1(embeds))
    out = F.relu(self.linear2(out))
    out = self.linear3(out)
    log_probs = F.log_softmax(out, dim=1)
    return log_probs


model = Beers(vocab_size, embedding_dim, batch_size)


model


# Define the training function


model.average_loss = []



epochs = range(100)


[epochs]



def train (model, train_data, epochs, word_to_index):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Training on GPU..")
    else:
        device = torch.device("cpu")
        print("Training on CPU..")
        
        model.to(device)
        
        for i in [epochs]:
            model.train()
            steps = 0
            print_every = 100
            running_loss = 0
            for feature, target in train_data:
                feature_tensor = torch.tensor([feature], dtype=torch.long)
                feature_tensor.to(device)
                target_tensor = torch.tensor([target], dtype=torch.long)
                target_tensor.to(device)
                model.zero_grad()
                log_probs = model(target_tensor)
                loss = criterion(log_probs, target_tensor)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                steps +=1
        
                if steps%print_every == 0:
                    model.eval()
                    average_loss.append(running_loss/print_every)
                    print("Epochs: {} / {}".format(i+1, epochs), "Training Loss: {: .3f}".format(running_loss/print_every))
                    running_loss = 0
                model.train
            return model


model = Beers(vocab_size, embedding_dim, batch_size)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

epochs = 100
device = 0

start_time = time.time()


print("training took {} minutes".format(round((start_time - time.time())/60),2))


# Create a batch generator on the train_data


from torch.utils.data.dataset import random_split


train_len = int(len(train_data) * 0.95)
valid_len = len(train_data) - train_len


train_data, valid_data = random_split(train_data, [train_len, valid_len])


examples = enumerate(train_data)
batch_idx, (example_data, example_targets) = next(examples)


example_targets


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


class TextTopic(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text, offsets):
        x = F.dropout(self.embedding(text, offsets), 0.3)
        x = F.dropout(self.fc(x), 0.3)
        return self.softmax(x)


model = Beers(vocab_size, embedding_dim, batch_size)

model

train(model, train_data, epochs, word_to_index)



from torch.utils.data import DataLoader


criterion = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.SGD(model.parameters(), lr=4.0)



scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)


train_data


def train_classification(model, criterion, optimizer, batch_size, device, scheduler=None, generate_batch=None):
    model.train()
    train_loss = 0
    train_acc = 0
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
    
    for feature, offsets, target_class in data:
        optimizer.zero_grad()
        feature, target_class = feature.to(device), target_class.to(device)
        output = model(feature, offsets)
        loss = criterion(output, target_class.long())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == target_class).sum().item()

    if scheduler:
        scheduler.step()

for epoch in range(epochs):
    train_loss, train_acc = train_classification(model, criterion, optimizer, batch_size=batch_size, device=device, scheduler=scheduler, generate_batch=generate_batch)
    valid_loss, valid_acc = test_classification(valid_data, model, criterion, batch_size=batch_size, device=device, generate_batch=generate_batch)

    print(f'Epoch: {epoch}')
    print(f'\t(train)\t|\tLoss: {train_loss:.4f}\t|\tAcc: {train_acc * 100:.1f}%')
    print(f'\t(valid)\t|\tLoss: {valid_loss:.4f}\t|\tAcc: {valid_acc * 100:.1f}%')


# Check Loss functions using keras

pip install keras


pip install tensorflow


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation




from tensorflow.keras.utils import to_categorical



from matplotlib import pyplot as plt


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()


X_train = np.asarray(data).astype('float32')
X_test = np.asarray(test[:1214782]).astype('float32')
y = np.asarray(y[:1214782]).astype('float32')



model = Sequential()

model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])




plt.show(model)


model.fit(data, y,
          epochs=100,
          callbacks=[plot_losses])


# Plot average loss


loss_plot = pd.DataFrame(model.average_loss)
loss_plot.plot()


# Push model to github


git init
git add .
commit -m "pytorch Assignment2"
git checkout master
git pull
git checkout pytorch_Assignment2

commit -m "Assignment2"



