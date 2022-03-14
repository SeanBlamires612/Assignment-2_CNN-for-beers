#!/usr/bin/env python
# coding: utf-8

# Import numpy and pandas and upload the data into a pandas dataframe

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


path = "C:\\Users\\z3066824\\OneDrive - UNSW\Documents\\"


# In[3]:


file = "beer_reviews.csv"


# In[4]:


df = pd.read_csv(path+file)


# View and clean the dataframe as needed

# In[5]:


df


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df_cleaned = df.copy()


# In[10]:


df_cleaned.drop('brewery_id', axis=1, inplace=True)


# In[11]:


df_cleaned.dropna(inplace=True)


# In[12]:


df_cleaned.reset_index(drop=True, inplace=True)


# In[13]:


df_cleaned


# Identify and transform the numeric columns within the cleaned dataframe

# In[14]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder


# In[15]:


num_cols = ['review_overall', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'beer_abv', 'beer_beerid']


# In[16]:


sc = StandardScaler()


# In[17]:


df_cleaned[num_cols] = sc.fit_transform(df_cleaned[num_cols])


# In[18]:


cat_cols = ['beer_style']


# In[19]:


cat_cols


# In[20]:


ohe = OneHotEncoder(sparse=False)


# In[21]:


X_cat = pd.DataFrame(ohe.fit_transform(df_cleaned[cat_cols]))


# In[22]:


X_cat.columns = ohe.get_feature_names(cat_cols)


# In[23]:


df_cleaned.drop(cat_cols, axis=1, inplace=True)


# Assign X and y variables for modelling

# In[24]:


X = pd.concat([df_cleaned, X_cat], axis=1)


# In[25]:


X.shape


# In[26]:


y = np.array(df['beer_beerid'])


# In[27]:


y.shape


# In[28]:


y.reshape(-1, 1)


# Import sk learn features and prepare testing and training datasets

# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


y_test = train_test_split(
    df['beer_beerid'], test_size=0.2, random_state=42)


# In[31]:


X_train, X_test = train_test_split(df_cleaned, test_size=0.2, random_state=8)


# In[32]:


from sklearn.dummy import DummyRegressor


# In[33]:


dummy_regr = DummyRegressor(strategy="mean")


# In[34]:


dummy_regr.fit(X, y[:1518478])


# In[35]:


dummy_regr.predict(y)


# In[36]:


dummy_regr.score(X, y[:1518478])


# In[37]:


baseline_model = DummyRegressor()


# 

# Develop a linear regression model fitting X and y

# In[58]:


y_base = baseline_model.fit(X, y[:1518478])


# In[39]:


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


# In[40]:


from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# In[41]:


class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns


# In[42]:


def fit(self, X, y=None):
        return self


# In[43]:


def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.columns].values


# Create a pipeline for the model

# In[44]:


simple_features = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
])


# In[45]:


simple_model = Pipeline([
    ('simple', simple_features),
    ('linear', LinearRegression()),
])


# In[46]:


num_transformer = Pipeline(
    steps=[
        ('scaler', StandardScaler())
    ]
)


# In[47]:


cat_transformer = Pipeline(
    steps=[
        ('one_hot_encoder', OneHotEncoder(sparse=False, drop='first'))
    ]
)


# In[48]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num_cols', num_transformer, num_cols),
        ('cat_cols', cat_transformer, cat_cols)
    ]
)


# In[49]:


simple_pipe = Pipeline(
    steps=[
        ('preprocessor', preprocessor)
    ]
)


# In[50]:


simple_pipe.fit(df)


# Define a probability function

# In[65]:


def positive_probability(baseline_model):
    def predict_proba(X):
        return model.predict_proba(X)[:, 1]
    return predictions


# In[66]:


from random import randrange


# In[67]:


from random import seed


# In[68]:


seed(1)


# In[69]:


train = X
test = y[:1518478]


# In[80]:


def zero_rule_algorithm_regression(train, test):
    output_values = [row[-1] for row in train]
    prediction = sum(num_cols) / float(len(num_cols))
    predicted = [prediction for i in range(len(test))]
    return predicted


# train a custom neural networks model

# In[56]:


conda create -n env_pytorch python=3.8


# In[54]:


pip install torchvision


# In[55]:


import torch
import torchvision


# Examine and enumerate the qualitative dataset

# In[84]:


data = df['beer_style']


# In[85]:


data[:10]


# In[86]:


set(data)


# In[87]:


vocab=set(data)


# In[88]:


vocab_size = len(data)


# In[89]:


vocab_size


# In[91]:


word_to_index = {word: i for i, word in enumerate(vocab)}
word_to_index


# In[92]:


data = [word_to_index[word] for word in data]


# In[93]:


data[:10]


# Batch and define training data

# In[94]:


batch_size = 5


# In[95]:


train_data = [([data[i], data[i+1], data[i+2], data[i+3], data[i+4]], data[i+5]) for i in range (vocab_size - batch_size)]


# In[96]:


train_data[:10]


# In[97]:


embedding_dim = 5


# Build the neural Network

# In[98]:


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


# In[106]:


class Beers(nn.Module):
    def __init__(self, vocab_size, embedding_dim, batch_size):
        super(Beers, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(batch_size*embedding_dim, 128)
        self.linear2 = nn.Linear (128,512)
        self.linear3 = nn.Linear(512, vocab_size)


# In[107]:


def forward(self, inputs):
    embeds = self.embeddings(inputs).view(1,-1)
    out = F. relu(self.linear1(embeds))
    out = F.relu(self.linear2(out))
    out = self.linear3(out)
    log_probs = F.log_softmax(out, dim=1)
    return log_probs


# In[108]:


model = Beers(vocab_size, embedding_dim, batch_size)


# In[109]:


model


# Define the training function

# In[144]:


model.average_loss = []


# In[117]:


epochs = range(100)


# In[130]:


[epochs]


# In[149]:


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


# In[150]:


model = Beers(vocab_size, embedding_dim, batch_size)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

epochs = 100
device = 0

start_time = time.time()


# In[151]:


print("training took {} minutes".format(round((start_time - time.time())/60),2))


# Create a batch generator on the train_data

# In[158]:


from torch.utils.data.dataset import random_split


# In[160]:


train_len = int(len(train_data) * 0.95)
valid_len = len(train_data) - train_len


# In[161]:


train_data, valid_data = random_split(train_data, [train_len, valid_len])


# In[162]:


examples = enumerate(train_data)
batch_idx, (example_data, example_targets) = next(examples)


# In[163]:


example_targets


# In[164]:


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


# In[165]:


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


# In[166]:


model = Beers(vocab_size, embedding_dim, batch_size)


# In[167]:


model


# In[177]:


train(model, train_data, epochs, word_to_index)


# 

# In[190]:


from torch.utils.data import DataLoader


# In[183]:


criterion = torch.nn.CrossEntropyLoss()


# In[184]:


optimizer = torch.optim.SGD(model.parameters(), lr=4.0)


# In[185]:


scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)


# In[194]:


train_data


# In[195]:


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


# In[196]:


for epoch in range(epochs):
    train_loss, train_acc = train_classification(model, criterion, optimizer, batch_size=batch_size, device=device, scheduler=scheduler, generate_batch=generate_batch)
    valid_loss, valid_acc = test_classification(valid_data, model, criterion, batch_size=batch_size, device=device, generate_batch=generate_batch)

    print(f'Epoch: {epoch}')
    print(f'\t(train)\t|\tLoss: {train_loss:.4f}\t|\tAcc: {train_acc * 100:.1f}%')
    print(f'\t(valid)\t|\tLoss: {valid_loss:.4f}\t|\tAcc: {valid_acc * 100:.1f}%')


# Check Loss functions using keras

# In[178]:


pip install keras


# In[197]:


pip install tensorflow


# In[199]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation


# In[209]:


from tensorflow.keras.utils import to_categorical


# In[235]:


from matplotlib import pyplot as plt


# In[200]:


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


# In[229]:


X_train = np.asarray(data).astype('float32')
X_test = np.asarray(test[:1214782]).astype('float32')
y = np.asarray(y[:1214782]).astype('float32')


# In[223]:


model = Sequential()

model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[232]:


plt.show(model)


# In[234]:


model.fit(data, y,
          epochs=100,
          callbacks=[plot_losses])


# Plot average loss

# In[155]:


loss_plot = pd.DataFrame(model.average_loss)
loss_plot.plot()


# Push model to github

# In[237]:


#In VS code:
#git add .
#git commit -m "pytorch Assignment2"
#git push --set-upstream origin gmm_pipeline
#git pull
#git checkout pytorch_Assignment2


# In[ ]:




