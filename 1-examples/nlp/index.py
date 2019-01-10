import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras import layers

# Ignore tensorflow cpu instruction set warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load text files
dirname = os.path.dirname(__file__)
filepath_dict = {
    'yelp': os.path.join(dirname, 'data/yelp_labelled.txt'),
    'amazon': os.path.join(dirname, 'data/amazon_cells_labelled.txt'),
    'imdb': os.path.join(dirname, 'data/imdb_labelled.txt'),
}

# Load texts into one dataframe
df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source
    df_list.append(df)

df = pd.concat(df_list)

# For every source (amazon/yelp/imdb) in the dataframe
for source in df['source'].unique():
    # Get sentences (x) and labels (y)
    df_source = df[df['source'] == source]
    sentences = df_source['sentence'].values
    y = df_source['label'].values

    # Split datasets
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences,
        y,
        test_size=0.25,
        random_state=1000
    )

    # CountVectorizer: Convert the sentences to a (sparse) matrix of token counts
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)

    X_train = vectorizer.transform(sentences_train)
    X_test = vectorizer.transform(sentences_test)

    # Logistic regression
    classifier = LogisticRegression(solver='lbfgs')
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)

    print(source)
    print("===========")

    print('Logistic regression: {:.4f}'.format(score))

    # NN
    input_dim = X_train.shape[1]

    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.fit(
        X_train,
        y_train,
        epochs=30,
        verbose=False,
        validation_data=(X_test, y_test),
        batch_size=10
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print('NN Testing:  {:.4f}'.format(accuracy))
