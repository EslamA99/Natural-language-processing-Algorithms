import gensim as gensim
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def get_x_y():
    data_set = load_files(r"txt_sentoken", categories=['neg', 'pos'])
    x, y = data_set.data, data_set.target
    x, y = shuffle(x, y)
    return x, y


def preprocess(doc):
    documents = []
    stemmer = nltk.WordNetLemmatizer()
    for sen in range(0, len(doc)):
        document = re.sub(r'\W', ' ', str(doc[sen]))
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        document = re.sub(r'^b\s+', '', document)
        document = document.lower()
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        documents.append(document)
    return documents


def avgEmbedding(x_data, m):
    returnedSentencesAverage = []
    for rev in x_data:
        doc = []
        for word in rev:
            if word in m.wv.key_to_index:
                doc.append(word)
        mean = np.mean(m.wv[doc], axis=0)
        returnedSentencesAverage.append(mean)
    return returnedSentencesAverage


print('Getting data from file with labeling...')
X, y = get_x_y()
X = preprocess(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


def svm(train, test, ytr, yte, alpha):
    print('SVM classifier with alpha = ' + str(alpha))
    clf = LinearSVC(C=alpha)
    clf.fit(train, ytr)
    y_pred = clf.predict(test)
    print('Accuracy: ', metrics.accuracy_score(yte, y_pred))


def logistic(train, test, ytr, yte, alpha):
    print('logistic classifier with alpha = ' + str(alpha))
    logistic = LogisticRegression(C=alpha)
    logistic.fit(train, ytr)
    y_pred = logistic.predict(test)
    print('Accuracy: ', metrics.accuracy_score(yte, y_pred))


def mlp(train, test, ytr, yte, hidden, mxIte):
    print('neural network MLP classifier with hidden layers = ' + str(hidden) + ' and maxIterations = ' + str(mxIte))
    clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=mxIte)
    clf.fit(train, ytr)
    y_pred = clf.predict(test)
    print('Accuracy: ', metrics.accuracy_score(yte, y_pred))


tfidfconverter = TfidfVectorizer(stop_words=stopwords.words('english'), use_idf=True)
tf_trained = tfidfconverter.fit_transform(X_train)
X_train11 = tf_trained.toarray()
X_test11 = tfidfconverter.transform(X_test).toarray()

svm(X_train11, X_test11, y_train, y_test, 1.0)
svm(X_train11, X_test11, y_train, y_test, 0.5)
#####################################################################################################
logistic(X_train11, X_test11, y_train, y_test, 1.0)
logistic(X_train11, X_test11, y_train, y_test, 0.5)
#####################################################################################################
mlp(X_train11, X_test11, y_train, y_test, 50, 1000)
mlp(X_train11, X_test11, y_train, y_test, 20, 900)
#####################################################################################################
print("-----------------------------------------------------------")
print()
print()


def corpus(X):
    corpus = []
    for string in X:
        words = string.split()
        grams = [" ".join(words[i:i + 1]) for i in range(0, len(words), 1)]
        corpus.append(grams)
    return corpus


X_train = corpus(X_train)
X_test = corpus(X_test)


#####################################################################################################
def runWord2Vec(vecSize, window, minCount, epchs, workers, sg):
    print('parameters = (vecSize=' + str(vecSize) + ',window=' + str(window) + ',minCount=' + str(
        minCount) + ',epochs=' + str(epchs) + ',workers=' + str(workers) + ',sg=' + str(sg) + ')')
    model = gensim.models.Word2Vec(
        X_train,
        vector_size=vecSize,
        window=window,
        min_count=minCount,
        epochs=epchs,
        workers=workers, sg=sg)

    x_train_after_embedding = avgEmbedding(X_train, model)
    x_test_after_embedding = avgEmbedding(X_test, model)
    svm(x_train_after_embedding, x_test_after_embedding, y_train, y_test, 1.0)
    svm(x_train_after_embedding, x_test_after_embedding, y_train, y_test, 0.5)
    #####################################################################################################
    logistic(x_train_after_embedding, x_test_after_embedding, y_train, y_test, 1.0)
    logistic(x_train_after_embedding, x_test_after_embedding, y_train, y_test, 0.5)
    #####################################################################################################
    mlp(x_train_after_embedding, x_test_after_embedding, y_train, y_test, 120, 1000)
    mlp(x_train_after_embedding, x_test_after_embedding, y_train, y_test, 150, 800)
    #####################################################################################################
    print("-----------------------------------------------------------")
    print()


#####################################################################################################
# print("change window and minCount parameters")
# runWord2Vec(200, 10, 2, 10, 5, 1)
# runWord2Vec(150, 8, 2, 10, 10, 1)
# runWord2Vec(300, 5, 2, 10, 10, 1)
# runWord2Vec(500, 10, 5, 10, 10, 1)
# print()
# print()
# print("change workers parameter")
# runWord2Vec(150, 10, 2, 10, 5, 1)
# runWord2Vec(300, 5, 2, 10, 10, 1)
# runWord2Vec(500, 10, 5, 10, 20, 1)
# print()
# print()
# print("change sg parameter")
# runWord2Vec(200, 10, 2, 10, 5, 0)
# runWord2Vec(500, 10, 5, 25, 10, 0)
# runWord2Vec(150, 10, 2, 20, 10, 0)
# print()
# print("change epochs parameter")
# runWord2Vec(150, 10, 2, 20, 10, 1)
# runWord2Vec(300, 5, 2, 10, 10, 1)
# runWord2Vec(200, 10, 5, 15, 10, 1)
# runWord2Vec(500, 10, 5, 25, 10, 1)

runWord2Vec(200, 10, 2, 10, 5, 1)
runWord2Vec(500, 10, 5, 25, 10, 1)
runWord2Vec(150, 10, 2, 20, 10, 1)
