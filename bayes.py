import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.likelihoods = {}
        self.vocab_size = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.vocab_size = n_features

        for c in self.classes:
            X_c = X[y == c]
            
            self.priors[c] = X_c.shape[0] / n_samples
            
            word_counts = np.sum(X_c, axis=0)
            total_words_in_class = np.sum(word_counts)
            
            self.likelihoods[c] = (word_counts + 1) / (total_words_in_class + self.vocab_size)

    def predict(self, X):
        predictions = []
        
        for x in X:
            posteriors = {}
            
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self.likelihoods[c]) * x)
                posteriors[c] = prior + likelihood

            predictions.append(max(posteriors, key=posteriors.get))
            
        return np.array(predictions)

def main():
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['class', 'message']

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['message']).toarray()
    y = df['class'].values

    print(f"Liczba próbek: {X.shape[0]}, Liczba słów (cech): {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)

    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    print(f"Dokładność: {accuracy_score(y_test, y_pred):.4f}")
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred))

    kf = KFold(n_splits=5, shuffle=True, random_state=5)
    accuracies = []
    
    fold = 1
    for train_index, val_index in kf.split(X):
        X_train_k, X_val_k = X[train_index], X[val_index]
        y_train_k, y_val_k = y[train_index], y[val_index]
        
        model_k = NaiveBayesClassifier()
        model_k.fit(X_train_k, y_train_k)
        predictions_k = model_k.predict(X_val_k)
        
        acc = accuracy_score(y_val_k, predictions_k)
        accuracies.append(acc)
        fold += 1

    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.xlabel('Przewidziana klasa')
    plt.ylabel('Rzeczywista klasa')
    plt.title('Macierz Konfuzji dla Klasyfikatora Bayesa')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, 6), accuracies, color='skyblue')
    plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', label=f'Średnia: {np.mean(accuracies):.4f}')
    plt.ylim(0.9, 1.0)
    plt.xlabel('Numer podgrupy')
    plt.ylabel('Dokładność')
    plt.title('Stabilność modelu w k-krotnej walidacji krzyżowej')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()