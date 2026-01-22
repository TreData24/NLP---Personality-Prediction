## Import Libraries
import argparse
import csv
import json
import os
import re
import string
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
import sklearn.metrics as metrics
import numpy as np

stop_words = nltk.corpus.stopwords.words('english')    
stop_words.append('*')
stop_words.append("...")
stop_words.append("--")
stop_words.append("''")
stop_words.append("``")

## Import Data
def convert_to_JSON(csvfile):
    print('\nConverting from CSV to JSON..')
    csvfile = open(csvfile, 'r')
    jsonfile = open(jsonpath, 'w')
    fieldnames = ("#AUTHID","STATUS","sEXT","sNEU","sAGR","sCON","sOPN", "cEXT", "cNEU", "cAGR", "cCON", "cOPN", "DATE", "NETWORKSIZE", "BETWEENNESS", "NBETWEENNESS", "DENSITY", "BROKERAGE", "NBROKERAGE", "TRANSITIVITY")
    reader = csv.DictReader( csvfile, fieldnames)
    for row in reader:
        json.dump(row, jsonfile)
        jsonfile.write('\n')

def read_and_clean_lines(JSONFile):
    print('\nReading and cleaning lines...')
    lines = []
    neuLines = []
    nonNeuLines = []
    NeurosisCat = []
    author_to_posts = {}
    for line in tqdm(open(JSONFile)):
        data = json.loads(line)
        status = re.sub('\s+',' ',data['STATUS'])
        if not data['#AUTHID'] in author_to_posts:
            author_to_posts[data['#AUTHID']]=''
        author_to_posts[data['#AUTHID']]=" * ".join([author_to_posts[data['#AUTHID']], status])
        lines.append(status)
        if data['cNEU'] == 'y':
            neuLines.append(status)
            NeurosisCat.append(1)
        else:
            nonNeuLines.append(status)
            NeurosisCat.append(0)
    print(author_to_posts)
    return [lines, neuLines, nonNeuLines, NeurosisCat]

def normalize_tokens(tokenlist):
    normalized_tokens = [token.lower().replace('_','+') for token in tokenlist   # lowercase, _ => +
                            if re.search('[^\s]', token) is not None            # ignore whitespace tokens
                            and not token.startswith("@")                       # ignore  handles
                    ]
    return normalized_tokens   

def ngrams(tokens, n):
    return_ngram = []
    for index in range(len(tokens)-n+1):
        result = tokens[index:(index+n)]
        return_ngram.append(result)
    return return_ngram

def filter_punctuation_bigrams(ngrams):
    punct = string.punctuation
    return [ngram   for ngram in ngrams   if ngram[0] not in punct and ngram[1] not in punct]

def filter_punctuation(tokens):
    punct = string.punctuation
    return [token   for token in tokens   if token not in punct]


def filter_stopword_bigrams(ngrams, stop_words):
    result = [ngram   for ngram in ngrams   if ngram[0] not in stop_words and ngram[1] not in stop_words]
    return result

def filter_stopword(tokens, stop_words):
    result = [token   for token in tokens   if token[0] not in stop_words]
    return result


def filter_and_normalize(lines, remove_stopword_bigrams, stopwords, remove_punctuation, n):
    toReturn = []
    for line in lines:
        tokens = TweetTokenizer().tokenize(line.lower())  # Convert text to lowercase and tokenize   word_tokenize
        # Normalize 
        normalized_tokens=normalize_tokens(tokens)
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in normalized_tokens]
        # Get bigrams
        bigrams = ngrams(lemmatized_tokens, n)
        # Filter out bigrams where either token is punctuation
        if remove_punctuation and n >= 2:
            filtered = filter_punctuation_bigrams(bigrams)
        else:
            filtered = bigrams
        # Optionally filter bigrams where either word is a stopword
        if remove_stopword_bigrams and n >= 2:
            filtered = filter_stopword_bigrams(filtered, stopwords)
        # Increment bigram counts
        formatted = [tuple(i) for i in filtered]
        toReturn.append(formatted)
    return toReturn

def filter_and_normalizeV2(line, remove_stopword_bigrams, stopwords, remove_punctuation):
    tokens = word_tokenize(line.lower())  # Convert text to lowercase and tokenize
    # Normalize 
    normalized_tokens=normalize_tokens(tokens)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in normalized_tokens]
    # Get bigrams
    # Filter out bigrams where either token is punctuation
    if remove_punctuation:
        filtered = filter_punctuation(lemmatized_tokens)
    else:
        filtered = lemmatized_tokens
    # Optionally filter bigrams where either word is a stopword
    if remove_stopword_bigrams:
        filtered = filter_stopword(filtered, stopwords)
    return filtered

## Normalize, Tokenize, Filter data
def collect_bigram_counts(lines):
    counter   = Counter()
    for formatted in lines:
        counter.update(formatted)
    return counter

def print_sorted_items(dict, n=10, order='descending'):
    if order == 'descending':
        multiplier = -1
    else:
        multiplier = 1
    ranked = sorted(dict.items(), key=lambda x: x[1] * multiplier)
    for key, value in ranked[:n] :
        print(key, value)

def split_training_set(lines, labels, test_size=0.3, random_seed=42):
    X_train, X_test, y_train, y_test = train_test_split(lines, labels, test_size=test_size, random_state=random_seed, stratify=labels)
    print("Training set label counts: {}".format(Counter(y_train)))
    print("Test set     label counts: {}".format(Counter(y_test)))
    return X_train, X_test, y_train, y_test

def convert_text_into_features(X, stopwords_arg, analyzefn="word", range=(1,2)):
    training_vectorizer = CountVectorizer(stop_words=stopwords_arg,
                                          analyzer=analyzefn,
                                          lowercase=True,
                                          ngram_range=range)
    X_features = training_vectorizer.fit_transform(X)
    return X_features, training_vectorizer
def whitespace_tokenizer(line):
    return line.split()
def convert_lines_to_feature_strings(lines, stop_words, remove_punctuation, remove_stopword_bigrams=True):

    print(" Converting from raw text to unigram and bigram features")
    if remove_stopword_bigrams:
        print(" Includes filtering stopword bigrams")
        
    print(" Initializing")
    all_features = []
    print(" Iterating through documents extracting unigram and bigram features")
    for line in tqdm(lines):

        normalized_tokens = filter_and_normalizeV2(line, remove_stopword_bigrams, stop_words, remove_punctuation)
        # Collect unigram tokens as features
        unigrams          = normalized_tokens
        
        # Collect string bigram tokens as features
        bigrams           = ngrams(normalized_tokens, 2) 

        bigram_tokens = ["_".join(bigram) for bigram in bigrams]

        feature_list   = unigrams + bigram_tokens

        feature_string = " ".join(feature_list)

        all_features.append(feature_string)


    # print(" Feature string for first document: '{}'".format(all_features[0]))
        
    return all_features

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Temporary')
    parser.add_argument('--infile',                     default = None,     action='store',      help="Input JSON file.")
    parser.add_argument('--remove_stopwords_bigrams',   default = True,     action='store',      help="Exclude Stopwords?")
    parser.add_argument('--remove_punctuation',         default = True,     action='store',      help="Exclude Punctuation?")
    parser.add_argument('--topN_to_show',               default = 15,     action='store',      help="How Many N-grams?")
    parser.add_argument('--print_top_ngrams',           default = True,     action='store',      help="Show class N-grams?")
    parser.add_argument('--use_sklearn_features', default=False, action='store_true', help="Use sklearn's feature extraction")
    parser.add_argument('--test_size',            default=0.3,   action='store',      help="Proportion (from 0 to 1) of items held out for final testing")
    parser.add_argument('--num_folds',            default=5,     action='store',      help="Number of folds for cross-validation (use 2 for just a train/test split)")
    parser.add_argument('--stratify',             default=False, action='store_true', help="Use stratified rather than plain cross-validation")
    parser.add_argument('--seed',                 default=13,    action='store',      help="Random seed")
    args = parser.parse_args()
    jsonpath = 'jsonfile.json'
    if not os.path.isfile(jsonpath):
        convert_to_JSON('wcpr_mypersonality2.csv')
    [lines, NeuLines, NonNeuLines, NeurosisCat] = read_and_clean_lines(jsonpath)
    print("Initializing NLTK")
    filteredAndNormalized = filter_and_normalize(lines, args.remove_stopwords_bigrams, stop_words, args.remove_punctuation,2)
    counter = collect_bigram_counts(filteredAndNormalized)

    if args.print_top_ngrams == True:
        print('Top N-grams of Population')
        print_sorted_items(counter, args.topN_to_show)
        filteredAndNormalizedNeuLines = filter_and_normalize(NeuLines, args.remove_stopwords_bigrams, stop_words, args.remove_punctuation,2)
        Neucounter = collect_bigram_counts(filteredAndNormalizedNeuLines)
        filteredAndNormalizedNonNeuLines = filter_and_normalize(NonNeuLines, args.remove_stopwords_bigrams, stop_words, args.remove_punctuation,2)
        NonNeucounter = collect_bigram_counts(filteredAndNormalizedNonNeuLines)
        print('Top N-grams of Neurotic Population')
        print_sorted_items(Neucounter,17)
        print('Top N-grams of NonNeurotic Population')
        print_sorted_items(NonNeucounter,args.topN_to_show)

    X                              = lines
    y                              = NeurosisCat
    X_train, X_test, y_train, y_test  = split_training_set(X, y, args.test_size)

    # Feature extraction is the same as done previously
    if args.use_sklearn_features:
        X_features_train, training_vectorizer = convert_text_into_features(X_train, stop_words, "word", range=(1,2))
        X_test_documents = X_test
    else:
        print("Creating feature strings for training data")
        X_train_feature_strings = convert_lines_to_feature_strings(X_train, stop_words, args.remove_punctuation)

        print("Creating feature strings for final test data")
        X_test_documents        = convert_lines_to_feature_strings(X_test,  stop_words, args.remove_punctuation)
        
        X_features_train, training_vectorizer = convert_text_into_features(X_train_feature_strings, stop_words, whitespace_tokenizer)
        X_features_test, training_vectorizer = convert_text_into_features(X_test_documents, stop_words, whitespace_tokenizer)

    # Create a k-fold cross validation object.
    print("Doing cross-validation splitting with stratify={}. Showing 10 indexes for items in train/test splits in {} folds.".format(args.stratify,args.num_folds))
    if args.stratify == True:
        kfold = sklearn.model_selection.StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    else:
        kfold =  sklearn.model_selection.KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed) # Replace this with appropriate function call to create cross-validation object
    for i, (train_index, test_index) in enumerate(kfold.split(X,y)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

    # Create the classifier object
    #classifier = LogisticRegression(solver='liblinear')
    classifier = sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)

    # Do cross-validation and look at mean/stdev of scores by calling cross_val_score()
    print("Running {}-fold cross-validation on {}% of the data, still holding out the rest for final testing.".format(args.num_folds,(1-args.test_size)*100))
    accuracy_scores = cross_val_score(classifier, X_features_train, y_train, scoring='accuracy', cv=kfold, n_jobs=-1) # Replace this line with your call to cross_val_score()
    print("accuracy scores = {}, mean = {}, stdev = {}".format(accuracy_scores, np.mean(accuracy_scores), np.std(accuracy_scores)))
    



