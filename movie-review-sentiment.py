import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import string


# classify a sample input sentence to positive / negative class and give the probability of
# belonging to each class, based on trained model

def classify_text(text):
    tokenized_text = tokenizer(text)
    featured_text = word_feats(tokenized_text)
    result = classifier.classify(featured_text)
    result_prob = classifier.prob_classify(featured_text)
    return result, result_prob.prob('pos'), result_prob.prob('neg')


# converting words inside input sentence ( set of words ) to a dictionary of [word , true] to set
# learning features

def word_feats(words):
    return dict([(word, True) for word in words])


# tokenize ( separate ) words of input text, leave out punctuations ( . , ? ! : " ' etc )
# and change to lower case format

def tokenizer(text):
    stops = list(string.punctuation)
    tokens = []
    for word in text:
        word.lower()
        if word not in stops:
            tokens.append(word)
    return tokens


# reading file ids for positive and negative classes from nltk movie_reviews package

pos_fileids = movie_reviews.fileids('pos')
neg_fileids = movie_reviews.fileids('neg')

pos_word_set = []
neg_word_set = []

# reading words for each file id from two classes

for file in pos_fileids:
    raw_words = movie_reviews.words(fileids=[file])
    raw_words = tokenizer(raw_words)
    temp_set = word_feats(raw_words)

    # give positive word dictionary 'pos' tag
    pos_word_set.append((temp_set, 'pos'))

for file in neg_fileids:
    raw_words = movie_reviews.words(fileids=[file])
    raw_words = tokenizer(raw_words)
    temp_set = word_feats(raw_words)

    # give negative word dictionary 'neg' tag
    neg_word_set.append((temp_set, 'neg'))

# split input data set with ratio of 0.8 to train and test data set ( 80% - train / 20% - test )

splitter = int(len(pos_word_set) * 0.8)

train_set = pos_word_set[:splitter] + neg_word_set[:splitter]
test_set = pos_word_set[splitter:] + neg_word_set[splitter:]

print("number of words in train set : ", len(train_set))
print("number of words in test set : ", len(test_set))
print("*******************************")

# we choose a Naive bayes classifier from nltk package and train with input data set

classifier = NaiveBayesClassifier.train(train_set)

# evaluate classifier accuracy metrics

print('Classifier Accuracy:', nltk.classify.util.accuracy(classifier, test_set))
print("*******************************")

# print out most effective features ( words ) in classification

classifier.show_most_informative_features()


# here you can input a sentence to test

while True:
    input_text = input("Enter a sentence to classify: ")
    result, result_prob_pos, result_prob_neg = classify_text(input_text)
    print("Your sentence belong to class:", result)
    print("probability of belong to class positive is :", result_prob_pos)
    print("probability of belong to class negative is :", result_prob_neg)

