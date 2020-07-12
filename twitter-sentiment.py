import nltk
from nltk import NaiveBayesClassifier
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re

# reading files data for positive and negative classes from nltk twitter_samples package
# the files are stored in json format

pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')


# performing tokenizing and cleaning steps :
# removing stop words - stemming words with porterstemmer - removing https and hashtags
# we use nltk package TweetTokenizer to tokenize ( separate ) words

def tokenize_tweet(tweet):
    stopwords_english = stopwords.words('english')
    stemmer = PorterStemmer()
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)  # https links

    tweet = re.sub(r'([@][\w_-]+)', '', tweet)  # @ for user names

    tweet = re.sub(r'^RT[\s]+', '', tweet)  # for retweets

    tweet = re.sub(r'#', '', tweet)  # hashtags

    tweet_tokens = tokenizer.tokenize(tweet)
    tweet_words = []

    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation:

            stem_word = stemmer.stem(word)
            tweet_words.append(stem_word)

    return tweet_words


# converting words of input sentence ( set of words ) to a dictionary of [word , true] to set
# learning features

def bag_of_words(tweet):
    words = tokenize_tweet(tweet)
    words_dictionary = dict([(word, True) for word in words])
    return words_dictionary


# classify a sample input sentence to positive / negative class and give the probability of
# belonging to each class, based on trained model

def classify_text(text):
    text = bag_of_words(text)
    result = classifier.classify(text)
    result_prob = classifier.prob_classify(text)
    return result, result_prob.prob('pos'), result_prob.prob('neg')


# give positive word dictionary 'pos' tag

pos_tweets_set = []
for tweet in pos_tweets:
    pos_tweets_set.append((bag_of_words(tweet), 'pos'))


# give negative word dictionary 'neg' tag

neg_tweets_set = []
for tweet in neg_tweets:
    neg_tweets_set.append((bag_of_words(tweet), 'neg'))


# split input data set to train and test data set ( leave 2000 for test set and the rest for train set )

test_set = pos_tweets_set[:1000] + neg_tweets_set[:1000]
train_set = pos_tweets_set[1000:] + neg_tweets_set[1000:]


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
    input_text = input("Enter a tweet to classify: ")
    result, result_prob_pos, result_prob_neg = classify_text(input_text)
    print("Your tweet belong to class:", result)
    print("probability of belong to class positive is :", result_prob_pos)
    print("probability of belong to class negative is :", result_prob_neg)

