import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# download data for nltk
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("vader_lexicon")
nltk.download('punkt')

def bag_of_words(sentence: list) -> dict:
    bag = {}
    for word in sentence:
        if bag.get(word) != None:
            bag[word] = bag[word] + 1
        else:
            bag[word] = 1
    return bag


analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    return analyzer.polarity_scores(text)


def segment(inp: str) -> list:
    return inp.split(".")


def tokenize(inp: str) -> list:
    return inp.split(" ")


def remove_special_characters(string: str) -> str:
    special_characters = "~`!@#$%^&*()_+{}|:\"<>?,./;'[]\\"
    for character in special_characters:
        string = string.replace(character, "")
    return string


def removal(words: list) -> list:
    result = []
    for word in words:
        result.append(stopwords.words(word))
        #result.append(remove_special_characters(word))


if __name__ == "__main__":
    data = input("Enter a data: ").lower()
    print()
    print("Segmentation")
    print()
    segmented = segment(data)
    print(segmented)
    print()
    print("Tokenization")
    print()
    tokenized = []
    for n, sentence in enumerate(segmented):
        if not sentence:
            continue
        print("Sentence " + str(n) + ": ", end="")
        tokenised = word_tokenize(sentence)
        tokenized.append(tokenised)
    print(tokenized)
    print()
    print("Lemmatizer")
    print()
    wnl = WordNetLemmatizer()
    lemmatized = []
    for sentence in tokenized:
        lemma = []
        for word in sentence:
            lemmatiz = wnl.lemmatize(word)
            lemma.append(lemmatiz)
        print(lemma)
        lemmatized.append(lemma)
    print(lemmatized)
    print()
    print("Removal of stopwords, special characters")
    print()
    stripped = []
    stopwords = set(stopwords.words("english"))
    for sentence in lemmatized:
        reduced = [w for w in sentence if not w.lower() in stopwords]
        stripped.append(reduced)
    print(stripped)
    print()
    print("Bag of words")
    bags = []
    for n, i in enumerate(stripped):
        print("Sentence " + str(n) + ": ", end="")
        bag = bag_of_words(i)
        print(bag)
        bags.append(bag)

    size = len(bags)
    unique = set()
    for n, i in enumerate(bags):
        for word in i:
            unique.add(word)

    print()
    print("Unique words")
    print()
    print(', '.join(unique))
    print()
    print("Term Frequency")
    tf = {}
    for n, i in enumerate(bags):
        for word in i:
            if tf.get(word) != None:
                tf[word] = tf[word] + 1
            else:
                tf[word] = 1
    
    for i in tf:
        print(i + ": " + str(tf[i]))
    print()
    print("TF-IDF")
    print()
    for word in tf:
        print(word + ": " + str(tf[word]) + "/" + str(size))
    print()
    print("Sentiment Analysis")
    for n, i in enumerate(stripped):
        print("Sentence " + str(n) + ": " + str(get_sentiment(" ".join(i))))
