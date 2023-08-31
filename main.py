import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("wordnet")
nltk.download("vader_lexicon")


def bag_of_words(sentences: list) -> dict:
    bag = {}
    for sentence in sentences:
        for word in sentence:
            if bag.get(word) != None:
                bag[word] = bag[word] + 1
            else:
                bag[word] = 1
    print(bag)


# initialize NLTK sentiment analyzer

analyzer = SentimentIntensityAnalyzer()


# create get_sentiment function


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
        result.append(remove_special_characters(word))


if __name__ == "__main__":
    data = input("Enter a data: ").lower()
    print()
    print("Segmentation")
    print()
    segmented = segment(data)
    print(segmented)
    print()
    print("Removal of stopwords, special characters")
    print()
    stripped = []
    for sentence in segmented:
        print(sentence)
        reduced = remove_special_characters(sentence)
        stripped.append(reduced)
    print("\n".join(stripped))
    print()
    print("Tokenization")
    print()
    tokenized = []
    for n, sentence in enumerate(stripped):
        if not sentence:
            continue
        print("Sentence " + str(n) + ": ", end="")
        tokenised = word_tokenize(sentence)
        tokenized.append(tokenised)
        print(tokenised)
    print()
    print("Lemmatizer")
    print()
    wnl = WordNetLemmatizer()
    lemmatized = []
    for sentence in tokenized:
        lemmatizedSentence = []
        for word in sentence:
            lemmatiz = wnl.lemmatize(word)
            lemmatizedSentence.append(lemmatiz)
        lemmatized.append(lemmatizedSentence)
    print(lemmatized)
    print()
    print("Bag of words")
    print(bag_of_words(lemmatized))

    print()
    print("Sentiment Analysis")
    for n, i in enumerate(lemmatized):
        print("Sentence " + str(n) + ": " + str(get_sentiment(" ".join(i))))
