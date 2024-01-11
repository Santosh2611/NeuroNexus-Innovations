# Import necessary libraries
import unittest
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import speech_recognition as sr

# Define a class for analyzing user input and generating responses
class ResponseAnalyzer:

    def __init__(self):
        self._download_nltk_data()  # Download NLTK data upon initialization

    # Method to download required NLTK data
    def _download_nltk_data(self):
        nltk.download('punkt', quiet=True)
        nltk.download('vader_lexicon', quiet=True)

    # Define patterns and corresponding responses for identifying spam content
    SPAM_FILTER = {
        r".*offer.*": "Spam detected: This looks like a promotional email. It has been blocked.",
        r".*win.*": "Spam detected: This looks like a scam. It has been blocked.",
        r".*urgent.*": "Spam detected: This message appears to be urgent. It has been blocked."
    }

    # Define patterns and corresponding responses for generating specific responses
    RESPONSES = {
        r".*hello.*": "Hello there! How can I help you today?",
        r".*how are you.*": "I'm just a bot, but thanks for asking! How can I assist you?",
        r".*bye.*": "Goodbye! Have a great day!"
    }

    # Method to preprocess the input text
    def preprocess_input(self, text):
        """Preprocesses the input text."""
        return ' '.join(word_tokenize(text)).strip().lower()

    # Method to check if the given text contains spam content
    def is_spam(self, text):
        """Checks if the text contains spam content."""
        for pattern, response in self.SPAM_FILTER.items():
            if re.search(re.compile(pattern), text):
                return True, response
        return False, None

    # Method to generate a response based on the user input
    def get_response(self, user_input):
        """Generates a response based on the user input."""
        sanitized_input = self.preprocess_input(user_input)
        is_spam, spam_response = self.is_spam(sanitized_input)
        
        if is_spam:
            return spam_response

        for pattern, response in self.RESPONSES.items():
            if re.match(re.compile(pattern), sanitized_input):
                return response

        return "I'm sorry, I didn't understand that. Can you please rephrase?"

    # Method to analyze the sentiment of the given text
    def analyze_sentiment(self, text):
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(text)
        compound_score = sentiment_score['compound']

        if compound_score > 0.05:
            return "Positive"
        elif compound_score < -0.05:
            return "Negative"
        else:
            return "Neutral"

    # Method to convert speech to text using the microphone
    def convert_speech_to_text(self, timeout=5):
        """Converts speech to text using the microphone."""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Please start speaking...")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=timeout)
                print("Recognizing speech...")
                return recognizer.recognize_google(audio)
            except sr.WaitTimeoutError:
                print("No speech detected within the specified time.")
                return ""
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service: {e}")
                return ""
            except sr.UnknownValueError:
                print("Unable to recognize speech")
                return ""

    # Method to get a response based on voice input
    def get_voice_input_response(self):
        user_input = self.convert_speech_to_text()
        return self.get_response(user_input) if user_input else "I'm sorry, I didn't catch that. Can you please repeat?"

# Class for unit testing the ResponseAnalyzer
class TestResponseAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = ResponseAnalyzer()

    # Unit test for hello response
    def test_hello_response(self):
        self.assertEqual(self.analyzer.get_response("Hello"), "Hello there! How can I help you today?")

    # Unit test for how are you response
    def test_how_are_you_response(self):
        self.assertEqual(self.analyzer.get_response("How are you doing?"), "I'm just a bot, but thanks for asking! How can I assist you?")

    # Unit test for goodbye response
    def test_bye_response(self):
        self.assertEqual(self.analyzer.get_response("Goodbye"), "Goodbye! Have a great day!")

    # Unit test for unknown input response
    def test_unknown_input_response(self):
        self.assertEqual(self.analyzer.get_response("Random input"), "I'm sorry, I didn't understand that. Can you please rephrase?")

    # Unit test for positive sentiment analysis
    def test_positive_sentiment(self):
        text = "I love this product. It's amazing!"
        self.assertEqual(self.analyzer.analyze_sentiment(text), "Positive")

    # Unit test for negative sentiment analysis
    def test_negative_sentiment(self):
        text = "This movie is terrible. I hated it."
        self.assertEqual(self.analyzer.analyze_sentiment(text), "Negative")

    # Unit test for neutral sentiment analysis
    def test_neutral_sentiment(self):
        text = "Maybe, you were right!"
        self.assertEqual(self.analyzer.analyze_sentiment(text), "Neutral")

if __name__ == '__main__':
    analyzer = ResponseAnalyzer()
    unittest.main()
    print(analyzer.get_voice_input_response())
