from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

class TextTokenizerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, max_words: int, max_length: int):
        self.max_words = max_words
        self.max_length = max_length

    def fit(self, X, y=None):
        self.tokenizer_ = Tokenizer(
            num_words=self.max_words,
            oov_token="<OOV>"
        )
        self.tokenizer_.fit_on_texts(X)
        return self

    def transform(self, X):
        sequences = self.tokenizer_.texts_to_sequences(X)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding="post",
            truncating="post"
        )

        return padded_sequences