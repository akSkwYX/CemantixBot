import random
from gensim.models import KeyedVectors as kv

class Game:
   def __init__(self, model_path):
      self.model = kv.load_word2vec_format(model_path, binary=True, unicode_errors="ignore")
      self.word = random.choice(self.model.index_to_key)
      self.guessed = []

   def try_guess(self, guess):
      if guess in self.model:
         similarity = self.model.similarity(self.word, guess)
         self.guessed.append(guess)
         return similarity
      else:
         print(f"'{guess}' is not in the model's vocabulary.")
         return None

   def word_from_vector(self, vector):
      return self.model.similar_by_vector(vector, topn=1)[0][0]

   def reveal(self):
      return self.word

   def random_word(self):
      return random.choice(self.model.index_to_key)

   def has_been_guess(self, guess):
      return guess in self.guessed
