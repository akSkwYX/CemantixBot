import numpy as np
import game
import network
import os

HISTORY_SIZE = 10
VECTOR_DIM = 500
INPUT_DIM = HISTORY_SIZE * (VECTOR_DIM + 1)

LEARNING_RATE = 0.001
NOISE_SCALE = 0.5

def build_state(guessed_words, game_instance):
   state = np.zeros((1, INPUT_DIM))

   sorted_guesses = sorted(guessed_words, key=lambda x: x[1], reverse=True)

   best_guesses = sorted_guesses[-HISTORY_SIZE:]

   for i, (word, score) in enumerate(best_guesses):
      vec = game_instance.get_vec(word)
      combined = np.concatenate([vec, [score]])
      start_idx = i * (VECTOR_DIM+1)
      end_idx = start_idx + (VECTOR_DIM+1)
      state[0, start_idx:end_idx] = combined

   return state

# Initialize bot
bot = network.Cemantix_Bot()
game_instance = game.Game("./frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin")
game_instance.word = "roi"

if os.path.exists("neural_network.npz"):
   bot.load_model("neural_network.npz")

avg_score = None
n_loop = 1000
alpha = 1/n_loop

history = []

current_word_str = game_instance.random_word()
current_word = game_instance.get_vec(current_word_str)
score = game_instance.try_guess(current_word_str)
avg_score = score
history.append((current_word_str, score))

target = game_instance.get_vec(game_instance.reveal())

best_score = -1
best_word = ""

print(f"Goal: {game_instance.reveal()}")
print(f"Start: {current_word_str}")

for i in range(n_loop):
   state_vector = build_state(history, game_instance)
   # Prediction
   predicted_vector = bot.forward(state_vector)

   # Explore
   noise = np.random.randn(1, 500) * NOISE_SCALE
   candidate_vector = predicted_vector + noise

   candidate_word = game_instance.word_from_vector(candidate_vector[0])

   if game_instance.has_been_guess(candidate_word):
      candidate_word = game_instance.random_word()

   score = game_instance.try_guess(candidate_word)
   history.append((candidate_word, score))


   # Backpropagation
   advantage = score - avg_score
   avg_score = (1 - alpha) * avg_score + (alpha * score)

   if advantage > 0:
      target = candidate_vector
      dvalues = 2 * (predicted_vector - target)
      learning_factor = (advantage * 10.0) ** 2
      bot.backward(dvalues, learning_factor * LEARNING_RATE)

      if advantage > 0.3:
         print("Good guess")
         for _ in range(20):
            bot.backward(dvalues, learning_factor * LEARNING_RATE)
      if advantage > 0.5:
         print("Great guess")
         for _ in range(50):
            bot.backward(dvalues, learning_factor * LEARNING_RATE)
      if advantage > 0.7:
         print("Excellent guess")
         for _ in range(100):
            bot.backward(dvalues, learning_factor * LEARNING_RATE)
      print(f"Turn: {i+1} Answer: {game_instance.word_from_vector(predicted_vector[0])} Guess: {candidate_word} Score: {score} Advantage: {advantage}")

   # Information
   if score > best_score:
      best_score = score
      best_word = candidate_word

   # Win
   if candidate_word == game_instance.reveal():
      print("Congratulations! You've guessed the word!")
      best_word = candidate_word
      best_score = score
      break

print(f"Best guessed word: {best_word} with score : {best_score}")
bot.save_model("neural_network.npz")
