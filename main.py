import numpy as np
import game
import network

LEARNING_RATE = 0.01
NOISE_SCALE = 0.05

bot = network.Cemantix_Bot()
game_instance = game.Game("./frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin")

current_word_str = game_instance.random_word()
best_score = -1.0
print(f"Goal: {game_instance.reveal()}")
print(f"Start: {current_word_str}")

for i in range(100):
   current_vector = game_instance.model[current_word_str]
   input_batch = np.expand_dims(current_vector, axis=0)
   predicted_vector = bot.forward(input_batch)

   noise = np.random.randn(1, 500) * NOISE_SCALE
   candidate_vector = predicted_vector + noise

   candidate_word = game_instance.word_from_vector(candidate_vector[0])

   if game_instance.has_been_guess(candidate_word):
      candidate_word = game_instance.random_word()

   score = game_instance.try_guess(candidate_word)

   print(f"Turn {i+1}: Guess: '{candidate_word}', Score: {score:.4f}")

   if candidate_word == game_instance.reveal():
      print("Congratulations! You've guessed the word!")
      break

   if score > best_score:
      print(f"Learning from {candidate_word}")
      best_score = score

      target = candidate_vector
      dvalues = 2 * (predicted_vector - target)

      bot.backward(dvalues, LEARNING_RATE)

      current_word_str = candidate_word
   else:
      pass
