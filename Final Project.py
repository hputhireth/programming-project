import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
glove_dir = os.path.join(script_dir, "glove.6B")
glove_path = os.path.join(glove_dir, "glove.6B.100d.txt")  # was 100d

embeddings_index = {}

with open(glove_path, "r", encoding="utf-8") as f:
    for line in f:
        values = line.split(" ")
        word = values[0]
        embeddings_index[word] = np.asarray(values[1:], dtype="float64")

print(f"Found {len(embeddings_index)} word vectors.")


def cosine_sim(A, B):
    return np.sum(A * B) / np.sqrt(np.sum(A ** 2) * np.sum(B ** 2))


king_v      = embeddings_index["king"]
queen_v     = embeddings_index["queen"]
potato_v    = embeddings_index["potato"]
woman_v     = embeddings_index["woman"]
man_v       = embeddings_index["man"]
male_v      = embeddings_index["male"]
female_v    = embeddings_index["female"]
programmer  = embeddings_index["programmer"]
doctor_v    = embeddings_index["doctor"]
physician_v = embeddings_index["physician"]

print(cosine_sim(king_v, queen_v))
print(cosine_sim(king_v, potato_v))
print(cosine_sim(doctor_v, physician_v))


embedding_calculation = woman_v - man_v + king_v
todrop = ["woman", "king", "man"]
print(get_closest(embedding_calculation, todrop))

embedding_calculation = doctor_v - male_v + female_v
todrop = ["female", "doctor", "male"]
print(get_closest(embedding_calculation, todrop))

embedding_calculation = female_v - male_v + programmer
todrop = ["woman", "programmer", "programmers", "man"]
print(get_closest(embedding_calculation, todrop))