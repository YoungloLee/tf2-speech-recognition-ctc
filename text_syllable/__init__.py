# coding: utf-8
import pickle

# loading
with open('./text_syllable/syllable_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# Mappings from symbol to numeric ID and vice versa:

tokens = list(tokenizer.index_word.values())


PAD = '<p>'
BLANK = '_'

tokens = [PAD] + tokens
# blank token (for CTC)
tokens.append(BLANK)

index_token = {idx: ch for idx, ch in enumerate(tokens)}
token_index = {ch: idx for idx, ch in enumerate(tokens)}
