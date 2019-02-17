from ztlearn.utils import *
from ztlearn.utils import get_sentence_tokens
from ztlearn.optimizers import register_opt
from ztlearn.dl.models import Sequential
from ztlearn.dl.layers import Embedding, Flatten, Dense

text_list = [
    'I like rally cars.',
    'This is good.',
    'This is bad.',
    'Rainy days are the worst.',
    'Mercedes is a good brand.',
    'Nobody has the patience for bad food.',
    'I believe Maasai Mara is the best.',
    'This is definately a bad idea.',
    'The trains in the city center suck!',
    'I have a new shinny app love it!',
    'Roads here are bad.',
    'The hospital has good service.',
]

paragraph = ' '.join(text_list)
targets = one_hot(np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]))

sentences_tokens, unique_words, longest_sentence = get_sentence_tokens(paragraph)



train_data, test_data, train_label, test_label = train_test_split(sentences_tokens,
                                                                  targets,
                                                                  test_size   = 0,
                                                                  random_seed = 5)

# optimizer definition
opt = register_opt(optimizer_name = 'sgd_momentum', momentum = 0.01, learning_rate = 0.01)

model = Sequential()
model.add(Embedding(unique_words, 4, input_length = longest_sentence))
model.add(Flatten())
model.add(Dense(2, activation = 'relu'))
model.compile(loss = 'mse', optimizer = opt)

model.summary('embedded sentences rnn')

'''
NOTE:
batch size should be equal the size of embedding
vectors and divisible by the training set size
'''

model_epochs = 500
fit_stats = model.fit(train_data,
                      train_label,
                      batch_size = 4,
                      epochs     = model_epochs)

# test out with the first sentence - sentences_tokens[0]
output_array = model.predict(np.expand_dims(sentences_tokens[0], axis=0))
print(np.argmax(output_array))

