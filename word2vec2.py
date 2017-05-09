from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
import tensorflow as tf
import shelve
# Step 1: Load the data.

filename='yelp.txt'
# Read the data into a list of strings.

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    max_length=0
    with open(filename,'r') as f:
        data=[]
        label=[]
        count=0
        for line in f:
            line_split=line.split('\t')
            tokens=line_split[1].split()
            data.extend(tokens)
            data.append('')
            max_length=max(max_length,len(tokens))
            if count>10000: break
            count+=1
    return data, max_length

words, max_length = read_data(filename)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 200000


def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 100
embedding_size = 300  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()


def build_pre_train(vocabulary_size, embedding_size, filename):
    pre_train_embeddings=np.array(2*(np.random.random((vocabulary_size,embedding_size))-0.5),np.float32)
    count=0
    with open(filename,'r') as f:
        for line in f:
            line_split=line.split()
            if line_split[0].lower() in dictionary:
                count+=1
                pre_train_embeddings[dictionary[line_split[0].lower()]]=np.array([float(i) for i in line_split[1:]])
                print count,'pretrain'
    return pre_train_embeddings

pre_train_embeddings=build_pre_train(vocabulary_size, embedding_size, 'GoogleNews-vectors-rcv_vocab.txt')
print 'finish pretrain'
with graph.as_default():

    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(pre_train_embeddings)
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 20000

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0
    final_embeddings = normalized_embeddings.eval()

null_embedding=np.zeros((1,embedding_size),np.float32)-1
print null_embedding.shape
print final_embeddings.shape
final_embeddings = np.concatenate((null_embedding,final_embeddings))
print final_embeddings.shape
print null_embedding
wvec=shelve.open('word2vec.db')
wvec['wv']=final_embeddings
wvec['dictionary']=dictionary
wvec['reverse_dictionary']=reverse_dictionary


from sklearn.neural_network import MLPClassifier


l2i=dict()
l2i['positive']=0
l2i['neutral']=1
l2i['negative']=2

input_x=[]
input_y=[]
with open(filename,'r') as f:
    data=[]

    label=[]
    count=0
    for line in f:
        line_split=line.split('\t')
        word=line_split[1].split()
        vec=np.array([0.0 for i in range(embedding_size)])
        vector_group=np.zeros(max_length,np.int32)
        sent_count=0
        for w in word:
            vec+=final_embeddings[dictionary[w]+1]
            vector_group[sent_count]=dictionary[w]+1
            sent_count+=1
        vec/=len(word)
        data.append(vec)
        target=np.zeros(3,np.float32)
        target[l2i[line_split[0]]]=1
        label.append(target)

        input_x.append(vector_group)
        input_y.append(target)


        if count>10000: break
        count+=1
        print count
    x=np.array(input_x)
    print x.shape
    wvec['input_x']=input_x
    wvec['input_y']=input_y
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50, 3), random_state=1)
    clf.fit(data[0:9000], label[0:9000])
    count=0
    for d in range(9001,10000):
        result=np.argmax(clf.predict(data[d]))
        ans=np.argmax(label[d])
        if result==ans:count+=1
        print result,ans
    print count,'accuracy'
    print average_loss

