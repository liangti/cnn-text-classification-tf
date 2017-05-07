# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import re

import numpy as np
# from six.moves import urllib
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import shelve
# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'

stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
             'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
             'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
             'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
             'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
             'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
             'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
             'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
             'through', 'during', 'before', 'after', 'above', 'below', 'to',
             'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
             'again', 'further', 'then', 'once', 'here', 'there', 'when',
             'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
             'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
             'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
             'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm',
             'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn',
             'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn',
             'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'])

# def maybe_download(filename, expected_bytes):
#     """Download a file if not present, and make sure it's the right size."""
#     if not os.path.exists(filename):
#         filename, _ = urllib.request.urlretrieve(url + filename, filename)
#     statinfo = os.stat(filename)
#     if statinfo.st_size == expected_bytes:
#         print('Found and verified', filename)
#     else:
#         print(statinfo.st_size)
#         raise Exception(
#             'Failed to verify ' + filename + '. Can you get to it with a browser?')
#     return filename
#
# filename = maybe_download('text8.zip', 31344016)

filename='text.txt'
# Read the data into a list of strings.


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    max_length=0
    with open(filename,'r') as f:
        data=[]
        label=[]
        count=0
        for line in f:
            line_split=line.lower().split('\t')
            #tokens=line_split[1].split()
            tokens=re.findall(r'(?u)\b\w\w+\b', line_split[1])
            tokens=[t for t in tokens if t not in stopwords]
            data.extend(tokens)
            max_length=max(max_length,len(tokens))
            #if count>10000: break
            count+=1
    return data, max_length

words, max_length = read_data(filename)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 400000


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
                print (count,'pretrain')
    return pre_train_embeddings

pre_train_embeddings=build_pre_train(vocabulary_size, embedding_size, 'GoogleNews-vectors-rcv_vocab.txt')
print ('finish pretrain')
with graph.as_default():

    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(pre_train_embeddings)
#             tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
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
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
#     optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.initialize_all_variables()

# Step 5: Begin training.
num_steps = 0

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    for step in range(num_steps):
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

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
#         if step % 10000 == 0:
#             sim = similarity.eval()
#             for i in xrange(valid_size):
#                 valid_word = reverse_dictionary[valid_examples[i]]
#                 top_k = 8  # number of nearest neighbors
#                 nearest = (-sim[i, :]).argsort()[1:top_k + 1]
#                 log_str = "Nearest to %s:" % valid_word
#                 for k in xrange(top_k):
#                     close_word = reverse_dictionary[nearest[k]]
#                     log_str = "%s %s," % (log_str, close_word)
#             print(log_str)
    final_embeddings = normalized_embeddings.eval()

null_embedding=np.zeros((1,embedding_size),np.float32)-1
# count=0.0
# for embed in final_embeddings:
#     null_embedding[0]+=embed
#     count+=1
# null_embedding[0]/=count
print (null_embedding.shape)
print (final_embeddings.shape)
final_embeddings = np.concatenate((null_embedding,final_embeddings))
print (final_embeddings.shape)
print (null_embedding)
wvec=shelve.open('word2vec_300v2.db')
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
        line_split=line.lower().split('\t')
        #word=line_split[1].split()
        word=re.findall(r'(?u)\b\w\w+\b', line_split[1])
        vec=np.array([0.0 for i in range(embedding_size)])
        vector_group=np.zeros(max_length,np.int32)
        sent_count=0
        for w in word:
            if w in stopwords or w not in dictionary: continue
            vec+=final_embeddings[dictionary[w]+1]
            vector_group[sent_count]=dictionary[w]+1
            sent_count+=1
#             print sent_count
        vec/=len(word)
        data.append(vec)
#         label.append(l2i[line_split[0]])
        target=np.zeros(3,np.float32)
        target[l2i[line_split[0]]]=1
        label.append(target)

        input_x.append(vector_group)
        input_y.append(target)


        #if count>10000: break
        count+=1
        print (count)
    x=np.array(input_x)
    print (x.shape)
    wvec['input_x']=input_x
    wvec['input_y']=input_y
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50, 3), random_state=1)
    clf.fit(data[0:9000], label[0:9000])
    count=0
    for d in range(9001,10000):
        result=np.argmax(clf.predict(data[d]))
        ans=np.argmax(label[d])
        if result==ans:count+=1
        print (result,ans)
    print (count,'accuracy')
    print (average_loss)



# Step 6: Visualize the embeddings.



# def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
#     assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
#     plt.figure(figsize=(18, 18))  # in inches
#     for i, label in enumerate(labels):
#         x, y = low_dim_embs[i, :]
#         plt.scatter(x, y)
#         plt.annotate(label,
#                      xy=(x, y),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
#
#     plt.savefig(filename)
#
# try:
#
#     from sklearn.manifold import TSNE
#     import matplotlib.pyplot as plt
#
#     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#     plot_only = 500
#     low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#     labels = [reverse_dictionary[i] for i in xrange(plot_only)]
#     plot_with_labels(low_dim_embs, labels)
#
# except ImportError:
#     print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
