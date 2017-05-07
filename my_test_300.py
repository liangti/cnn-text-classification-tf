'''
Created on May 2, 2017

@author: uuisafresh
'''


import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from my_cnn import TextCNN
import shelve
#from tensorflow.contrib import learn



# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 1024, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
# x_text, y = data_helper.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
x_text=[]
y=[]
l2i=dict()
l2i['positive']=0
l2i['neutral']=1
l2i['negative']=2
wvec=shelve.open('word2vec_300.db',writeback=False)
# with open('yelp.txt', "r") as f:
#     count=0
#     for line in f:
#         line_split=line.split('\t')
#         if len(line_split)==0: continue
#         target=np.array([0 for i in range(3)])
#         target[l2i[line_split[0]]]=1
#         y += [target]
#         wid=[]
#         for word in line_split[1].split():
#             wid.append(wvec['dictionary'][word])
#         x_text += wid
# #                 print [[w.lower().decode("utf-8").encode('latin1') for w in splitted]]
#         if count%100==0: print count
#         if count>3000: break
#         count+=1
#         print count
x_text=wvec['input_x']
y=wvec['input_y']
# Build vocabulary
max_document_length = max([len(x) for x in x_text])
print (max_document_length,'max_document')
#vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

#print (len(vocab_processor.vocabulary_),'vocabulary')
x = np.array(x_text)
y = np.array(y)
print (x.shape)


# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
# print shuffle_indices
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
#print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
print("Number of Class: {:d}".format(y_train.shape[1]))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=10,#len(vocab_processor.vocabulary_),
            embedding_size=300,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            word2vec=wvec['wv'],
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                #grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                #sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        #grad_summaries_merged = tf.summary.merge(grad_summaries)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        #loss_summary = tf.summary.scalar("loss", cnn.loss)
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        #acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        #train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        #train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        #train_summary_writer = tf.train.SummaryWriter(train_summary_dir,
        #                                              sess.graph.as_graph_def())

        # Dev summaries
        #dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        #dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        #dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir,
        #                                            sess.graph.as_graph_def())

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        #checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        #checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        #if not os.path.exists(checkpoint_dir):
        #    os.makedirs(checkpoint_dir)
        #saver = tf.train.Saver(tf.all_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        #vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        #sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1#FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #if writer:
            #    writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helper.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=None)
                print("")
            #if current_step % FLAGS.checkpoint_every == 0:
            #    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            #    print("Saved model checkpoint to {}\n".format(path))
