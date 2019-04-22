from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import io
import math
import os
import random
from six.moves import urllib

from IPython.display import clear_output, Image, display, HTML

import tensorflow as tf 
import tensorflow_hub as hub 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sk_metrics
import tarfile
import time
import pdb

IMAGE_DIR = './animal_database'
TRAIN_FRACTION = 0.8
RANDOM_SEED = 2019

def download_images():
	if not os.path.exists(IMAGE_DIR):
		DOWNLOAD_URL = 'https://www.csc.kth.se/~heydarma/datasets/animal_database.tar.gz'
		print('Downloading images from %s...' %DOWNLOAD_URL)
		urllib.request.urlretrieve(DOWNLOAD_URL, 'animal_database.tgz')
		tf = tarfile.open('animal_database.tgz')
		tf.extractall()
	print('Images are located in %s' %IMAGE_DIR)


def make_train_and_test_sets():
	train_ex, test_ex = [], []
	shuffler = random.Random(RANDOM_SEED)
	is_root = True
	for (dirname, subdirs, filenames) in tf.gfile.Walk(IMAGE_DIR):
		if is_root:
			subdirs = sorted(subdirs)
			classes = collections.OrderedDict(enumerate(subdirs))
			label_to_class = dict([(x, i) for i, x in enumerate(subdirs)])
			is_root = False
		else:
			filenames.sort()
			shuffler.shuffle(filenames)
			full_filenames = [os.path.join(dirname, f) for f in filenames]
			label = dirname.split('\\')[-1]
			label_class = label_to_class[label]
			ex = list(zip(full_filenames, [label_class] * len(filenames)))
			num_train = int(len(filenames) * TRAIN_FRACTION)
			train_ex.extend(ex[:num_train])
			test_ex.extend(ex[num_train:])
	shuffler.shuffle(train_ex)
	shuffler.shuffle(test_ex)
	return train_ex, test_ex, classes

def strip_consts(graph_def, max_const_size = 32):
	strip_def = tf.GraphDef()
	for node in graph_def.node:
		stripped_node = strip_def.node.add()
		stripped_node.MergeFrom(node)
		if stripped_node.op == 'Const':
			tensor = stripped_node.attr['value'].tensor
			size = len(tensor.tensor_content)
			if size > max_const_size:
				tensor.tensor_content = '<stripped %d bytes>'%size
	return strip_def

def show_graph(graph_def, max_const_size = 32):
	if hasattr(graph_def, 'as_graph_def'):
		graph_def = graph_def.as_graph_def()
	strip_def = strip_consts(graph_def, max_const_size = max_const_size)
	code = """
		<script>
			function load() {{
				document.getElementById("{id}").pbtxt = {data};
			}}
		</script>
		<link rel = "import" href = "https://tensorboard.appspot.com/tf-graph-basic.build.html" onload = load()>
		<div style = "height:600px">
			<tf-graph-basic id = "{id}"></tf-graph-basic>
		</div>
	""".format(data = repr(str(strip_def)), id = 'graph' + str(np.random.rand()))
	iframe = """
		<iframe seamless style = "width:1200px;height:620px;border:0" srcdoc = "{}"></iframe>
	""".format(code.replace('"', '&quot;'))
	display(HTML(iframe))

download_images()
TRAIN_EXMAPLES, TEST_EXAMPLES, CLASSES = make_train_and_test_sets()
NUM_CLASSES = len(CLASSES)

print('\nThe dataset has %d label classes: %s' %(NUM_CLASSES, CLASSES.values()))
print('There are %d training images' %len(TRAIN_EXMAPLES))
print('There are %d test images' %len(TEST_EXAMPLES))

def get_label(example):
	return example[1]

def get_class(example):
	return CLASSES[get_label(example)]

def get_encoded_image(example):
	image_path = example[0]
	return tf.gfile.GFile(image_path, 'rb').read()

def get_image(example):
	return plt.imread(io.BytesIO(get_encoded_image(example)), format = 'jpg')

def display_images(images_and_classes, cols = 5):
	rows = int(math.ceil(len(images_and_classes) / cols))
	fig = plt.figure()
	fig.set_size_inches(cols * 3, rows * 3)
	for i, (image, image_class) in enumerate(images_and_classes):
		plt.subplot(rows, cols, i + 1)
		plt.axis('off')
		plt.imshow(image)
		plt.title(image_class)

NUM_IMAGES = 15 #@param {type: 'integer'}
# display_images([(get_image(example), get_class(example)) for example in TRAIN_EXMAPLES[:NUM_IMAGES]])

LEARNING_RATE = 0.01

tf.reset_default_graph()

image_module = hub.Module('https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/2')

encoded_images = tf.placeholder(tf.string, shape = [None])
image_size = hub.get_expected_image_size(image_module)

def decode_and_resize_image(encoded):
	decoded = tf.image.decode_jpeg(encoded, channels = 3)
	decoded = tf.image.convert_image_dtype(decoded, tf.float32)
	return tf.image.resize_images(decoded, image_size)

batch_images = tf.map_fn(decode_and_resize_image, encoded_images, dtype = tf.float32)

features = image_module(batch_images)

def create_model(features):
	layer = tf.layers.dense(inputs = features, units = NUM_CLASSES, activation = None)
	return layer

logits = create_model(features)
labels = tf.placeholder(tf.float32, [None, NUM_CLASSES])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels)
cross_entropy_mean = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)
train_op = optimizer.minimize(loss = cross_entropy_mean)

probabilities = tf.nn.softmax(logits)

prediction = tf.argmax(probabilities, 1)
correct_prediction = tf.equal(prediction, tf.argmax(labels, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

show_graph(tf.get_default_graph())

NUM_TRAIN_STEPS = 100
TRAIN_BATCH_SIZE = 10
EVAL_EVERY = 10

def get_batch(batch_size = None, test = False):
	examples = TEST_EXAMPLES if test else TRAIN_EXMAPLES
	batch_examples = random.sample(examples, batch_size) if batch_size else examples
	return batch_examples

def get_images_and_labels(batch_examples):
	images = [get_encoded_image(x) for x in batch_examples]
	one_hot_labels = [get_label_one_hot(x) for x in batch_examples]
	return images, one_hot_labels

def get_label_one_hot(example):
	one_hot_vector = np.zeros(NUM_CLASSES)
	np.put(one_hot_vector, get_label(example), 1)
	return one_hot_vector

with tf.Session() as s:
	s.run(tf.global_variables_initializer())
	for i in range(NUM_TRAIN_STEPS):
		train_batch = get_batch(batch_size = TRAIN_BATCH_SIZE)
		batch_images, batch_labels = get_images_and_labels(train_batch)
		feature_vector = s.run(features, feed_dict = {encoded_images: batch_images, labels: batch_labels})
		train_loss, _, train_accuracy = s.run([cross_entropy_mean, train_op, accuracy], feed_dict = {encoded_images: batch_images, labels: batch_labels})
		is_final_step = (i == (NUM_TRAIN_STEPS - 1))
		if i % EVAL_EVERY == 0 or is_final_step:
			test_batch = get_batch(batch_size = None, test = True)
			batch_images, batch_labels = get_images_and_labels(test_batch)
			test_loss, test_accuracy, test_prediction, correct_predicate = s.run([cross_entropy_mean, accuracy, prediction, correct_prediction], feed_dict = {encoded_images: batch_images, labels: batch_labels})
			print('Test accuracy at step %s: %.2f%%' %(i, (test_accuracy * 100)))

def show_confusion_matrix(test_labels, predictions):
	confusion = sk_metrics.confusion_matrix(np.argmax(test_labels, axis = 1), predictions)
	confusion_normalized = confusion.astype("float") / confusion.sum(axis = 1)
	axis_labels = list(CLASSES.values())
	ax = sns.heatmap(confusion_normalized, xticklabels = axis_labels, yticklabels = axis_labels, cmap = 'Blues', annot = True, fmt = '.2f', square = True)
	plt.title("Confusion Matrix")
	plt.ylabel("True labels")
	plt.xlabel("Predicted labels")
	plt.show()

show_confusion_matrix(batch_labels, test_prediction)
