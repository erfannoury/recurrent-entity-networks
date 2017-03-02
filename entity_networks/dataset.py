from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import tensorflow as tf

class Dataset(object):

    def __init__(self, dataset_path, batch_size):
        self.dataset_dir = os.path.dirname(dataset_path)
        self.batch_size = batch_size
        self.examples_per_epoch = 10000

        with open(dataset_path) as f:
            metadata = json.load(f)

        self.max_sentence_length = metadata['max_sentence_length']
        self.max_story_length = metadata['max_story_length']
        self.max_query_length = metadata['max_query_length']
        self.dataset_size = metadata['dataset_size']
        self.vocab_size = metadata['vocab_size']
        self.tokens = metadata['tokens']
        self.datasets = metadata['datasets']

    @property
    def steps_per_epoch(self):
        # It should return the number of mini-batches in a single epoch. And it
        # depends on whether the last batch will be smaller than the rest or not.
        # In the implementation of the `read_batch_record_features` function, if
        # the `num_epochs` argument is `None` (and therefore it will serve
        # mini-batches indefinitely), a smaller final mini-batch is not allowed.
        # However, when the number of epochs is limited, the final mini-batch can
        # be smaller than other mini-batches. Given that in this repo training
        # doesn't go for an indefinite number of epochs, therefore it is assumed
        # that the last mini-batch can be smaller than the previous ones.
        if self.examples_per_epoch % self.batch_size:
            return self.examples_per_epoch // self.batch_size
        else:
            return self.examples_per_epoch // self.batch_size + 1

    def get_input_fn(self, name, num_epochs, shuffle):
        def input_fn():
            features = {
                "story": tf.FixedLenFeature([self.max_story_length, self.max_sentence_length], dtype=tf.int64),
                "query": tf.FixedLenFeature([1, self.max_query_length], dtype=tf.int64),
                "answer": tf.FixedLenFeature([], dtype=tf.int64),
            }

            dataset_path = os.path.join(self.dataset_dir, self.datasets[name])
            features = tf.contrib.learn.read_batch_record_features(dataset_path,
                features=features,
                batch_size=self.batch_size,
                randomize_input=shuffle,
                num_epochs=num_epochs)

            story = features['story']
            query = features['query']
            answer = features['answer']

            return {'story': story, 'query': query}, answer
        return input_fn
