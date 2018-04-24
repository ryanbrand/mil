"""
Supervised Reptile learning and evaluation on arbitrary
datasets.
"""

import random

import tensorflow as tf
import numpy as np

from variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                        VariableState)

class Reptile:
    """
    A meta-learning session.

    Reptile can operate in two evaluation modes: normal
    and transductive. In transductive mode, information is
    allowed to leak between test samples via BatchNorm.
    Typically, MAML is used in a transductive manner.
    """
    def __init__(self, session, graph, variables=None, transductive=False, pre_step_op=None):
        self.session = session
        # variable state creates placeholders for each variables and assign ops
        # lets you restore and export variables
        self._model_state = VariableState(self.session, graph, variables or tf.trainable_variables())
        global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self._full_state = VariableState(self.session, graph, global_vars)
        self._transductive = transductive
        self._pre_step_op = pre_step_op

    # pylint: disable=R0913,R0914
    # TODO: main need to modify for few shot learning to account for imitation version of num_classes / num_shots
    # num_classes (num tasks to sample per outter loop), num_shots (num demonstrations per task)
    def train_step(self,
                   dataset,
                   state_ph,
                   obs_ph,
                   label_ph,
                   minimize_op,
                   loss_op,
                   writer,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   step,
                   meta_step_size,
                   meta_batch_size):
        """
        Perform a Reptile training step.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          meta_step_size: interpolation coefficient.
          meta_batch_size: how many inner-loops to run.
        """
        old_vars = self._model_state.export_variables()
        new_vars = []
        losses   = []
        print('taking train step')
        for _ in range(meta_batch_size):
            # sample n classes and k+1 examples from each class
            # k*n for training and 1*n for testing
            mini_dataset = _sample_mini_dataset_mil(dataset, num_classes, num_shots)
            # for each task in the mini_dataset
            for batch in _mini_batches(mini_dataset, inner_batch_size, inner_iters):
                gifs    = np.concatenate([t[0] for t in batch], axis=0)
                states  = np.concatenate([t[1] for t in batch], axis=0)
                actions = np.concatenate([t[2] for t in batch], axis=0)
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                # take a gradient step
                feed_dict = {state_ph: states, obs_ph : gifs, label_ph: actions}
                print('taking gradient step')
                _, loss = self.session.run([minimize_op, loss_op], feed_dict=feed_dict)
                print('took graident step on loss:', loss)
                losses.append(loss)
            # store the new variables
            new_vars.append(self._model_state.export_variables())
            self._model_state.import_variables(old_vars)
        # update old variables 
        new_vars = average_vars(new_vars)
        self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))
        # add loss summary
        summary = tf.Summary()
        print('ave loss:', np.mean(losses))
        summary.value.add(tag='ave_loss', simple_value=np.mean(losses))
        writer.add_summary(summary, step)

    def evaluate(self,
                 dataset, # (train_example, test_example)
                 state_ph,
                 obs_ph,
                 label_ph,
                 minimize_op,
                 predictions, # not using predictions for now
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters):
        """
        Run a single evaluation of the model.

        Samples a few-shot learning task and measures
        performance.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          predictions: a Tensor of integer label predictions.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.

        Returns:
          The number of correctly predicted samples.
            This always ranges from 0 to num_classes.
        """
        # get train and test split, assuming that test is always one example from each class
        # we know that we only use two examples so ignore this
        #train_set, test_set = _split_train_test(
        #    _sample_mini_dataset(dataset, num_classes, num_shots+1))

        train_example, test_example = dataset
        statea, obsa, actiona = train_example
        train_feed_dict = {
            state_ph : statea,
            obs_ph   : obsa,
            label_ph : actiona
        }
        #print('statea:', statea.shape, 'obsa:', obsa.shape, 'actiona:', actiona.shape)
        stateb, obsb = test_example
        test_feed_dict = {
            state_ph : stateb,
            obs_ph   : obsb
        }
        #print('stateb:', stateb.shape, 'obsb:', obsb.shape)
        # save model variables for update
        old_vars = self._full_state.export_variables()
        # removed for reptile
        #for batch in _mini_batches(train_set, inner_batch_size, inner_iters):
        #    inputs, labels = zip(*batch)

        for i in range(inner_iters):
            if self._pre_step_op:
                self.session.run(self._pre_step_op)
            self.session.run(minimize_op, feed_dict=train_feed_dict)

        # compute predicted values for the newly trained model
        # TODO: should the data be passed in together for some reason?
        # test_preds = self._test_predictions(train_set, test_set, input_ph, predictions)
        # num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
        # try non-transductive procedure
        if self._transductive == False: # there appears to be a very small difference when using this option
            all_state, all_obs = np.concatenate([statea, stateb], axis=0), np.concatenate([obsa, obsb], axis=0)
            action = self.session.run(predictions, feed_dict={state_ph : all_state, obs_ph : all_obs})
            action = action[-1]
        else:
            action = self.session.run(predictions, feed_dict=test_feed_dict)
        # reset back to the old variables for the next evaluation
        self._full_state.import_variables(old_vars)
        #return num_correct
        return action

    # TODO: figure out if we should evaluate transductively
    def _test_predictions(self, train_set, test_set, input_ph, predictions):
        if self._transductive:
            inputs, _ = zip(*test_set)
            return self.session.run(predictions, feed_dict={input_ph: inputs})
        res = []
        for test_sample in test_set:
            inputs, _ = zip(*train_set)
            inputs += (test_sample[0],)  # this passes in the training set and the test set?
            res.append(self.session.run(predictions, feed_dict={input_ph: inputs})[-1])
        return res

class FOML(Reptile):
    """
    A basic implementation of "first-order MAML" (FOML).

    FOML is similar to Reptile, except that you use the
    gradient from the last mini-batch as the update
    direction.

    There are two ways to sample batches for FOML.
    By default, FOML samples batches just like Reptile,
    meaning that the final mini-batch may overlap with
    the previous mini-batches.
    Alternatively, if tail_shots is specified, then a
    separate mini-batch is used for the final step.
    This final mini-batch is guaranteed not to overlap
    with the training mini-batches.
    """
    def __init__(self, tail_shots=None, *args, **kwargs):
        """
        Create a first-order MAML session.

        Args:
          args: args for Reptile.
          tail_shots: if specified, this is the number of
            examples per class to reserve for the final
            mini-batch.
          kwargs: kwargs for Reptile.
        """
        super(FOML, self).__init__(*args, **kwargs)
        self.tail_shots = tail_shots

    # pylint: disable=R0913,R0914
    def train_step(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   meta_step_size,
                   meta_batch_size):
        old_vars = self._model_state.export_variables()
        updates = []
        for _ in range(meta_batch_size):
            mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
            for batch in self._mini_batches(mini_dataset, inner_batch_size, inner_iters):
                inputs, labels = zip(*batch)
                last_backup = self._model_state.export_variables()
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
            updates.append(subtract_vars(self._model_state.export_variables(), last_backup))
            self._model_state.import_variables(old_vars)
        update = average_vars(updates)
        self._model_state.import_variables(add_vars(old_vars, scale_vars(update, meta_step_size)))

    def _mini_batches(self, mini_dataset, inner_batch_size, inner_iters):
        """
        Generate inner-loop mini-batches for the task.
        """
        if self.tail_shots is None:
            for value in _mini_batches(mini_dataset, inner_batch_size, inner_iters):
                yield value
            return
        train, tail = _split_train_test(mini_dataset, test_shots=self.tail_shots)
        for batch in _mini_batches(train, inner_batch_size, inner_iters - 1):
            yield batch
        yield tail

def _sample_mini_dataset_mil(dataset, num_classes, num_shots):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """
    shuffled = list(dataset)
    random.shuffle(shuffled)
    for class_idx, class_obj in enumerate(shuffled[:num_classes]):
        gifs, states, actions = class_obj.sample(num_shots)
        for shot_idx in range(num_shots):
            start_idx, end_idx = shot_idx*class_obj.T, (shot_idx + 1)*class_obj.T
            g, s, a = gifs[start_idx:end_idx], states[start_idx:end_idx], actions[start_idx:end_idx]
            yield (g, s, a)

def _sample_mini_dataset(dataset, num_classes, num_shots):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """
    shuffled = list(dataset)
    random.shuffle(shuffled)
    for class_idx, class_obj in enumerate(shuffled[:num_classes]):
        for sample in class_obj.sample(num_shots):
            yield (sample, class_idx)

def _mini_batches(samples, batch_size, num_batches):
    """
    Generate mini-batches from some data.

    Returns:
      An iterable of sequences of (input, label) pairs,
        where each sequence is a mini-batch.
    """
    cur_batch = []
    samples = list(samples)
    batch_count = 0
    while True:
        random.shuffle(samples)
        for sample in samples:
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:
                continue
            yield cur_batch
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return

def _split_train_test(samples, test_shots=1):
    """
    Split a few-shot task into a train and a test set.

    Args:
      samples: an iterable of (input, label) pairs.
      test_shots: the number of examples per class in the
        test set.

    Returns:
      A tuple (train, test), where train and test are
        sequences of (input, label) pairs.
    """
    train_set = list(samples)
    test_set = []
    labels = set(item[1] for item in train_set)
    for _ in range(test_shots):
        for label in labels:
            for i, item in enumerate(train_set):
                if item[1] == label:
                    del train_set[i]
                    test_set.append(item)
                    break
    if len(test_set) < len(labels) * test_shots:
        raise IndexError('not enough examples of each class for test set')
    return train_set, test_set
