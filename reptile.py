import random
import tensorflow as tf
import numpy as np

from variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars, VariableState)

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
        self._model_state = VariableState(self.session, graph, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session, graph,
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self._transductive = transductive
        self._pre_step_op = pre_step_op

    # pylint: disable=R0913,R0914
    def train_step(self,
                   dataset,
                   state_ph,
                   label_ph,
                   minimize_op,
                   loss_op,
                   writer,
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
        actiona, statea = dataset
        for task_idx in range(meta_batch_size):
            action_batch = actiona[task_idx, :, :]
            state_batch  = statea[task_idx, :, :]
            # inner loop update
            if self._pre_step_op:
                self.session.run(self._pre_step_op)
            feed_dict = {state_ph : state_batch, label_ph : action_batch}
            print('taking gradient step')
            _, loss = self.session.run([minimize_op, loss_op], feed_dict=feed_dict)
            print('took graident step on loss:', loss)
            losses.append(loss)
            # export vars and import old vars
            new_vars.append(self._model_state.export_variables())
            self._model_state.import_variables(old_vars)
        # meta update
        new_vars = average_vars(new_vars)
        self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))
        # add loss summary
        summary = tf.Summary()
        print('ave loss:', np.mean(losses))
        summary.value.add(tag='ave_loss', simple_value=np.mean(losses))
        writer.add_summary(summary, step)


