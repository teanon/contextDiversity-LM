# -*- coding: utf-8 -*-
import tensorflow as tf
import reader
import time
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("gpk",1,"Interval value")
flags.DEFINE_float("alpha",10.0,"the  coefficient of innerProLoss")
FLAGS = flags.FLAGS


class PTBInput(object):
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
                            data, batch_size, num_steps, name=name)
         


class PTBModel(object):
    def __init__(self, is_training, config, input_):
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                        size, forget_bias=0.0, state_is_tuple=True)  
        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                               lstm_cell(), output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell(
                [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True )
        
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        
        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable(
                            "embedding", [vocab_size, size], dtype=tf.float32 )
             
            inputs = tf.nn.embedding_lookup(self.embedding, input_.input_data)
            
        
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state )
                outputs.append(cell_output)
        self.output = tf.reshape(tf.concat(outputs, 1,name="outputs"), [-1, size],name="output") #shape=(700, 1500)
         
        softmax_w = tf.get_variable(
                        "softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.matmul(self.output, softmax_w) + softmax_b
        innerPro=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(logits[config.gap_k::,:], axis=1)+1e-37,
                                                logits=tf.nn.softmax(logits[0:batch_size*num_steps-config.gap_k:,:], axis=1)+1e-37))/(batch_size*num_steps-config.gap_k)
         
        innerProLoss=1.0/(0.00001+tf.exp(innerPro)) 
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(input_.targets, [-1])],
                [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        finalLoss=cost+config.alpha*innerProLoss
        if not is_training: #
            return
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(finalLoss, tvars),
                                config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        self._new_lr = tf.placeholder(
                        tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input
    @property
    def initial_state(self):
        return self._initial_state
    @property
    def cost(self):
        return self._cost
    @property
    def final_state(self):
        return self._final_state
    @property
    def lr(self):
        return self._lr
    @property
    def train_op(self):
        return self._train_op

class SmallConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 11
    keep_prob = 1.0
    lr_decay = 0.35
    batch_size = 20
    vocab_size = 10000
    gap_k=FLAGS.gpk #Interval value
    alpha=FLAGS.alpha  # the weight coefficient of innerProLoss
# class MediumConfig(object):
#     init_scale = 0.05
#     learning_rate = 1.0
#     max_grad_norm = 5
#     num_layers = 2
#     num_steps = 35
#     hidden_size = 650
#     max_epoch = 11
#     max_max_epoch = 38
#     keep_prob = 0.5
#     lr_decay = 0.78   
#     batch_size = 20
#     vocab_size = 10000
#     gap_k=5
class LargeConfig(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10   
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 18
    max_max_epoch = 47
    keep_prob = 0.35  
    lr_decay = 0.78
    batch_size = 20
    vocab_size = 10000
    gap_k=FLAGS.gpk #Interval value
    alpha=FLAGS.alpha  # the weight coefficient of innerProLoss

def run_epoch(session, model, eval_op=None, verbose=False):
    
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
     
    fetches = {
               "cost": model.cost,
               "final_state": model.final_state,
               }
    if eval_op is not None:
        fetches["eval_op"] = eval_op
    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]
        costs += cost
        iters += model.input.num_steps
        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))
             
    return np.exp(costs / iters)

raw_data = reader.ptb_raw_data('data/penn/')
train_data, valid_data, test_data, _ = raw_data 
#  LargeConfig  SmallConfig
config = LargeConfig()
eval_config = LargeConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer ):
            m = PTBModel(is_training=True, config=config, input_=train_input)
    
    with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config, input_=valid_input )

    with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = PTBModel(is_training=False, config=eval_config,
                            input_=test_input )

    
    init = tf.global_variables_initializer()
 
    saver = tf.train.Saver()
    sv = tf.train.Supervisor(logdir="models/", init_op=init)   
    saver=sv.saver    
    
    with sv.managed_session() as session:
        
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                              verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
        test_perplexity = run_epoch(session, mtest)
        print("Test Perplexity: %.3f" % test_perplexity)
        saver.save(session,'models/PTB')
         
