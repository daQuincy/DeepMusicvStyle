# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:47:20 2019
@author: YQ
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from rnn import HyperLSTMCell
from rnn import LayerNormLSTMCell as LSTMCell

ohc = tfp.distributions.OneHotCategorical
seq2seq = tf.contrib.seq2seq
w_init = tf.contrib.layers.xavier_initializer()

class MusicVAE:
    def __init__(self, x_depth=[89, 33, 33],
                 enc_rnn_dim=512, enc_hyper_unit=256, enc_dropout=0.1,
                 dec_rnn_dim=1024, dec_hyper_unit=256, dec_dropout=0.2,
                 enc_rnn_layer=1, dec_rnn_layer=1,
                 enc_rnn="hyperlstm", dec_rnn="hyperlstm",
                 attention=0,
                 cont_dim=256, cat_dim=2, mu_force=2.0,
                 gumbel=0.05, style_embed_dim=256,
                 training=True,
                 beta_anneal_steps=1000, kl_reg=1.0
                 ):
        self.features = ["pitch", "dt", "duration", "velocity"]
        self.x_depth = x_depth
        self.x_dim = np.sum(x_depth)
        
        self.enc_rnn_dim = enc_rnn_dim
        self.enc_hyper_unit = enc_hyper_unit
        self.enc_dropout = 1 - enc_dropout
        self.enc_rnn = enc_rnn
        self.enc_rnn_layer = enc_rnn_layer
        
        self.dec_rnn_dim = dec_rnn_dim
        self.dec_hyper_unit = dec_hyper_unit
        self.dec_dropout = 1 - dec_dropout
        self.dec_rnn = dec_rnn
        self.dec_rnn_layer = dec_rnn_layer
        
        self.attention = attention
        
        self.cont_dim = cont_dim
        self.cat_dim = cat_dim
        self.mu_force = mu_force
        
        self.style_embed_dim = style_embed_dim
        self.gumbel = gumbel
        
        self.training = training
        self.beta_anneal_steps = beta_anneal_steps
        
        self.kl_reg = kl_reg
        
        self.summaries = []
        
        # https://datascience.stackexchange.com/questions/29851/one-hot-encoding-vs-word-embeding-when-to-choose-one-or-another
        self.pitch_embedding = tf.Variable(tf.random_uniform([89, 32], -1.0, 1.0), 
                                           name="pitch_embedding")
        self.style_embedding = tf.Variable(
            tf.random_uniform([self.cat_dim, self.style_embed_dim],  -1.0, 1.0), 
            name="style_embedding")
        
    def kl_cost(self, enc_out, mode="cont"):
        if mode == "cont":
            mu, log_z_var = enc_out
            loss = -0.5 * tf.reduce_sum(1 + log_z_var - tf.square(mu) - tf.exp(log_z_var), axis=-1)
        elif mode == "cat":
            alpha = tf.nn.softmax(enc_out)
            log_dim = tf.math.log(tf.cast(tf.shape(enc_out)[-1], tf.float32))
            neg_entropy = tf.reduce_sum(alpha * tf.math.log(alpha + 1e-10), axis=-1)
            loss = log_dim + neg_entropy
        
        return loss
    
    def rnn_cell(self, rnn, n_units, hyper_unit, dropout, n_layers=1, input_dropout=False, device=None):
        if rnn == "hyperlstm":
            rnn = tf.nn.rnn_cell.MultiRNNCell([
                        HyperLSTMCell(n_units,
                                      hyper_num_units=hyper_unit,
                                      dropout_keep_prob=dropout if self.training else 1.0, 
                                      use_recurrent_dropout=self.training) 
                        for _ in range(n_layers)])
        elif rnn == "lstm":
            rnn = tf.nn.rnn_cell.MultiRNNCell([
                    LSTMCell(n_units,
                             dropout_keep_prob=dropout if self.training else 1.0,
                             use_recurrent_dropout=self.training)
                        for _ in range(n_layers)])
    
        if input_dropout:
            keep_prob = dropout if self.training else 1.0
            rnn = tf.nn.rnn_cell.DropoutWrapper(rnn, input_keep_prob=keep_prob)
    
        if device is not None:
            rnn = tf.nn.rnn_cell.DeviceWrapper(rnn, device)
        
        return rnn
    
    def get_initial_rnn_state(self, z, rnn, rnn_dim, rnn_layer):
        init_state = []
        if rnn == "hyperlstm":
            init_state = []
            for i in range(rnn_layer):
                tmp = tf.layers.dense(z, 2*(self.dec_rnn_dim+self.dec_hyper_unit), activation=tf.nn.elu, 
                                      name="dec_init_state_{}".format(i), kernel_initializer=w_init)
                init_state.append(tmp)
            
        elif rnn == "lstm":
            for i in range(rnn_layer):
                tmp = tf.layers.dense(z, rnn_dim*2, name="rnn_state_{}".format(i), activation=tf.nn.tanh,
                                      kernel_initializer=w_init)
                init_state.append(tmp)
                
        init_state = tuple(init_state)
            
        return init_state
        
    def encoder(self, x, seq_len):
        with tf.variable_scope("encoder"):
            cell_fw = self.rnn_cell(self.enc_rnn, self.enc_rnn_dim, 
                                    self.enc_hyper_unit, self.enc_dropout, 
                                    self.enc_rnn_layer)
            cell_bw = self.rnn_cell(self.enc_rnn, self.enc_rnn_dim, 
                                    self.enc_hyper_unit, self.enc_dropout, 
                                    self.enc_rnn_layer)
            
            # sequence_length exclude the <end> token
            outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x,
                                                                            dtype=tf.float32,
                                                                            sequence_length=seq_len-1)    
            
            states = tf.concat([state_fw[-1], state_bw[-1]], axis=-1)
                
        with tf.variable_scope("enc_heads"):
            cont_head = tf.layers.dense(states, 512, name="cont_head", 
                                        kernel_initializer=w_init, activation=tf.nn.relu)
            z_mean = tf.layers.dense(cont_head, self.cont_dim, name="z_mean", kernel_initializer=w_init)
            log_z_var = tf.layers.dense(cont_head, self.cont_dim, name="z_variance", 
                                        kernel_initializer=w_init)
            
            cat_head = tf.layers.dense(states, 512, name="cat_head", 
                                       kernel_initializer=w_init, activation=tf.nn.relu)
            z_cat_logit = tf.layers.dense(cat_head, self.cat_dim, name="z_cat_logit", 
                                          kernel_initializer=w_init)
            
        return z_mean, log_z_var, z_cat_logit
        
    def decoder(self, z, x, seq_len):
        start_token = tf.zeros((tf.shape(z)[0], 1, self.x_dim), dtype=tf.float32)
        start_token = self.embedding_lookup(start_token, self.pitch_embedding)
        
        mask = tf.stop_gradient(tf.sequence_mask(seq_len, dtype=tf.bool))
        
        x = tf.concat([start_token, x[:, :-1, :]], axis=1)
        
        if self.attention > 0:
            key = tf.keras.layers.Dense(self.attention)
            value = tf.keras.layers.Dense(self.attention)
            
            attn = tf.keras.layers.Attention(use_scale=True, causal=True)
            keys = key(x)
            values = value(x)
            
            if self.training:
                keys = tf.keras.layers.Dropout(0.2)(keys)
                values = tf.keras.layers.Dropout(0.2)(values)
            
            x = tf.concat([x, attn([keys, values], [mask, mask])], axis=-1)
        
            start_token = tf.concat([start_token, 
                                     tf.zeros((tf.shape(z)[0], 1, self.attention), 
                                              dtype=tf.float32)], axis=-1)
        initialize_fn = lambda: (tf.zeros((tf.shape(z)[0],), tf.bool), tf.squeeze(start_token, 1))
        
        def sample_fn(time, outputs, state):
            logits = dense(outputs)
            logits = tf.split(logits, self.x_depth, axis=-1)
            
            samples = []
            for (logit, depth) in zip(logits, self.x_depth):
                if depth == 1:
                    tmp = logit
                else:
                    tmp = ohc(logits=logit/self.temperature, dtype=tf.float32).sample()
                samples.append(tmp)
            samples = tf.concat(samples, axis=-1)
            
            return samples
        
        def next_input_fn(time, outputs, state, sample_ids):
            outputs = sample_fn(None, outputs, None)
            
            finished, _, _ = tf.split(outputs, self.x_depth, axis=-1)
            finished = tf.argmax(finished, -1, output_type=tf.int32)
            finished = tf.math.equal(finished, 88)
            
            outputs = self.embedding_lookup(outputs, self.pitch_embedding)
            
            if self.attention > 0:
                k = key(outputs)
                v = value(outputs)
                outputs = tf.concat([outputs, attn([k, v])], axis=-1)
            
            return finished, outputs, state
        
        with tf.variable_scope("decoder"):
            dense = tf.layers.Dense(self.x_dim, name="logit_dense", kernel_initializer=w_init)
            init_state = self.get_initial_rnn_state(z, self.dec_rnn, self.dec_rnn_dim, self.dec_rnn_layer)
            cell = self.rnn_cell(self.dec_rnn, self.dec_rnn_dim, self.dec_hyper_unit, self.dec_dropout, 
                                 self.dec_rnn_layer, input_dropout=False)
            
            train_out, _ = tf.nn.dynamic_rnn(cell, x, sequence_length=seq_len, initial_state=init_state)
            logits = dense(train_out)
            logits_split = tf.split(logits, self.x_depth, axis=-1)
            train_out = [tf.argmax(x, axis=-1, output_type=tf.int32) for x in logits_split]
            train_out = tf.stack(train_out, axis=-1)
            
            helper = seq2seq.CustomHelper(initialize_fn, sample_fn, next_input_fn, 
                                          sample_ids_shape=(self.x_dim), sample_ids_dtype=tf.float32)
            decoder = seq2seq.BasicDecoder(cell, helper, init_state)
            outputs, final_state, final_seq_len = seq2seq.dynamic_decode(decoder, maximum_iterations=512)
            sample_outputs = outputs.sample_id
            sample_outputs = tf.split(sample_outputs, self.x_depth, axis=-1)
            final_outputs = [tf.argmax(x, axis=-1, output_type=tf.int32) for x in sample_outputs]
            final_outputs = tf.stack(final_outputs, axis=-1)
            
            final_state = final_state[-1]
            
        return logits, final_outputs, final_seq_len, final_state
    
    def reconstruction_loss(self, logits, X, X_len):
        logits = tf.split(logits, self.x_depth, axis=-1)
        X = tf.split(X, self.x_depth, axis=-1)
        mask = tf.stop_gradient(tf.sequence_mask(X_len, dtype=tf.float32))
        loss = []
        for (logit, x, depth, feature) in zip(logits, X, self.x_depth, self.features):
            if depth == 1:
                tmp_loss = tf.losses.mean_squared_error(tf.squeeze(x, axis=-1), 
                                                        tf.squeeze(logit, axis=-1), weights=mask)
            else:
                tmp_loss = tf.nn.softmax_cross_entropy_with_logits(labels=x, logits=logit)
                tmp_loss = tf.losses.compute_weighted_loss(tmp_loss, weights=mask)
                if feature == "pitch":
                    tmp_loss = tmp_loss * 3
            loss.append(tmp_loss)
            self.summaries.append(tf.summary.scalar(feature+"_loss", tmp_loss))
            
        return tf.reduce_sum(loss), loss
    
    def reconstruction_accuracy(self, logits, X, X_len):
        logits = tf.split(logits, self.x_depth, axis=-1)
        X = tf.split(X, self.x_depth, axis=-1)
        mask = tf.sequence_mask(X_len, dtype=tf.float32)
        accuracy = []
        for (logit, x, feature, depth) in zip(logits, X, self.features, self.x_depth):
            if depth == 1:
                continue
            tmp_x = tf.argmax(x, axis=-1, output_type=tf.int32)
            tmp_logit = tf.argmax(logit, axis=-1, output_type=tf.int32)
            tmp_acc = tf.contrib.metrics.accuracy(tmp_x, tmp_logit, 
                                                  mask, feature+"_accuracy")
            accuracy.append(tmp_acc)
            self.summaries.append(tf.summary.scalar(feature+"_acc", tmp_acc))
        
        return accuracy
    
    def sample(self, enc_out, mode="cont"):
        if mode == "cont":
            z_mean, log_z_var = enc_out
            z_sigma = tf.sqrt(tf.exp(log_z_var))
            eps = tf.random_normal(tf.shape(z_mean), 0.0, 1.0, tf.float32)
            z = z_mean + tf.multiply(z_sigma, eps)
            
        elif mode == "cat":
            if self.training:
                unif = tf.random.uniform(shape=tf.shape(enc_out))
                gumbel_noise = -tf.math.log(-tf.math.log(unif + 1e-10) + 1e-10)
                logit = (enc_out + gumbel_noise) / self.gumbel
                z = tf.nn.softmax(logit)
            else:
                z = ohc(logits=enc_out, dtype=tf.float32).sample()
            
        return z
    
    def embedding_lookup(self, X, lookup_table):        
        X_split = tf.split(X, self.x_depth, -1)
        p = tf.nn.embedding_lookup(lookup_table, tf.argmax(X_split[0], axis=-1))
        x = tf.concat([p] + X_split[1:], -1)
        
        return x
    
    def build(self, X=None, S=None, labels=None, gpu="/gpu:0"):
        """
        X: Onehot encoded MIDI representation of notes.
        S: sequence length
        labels: genre/style labels
        """
        with tf.device(gpu):
            if X is None:
                self.X = tf.placeholder(tf.float32, (None, None, self.x_dim))
                self.S = tf.placeholder(tf.int32, (None,))
                self.labels = tf.placeholder(tf.int32, (None,))
            else:
                self.X = X
                self.S = S
                self.labels = labels
                
            labels = tf.one_hot(self.labels, self.cat_dim)
            model_input = tf.split(self.X, self.x_depth, axis=-1)
            model_input = [tf.argmax(x, axis=-1, output_type=tf.int32) for x in model_input]
            self.model_input = tf.stack(model_input, axis=-1)
            
            self.temperature = tf.placeholder_with_default(1.0, shape=[], name="temperature")
            self.step = tf.train.create_global_step()
            
            beta = 1.0 - self.beta_anneal_steps / (self.beta_anneal_steps + tf.exp(self.step / self.beta_anneal_steps))
            beta = tf.cast(beta, tf.float32)
            
            # embed features
            X_embed = self.embedding_lookup(self.X, self.pitch_embedding)
            
            z_mean, log_z_var, z_cat_logit = self.encoder(X_embed, self.S)
            self.z_cont = self.sample([z_mean, log_z_var], mode="cont") 
            self.z_cat = self.sample(z_cat_logit, mode="cat") 
            z_cat = tf.matmul(self.z_cat, self.style_embedding)
            if self.style_embed_dim == self.cont_dim:
                z = self.z_cont + z_cat
            else:
                z = tf.concat([self.z_cont, z_cat], axis=-1)
            
            logits, self.output, self.len, dec_state = self.decoder(z, X_embed, self.S)
                            
            with tf.variable_scope("losses"):
                self.recon_loss, self.feature_loss = self.reconstruction_loss(logits, self.X, self.S)
                
                z_mean_mu = tf.reduce_mean(z_mean, 0)
                z_mean_mu = tf.tile(tf.expand_dims(z_mean_mu, 0), [tf.shape(self.z_cont)[0], 1])
                mu_loss = tf.nn.relu(self.mu_force - tf.losses.mean_squared_error(z_mean_mu, z_mean))
                
                cont_kl_cost = tf.reduce_mean(self.kl_cost([z_mean, log_z_var], mode="cont")/self.cont_dim)
                
                # categorical z
                cat_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=z_cat_logit)
                cat_loss = tf.reduce_mean(cat_loss)
                
                
                self.kl_loss = cont_kl_cost
                            
                self.loss = 0
                self.loss += self.recon_loss + mu_loss
                self.loss += cont_kl_cost * beta * self.kl_reg
                self.loss += cat_loss
                
                self.summaries.append(tf.summary.scalar("total_loss", self.loss))
                self.summaries.append(tf.summary.scalar("cat_loss", cat_loss))
                self.summaries.append(tf.summary.scalar("cont_kl_cost", cont_kl_cost))
                self.summaries.append(tf.summary.scalar("mu_loss", mu_loss))
                
            with tf.variable_scope("accuracies"):
                self.accuracies = self.reconstruction_accuracy(logits, self.X, self.S)
            
            if self.training:
                # optimizer... training...
                self.learning_rate = tf.maximum(5e-4 * 0.95 ** ((self.step - 10000) / 5000), 1e-4)
                opt = tf.train.AdamOptimizer(self.learning_rate)
                g, v = zip(*opt.compute_gradients(self.loss, tf.trainable_variables()))
                g, _ = tf.clip_by_global_norm(g, 1.0)
                gvs = zip(g, v)
                self.op = opt.apply_gradients(gvs, global_step=self.step)
                
                self.init = tf.global_variables_initializer()
            
                with tf.variable_scope("misc"):
                    self.summaries.append(tf.summary.scalar("beta", beta))
                    self.summaries.append(tf.summary.scalar("learning_rate", self.learning_rate))
                
            self.summ_op = tf.summary.merge(self.summaries)
            self.saver = tf.train.Saver(max_to_keep=20)
            
            restore_ops = [self.temperature, self.output, self.len,
                           self.z_cont, self.z_cat]
            for restore_op in restore_ops:
                tf.add_to_collection("restore_ops", restore_op)
            
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print("Total learnable parameters: {}".format(total_parameters))
        
if __name__ == "__main__":
    tf.reset_default_graph()
    m = MusicVAE()
    m.build()