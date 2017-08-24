import tensorflow as tf
import numpy as np
import os
import sys
import glob
import time
from operations import *
from utils import *

class EBGAN():
    def __init__(self, args, sess):
        self.args = args
        self.sess = sess

        self.build_model()
        self.args.is_training = True

    def build_model(self):
        self.input_imgs = tf.placeholder(tf.float32, [self.args.batch_size] + [self.args.target_size, self.args.target_size, self.args.num_channel], name='input_image')
        self.z = tf.placeholder(tf.float32, [None, self.args.z_dim], name='noise_z')
        tf.summary.histogram('z', self.z)

        # Define generator
        self.g = self.generator(self.z)
        tf.summary.image('Gen', self.g)
        self.real_ae, self.real_embedding = self.discriminator(self.input_imgs)
        tf.summary.image('Decoder_real', self.real_ae)
        self.gen_ae, self.gen_embedding = self.discriminator(self.g, reuse=True)
        tf.summary.image('Decoder_fake', self.gen_ae)
        self.generated_sample = self.generator(self.z, reuse=True, sampling=True)

        '''
        Loss term
        1. Minimum Squared Error : D(x) = ||dec(enc(x)) - x||
        2. Pull away Term : adding to MSE only used in the generator loss'''
        self.gen_loss = tf.sqrt(tf.reduce_sum(tf.square(self.gen_ae - self.g)))/self.args.batch_size
        if self.args.pt:
            '''PT operates on mini-batch and to orthogonalize the pairwise representation for embedding for varied face, [batch, embedding]'''
            norm = tf.reduce_sum(tf.square(self.gen_embedding), axis=1, keep_dims=True) # Keep_dims to normalize, to divide
            normalized_embedding = self.gen_embedding / norm # [batch, embedding_size]
            pt = tf.matmul(normalized_embedding, normalized_embedding, transpose_b=True) # s_i(transpose)*s_j, make [batch, batch]
            diagonal_part = tf.diag_part(pt)
            diag_mtx_pt = tf.diag(diagonal_part)
            similarity = pt - diag_mtx_pt
            pt_loss = (tf.reduce_sum(tf.square(similarity))/2)/(self.args.batch_size*(self.args.batch_size-1))
            self.gen_loss += self.args.pt_weight*pt_loss
        tf.summary.scalar('generator_loss', self.gen_loss)
        self.real_loss = tf.sqrt(tf.reduce_sum(tf.square(self.real_ae-self.input_imgs)))/self.args.batch_size
        self.disc_loss = tf.maximum(self.args.margin-self.gen_loss, 0) + self.real_loss
        tf.summary.scalar('discriminator_loss', self.disc_loss)

        tr_vrbs = tf.trainable_variables()
        self.sess.run(tf.global_variables_initializer())
        for i in tr_vrbs:
            print(i.op.name)

        self.g_vars = [v for v in tr_vrbs if v.name.startswith('generator')]
        self.d_vars = [v for v in tr_vrbs if v.name.startswith('discriminator')]

        d_optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
        d_grads = d_optimizer.compute_gradients(self.disc_loss, var_list=self.d_vars) # Returns (gradient, variable) pair, gradient can be none
        for grad, var in d_grads:
            if grad is not None:
                tf.summary.histogram(var.op.name+'/gradient', grad) # var.op.name -> 'var', var.name => 'var:0'
        self.d_opt_target = d_optimizer.apply_gradients(d_grads)

        g_optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
        g_grads = g_optimizer.compute_gradients(self.gen_loss, var_list=self.g_vars)
        for grad, var in g_grads:
            if grad is not None:
                tf.summary.histogram(var.op.name+'/gradient', grad)
        self.g_opt_target = g_optimizer.apply_gradients(g_grads)

        self.saver = tf.train.Saver()


    def generator(self,z, reuse=False, sampling=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()
            if sampling:
                self.args.is_training = False

            # self.args.final_dim=128
            z_flatten = linear(z, self.args.target_size//16 * self.args.target_size//16 * self.args.final_dim * 8, name='linear')
            deconv1 = tf.reshape(z_flatten, [-1, int(self.args.target_size/16), int(self.args.target_size/16), self.args.final_dim*8])
            deconvolution1 = deconv2d(deconv1, output_shape=[self.args.batch_size, int(self.args.target_size/8), int(self.args.target_size/8), self.args.final_dim*4], filter_height=5, filter_width=5, name='gen1')
            deconv1_batch = batch_wrapper(deconvolution1, self.args.is_training, name='gen_batch1')
            deconv2 = relu(deconv1_batch, name='gen_relu1')

            deconvolution2 = deconv2d(deconv2, output_shape=[self.args.batch_size, int(self.args.target_size/4), int(self.args.target_size/4), self.args.final_dim*2], filter_height=5, filter_width=5, name='gen2')
            deconv2_batch = batch_wrapper(deconvolution2, self.args.is_training, name='gen_batch2')
            deconv3 = relu(deconv2_batch, name='gen_relu2')

            deconvolution3 = deconv2d(deconv3, output_shape=[self.args.batch_size, int(self.args.target_size/2), int(self.args.target_size/2), self.args.final_dim], filter_height=5, filter_width=5, name='gen3')
            deconv3_batch = batch_wrapper(deconvolution3, self.args.is_training, name='gen_batch3')
            deconv4 = relu(deconv3_batch, name='gen_relu3')

            deconvolution4 = deconv2d(deconv4, output_shape=[self.args.batch_size, self.args.target_size, self.args.target_size, self.args.num_channel], name='gen4')

            return tf.nn.tanh(deconvolution4, name='generator_result')


    def discriminator(self, input_image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            embedding = self.encoder(input_image)
            auto_encoder = self.decoder(embedding)
            return auto_encoder, embedding


    # Encoder : (64) 4c2s- (128)4c2s - (256)4c2s, self.args.disc_channel : 64
    def encoder(self, input_image):
        with tf.variable_scope('encoder'):
            enc1 = conv2d(input_image, output_channel=self.args.disc_channel, filter_height=4, filter_width=4, stride_hor=2, stride_ver=2, name='enc1') # [batch, 32, 32, 64]
            enc_batch1 = batch_wrapper(enc1, self.args.is_training, name='enc_batch1')
            enc_layer2 = leakyrelu(enc_batch1, name='enc_lrelu1')

            enc2 = conv2d(enc_layer2, output_channel=self.args.disc_channel*2, filter_height=4, filter_width=4, stride_hor=2, stride_ver=2, name='enc2') # [batch, 16, 16, 128]
            enc_batch2 = batch_wrapper(enc2, self.args.is_training, name='enc_batch2')
            enc_layer3 = leakyrelu(enc_batch2, name='enc_lrelu2')

            enc3 = conv2d(enc_layer3, output_channel=self.args.disc_channel*4, filter_height=4, filter_width=4, stride_hor=2, stride_ver=2, name='enc3') # [batch, 8,8, 256]
            enc_batch3 = batch_wrapper(enc3, self.args.is_training, name='enc_batch3')
            enc_layer4 = leakyrelu(enc_batch3, name='enc_lrelu3')

            embedding = tf.reshape(enc_layer4, [self.args.batch_size, -1], name='embedding')

            return embedding

    # Decoder :(128)4c2s - (64)4c2s - (3)4c2s, embedding size : [batch, 8*8*256]
    def decoder(self, embedding):
        with tf.variable_scope('decoder'):
            reshaped = tf.reshape(embedding, [self.args.batch_size, int(self.args.target_size/8), int(self.args.target_size/8), self.args.disc_channel*4]) # [batch, 8,8,256]

            dec1 = deconv2d(reshaped, output_shape=[self.args.batch_size, int(self.args.target_size/4), int(self.args.target_size/4), self.args.disc_channel*2], filter_height=4, filter_width=4, stride_hor=2, stride_ver=2, name='dec1') # [batch, 16, 16, 128]
            dec_batch1 = batch_wrapper(dec1, self.args.is_training, name='dec_batch1')
            dec_layer2 = leakyrelu(dec_batch1, name='dec_lrelu1')

            dec2 = deconv2d(dec_layer2, output_shape=[self.args.batch_size, int(self.args.target_size/2), int(self.args.target_size/2), self.args.disc_channel], filter_height=4, filter_width=4, stride_hor=2, stride_ver=2, name='dec2') # [batch, 32, 32, 64]
            dec_batch2 = batch_wrapper(dec2, self.args.is_training, name='dec_batch2')
            dec_layer3 = leakyrelu(dec_batch2, name='dec_lrelu2')

            dec3 = deconv2d(dec_layer3, output_shape=[self.args.batch_size, int(self.args.target_size), int(self.args.target_size), self.args.num_channel], filter_height=4, filter_width=4, stride_hor=2, stride_ver=2, name='dec3')

            return tf.nn.tanh(dec3, name='decoder_output')

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('./tensorboard_log', self.sess.graph)

        self.train_count = 0
        self.real_datas = glob.glob(os.path.join(self.args.data_path, '*.jpg'))

        self.sample_z = np.random.uniform(-1, 1, [self.args.showing_height * self.args.showing_width, self.args.z_dim])

        if self.load():
            print('Checkpoint_loaded')
        else:
            print('Checkpoint load failed')

        for epoch in range(self.args.num_epoch):
            self.start_time = time.time()
            print('Epoch %d' % (epoch+1))
            shuffle = np.random.permutation(len(self.real_datas))[:self.args.partition_index]
            batches = np.asarray(self.real_datas)[shuffle]
            trainingsteps_per_epoch = self.args.partition_index // self.args.batch_size
            for index in range(trainingsteps_per_epoch):
                self.train_count += 1
                batch_z = np.random.uniform(-1,1, [self.args.batch_size, self.args.z_dim])
                batch_input = batches[self.args.batch_size*index: self.args.batch_size*(index+1)]
                batch_real = [get_image(one_input, self.args.input_size, self.args.target_size) for one_input in batch_input]

                discriminator_loss, summaries, _ = self.sess.run([self.disc_loss, self.summary_op, self.d_opt_target], feed_dict={self.z:batch_z, self.input_imgs:batch_real})
                self.summary_writer.add_summary(summaries, self.train_count)
                _ = self.sess.run([self.g_opt_target], feed_dict={self.z:batch_z})
                generator_loss, _ = self.sess.run([self.gen_loss, self.g_opt_target], feed_dict={self.z:batch_z})

                print('Epoch %d, Generator loss : %3.4f, Discriminator loss : %3.4f at time : %3.4f' % (epoch+1, generator_loss, discriminator_loss, time.time()-self.start_time))

            if np.mod(epoch+1, self.args.save_interval) == 0:
                G_sample = self.sess.run(self.generated_sample, feed_dict={self.z:self.sample_z})
                save_image(G_sample, [self.args.showing_height, self.args.showing_width], '{}/train_{:2d}epoch.png'.format(self.args.sample_dir, epoch+1))
                self.save(self.train_count)
                self.args.is_training = True

    def generator_test(self):
        self.load()
        z_test = np.random.uniform(-1,1, [self.args.showing_height*self.args.showing_width, self.args.z_dim])
        generated = self.sess.run(self.generated_sample, feed_dict={self.z: z_test})
        save_image(generated, [self.args.showing_height, self.args.showing_width], './{}/test.jpg'.format(self.args.sample_dir))

    @property
    def model_dir(self):
        return "{}batch_{}zdim".format(self.args.batch_size, self.args.z_dim)

    def save(self, global_step):
        model_name='EBGAN'
        checkpoint_path = os.path.join(self.args.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        self.saver.save(self.sess, os.path.join(checkpoint_path, model_name), global_step=global_step)
        print('Save checkpoint at %d step' % global_step)

    def load(self):
        checkpoint_path = os.path.join(self.args.checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.train_count = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(checkpoint_path, ckpt_name))
            print('Checkpoint loaded at %d steps' % (self.train_count))
            return True
        else:
            self.train_count = 0
            return False
