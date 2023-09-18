import sys
import numpy as np
import PIL.Image
from tqdm import tqdm
from tensorflow.data.experimental import prefetch_to_device, shuffle_and_repeat, map_and_batch
from ops import *
from utils import *


class StyleGAN(object):

    def __init__(self, sess, args):
        self.phase = args.phase
        self.progressive = args.progressive
        self.model_name = "StyleGAN"
        self.sess = sess
        self.dataset_name = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir

        self.iteration = args.iteration * 10000
        self.max_iteration = args.max_iteration * 10000

        self.batch_size = args.batch_size
        self.img_size = args.img_size

        """ Hyper-parameter"""
        self.start_res = args.start_res
        self.resolutions = resolution_list(self.img_size) # [4, 8, 16, 32, 64, 128, 256, 512, 1024 ...]
        self.featuremaps = featuremap_list(self.img_size) # [512, 512, 512, 512, 256, 128, 64, 32, 16 ...]

        if not self.progressive :
            self.resolutions = [self.resolutions[-1]]
            self.featuremaps = [self.featuremaps[-1]]
            self.start_res = self.resolutions[-1]

        self.gpu_num = 1

        self.z_dim = 512
        self.w_dim = 512
        self.n_mapping = 8

        self.w_ema_decay = 0.995 # Decay for tracking the moving average of W during training
        self.style_mixing_prob = 0.9 # Probability of mixing styles during training
        self.truncation_psi = 0.9 # Style strength multiplier for the truncation trick
        self.truncation_cutoff = 8 # Number of layers for which to apply the truncation trick

        self.batch_size_base = 4
        self.learning_rate_base = 0.001

        self.train_with_trans = {4: False, 8: False, 16: True, 32: True, 64: True, 128: True, 256: True, 512: True, 1024: True}
        self.batch_sizes = get_batch_sizes(self.gpu_num)

        self.end_iteration = get_end_iteration(self.iteration, self.max_iteration, self.train_with_trans, self.resolutions, self.start_res)

        self.g_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        self.d_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

        self.sn = args.sn

        self.print_freq = {4: 1000, 8: 1000, 16: 1000, 32: 1000, 64: 1000, 128: 3000, 256: 5000, 512: 10000, 1024: 10000}
        self.save_freq = {4: 1000, 8: 1000, 16: 1000, 32: 1000, 64: 1000, 128: 3000, 256: 5000, 512: 10000, 1024: 10000}

        self.print_freq.update((x, y // self.gpu_num) for x, y in self.print_freq.items())
        self.save_freq.update((x, y // self.gpu_num) for x, y in self.save_freq.items())

        self.dataset = self.dataset_name


    ##################################################################################
    # Generator
    ##################################################################################

    def g_mapping(self, z, n_broadcast, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('g_mapping', reuse=reuse):
            # normalize input first
            x = pixel_norm(z)

            # run through mapping network
            for ii in range(self.n_mapping):
                with tf.variable_scope('FC_{:d}'.format(ii)):
                    x = fully_connected(x, units=self.w_dim, gain=np.sqrt(2), lrmul=0.01, sn=self.sn)
                    x = apply_bias(x, lrmul=0.01)
                    x = lrelu(x, alpha=0.2)

            # broadcast to n_layers
            with tf.variable_scope('Broadcast'):
                x = tf.tile(x[:, np.newaxis], [1, n_broadcast, 1])

        return x


    @tf.function
    def g_synthesis(self, w_broadcasted, alpha, resolutions, featuremaps, noise_list, noise_list_, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('g_synthesis', reuse=reuse):
            coarse_styles, middle_styles, fine_styles = get_style_class(resolutions, featuremaps)
            layer_index = 2

            """ initial layer """
            res = resolutions[0]
            n_f = featuremaps[0]

            x = synthesis_const_block(res, w_broadcasted, n_f, noise_list[0], noise_list_[0], self.sn)

            """ remaining layers """
            if self.progressive :
                images_out = torgb(x, res=res, sn=self.sn)
                coarse_styles.pop(res, None)

                # Coarse style [4 ~ 8]
                # pose, hair, face shape

                for res, n_f in coarse_styles.items():
                    x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, noise_list[int(layer_index/2)], noise_list_[int(layer_index/2)], sn=self.sn)
                    img = torgb(x, res, sn=self.sn)
                    images_out = upscale2d(images_out)
                    images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)

                    layer_index += 2

                # Middle style [16 ~ 32]
                # facial features, eye
                for res, n_f in middle_styles.items():
                    x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, noise_list[int(layer_index/2)], noise_list_[int(layer_index/2)], sn=self.sn)
                    img = torgb(x, res, sn=self.sn)
                    images_out = upscale2d(images_out)
                    images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)

                    layer_index += 2

                # Fine style [64 ~ 1024]
                # color scheme
                for res, n_f in fine_styles.items():
                    x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, noise_list[int(layer_index/2)], noise_list_[int(layer_index/2)], sn=self.sn)
                    img = torgb(x, res, sn=self.sn)
                    images_out = upscale2d(images_out)
                    images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)

                    layer_index += 2

            else :
                layer_index = 0
                for res, n_f in zip(resolutions[1:], featuremaps[1:]) :
                    x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, noise_list[int(layer_index/2)+1], noise_list_[int(layer_index/2)+1], sn=self.sn)
                    layer_index += 2
                images_out = torgb(x, resolutions[-1], sn=self.sn)

            return images_out


    def generator(self, z, alpha, target_img_size, is_training=True, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("generator", reuse=reuse):
            resolutions = resolution_list(target_img_size)
            featuremaps = featuremap_list(target_img_size)
            
            w_avg = tf.get_variable('w_avg', shape=[self.w_dim],
                                    dtype=tf.float32, initializer=tf.initializers.zeros(),
                                    trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_TOWER)

            """ mapping layers """
            n_broadcast = len(resolutions) * 2
            w_broadcasted = self.g_mapping(z, n_broadcast)

            if is_training:
                """ apply regularization techniques on training """
                # update moving average of w
                w_broadcasted = self.update_moving_average_of_w(w_broadcasted, w_avg)

                # perform style mixing regularization
                w_broadcasted = self.style_mixing_regularization(z, w_broadcasted, n_broadcast, resolutions)

            else :
                """ apply truncation trick on evaluation """
                w_broadcasted = self.truncation_trick(n_broadcast, w_broadcasted, w_avg, self.truncation_psi)

            """ synthesis layers """
            noise_list = []
            noise_list_ = []
            for i in range(len(resolutions)):
                noise_list.append(tf.random_normal([self.batch_size, resolutions[i], resolutions[i], 1]))
                noise_list_.append(tf.random_normal([self.batch_size, resolutions[i], resolutions[i], 1]))

            x = self.g_synthesis(w_broadcasted, alpha, resolutions, featuremaps, noise_list, noise_list_)

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, alpha, target_img_size, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("discriminator", reuse=reuse):
            resolutions = resolution_list(target_img_size)
            featuremaps = featuremap_list(target_img_size)

            r_resolutions = resolutions[::-1]
            r_featuremaps = featuremaps[::-1]

            """ set inputs """
            x = fromrgb(x_init, r_resolutions[0], r_featuremaps[0], self.sn)

            """ stack discriminator blocks """
            for index, (res, n_f) in enumerate(zip(r_resolutions[:-1], r_featuremaps[:-1])):
                res_next = r_resolutions[index + 1]
                n_f_next = r_featuremaps[index + 1]

                x = discriminator_block(x, res, n_f, n_f_next, self.sn)

                if self.progressive :
                    x_init = downscale2d(x_init)
                    y = fromrgb(x_init, res_next, n_f_next, self.sn)
                    x = smooth_transition(y, x, res, r_resolutions[0], alpha)

            """ last block """
            res = r_resolutions[-1]
            n_f = r_featuremaps[-1]

            logit = discriminator_last_block(x, res, n_f, n_f, self.sn)

            return logit

    ##################################################################################
    # Technical skills
    ##################################################################################


    def update_moving_average_of_w(self, w_broadcasted, w_avg):
        with tf.variable_scope('WAvg'):
            batch_avg = tf.reduce_mean(w_broadcasted[:, 0], axis=0)
            update_op = tf.assign(w_avg, lerp(batch_avg, w_avg, self.w_ema_decay))

            with tf.control_dependencies([update_op]):
                w_broadcasted = tf.identity(w_broadcasted)

        return w_broadcasted


    def style_mixing_regularization(self, z, w_broadcasted, n_broadcast, resolutions):
        with tf.name_scope('style_mix'):
            z2 = tf.random_normal(tf.shape(z), dtype=tf.float32)
            w_broadcasted2 = self.g_mapping(z2, n_broadcast)
            layer_indices = np.arange(n_broadcast)[np.newaxis, :, np.newaxis]
            last_layer_index = (len(resolutions)) * 2

            mixing_cutoff = tf.cond(tf.random_uniform([], 0.0, 1.0) < self.style_mixing_prob,
                lambda: tf.random_uniform([], 1, last_layer_index, dtype=tf.int32),
                lambda: tf.constant(last_layer_index, dtype=tf.int32))

            w_broadcasted = tf.where(tf.broadcast_to(layer_indices < mixing_cutoff, tf.shape(w_broadcasted)),
                                     w_broadcasted,
                                     w_broadcasted2)
        return w_broadcasted


    def truncation_trick(self, n_broadcast, w_broadcasted, w_avg, truncation_psi):
        with tf.variable_scope('truncation'):
            layer_indices = np.arange(n_broadcast)[np.newaxis, :, np.newaxis]
            ones = np.ones(layer_indices.shape, dtype=np.float32)
            coefs = tf.where(layer_indices < self.truncation_cutoff, truncation_psi * ones, ones)
            w_broadcasted = lerp(w_avg, w_broadcasted, coefs)

        return w_broadcasted

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph """
        if self.phase == 'train':
            self.d_loss_per_res = {}
            self.g_loss_per_res = {}
            self.generator_optim = {}
            self.discriminator_optim = {}
            self.alpha_summary_per_res = {}
            self.d_summary_per_res = {}
            self.g_summary_per_res = {}
            self.train_fake_images = {}

            for res in self.resolutions[self.resolutions.index(self.start_res):]:
                g_loss_per_gpu = []
                d_loss_per_gpu = []
                train_fake_images_per_gpu = []

                batch_size = self.batch_sizes.get(res, self.batch_size_base)
                global_step = tf.get_variable('global_step_{}'.format(res), shape=[], dtype=tf.float32,
                                              initializer=tf.initializers.zeros(),
                                              trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_TOWER)
                alpha_const, zero_constant = get_alpha_const(self.iteration // 2, batch_size * self.gpu_num, global_step)

                # smooth transition variable
                do_train_trans = self.train_with_trans[res]

                alpha = tf.get_variable('alpha_{}'.format(res), shape=[], dtype=tf.float32,
                                        initializer=tf.initializers.ones() if do_train_trans else tf.initializers.zeros(),
                                        trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_TOWER)

                if do_train_trans:
                    alpha_assign_op = tf.assign(alpha, alpha_const)
                else:
                    alpha_assign_op = tf.assign(alpha, zero_constant)

                with tf.control_dependencies([alpha_assign_op]):
                    for gpu_id in range(self.gpu_num):
                        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                                # images
                                gpu_device = '/gpu:{}'.format(gpu_id)
                                image_class = ImageData(res)
                                inputs = tf.data.Dataset.from_tensor_slices(self.dataset)


                                inputs = inputs. \
                                    apply(map_and_batch(image_class.image_processing, batch_size, num_parallel_batches=16, drop_remainder=True)). \
                                    apply(prefetch_to_device(gpu_device, None))
                                    # When using dataset.prefetch, use buffer_size=None to let it detect optimal buffer size

                                inputs_iterator = inputs.make_one_shot_iterator()

                                real_img = inputs_iterator.get_next()
                                z = tf.random_normal(shape=[batch_size, self.z_dim])

                                fake_img = self.generator(z, alpha, res)
                                real_img = smooth_crossfade(real_img, alpha)

                                real_logit = self.discriminator(real_img, alpha, res)
                                fake_logit = self.discriminator(fake_img, alpha, res)

                                # compute loss
                                d_loss, g_loss = compute_loss(real_img, real_logit, fake_logit)

                                d_loss_per_gpu.append(d_loss)
                                g_loss_per_gpu.append(g_loss)
                                train_fake_images_per_gpu.append(fake_img)

                print("Create graph for {} resolution".format(res))

                # prepare appropriate training vars
                d_vars, g_vars = filter_trainable_variables(res)

                d_loss = tf.reduce_mean(d_loss_per_gpu)
                g_loss = tf.reduce_mean(g_loss_per_gpu)

                d_lr = self.d_learning_rates.get(res, self.learning_rate_base)
                g_lr = self.g_learning_rates.get(res, self.learning_rate_base)

                if self.gpu_num == 1 :
                    colocate_grad = False
                else :
                    colocate_grad = True

                d_optim = tf.train.AdamOptimizer(d_lr, beta1=0.0, beta2=0.99, epsilon=1e-8).minimize(d_loss,
                                                                                                     var_list=d_vars,
                                                                                                     colocate_gradients_with_ops=colocate_grad)

                g_optim = tf.train.AdamOptimizer(g_lr, beta1=0.0, beta2=0.99, epsilon=1e-8).minimize(g_loss,
                                                                                                     var_list=g_vars,
                                                                                                     global_step=global_step,
                                                                                                     colocate_gradients_with_ops=colocate_grad)

                self.discriminator_optim[res] = d_optim
                self.generator_optim[res] = g_optim

                self.d_loss_per_res[res] = d_loss
                self.g_loss_per_res[res] = g_loss

                self.train_fake_images[res] = tf.concat(train_fake_images_per_gpu, axis=0)

                """ Summary """
                self.alpha_summary_per_res[res] = tf.summary.scalar("alpha_{}".format(res), alpha)

                self.d_summary_per_res[res] = tf.summary.scalar("d_loss_{}".format(res), self.d_loss_per_res[res])
                self.g_summary_per_res[res] = tf.summary.scalar("g_loss_{}".format(res), self.g_loss_per_res[res])

        else :
            """" Testing """
            test_z = tf.random_normal(shape=[self.batch_size, self.z_dim])
            alpha = tf.constant(0.0, dtype=tf.float32, shape=[])
            self.fake_images = self.generator(test_z, alpha=alpha, target_img_size=self.img_size, is_training=False)
            #self.output_img = tf.placeholder(dtype=tf.float32, shape=[None, self.img_size, self.img_size, 3])
            #self.score = self.discriminator(self.output_img, alpha, self.img_size)
            self.saver = tf.train.Saver()


    @property
    def model_dir(self):

        if self.sn :
            sn = '_sn'
        else :
            sn = ''

        if self.progressive :
            progressive = '_progressive'
        else :
            progressive = ''

        return "{}_{}_{}to{}{}{}".format(self.model_name, self.dataset_name, self.start_res, self.img_size, progressive, sn)


    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)


    def load(self, checkpoint_dir):
        #print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            #print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            #print(" [*] Failed to find a checkpoint")
            return False, 0


    # Find optimized w+ space and noise space (image2stylegan & image2stylegan++)
    def find_z(self, img_x, img_y, M, args):

        # Config
        lambda_mse = args.lambda_mse
        lambda_p = args.lambda_p
        lambda_mse1 = args.lambda_mse1
        lambda_mse2 = args.lambda_mse2
        epoch_w = args.epoch_w
        epoch_n = args.epoch_n

        # Load model and check
        could_load, _ = self.load(self.checkpoint_dir)
        style_model = get_style_model()
        percep_model = get_perceptual_model()

        x = tf.cast(np.array(img_x), tf.float32)
        y = tf.cast(np.array(img_y), tf.float32)
        
        # Some configuration
        resolutions = resolution_list(self.img_size)
        featuremaps = featuremap_list(self.img_size)
        n_broadcast = len(resolutions) * 2

        if self.phase.lower() == 'inpainting':
            M_ = cv2.GaussianBlur(M, (701, 701), 0)
            M_ = np.expand_dims(M_, -1)

        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):

            # Random generate 10000 z codes. Passing through g_mapping, all of the w codes are averaged into one
            z_all = tf.random.uniform(shape=[10000, self.z_dim], minval=-1, maxval=1)
            w = tf.expand_dims(tf.reduce_mean(self.g_mapping(z_all, n_broadcast), axis=0), axis=0)
            w = tf.reshape(w, [1, 18, self.z_dim])

            w_stack = []
            if self.phase.lower() == 'inpainting':
                for i in range(18):
                    if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 17]:
                        constant_w = tf.Variable(w[0, i, :], name=f'project_w_{i}')
                        w_stack.append(constant_w)
                    else:
                        variable_w = tf.Variable(w[0, i, :], trainable=False, name=f'constant_{i}')
                        w_stack.append(variable_w)
                w_init_x = tf.expand_dims(tf.stack(w_stack), axis=0)

            else:
                w_init_x = tf.Variable(w, name='project_w')

            # Find noise space
            x_noise_list = []
            x_noise_list_ = []
            for i in range(len(resolutions)):
                x_noise_list.append(tf.Variable(tf.random.normal([1, resolutions[i], resolutions[i], 1]), name=f'project_noise{i}'))
                x_noise_list_.append(tf.Variable(tf.random.normal([1, resolutions[i], resolutions[i], 1]), name=f'project_noise_{i}'))

            alpha = tf.constant(0.0, dtype=tf.float32, shape=[])

            # Generated image
            x_fake_Wl = self.g_synthesis(w_init_x, alpha, resolutions, featuremaps, x_noise_list, x_noise_list_)[0]
            y_fake_Wl = self.g_synthesis(w_init_x, alpha, resolutions, featuremaps, x_noise_list, x_noise_list_)[0]

            if self.phase.lower() == 'inpainting':
                noise_list = []
                noise_list_ = []
                for i in range(len(resolutions)):
                    noise_list.append(tf.Variable(tf.random.normal([1, resolutions[i], resolutions[i], 1]), trainable=False, name=f'constant_noise{i}'))
                    noise_list_.append(tf.Variable(tf.random.normal([1, resolutions[i], resolutions[i], 1]), trainable=False, name=f'constant_noise{i}'))
                x_fake = self.g_synthesis(w_init_x, alpha, resolutions, featuremaps, noise_list, noise_list_)[0]

        # Variable: w+ space and noise space
        vars_w = [var for var in tf.trainable_variables() if 'project_w' in var.name]
        vars_n = [var for var in tf.trainable_variables() if 'project_noise' in var.name]

        # Lots of losses
        if self.phase.lower() == 'reconstruction':
            mse_loss_x = lambda_mse * cal_loss(x, x_fake_Wl * M)
            percep_loss = Cal_percep_loss(x, x_fake_Wl, M, percep_model, Lambda=lambda_p)
            mse_loss_x_n = lambda_mse * cal_loss(x, x_fake_Wl*M)
            loss_Wl = mse_loss_x + percep_loss
            loss_Mkn = mse_loss_x_n

        elif self.phase.lower() == 'crossover':
            mse_loss_x = lambda_mse * cal_loss(x*M, x_fake_Wl*M)
            mse_loss_y = lambda_mse * cal_loss(y*(1-M), y_fake_Wl*(1-M))
            style_loss = lambda_p * Cal_style_loss(y, y_fake_Wl, (1-M), style_model)
            percep_loss = lambda_p * Cal_percep_loss(x, x_fake_Wl, M, percep_model)
            mse_loss_x_n = lambda_mse1 * cal_loss(x*M, x_fake_Wl*M)
            mse_loss_y_n = lambda_mse2 * cal_loss(y*(1-M), y_fake_Wl*(1-M))
            loss_Wl = mse_loss_x + mse_loss_y + style_loss + percep_loss
            loss_Mkn = mse_loss_x_n + mse_loss_y_n

        elif self.phase.lower() == 'inpainting':
            mse_loss_x = lambda_mse * cal_loss(x*M, x_fake_Wl*M)
            percep_loss = lambda_p * Cal_percep_loss(x, x_fake_Wl, M, percep_model)
            mse_loss_x_n = lambda_mse1 * cal_loss(x*M_, x_fake_Wl*M_)
            mse_loss_y_n = lambda_mse2 * cal_loss(x_fake*(1-M_), x_fake_Wl*(1-M_))
            loss_Wl = mse_loss_x + percep_loss
            loss_Mkn = mse_loss_x_n + mse_loss_y_n

        elif self.phase.lower() == 'style_transfer':
            mse_loss_x = lambda_mse * cal_loss(x, x_fake_Wl)
            percep_loss_x = lambda_p * Cal_percep_loss(x, x_fake_Wl, M, percep_model)
            loss_Wl = mse_loss_x + percep_loss_x
            loss_Mkn = loss_Wl

        # Learning rate (default=0.01)
        optimizer1 = tf.train.AdamOptimizer(learning_rate=0.01)
        optimizer2 = tf.train.AdamOptimizer(learning_rate=5)
        opt_Wl = optimizer1.minimize(loss_Wl, var_list=[vars_w])
        opt_Mkn = optimizer2.minimize(loss_Mkn, var_list=[vars_n])

        tf.global_variables_initializer().run()
        percep_model.load_weights('./checkpoint/vgg_perceptual.h5')
        style_model.load_weights('./checkpoint/vgg_style.h5')
        could_load, _ = self.load(self.checkpoint_dir)

        # Optimization
        if self.phase.lower() != 'style_transfer':
            for i in tqdm(range(epoch_w)):
                self.sess.run([opt_Wl])
            for i in tqdm(range(epoch_n)):
                self.sess.run([opt_Mkn])
        else:
            for i in tqdm(range(epoch_w)):
                self.sess.run([opt_Wl])

        # Get optimized latent code
        w_x, x_opt_noise_fake, x_opt_noise_fake_ = self.sess.run([w_init_x,
                                                                x_noise_list,
                                                                x_noise_list_])
        x_opt_noise = []
        x_opt_noise_ = []
        for i in range(len(resolutions)):
            x_opt_noise.append(x_opt_noise_fake[i])
            x_opt_noise_.append(x_opt_noise_fake_[i])

        return w_x, x_opt_noise, x_opt_noise_


    def reconstruction(self, latent_code, opt_noise, opt_noise_):

        # Some configuration
        resolutions = resolution_list(self.img_size)
        featuremaps = featuremap_list(self.img_size)
        alpha = tf.constant(0.0, dtype=tf.float32, shape=[])

        # Reconstruct image
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            fake_image = self.g_synthesis(np.array(latent_code), alpha, resolutions, featuremaps, opt_noise, opt_noise_)
        sample = self.sess.run(fake_image)
        return sample[0, :, :, :]