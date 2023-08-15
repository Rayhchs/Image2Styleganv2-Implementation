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
        self.gpu_id = args.gpu_id
        self.sess = sess
        self.dataset_name = args.dataset
        self.data_folder = args.test_folder
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

        self.gpu_num = 1#args.gpu_num

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

        if args.phase.lower() != 'augmentation':
            print()

            print("##### Information #####")
            print("# dataset : ", self.dataset_name)
            print("# gpu : ", self.gpu_num)
            print("# batch_size in train phase : ", self.batch_sizes)
            print("# batch_size in test phase : ", self.batch_size)

            print("# start resolution : ", self.start_res)
            print("# target resolution : ", self.img_size)
            print("# iteration per resolution : ", self.iteration)

            print("# progressive training : ", self.progressive)
            print("# spectral normalization : ", self.sn)

            print()

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


    # Find optimized w space (image2stylegan++)
    def find_z(self, img_x, img_y, M, lambdax=1, lambday=1, vgg_size=128, epoch=1000):

        # Load model and check
        could_load, _ = self.load(self.checkpoint_dir)
        # if not could_load:
        #     sys.exit(" [*] Checkpoint Load Failed")

        x = tf.cast(np.array(img_x), tf.float32)
        y = tf.cast(np.array(img_y), tf.float32)
        
        # Some configuration
        resolutions = resolution_list(self.img_size)
        featuremaps = featuremap_list(self.img_size)
        n_broadcast = len(resolutions) * 2

        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):

            # Create style and perceptual model
            style_model = get_style_model()
            percep_model = get_perceptual_model()#[get_vgg_conv1_1(), get_vgg_conv1_2(), get_vgg_conv2_2(), get_vgg_conv3_3()]

            # Random generate 20000 z codes. Passing through g_mapping, all of the w codes are averaged into one
            z_all = tf.random.normal(shape=[10000, self.z_dim])
            w = tf.expand_dims(tf.reduce_mean(self.g_mapping(z_all, n_broadcast), axis=0), axis=0)
            w = tf.reshape(w, [1, 18, self.z_dim])

            # initial w code
            w_init_x = tf.Variable(w, name='project_w_x')
            w_init_y = tf.Variable(w, name='project_w_y')

            # Find noise space
            x_noise_list = []
            x_noise_list_ = []
            y_noise_list = []
            y_noise_list_ = []
            for i in range(len(resolutions)):
                x_noise_list.append(tf.Variable(tf.random_normal([1, resolutions[i], resolutions[i], 1]), name=f'xproject_noise{i}'))
                x_noise_list_.append(tf.Variable(tf.random_normal([1, resolutions[i], resolutions[i], 1]), name=f'xproject_noise_{i}'))
                y_noise_list.append(tf.Variable(tf.random_normal([1, resolutions[i], resolutions[i], 1]), name=f'yproject_noise{i}'))
                y_noise_list_.append(tf.Variable(tf.random_normal([1, resolutions[i], resolutions[i], 1]), name=f'yproject_noise_{i}'))

            alpha = tf.constant(0.0, dtype=tf.float32, shape=[])

            # Generated image
            x_fake = self.g_synthesis(w_init_x, alpha, resolutions, featuremaps, x_noise_list, x_noise_list_)[0]
            y_fake = self.g_synthesis(w_init_y, alpha, resolutions, featuremaps, y_noise_list, y_noise_list_)[0]

        # Variable: w+ space and noise space
        vars_w = [var for var in tf.trainable_variables() if 'project_w' in var.name]
        vars_n = [var for var in tf.trainable_variables() if 'project_noise' in var.name]

        # Lots of losses
        mse_loss_x = lambdax * self.cal_loss(x, x_fake * M)
        mse_loss_y = lambday * self.cal_loss(y, y_fake * (1-M))
        style_loss = Cal_style_loss(y, y_fake, M, style_model, Lambda=1)
        percep_loss = Cal_percep_loss(x, x_fake, M, percep_model, Lambda=1)
        loss = mse_loss_x + mse_loss_y + style_loss + percep_loss

        # Learning rate (default=0.01)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        opt = optimizer.minimize(loss, var_list=[vars_w, vars_n])

        tf.global_variables_initializer().run()
        could_load, _ = self.load(self.checkpoint_dir)

        # Optimization
        for i in tqdm(range(epoch)):
            with tf.xla.experimental.jit_scope():
                self.sess.run([opt])

        # Get optimized latent code
        w_x, x_opt_noise_fake, x_opt_noise_fake_, w_y, y_opt_noise_fake, y_opt_noise_fake_ = self.sess.run([w_init_x,
                                                                                                            x_noise_list,
                                                                                                            x_noise_list_,
                                                                                                            w_init_y,
                                                                                                            y_noise_list, 
                                                                                                            y_noise_list_])

        x_opt_noise = []
        x_opt_noise_ = []
        y_opt_noise = []
        y_opt_noise_ = []
        for i in range(len(resolutions)):
            x_opt_noise.append(x_opt_noise_fake[i])
            x_opt_noise_.append(x_opt_noise_fake_[i])
            y_opt_noise.append(y_opt_noise_fake[i])
            y_opt_noise_.append(y_opt_noise_fake_[i])


        return w_x, x_opt_noise, x_opt_noise_


    @tf.function
    def cal_loss(self, y, y_):
        return tf.reduce_mean(tf.pow(y_- y,2))


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


    def pose_transition(self, latent_code, latent_code_flip, opt_noise, opt_noise_flip, opt_noise_, opt_noise_flip_, name, interpolate_num=10):

        # Some configuration
        resolutions = resolution_list(self.img_size)
        featuremaps = featuremap_list(self.img_size)
        alpha = tf.constant(0.0, dtype=tf.float32, shape=[])

        # Interpolate latent code and noise space
        new_latents = []
        names = []
        for j in range(1, interpolate_num+1):

            r1 = j/interpolate_num
            r2 = 1 - (j/interpolate_num)
            new_latent = (latent_code * r1) + (latent_code_flip * r2)
            new_latents.extend(new_latent)
            for n in name:
                names.append('{}_p{}.jpg'.format(n, j))

        new_noise = []
        new_noise_ = []
        for k in range(len(opt_noise)):
            temp = []
            temp_ = []
            for j in range(1, interpolate_num+1):

                r1 = j/interpolate_num
                r2 = 1 - (j/interpolate_num)
                temp.extend((opt_noise[k] * r1) + (opt_noise_flip[k] * r2))
                temp_.extend((opt_noise_[k] * r1) + (opt_noise_flip_[k] * r2))
            new_noise.append(np.array(temp))
            new_noise_.append(np.array(temp_))

        #tf.global_variables_initializer().run()
        #could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            fake_images = self.g_synthesis(np.array(new_latents), alpha, resolutions, featuremaps, new_noise, new_noise_)
        samples = self.sess.run(fake_images)
        # Filter out some inappropriate images using stylegan discriminator
        #score = self.sess.run(self.score, feed_dict={self.output_img: samples})

        # -0.25 is experience value
        for k in range(len(names)):
            if True:#score[k,:] > -0.25:
                save_images(samples[k:k+1, :, :, :], [1, 1], names[k])