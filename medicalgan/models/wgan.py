import pickle
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import RMSprop


class WGAN(Model):
    """
    Creates a WGAN model given a model architecuture.
    Ultilises Wassertein loss and multiple critics (discriminators).
    Model architecture consists of generator and discriminator networks.
    """
    def __init__(self, architecture, img_shape, batch_size=32, z_dim=128, n_critics=5, show_summary=False):
        super(WGAN, self).__init__()
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.n_critics = n_critics
        self.show_summary = show_summary
        self.generator = architecture.generator
        self.discriminator = architecture.discriminator

    def compile(self, gen_opt, dis_opt, loss):
        super(WGAN, self).compile()

        # parameters as per original WGAN paper
        opt = RMSprop(learning_rate=0.00005)
        loss = self.wasserstein_loss

        self.gen_opt = opt
        self.dis_opt = opt
        self.loss = loss
        self.gen_loss = loss # {"gen_loss": loss}
        self.dis_loss = loss # {"dis_loss": loss}

    def wasserstein_loss(self, y_true, y_pred):
        """
        Implements Wasserstein loss function.
        """
        return K.mean(y_true * y_pred)

    def train_step(self, X):
        """
        Goal: 
            Train GAN discriminator and generator networks on image batch.
        Args: 
            X: training image batch.
        Steps:
            a) Train critics (discriminators):
                for n_critics:
                    1. generate batch size random latent spaces.
                    2. input latent spaces to generator to generate images.
                    3. classify batch X and generated images with discriminator.
                    4. calculate discriminator loss and update discriminator weights.
            b) Train generator:
                1. generate batch size latent spaces.
                2. input latent spaces to generator to generate images.
                3. classify generated images with discriminator.
                4. calculate generator loss and update generator weights.
        """

        # batch_size = tf.shape(X)[0]

        for c in range(self.n_critics):

            z0 = tf.random.normal(shape=(self.batch_size, self.z_dim))
            X_gen = self.generator(z0)    
            y_gen = tf.zeros((self.batch_size, 1))
            y_real = tf.ones((self.batch_size, 1))

            # X_all = tf.concat([X_gen, X], axis=0)
            # y_all = tf.concat([tf.ones((self.batch_size, 1)), tf.zeros((self.batch_size, 1))], axis=0)
            # y_all += 0.05 * tf.random.uniform(tf.shape(y_all))

            with tf.GradientTape() as tape:
                gen_preds = self.discriminator(X_gen)
                real_preds = self.discriminator(X)
                gen_dis_loss = self.loss(y_gen, gen_preds)
                real_dis_loss = self.loss(y_real, real_preds)
                dis_loss = 0.5 * tf.add(gen_dis_loss, real_dis_loss)

            dis_grads = tape.gradient(dis_loss, self.discriminator.trainable_weights)
            self.dis_opt.apply_gradients(zip(dis_grads, self.discriminator.trainable_weights))
        
        z1 = tf.random.normal(shape=(self.batch_size, self.z_dim))
        y_gen = tf.zeros((self.batch_size, 1))
        with tf.GradientTape() as tape:
            gen_preds = self.discriminator(self.generator(z1))
            gen_loss = self.loss(y_gen, gen_preds)        
        gen_grads = tape.gradient(gen_loss, self.generator.trainable_weights)
        self.gen_opt.apply_gradients(zip(gen_grads, self.generator.trainable_weights))

        return {"dis_los": dis_loss, "gen_loss": gen_loss}