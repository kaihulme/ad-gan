import pickle
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.metrics import Mean


class GAN(Model):
    """
    Creates a GAN model given a model architecuture.
    Model architecture consists of generator and discriminator networks.
    """
    def __init__(self, architecture, img_shape, batch_size=32, z_dim=128, show_summary=False):
        super(GAN, self).__init__()
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.show_summary = show_summary
        self.generator = architecture.generator
        self.discriminator = architecture.discriminator

    def compile(self, gen_opt, dis_opt, loss):
        super(GAN, self).compile()
        self.gen_opt = gen_opt
        self.dis_opt = dis_opt
        self.loss = loss
        self.gen_loss = Mean(name="gen_loss")
        self.dis_loss = Mean(name="dis_los")

    def train_step(self, X):
        """
        Goal: 
            Train GAN discriminator and generator networks on image batch.
        Args: 
            X: training image batch.
        Steps:
            a) Train discriminator:
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
        batch_size = tf.shape(X)[0]
        z0 = tf.random.normal(shape=(batch_size, self.z_dim))
        X_gen = self.generator(z0)    
        X_all = tf.concat([X_gen, X], axis=0)
        y_all = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        y_all += 0.05 * tf.random.uniform(tf.shape(y_all))
        with tf.GradientTape() as tape:
            preds = self.discriminator(X_all)
            dis_loss = self.loss(y_all, preds)        
        grads = tape.gradient(dis_loss, self.discriminator.trainable_weights)
        self.dis_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        z1 = tf.random.normal(shape=(batch_size, self.z_dim))
        y_gen = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            preds = self.discriminator(self.generator(z1))
            gen_loss = self.loss(y_gen, preds)        
        grads = tape.gradient(gen_loss, self.generator.trainable_weights)
        self.gen_opt.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        self.dis_loss.update_state(dis_loss)
        self.gen_loss.update_state(gen_loss)
        return {"dis_los": self.dis_loss.result(), "gen_loss": self.gen_loss.result()}