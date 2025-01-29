# Step 1: Install TensorFlow (if not already installed)
# Run the following command in your terminal or command prompt:
# pip install tensorflow

# Step 2: Import required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import imageio
import os
from tensorflow.keras.datasets import mnist

# Hyperparameters
batch_size = 100
latent_dim = 20
epochs = 50
num_classes = 10
img_dim = 28
filters = 16
intermediate_dim = 256

# Load MNIST dataset
(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape((-1, img_dim, img_dim, 1))
x_test = x_test.reshape((-1, img_dim, img_dim, 1))

# Build the model
x = Input(shape=(img_dim, img_dim, 1))
h = x

for i in range(2):
    filters *= 2
    h = Conv2D(filters=filters, kernel_size=3, strides=2, padding="same")(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(h)
    h = LeakyReLU(0.2)(h)

h_shape = h.shape[1:]  # Use .shape instead of K.int_shape
h = Flatten()(h)
z_mean = Dense(latent_dim)(h)  # Mean of p(z|x)
z_log_var = Dense(latent_dim)(h)  # Variance of p(z|x)

encoder = Model(x, z_mean)  # Encoder model

# Decoder
z = Input(shape=(latent_dim,))
h = z
h = Dense(np.prod(h_shape))(h)
h = Reshape(h_shape)(h)

for i in range(2):
    h = Conv2DTranspose(filters=filters, kernel_size=3, strides=1, padding="same")(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding="same")(h)
    h = LeakyReLU(0.2)(h)
    filters //= 2

x_recon = Conv2DTranspose(
    filters=1, kernel_size=3, activation="sigmoid", padding="same"
)(h)

decoder = Model(z, x_recon)  # Decoder model
generator = decoder

# Classifier
z = Input(shape=(latent_dim,))
y = Dense(intermediate_dim, activation="relu")(z)
y = Dense(num_classes, activation="softmax")(y)

classifier = Model(z, y)  # Classifier model


# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(z_log_var / 2) * epsilon


# Reparameterization layer
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
x_recon = decoder(z)
y = classifier(z)


# Gaussian layer
class Gaussian(Layer):
    """A simple layer to define the mean parameters for q(z|y)."""

    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(Gaussian, self).__init__(**kwargs)

    def build(self, input_shape):
        latent_dim = input_shape[-1]
        self.mean = self.add_weight(
            name="mean", shape=(self.num_classes, latent_dim), initializer="zeros"
        )

    def call(self, inputs):
        z = inputs  # z.shape = (batch_size, latent_dim)
        z = tf.expand_dims(z, 1)  # Expand dims for broadcasting
        return z - tf.expand_dims(self.mean, 0)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes, input_shape[-1])


gaussian = Gaussian(num_classes)
z_prior_mean = gaussian(z)

# Build the VAE model
vae = Model(x, [x_recon, z_prior_mean, y])

# Define the loss function
lamb = 2.5  # Weight for reconstruction loss
xent_loss = 0.5 * tf.reduce_mean((x - x_recon) ** 2, axis=[1, 2, 3])
kl_loss = -0.5 * (z_log_var - tf.square(z_prior_mean))
kl_loss = tf.reduce_mean(
    tf.einsum("bij,bjk->bik", tf.expand_dims(y, 1), kl_loss), axis=0
)
cat_loss = tf.reduce_mean(y * tf.math.log(y + K.epsilon()), axis=1)
vae_loss = (
    lamb * tf.reduce_sum(xent_loss) + tf.reduce_sum(kl_loss) + tf.reduce_sum(cat_loss)
)

vae.add_loss(vae_loss)
vae.compile(optimizer="adam")
vae.summary()

# Train the model
vae.fit(
    x_train,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, None),
)

# Evaluate the model
means = gaussian.mean.numpy()
x_train_encoded = encoder.predict(x_train)
y_train_pred = classifier.predict(x_train_encoded).argmax(axis=1)
x_test_encoded = encoder.predict(x_test)
y_test_pred = classifier.predict(x_test_encoded).argmax(axis=1)


# Helper functions for visualization
def cluster_sample(path, category=0):
    """Visualize samples clustered into the same class."""
    n = 8
    figure = np.zeros((img_dim * n, img_dim * n))
    idxs = np.where(y_train_pred == category)[0]
    for i in range(n):
        for j in range(n):
            digit = x_train[np.random.choice(idxs)]
            digit = digit.reshape((img_dim, img_dim))
            figure[i * img_dim : (i + 1) * img_dim, j * img_dim : (j + 1) * img_dim] = (
                digit
            )
    imageio.imwrite(path, figure * 255)


def random_sample(path, category=0, std=1):
    """Generate random samples conditioned on the clustering result."""
    n = 8
    figure = np.zeros((img_dim * n, img_dim * n))
    for i in range(n):
        for j in range(n):
            noise_shape = (1, latent_dim)
            z_sample = np.random.randn(*noise_shape) * std + means[category]
            x_recon = generator.predict(z_sample)
            digit = x_recon[0].reshape((img_dim, img_dim))
            figure[i * img_dim : (i + 1) * img_dim, j * img_dim : (j + 1) * img_dim] = (
                digit
            )
    imageio.imwrite(path, figure * 255)


# Save samples
if not os.path.exists("samples"):
    os.mkdir("samples")

for i in range(10):
    cluster_sample(f"samples/cluster_class_{i}.png", i)
    random_sample(f"samples/random_sample_class_{i}.png", i)


# Calculate accuracy
def calculate_accuracy(y_true, y_pred):
    right = 0
    for i in range(num_classes):
        _ = np.bincount(y_true[y_pred == i])
        right += _.max()
    return right / len(y_true)


train_acc = calculate_accuracy(y_train_, y_train_pred)
test_acc = calculate_accuracy(y_test_, y_test_pred)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
