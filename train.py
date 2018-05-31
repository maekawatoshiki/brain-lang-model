import collections
import nltk
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
 
    #To check if the SOM has been trained
    _trained = False
 
    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):
        """
        Initializes all necessary components of the TensorFlow
        Graph.
 
        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """

        self.r_min = 0.0
        self.r_max = 0.0
        self.g_min = 0.0
        self.g_max = 0.0
        self.b_min = 0.0
        self.b_max = 0.0
 
        #Assign required variables first
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))
 
        ##INITIALIZE GRAPH
        self._graph = tf.Graph()
 
        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():
 
            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE
 
            #Randomly initialized weightage vectors for all neurons,
            #stored together as a matrix Variable of size [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random_normal(
                [m*n, dim]))
 
            #Matrix of size [m*n, 2] for SOM grid locations
            #of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))
 
            ##PLACEHOLDERS FOR TRAINING INPUTS
            #We need to assign them as attributes to self, since they
            #will be fed in during training
 
            #The training vector
            self._vect_input = tf.placeholder("float", [dim])
            #Iteration number
            self._iter_input = tf.placeholder("float")
 
            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            #Only the final, 'root' training op needs to be assigned as
            #an attribute to self, since all the rest will be executed
            #automatically during training
 
            #To compute the Best Matching Unit given a vector
            #Basically calculates the Euclidean distance between every
            #neuron's weightage vector and the input, and returns the
            #index of the neuron which gives the least value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weightage_vects, tf.stack(
                    [self._vect_input for i in range(m*n)])), 2), 1)),
                                  0)
 
            #This will extract the location of the BMU based on the BMU's
            #index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]))),
                                 [2])
 
            #To compute the alpha and sigma values based on iteration
            #number
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input,
                                                  self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)
 
            #Construct the op that will generate a vector with learning
            #rates for all neurons, based on iteration number and location
            #wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vects, tf.stack(
                    [bmu_loc for i in range(m*n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)
 
            #Finally, the op that will use learning_rate_op to update
            #the weightage vectors of all neurons based on a particular
            #input
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                                               for i in range(m*n)])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(m*n)]),
                       self._weightage_vects))                                         
            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)                                       
 
            ##INITIALIZE SESSION
            self._sess = tf.Session()
 
            ##INITIALIZE VARIABLES
            init_op = tf.initialize_all_variables()
            self._sess.run(init_op)
 
    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        #Nested iterations over both dimensions
        #to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
 
    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """
 
        #Training iterations
        for iter_no in range(self._n_iterations):
            #Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})
 
        #Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            a = self._weightages[i]
            count = 0
            r = 0.0
            g = 0.0
            b = 0.0
            while count < 45:
                r += a[count]
                count += 1
            while count < 90:
                g += a[count]
                count += 1
            while count < 128:
                b += a[count]
                count += 1
            self.r_min = min(self.r_min, r)
            self.r_max = max(self.r_max, r)
            self.g_min = min(self.r_min, g)
            self.g_max = max(self.r_max, g)
            self.b_min = min(self.r_min, b)
            self.b_max = max(self.r_max, b)

            b = [
                    (r + abs(self.r_min)) / (abs(self.r_max) + abs(self.r_min)) * 256,
                    (g + abs(self.g_min)) / (abs(self.g_max) + abs(self.g_min)) * 256,
                    (b + abs(self.b_min)) / (abs(self.b_max) + abs(self.b_min)) * 256,
                ]

            centroid_grid[loc[0]].append(b)
        self._centroid_grid = centroid_grid
 
        self._trained = True
 
    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid
 
    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """
 
        if not self._trained:
            raise ValueError("SOM not trained yet")
 
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect-
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])
 
        return to_return

#For plotting the images
from matplotlib import pyplot as plt
 
#Train SOM with n iterations
som = SOM(60, 60, 128, 4)

# Word embedding
def word_embedding(words):
    vocabulary = collections.Counter(words).most_common()
    vocabulary_dictionary = dict()
    for word, _ in vocabulary:
        # Assign a numerical unique value to each word inside vocabulary 
        vocabulary_dictionary[word] = len(vocabulary_dictionary)
    rev_vocabulary_dictionary = dict(zip(vocabulary_dictionary.values(), vocabulary_dictionary.keys()))
    return vocabulary_dictionary, rev_vocabulary_dictionary


# Build Training data. For example if X = ['long', 'ago', ','] then Y = ['the']
def sampling(words, vocabulary_dictionary, window):
    X = []
    Y = []
    sample = []
    for index in range(0, len(words) - window):
        for i in range(0, window):
            sample.append(vocabulary_dictionary[words[index + i]])
            if (i + 1) % window == 0:
                X.append(sample)
                Y.append(vocabulary_dictionary[words[index + i + 1]])
                sample = []
    return X,Y


with open("data.txt") as f:
    content = f.read()
words = nltk.tokenize.word_tokenize(content)
vocabulary_dictionary, reverse_vocabulary_dictionary = word_embedding(words)

window = 5
num_classes = len(vocabulary_dictionary)
timesteps = window
num_hidden = 128
num_input = 1
batch_size = 10
iteration = 20000


training_data, label = sampling(words, vocabulary_dictionary, window)


# RNN output node weights and biases
weights = {
    'in': tf.Variable(tf.truncated_normal([5,timesteps,1], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'in': tf.Variable(tf.Variable(tf.constant(0.1, shape=[num_classes]))),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# tf graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])


def RNN(x, weights, biases):

    # Unstack to get a list of 'timesteps' tensors, each tensor has shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Build a LSTM cell
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get LSTM cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    # return tf.matmul(outputs[-1], weights['out']) + biases['out']
    return (outputs, states)

outputs, states = RNN(X, weights, biases)

logits = tf.matmul(outputs[-1], weights['out']) + biases['out']

prediction = tf.nn.softmax(logits)

# Loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss_op)
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

shower = tf.argmax(prediction,1)

# Initialize the variables with default values
init = tf.global_variables_initializer()

memory_for_som = []

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for i in range(iteration):
        last_batch = len(training_data) % batch_size
        training_steps = int((len(training_data) / batch_size) + 1)
        for step in range(training_steps):
            X_batch = training_data[(step * batch_size) :((step + 1) * batch_size)]
            Y_batch = label[(step * batch_size) :((step + 1) * batch_size)]
            Y_batch_encoded = []
            for x in Y_batch:
                on_hot_vector = np.zeros([num_classes], dtype=float)
                on_hot_vector[x] = 1.0
                Y_batch_encoded = np.concatenate((Y_batch_encoded,on_hot_vector))
            if len(X_batch) < batch_size:
                X_batch = np.array(X_batch)
                X_batch = X_batch.reshape(last_batch, timesteps, num_input)
                Y_batch_encoded = np.array(Y_batch_encoded)
                Y_batch_encoded = Y_batch_encoded.reshape(last_batch, num_classes)
            else:
                X_batch = np.array(X_batch)
                X_batch = X_batch.reshape(batch_size, timesteps, num_input)
                Y_batch_encoded = np.array(Y_batch_encoded)
                Y_batch_encoded = Y_batch_encoded.reshape(batch_size, num_classes)
            _, acc, loss, onehot_pred = sess.run([train_op, accuracy, loss_op, logits], feed_dict={X: X_batch, Y: Y_batch_encoded})
            state1 = sess.run(states, feed_dict={X: X_batch, Y: Y_batch_encoded})

        som.train([state1[0][0]])

        print("Step " + str(i) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.2f}".format(acc * 100))

        if (i+1) % 5 == 0:
            saver = tf.train.Saver()
            saver.save(sess, 'model' + str(i))

        if (i+1) % 1 == 0:
            image_grid = som.get_centroids()
            plt.imshow(image_grid)
            plt.savefig('/home/groupdeep/Desktop/SOM Figures/figure-' + str(i))
            # plt.show()
            # # som.train(memory_for_som)
            # user_input = input().split(' ')
            # new_batch = []
            # for ui in user_input:
            #     new_batch.append([vocabulary_dictionary[ui]])
            # predicted_word = ""
            # for _ in range(10):
            #     output = sess.run([shower], feed_dict={X: [new_batch], Y: Y_batch_encoded})
            #     if len(output) > 0:
            #         if len(output[0]) > 0:
            #             a = output[0][0]
            #             predicted_word = reverse_vocabulary_dictionary[a % num_classes]
            #             print("predicted: " + predicted_word)
            #             new_batch.pop(0)
            #             new_batch.append([vocabulary_dictionary[predicted_word]])
