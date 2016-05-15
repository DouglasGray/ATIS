import numpy as np


# Sigmoid activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Softmax activation function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


# Shuffle the inputs and labels
def shuffle(x, y):
    z = [[a, b] for a, b in zip(x, y)]
    np.random.shuffle(z)
    return z


# Given a list of word indexes return a list of list of indexes which are the context window
def context_window(s, win):
    assert (win % 2) == 1

    pad = [-1] * (win // 2)
    s = np.concatenate((pad, s, pad))

    return [s[i:i + win] for i in range(len(s) - win + 1)]


# Given a list containing context windows and no steps to backpropagate over return a minibatch for training
def minibatch(cw, bt):
    batch = [cw[max(0, i-bt+1):i+1] for i in range(len(cw))]
    return map(lambda x: np.asarray(x), batch)


# Simple Elman recurrent neural network, three layers with a recurrent hidden layer
class Network:
    def __init__(self, nh, no, nv, de, win):
        self.ni = de * win
        self.nh = nh
        self.no = no
        self.de = de
        self.win = win

        # Set weights and biases
        self.Wh = np.random.randn(nh, nh)
        self.Wx = np.random.randn(nh, self.ni) / np.sqrt(self.ni)
        self.W = np.random.randn(no, nh) / np.sqrt(nh)
        self.bh = np.random.randn(nh, 1)
        self.b = np.random.randn(no, 1)

        # Word embedding matrix to be learned
        self.E = np.random.rand(nv + 1, de)

    # Given a list of context windows form the network inputs using the embedding matrix
    def embed(self, cw):
        return [self.E[w].reshape(self.de * self.win, 1) for w in cw]
        # return self.E[cw].reshape(cw.shape[0], self.de * self.win)

    def train(self, train_x, train_y, val_x, val_y, lr, epochs, bt):
        e = 0
        while e < epochs:
            for s, y in zip(train_x, train_y):
                eta = lr / len(s)

                cw = context_window(s, self.win)
                x = self.embed(cw)

                o, h, z = self.feed_forward(x)

                grad = self.bptt(o, h, z, cw, x, y, bt)

                self.update(grad, eta)

            if val_x and val_y:
                acc = self.test(val_x, val_y)
                print('Accuracy after epoch ' + str(e) + ': ' + "%0.2f" % acc)

            e += 1

    def update_mini_batch(self, train_data, lr, bt):
        for s, y in train_data:
            eta = lr / len(s)

            cw = context_window(s, self.win)
            x = self.embed(cw)

            o, h, z = self.feed_forward(x)

            grad = self.bptt(o, h, z, cw, x, y, bt)

            self.update(grad, eta)

    def test(self, test_x, test_y):
        acc = 0.0
        ntotal = 0.0
        for s, y in zip(test_x, test_y):
            ntotal += len(s)
            cw = context_window(s, self.win)
            x = self.embed(cw)

            o, h, z = self.feed_forward(x)

            for ix, o_t in enumerate(o):
                if o_t.argmax() == y[ix]:
                    acc += 1.0

        return acc / ntotal

    def bptt(self, o, h, z, cw, x, y, bt):
        # Store the gradients
        dWx = np.zeros(self.Wx.shape)
        dWh = np.zeros(self.Wh.shape)
        dW = np.zeros(self.W.shape)
        dE = np.zeros(self.E.shape)

        t_max = len(y)
        for t in range(t_max):
            # Delta at outer layer, use it to find the gradient for the output weights
            delta_o = o[t]
            delta_o[y[t]] -= 1

            dW += np.outer(delta_o, sigmoid(z[t]))

            # Delta of hidden layer at time t
            delta_t = np.dot(self.W.T, delta_o) * sigmoid_prime(z[t])

            # Calculate gradient for the hidden layer, embedding and input weights working backwards through time
            for bs in range(max(0, t - bt), t + 1)[::-1]:
                dWx += np.outer(delta_t, x[bs])

                if bs > 0:
                    dWh += np.outer(delta_t, sigmoid(z[bs - 1]))

                delta_t = np.dot(self.Wh.T, delta_t) * sigmoid_prime(z[bs - 1])

                dE_t = np.dot(self.Wx.T, delta_t) * np.ones((self.ni, 1))

                for ix, w in enumerate(cw[t]):
                    dE_w = dE_t[ix*self.de:(ix+1)*self.de]
                    dE[w, :] += dE_w.reshape((self.de,))

        return {'dWx': dWx, 'dWh': dWh, 'dW': dW, 'dE': dE}

    def update(self, grad, eta):
        self.Wx -= eta * grad['dWx']
        self.Wh -= eta * grad['dWh']
        self.W -= eta * grad['dW']
        self.E -= eta * grad['dE']

        # Normalise the embedding matrix
        self.E = self.E / np.sum(self.E**2, axis=1)[:, np.newaxis]

    def feed_forward(self, inputs):
        os = []
        hs = [np.zeros((self.nh, 1))]
        zs = []
        for x in inputs:
            z = np.dot(self.Wx, x) + np.dot(self.Wh, hs[-1])
            h = sigmoid(z)
            o = softmax(np.dot(self.W, h))
            zs.append(z)
            hs.append(h)
            os.append(o)

        return np.asarray(os), np.asarray(hs), np.asarray(zs)

    @staticmethod
    def cost(a, y):
        return -np.log(a[y])

    @staticmethod
    def delta(a, y):
        return a - y



