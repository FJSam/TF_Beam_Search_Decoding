import tensorflow as tf
import numpy as np
import queue
from collections import deque
import operator
from queue import PriorityQueue




## Utility functions
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))  #expected input shape of inception model
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# Create the base pre-trained model
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet') # include_top set to False removes the Softmax which performs image classification - no need for that in this application
new_input = image_model.input  #list of the input tensors of the InceptionV3 model
hidden_layer = (image_model.layers[-1].output) # list of the output tensors of the InceptionV3 model
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# create tokenizer
# Choose the top 10000 words from the vocabulary
top_k = 10000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(your_vocabulary_list)  # creates a word to index dictionary
train_seqs = tokenizer.texts_to_sequences(your_vocabulary_list)
max_length = calc_max_length(train_seqs)

SOS_token = tokenizer.word_index['<start>']
EOS_token = tokenizer.word_index['<end>']
MAX_LENGTH = max_length

# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64

embedding_dim = 256 #output of embedding layer
units = 512 #number of hidden neurons

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(cnn_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                            self.W2(hidden_with_time_axis)))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Base_RNN_Decoder(tf.keras.Model):
    '''
        Illustrative decoder with attention
    '''
    def __init__(self, embedding_dim, units, vocab_size):
        super(Base_RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class CNN_Encoder(tf.keras.Model):
    '''
        Illustrative image encoder
    '''
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

cnn_encoder = CNN_Encoder(embedding_dim)
decoder = Base_RNN_Decoder(embedding_dim, units, vocab_size)


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length,
                 attention_weights):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.attention_weights = attention_weights

    def __lt__(self, other):
        return self.logp < other.logp

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


 def beam_decoding(self, image, beam_width=10):

    '''
    :param image: image to caption
    :return: decoded_batch - predicted sentence list and attention weights per sentence ouput
    :return: attention plot -  attention weights per sentence ouput
    '''
    beam_width = beam_width 
    topk = 1  # number of sentences to generate
    decoder_hiddens = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(load_image(image)[0], 0) # preprocess image
    img_tensor_val_features = image_features_extract_model(temp_input) # extract features from image
    img_tensor_val = tf.reshape(img_tensor_val_features, (img_tensor_val_features.shape[0], -1, img_tensor_val_features.shape[3]))
    encoder_output = encoder(img_tensor_val)
    attention_weights = np.zeros((max_length, attention_features_shape))
    decoded_batch = []


    # LSTM case
    if isinstance(decoder_hiddens, tuple):
        decoder_hidden = (tf.expand_dims( decoder_hiddens[0][:, idx, :], 0),
                            tf.expand_dims( decoder_hiddens[1][:, idx, :], 0))
    # GRU case - default
    else: 
        decoder_hidden = decoder_hiddens

    # Start with the start of the sentence token
    decoder_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

    # Number of sentence to generate
    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))
    
    # starting node -  hidden vector, previous node, word id, logp, length, attention 
    node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1, attention_weights)
    nodes = PriorityQueue()

    # start the queue
    nodes.put((-node.eval(), node))
    qsize = 1

    # start beam search
    while True:
        # give up when decoding takes too long
        if qsize > 3000: break

        # fetch the best node
        score, n = nodes.get()
        decoder_input = n.wordid
        decoder_hidden = n.h
        attention_weights = n.attention_weights

        if n.wordid.numpy()[0] == EOS_token and n.prevNode != None:
            endnodes.append((score, n))
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break
            else:
                continue
    
        # decode for one step using decoder
        decoder_output, decoder_hidden, attention_weights = decoder(decoder_input, encoder_output, decoder_hidden)
        attention_weights = tf.reshape(attention_weights, (-1, )).numpy()

        # PUT HERE REAL BEAM SEARCH OF TOP
        log_prob, indexes = tf.math.top_k(decoder_output, k=beam_width) # similar to argmax, get top 10 words probs
        nextnodes = []

        for new_k in range(beam_width):
            decoded_t = tf.reshape(indexes[0][new_k], (1, -1))
            log_p = log_prob[0][new_k].numpy()

            node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1, attention_weights)
            score = -node.eval()
            nextnodes.append((score, node))

        # put them into queue
        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]
            nodes.put((score, nn))
            # increase qsize
        qsize += len(nextnodes) - 1

    # choose nbest paths, back trace them
    if len(endnodes) == 0:
        endnodes = [nodes.get() for _ in range(topk)]

    utterances = []
    attention_plots = []
    prob_outputs = []
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        utterance = []
        attention_plot = []
        prob_output = []
        utterance.append(n.wordid)
        attention_plot.append(n.attention_weights)
        prob_output.append(decoder_output)

        # back trace
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(n.wordid)
            attention_plot.append(n.attention_weights)
            prob_output.append(decoder_output)

        utterance = utterance[::-1]
        utterances.append(utterance)
        attention_plot = attention_plot[::-1]
        attention_plots.append(attention_plot)
        prob_output = prob_output[::-1]
        prob_outputs.append(prob_output)

    decoded_batch.append(utterances)

    decoded_batch = [tokenizer.index_word[idx[0][0].numpy()] for idx in decoded_batch[0][0]]
    attention_plots = [att for att in attention_plots[0]]
    return decoded_batch, attention_plots



 def greedy_decoding(image, attention_features_shape, technique='ArgMax'):
    attention_plot = np.zeros((max_length, attention_features_shape))
    hidden = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(load_image(image)[0], 0) # preprocess image
    img_tensor_val = image_features_extract_model(temp_input) # extract features from image
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        if technique == 'ArgMax':
            dec_input = tf.expand_dims(tf.argmax(predictions, -1), 0)  
            predicted_id = dec_input.numpy()[0][0]
            result.append(tokenizer.index_word[predicted_id])
        
        elif technique == 'RandSamp':
            predicted_id = tf.random.categorical(predictions, 1, seed=tf.random.set_seed(50)).numpy()[0][0] #seed is set to generate the same caption everytime
            result.append(tokenizer.index_word[predicted_id])
            dec_input = tf.expand_dims([predicted_id], 0)
      
        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


