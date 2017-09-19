# Language Translation - English to French

This project is in the realm of neural network machine translation. I'll be training a sequence to sequence model on a dataset of English and French sentences that can translate new sentences from English to French.

The training data is located in `data/small_vocab_en` & `data/small_vocab_fr`. 

Here are some Dataset Stats: 

```
Roughly the number of unique words: 227

Number of sentences: 137861

Average number of words in a sentence: 13.225
```

This project was built using Floydhub:
`floyd run --env tensorflow-1.1.0 --gpu  --mode jupyter`

Here's a step-by-step algorithm to build a Sequence-to-Sequence model by implementing the following functions below:

+ **model_inputs**: Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences

	+ Input text placeholder named "input" using the TF Placeholder name parameter with rank 2.
	+ Targets placeholder with rank 2.
	+ Learning rate placeholder with rank 0.
	+ Keep probability placeholder named "keep_prob" using the TF Placeholder name parameter with rank 0.
	+ Target sequence length placeholder named "target_sequence_length" with rank 1
	+ Max target sequence length tensor named "max_target_len" getting its value from applying tf.reduce_max on the target_sequence_length placeholder. Rank 0.
	+ Source sequence length placeholder named "source_sequence_length" with rank 1

+ **process_decoder_input**: Preprocess target data for encoding
	+ Remove the last word id from each batch in target_data and concat the GO ID to the begining of each batch.

+ **encoding_layer**: Implement encoding_layer() to create a Encoder RNN layer:
	+ Embed the encoder input using tf.contrib.layers.embed_sequence
	+ Construct a stacked tf.contrib.rnn.LSTMCell wrapped in a tf.contrib.rnn.DropoutWrapper
	+ Pass cell and embedded input to tf.nn.dynamic_rnn()

+ **decoding_layer_train**: Create a decoding layer for training

	+ Create a tf.contrib.seq2seq.TrainingHelper
	+ Create a tf.contrib.seq2seq.BasicDecoder
	+ Obtain the decoder outputs from tf.contrib.seq2seq.dynamic_decode
  
+ **decoding_layer_infer**: Create inference decoder

	+ Create a tf.contrib.seq2seq.GreedyEmbeddingHelper
	+ Create a tf.contrib.seq2seq.BasicDecoder
	+ Obtain the decoder outputs from tf.contrib.seq2seq.dynamic_decode

+ **decoding_layer**: Implement decoding_layer() to create a Decoder RNN layer.

	+ Embed the target sequences
	+ Construct the decoder LSTM cell (just like you constructed the encoder cell above)
	+ Create an output layer to map the outputs of the decoder to the elements of our vocabulary
	+ Use the your decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length, max_target_sequence_length, output_layer, keep_prob) function to get the training logits.
	+ Use your decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, max_target_sequence_length, vocab_size, output_layer, batch_size, keep_prob) function to get the inference logits.

+ **seq2seq_model**: Build the Sequence-to-Sequence part of the neural network

	+ Encode the input using your encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,  source_sequence_length, source_vocab_size, encoding_embedding_size).
	+ Process target data using your process_decoder_input(target_data, target_vocab_to_int, batch_size) function.
	+ Decode the encoded input using your decoding_layer(dec_input, enc_state, target_sequence_length, max_target_sentence_length, rnn_size, num_layers, target_vocab_to_int, target_vocab_size, batch_size, keep_prob, dec_embedding_size) function.

### Tuning Hyperparameters

Tune the following parameters:

```
Set epochs to the number of epochs.
Set batch_size to the batch size.
Set rnn_size to the size of the RNNs.
Set num_layers to the number of layers.
Set encoding_embedding_size to the size of the embedding for the encoder.
Set decoding_embedding_size to the size of the embedding for the decoder.
Set learning_rate to the learning rate.
Set keep_probability to the Dropout keep probability
Set display_step to state how many steps between each debug output statement

# Number of Epochs
epochs = 12
# Batch Size
batch_size = 512
# RNN Size
rnn_size = 256
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 227
decoding_embedding_size = 227
# Learning Rate
learning_rate = .005
# Dropout Keep Probability
keep_probability = .75
display_step = 10
```

### Build the graph, batch and pad the source and target sequences and train...


### Model Training Results
```

Epoch   0 Batch   10/269 - Train Accuracy: 0.3357, Validation Accuracy: 0.3982, Loss: 3.0391
Epoch   0 Batch   20/269 - Train Accuracy: 0.4101, Validation Accuracy: 0.4611, Loss: 2.5589
...
Epoch  11 Batch  220/269 - Train Accuracy: 0.9872, Validation Accuracy: 0.9727, Loss: 0.0098
Epoch  11 Batch  230/269 - Train Accuracy: 0.9862, Validation Accuracy: 0.9687, Loss: 0.0104
Epoch  11 Batch  240/269 - Train Accuracy: 0.9861, Validation Accuracy: 0.9780, Loss: 0.0083
Epoch  11 Batch  250/269 - Train Accuracy: 0.9874, Validation Accuracy: 0.9763, Loss: 0.0088
Epoch  11 Batch  260/269 - Train Accuracy: 0.9939, Validation Accuracy: 0.9802, Loss: 0.0089
Model Trained and Saved
```

### Translate 

This will translate translate_sentence from English to French.

`translate_sentence = 'he saw a old yellow truck .'`

```
INFO:tensorflow:Restoring parameters from checkpoints/dev
Input
  Word Ids:      [143, 38, 214, 230, 227, 90, 223]
  English Words: ['he', 'saw', 'a', 'old', 'yellow', 'truck', '.']

Prediction
  Word Ids:      [199, 354, 74, 126, 118, 162, 0, 0, 0, 0, 0, 0, 0, 0]
  French Words: il a vu un vieux camion <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
```