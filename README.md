# TF_Beam_Search_Decoding for image captioning

Tensorflow implementation of beam search decoding for seq2seq image captioning models based on https://github.com/budzianowski/PyTorch-Beam-Search-Decoding. Decoding goes seperately for each sentence and stores the nodes in prioritized queue.

Usage: You can specify additional reward for decoding through BeamSearchNode.eval. Works for model with and without attention.
