import numpy as np
from typing import List, Union

from encoding.base import BaseEncoding


class OneHotEncoding(BaseEncoding):
    """
    One-hot encoding implementation for sequences.

    This class provides methods to encode sequences into one-hot vectors and
    decode one-hot vectors back into sequences.
    """

    def encode(self, sequences: List[str]) -> np.ndarray:
        """
        Encode a list of sequences into one-hot vectors.

        Args:
            sequences (List[str]): List of sequences to encode.

        Returns:
            np.ndarray: Encoded sequences as a 2D numpy array.
        """
        max_length = max(len(seq) for seq in sequences)
        encoded = np.zeros((len(sequences), max_length, len(self.config.alphabet)))
        for i, seq in enumerate(sequences):
            for j, char in enumerate(seq):
                if char in self.config.alphabet:
                    encoded[i, j, self.config.alphabet.index(char)] = 1
        return encoded.reshape(len(sequences), -1)

    def decode(self, encoded_sequences: Union[List[np.ndarray], np.ndarray]) -> List[str]:
        """
        Decode one-hot encoded sequences back into string sequences.

        Args:
            encoded_sequences (Union[List[np.ndarray], np.ndarray]): Encoded sequences to decode.

        Returns:
            List[str]: Decoded sequences as a list of strings.
        """
        if isinstance(encoded_sequences, list):
            encoded_sequences = np.array(encoded_sequences)
        if encoded_sequences.ndim == 1:
            encoded_sequences = encoded_sequences.reshape(1, -1)
        seq_length = encoded_sequences.shape[1] // len(self.config.alphabet)
        decoded = []
        for seq in encoded_sequences:
            seq_2d = seq.reshape(seq_length, len(self.config.alphabet))
            decoded_seq = ''.join(self.config.alphabet[np.argmax(seq_2d[i])] for i in range(seq_length))
            decoded.append(decoded_seq.rstrip())  # Remove trailing spaces
        return decoded
