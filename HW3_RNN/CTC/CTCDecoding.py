import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1
        B = y_probs.shape[2]
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        for t in range(y_probs.shape[1]):
            path_prob *= np.max(y_probs[:, t, :])
            decoded_path.append(np.argmax(y_probs[:, t, :]))

        # compress
        compressed = []
        for i in range(len(decoded_path)):
            if decoded_path[i] != blank:
                if i == 0 or decoded_path[i] != decoded_path[i - 1]:
                    compressed.append(decoded_path[i])

        decoded_path = []
        for i in compressed:
            decoded_path.append(self.symbol_set[i - 1])
        
        decoded_path = ''.join(decoded_path)
        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        bestPath, FinalPathScore = None, None
        for t in range(T):
            if t == 0:
                paths = [([], 1)]
            else:
                new_paths = []
                for path, score in paths:
                    for i in range(y_probs.shape[0]):
                        new_path = path + [i]
                        new_score = score * y_probs[i, t, 0]
                        new_paths.append((new_path, new_score))
                new_paths = sorted(new_paths, key=lambda x: x[1], reverse=True)
                paths = new_paths[:self.beam_width]
        bestPath, FinalPathScore = paths[0]

        # compress
        compressed = []
        for i in range(len(bestPath)):
            if bestPath[i] != 0:
                if i == 0 or bestPath[i] != bestPath[i - 1]:
                    compressed.append(bestPath[i])
        bestPath = []
        for i in compressed:
            bestPath.append(self.symbol_set[i - 1])
        bestPath = ''.join(bestPath)

        return bestPath, FinalPathScore

