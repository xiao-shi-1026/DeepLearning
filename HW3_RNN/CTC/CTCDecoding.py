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

        blank = 0
        T = y_probs.shape[1]
        bestPath, FinalPathScore = None, None
        
        # initialize beam -> list of lists e.g. [[2], [1]]
        # Each lists inside the list is a path. Each number represents the index of the symbol in the symbol_set
        path = [[i] for i in np.argsort(y_probs[:, 0, :].flatten())[::-1][:self.beam_width]]

        # initialize probability for each path -> list of lists
        # Each lists inside the list is a prob of that path. The index of prob is the same as the index of the path

        prob = [i for i in np.sort(y_probs[:, 0, :].flatten())[::-1][:self.beam_width]]

        # update the path and prob for each time step
        for t in range(1, T):
            # new path could only be the highest beam_width th. Use the same method to initialize new_path
            new_path = []
            for i in range(len(path)):
                for j in range(len(y_probs[:, t, :].flatten())):
                    copypath = path[i].copy()
                    copypath.append(j)
                    new_path.append(copypath)
            new_prob = [i * j for i in prob for j in y_probs[:, t, :].flatten()]

            # compress the path. The sequence of if loop is important, as we don't want to
            # delete blank in the middle of the path but only delete the last blank.
            for tmp_path in new_path:
                
                if len(tmp_path) >= 2 and tmp_path[-2] == blank:
                    tmp_path.pop(-2)
                    continue
                
                if t == T - 1 and tmp_path[-1] == blank:
                    tmp_path.pop()
                    continue

                if tmp_path[-1] == tmp_path[-2]:
                    tmp_path.pop()
                    continue

                

            path_dict = {}
            
            for i in range(len(new_path)):
                if tuple(new_path[i].copy()) not in path_dict:
                    path_dict[tuple(new_path[i].copy())] = new_prob[i].copy()
                else:
                    path_dict[tuple(new_path[i].copy())] += new_prob[i].copy()
            before_prune = path_dict.copy()

            # return top beam_width paths
            path = []
            prob = []
            for i in range(self.beam_width):
                max_key = max(path_dict, key=path_dict.get)
                path.append(list(max_key))
                prob.append(path_dict[max_key])
                del path_dict[max_key]
            
        bestPath = path[np.argmax(prob)]
        
        # delete 0 in bestPath
        bestPath = [i for i in bestPath if i != 0]
        bestPath = [self.symbol_set[i - 1] for i in bestPath]
        bestPath = ''.join(bestPath)
       
        # change the key of finalpathscore to string
        FinalPathScore = {}

        for key, value in before_prune.items():
            newkey = [self.symbol_set[i - 1] if i != 0 else '' for i in key] # The HW3 requires the blank to be ''
            newkey = ''.join(newkey)
            FinalPathScore[newkey] = value

        return bestPath, FinalPathScore

