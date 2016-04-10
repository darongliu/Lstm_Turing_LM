import cPickle
import numpy as np

### Utilities:
class Vocab:
    __slots__ = ["word2index", "index2word", "unknown"]
    
    def __init__(self, index2word = None):
        self.word2index = {}
        self.index2word = []
        
        # add unknown word:
        self.add_words(["**UNKNOWN**"])
        self.unknown = 0
        
        if index2word is not None:
            self.add_words(index2word)
                
    def add_words(self, words):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.word2index)
                self.index2word.append(word)
                       
    def __call__(self, line):
        """
        Convert from numerical representation to words
        and vice-versa.
        """
        if type(line) is np.ndarray:
            return " ".join([self.index2word[word] for word in line])
        if type(line) is list:
            if len(line) > 0:
                if line[0] is int:
                    return " ".join([self.index2word[word] for word in line])
            indices = np.zeros(len(line), dtype=np.int32)
        else:
            line = line.split(" ")
            indices = np.zeros(len(line), dtype=np.int32)
        
        for i, word in enumerate(line):
            indices[i] = self.word2index.get(word, self.unknown)
            
        return indices
    
    @property
    def size(self):
        return len(self.index2word)
    
    def __len__(self):
        return len(self.index2word)


def pad_into_matrix(rows, padding = 0):
    if len(rows) == 0:
        return np.array([0, 0], dtype=np.int32)
    lengths = map(len, rows)
    width = max(lengths)
    height = len(rows)
    mat = np.empty([height, width], dtype=rows[0].dtype)
    mat.fill(padding)
    for i, row in enumerate(rows):
        mat[i, 0:len(row)] = row
    return mat, list(lengths) #can add a mask


if __name__ == "__main__":

	# get sample data for vocab test
	lines = cPickle.load( open("sample_data.p","rb"))

	# build vocab
	vocab = Vocab()
	for line in lines:
		vocab.add_words(line.split(" "))

	# transform into big numerical matrix of sentences:
	numerical_lines = []
	for line in lines:
		numerical_lines.append(vocab(line))
	numerical_lines, numerical_lengths = pad_into_matrix(numerical_lines)

	print vocab.word2index
	print lines[10]
	print numerical_lines[10]	



