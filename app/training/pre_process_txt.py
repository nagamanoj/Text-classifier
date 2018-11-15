import numpy as np
import re
import app.training.word_vec as vec


class PreProcess:

    def process_data(self, text, prefered_length):
        processed_data = []
        processed_data.append(self.padd_data(self, self.clean_data(self, text), prefered_length))
        return processed_data

    def clean_data(self, text):
        lower = text.lower()
        lower = re.sub(r"[^a-z ]+", "", lower)
        lower.strip()
        string_array = lower.split(' ')
        final_array = []
        for word in string_array:
            if word:
                final_array.append(vec.Word2Vec.get_vector(vec.Word2Vec, word))

        return final_array

    def padd_data(self, arr, prefered_length):

        if len(arr) < prefered_length:
            print('padding array with zero')
            while len(arr) < prefered_length:
                arr.insert(0, np.zeros(300))
        elif len(arr) > prefered_length:
            print('truncating the array')
            while len(arr) > prefered_length:
                arr.pop(0)

        return arr
