import json

class LabelEncoder:
    def __init__(self):
        self.mapping = {}      # Text -> Number
        self.reverse_mapping = {} # Number -> Text

    def fit(self, data):
        unique_values = sorted(list(set(data)))
        for index, value in enumerate(unique_values):
            self.mapping[value] = index
            self.reverse_mapping[index] = value
        return self

    def transform(self, data):
        return [self.mapping[v] for v in data]

    def inverse_transform(self, data):
        return [self.reverse_mapping[v] for v in data]

    def save(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({'mapping': self.mapping, 'reverse_mapping': self.reverse_mapping}, f)

    def load(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.mapping = data['mapping']
            self.reverse_mapping = data['reverse_mapping']
        return self