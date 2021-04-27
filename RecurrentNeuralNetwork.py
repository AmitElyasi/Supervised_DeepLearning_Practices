from collections import Counter
from torch.utils.data import DataLoader
from utils import *


class Model(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, vocab_size, dropout, model_type, device):
        super(Model, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.model_type = model_type
        self.device = device

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
        )
        if model_type == "lstm":
            self.lstm = nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=self.hidden_size,
                dropout= self.dropout,
                batch_first=True
            )
        if model_type == "rnn":
            self.rnn = nn.RNN(
                input_size=self.embedding_size,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                batch_first=True
            )
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        if self.model_type == "lstm":
            output, state = self.lstm(embed, prev_state)
            logits = self.fc(output)
            return logits, state
        if self.model_type == "rnn":
            output, state = self.rnn(embed, prev_state)
            logits = self.fc(output)
            return logits, state


    def init_state(self, chars, index):
        return (
            torch.tensor(chars[index:index + self.hidden_size]),
            torch.tensor(chars[index + 1:index + self.hidden_size + 1]),
        )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab_path, train_data_path, sequence_length):
        self.vocab = self.load_vocab(vocab_path)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.chars = self.load_chars(train_data_path)
        self.sequence_length = sequence_length

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
            return vocab

    def load_chars(self, train_data_path):
        train_data = torch.load(train_data_path)
        return train_data.tolist()

    def get_unique_chars(self):
        char_counts = Counter(self.chars)
        return sorted(char_counts, key=char_counts.get, reverse=True)

    def __len__(self):
        return len(self.chars) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.chars[index:index + self.sequence_length]),
            torch.tensor(self.chars[index+1:index + self.sequence_length + 1]),
        )



def train(model_type):
    embedding_size = 100
    hidden_size = 100
    num_layers = 1
    dropout = 0.005
    sequence_length = 32
    batch_size = 1

    device = get_training_device(cuda_num=1)
    dataset = Dataset("./q4_data/vocab.pkl", "./q4_data/shakespeare_data.pt", sequence_length)
    vocab_size = len(dataset.vocab)

    model = Model(
        embedding_size,
        hidden_size,
        num_layers,
        vocab_size,
        dropout,
        model_type,
        device
    ).to(device)
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    state_h, state_c = model.init_state(dataset.chars, 0)

    for batch, (x, y) in enumerate(dataloader):
        print(x.shape)
        print(y.shape)
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        y_pred_t = y_pred.transpose(1, 2)

        symbol_tensor = torch.argmax(y_pred_t, axis=1)
        symbol_list = [t.item() for t in symbol_tensor[0]]
        decoded = [dataset.reverse_vocab[t] for t in symbol_list]
        print("".join(decoded))

        loss = criterion(y_pred_t, y)

        state_h = state_h.detach()
        state_c = state_c.detach()

        loss.backward()
        optimizer.step()

        print({'batch': batch, 'loss': loss.item()})



print("Running LSTM:")
train("lstm")
print("Running RNN:")
train("rnn")