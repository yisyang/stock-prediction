import torch


class SimpleLstm(torch.nn.Module):
    def __init__(self, x_seq_length, x_features, y_seq_length, n_hidden=24, device=None):
        super().__init__()
        self.x_seq_length = x_seq_length
        self.x_features = x_features
        self.y_seq_length = y_seq_length
        self.n_hidden = n_hidden  # Number of neurons in hidden layer
        self.batch_size = None  # Automatically calculated in forward
        self.n_lstm_layers = 1
        self.device = device or torch.device('cpu')

        # Out = (batch_size, x_seq_length, n_hidden)
        self.l_lstm = torch.nn.LSTM(input_size=x_features,
                                    hidden_size=self.n_hidden,
                                    num_layers=self.n_lstm_layers,
                                    batch_first=True)

        self.l_dropout = torch.nn.Dropout(p=0.3)

        # Out = (batch_size, y_seq_length)
        self.l_linear = torch.nn.Linear(self.n_hidden * self.x_seq_length, self.y_seq_length)

        # Send to GPU if needed.
        self.to(device=self.device)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(params=self.parameters(), lr=0.001, weight_decay=0.005)

    def init_hidden(self):
        # For lack of better knowledge let's start with zeros.
        hidden_state = torch.zeros(self.n_lstm_layers, self.batch_size, self.n_hidden, device=self.device)
        cell_state = torch.zeros(self.n_lstm_layers, self.batch_size, self.n_hidden, device=self.device)
        return hidden_state, cell_state

    def forward(self, x, hidden_state=None):
        # Set batch_size based on input
        self.batch_size = len(x)
        if hidden_state is None:
            hidden_state = self.init_hidden()
        # print(x[:2])
        lstm_out, hidden_state_next = self.l_lstm(x, hidden_state)
        # print('lstm_out shape', lstm_out.shape)  # [25, 30, 24]
        # print(lstm_out[:2])
        # res = lstm_out.contiguous().view(self.batch_size, -1)
        res = lstm_out.reshape(self.batch_size, self.x_seq_length * self.n_hidden)
        # print('res shape', res.shape)  # [25, 720]
        out = self.l_linear(res)
        # print('out shape', out.shape)  # [25, 5]
        return out, hidden_state_next

    def get_loss(self, pred, actual):
        # print(pred.shape)
        # print(actual.shape)
        loss = self.criterion(pred, actual)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
