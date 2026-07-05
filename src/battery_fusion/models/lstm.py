import torch
from torch import nn
from torch.nn import functional as F


class RdfLSTMRegressor(nn.Module):
    """Legacy RDF sequence encoder from the original single-modal script."""

    def __init__(
        self,
        input_size: int = 400,
        hidden_size: int = 256,
        output_size: int = 1,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        if input_dim is not None and input_size == 400:
            input_size = input_dim
        if hidden_dim is not None and hidden_size == 256:
            hidden_size = hidden_dim
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, rdf: torch.Tensor, hidden=None) -> torch.Tensor:
        if hidden is None:
            hidden = self.init_hidden(1, device=rdf.device)
        output, _hidden = self.lstm_cell(rdf.float(), hidden)
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.bn1(output)
        output = F.relu(output)
        return self.fc2(output).squeeze(-1)

    def init_hidden(self, batch_size: int, device=None):
        return (
            torch.zeros(batch_size, self.hidden_size, device=device),
            torch.zeros(batch_size, self.hidden_size, device=device),
        )
