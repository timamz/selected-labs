import torch
from typing import Type
from torch import nn
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, 
            embedding_dim=embed_size, 
            padding_idx=dataset.pad_id
        )
        
        self.rnn = rnn_type(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            dropout=0.2,
            batch_first=True
        )
        
        self.linear = nn.Linear(
            in_features=hidden_size, 
            out_features=self.vocab_size
        )

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        
        embeds = self.embedding(indices)
        outputs, _ = self.rnn(embeds)
        logits = self.linear(outputs)
        
        max_seq_len = lengths.max().item()
        logits = logits[:, :max_seq_len, :]
        
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        device = next(self.parameters()).device

        prefix_tokens = [self.dataset.bos_id] + self.dataset.text2ids(prefix)
        h = None

        for token in prefix_tokens:
            input_tensor = torch.tensor([[token]], device=device)
            embed = self.embedding(input_tensor)
            output, h = self.rnn(embed, h)

        generated_tokens = prefix_tokens[:]
        it = 0

        while generated_tokens[-1] != self.dataset.eos_id and it < self.max_length:
            input_tensor = torch.tensor([[generated_tokens[-1]]], device=device)
            embed = self.embedding(input_tensor)
            output, h = self.rnn(embed, h)
            logits = self.linear(output.squeeze(1)) / temp
            probs = torch.softmax(logits, dim=1)
            output_token = torch.multinomial(probs, num_samples=1).item()
            generated_tokens.append(output_token)
            it += 1

        return self.dataset.ids2text(generated_tokens)
