import sys
import torch
import unittest
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("./src/")

from positional_encoding import PositionalEncoding
from feedforward_network import FeedForwardNetwork
from layer_normalization import LayerNormalization
from multihead_attention import MultiHeadAttention
from scaled_dot_product import scaled_dot_product_attention


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.total_texts = 400
        self.batch_size = 40
        self.nheads = 8
        self.sequence_length = 200
        self.feedforward = 2048
        self.dimension = 512
        self.constant = 10000
        self.dropout = 0.1
        self.activation = "relu"

        self.query = torch.randn(self.batch_size, self.sequence_length, self.dimension)
        self.key = torch.randn(self.batch_size, self.sequence_length, self.dimension)
        self.value = torch.randn(self.batch_size, self.sequence_length, self.dimension)

        self.positional_encoding = PositionalEncoding(
            sequence_length=self.sequence_length,
            dimension=self.dimension,
            constant=self.constant,
        )

        self.attention = scaled_dot_product_attention(
            query=self.query.view(
                self.query.size(0),
                self.nheads,
                self.sequence_length,
                self.dimension // self.nheads,
            ),
            key=self.key.view(
                self.key.size(0),
                self.nheads,
                self.sequence_length,
                self.dimension // self.nheads,
            ),
            value=self.value.view(
                self.value.size(0),
                self.nheads,
                self.sequence_length,
                self.dimension // self.nheads,
            ),
        )

        self.network = FeedForwardNetwork(
            in_features=self.dimension,
            out_features=self.feedforward,
            activation=self.activation,
        )

        self.layernorm = LayerNormalization(
            normalized_shape=self.dimension,
        )

        self.multihead_attention = MultiHeadAttention(
            dimension=self.dimension,
            nheads=self.nheads,
            dropout=self.dropout,
        )

    def test_positional_encoding(self):
        embedding_layer = nn.Embedding(
            num_embeddings=self.sequence_length, embedding_dim=self.dimension
        )

        embedding = embedding_layer(
            torch.randint(
                0, self.sequence_length, (self.total_texts, self.sequence_length)
            )
        )

        self.assertEqual(
            embedding.size(), (self.total_texts, self.sequence_length, self.dimension)
        )

        positional_encoding = self.positional_encoding(x=embedding)

        self.assertEqual(
            positional_encoding.size(), (1, self.sequence_length, self.dimension)
        )

        embeddings_with_positional = torch.add(embedding, positional_encoding)

        self.assertEqual(
            embeddings_with_positional.size(),
            (400, self.sequence_length, self.dimension),
        )

    def test_positional_encoding_with_dataloader(self):
        embedding_layer = nn.Embedding(
            num_embeddings=self.sequence_length, embedding_dim=self.dimension
        )
        embedding = embedding_layer(
            torch.randint(
                0, self.sequence_length, (self.total_texts, self.sequence_length)
            )
        )

        dataloader = DataLoader(
            dataset=list(embedding), batch_size=self.batch_size, shuffle=True
        )

        data = next(iter(dataloader))

        self.assertEqual(
            data.size(), (self.batch_size, self.sequence_length, self.dimension)
        )

        positional_encoding = self.positional_encoding(x=data)

        embeddings_with_positional = torch.add(data, positional_encoding)

        self.assertEqual(
            embeddings_with_positional.size(),
            (self.batch_size, self.sequence_length, self.dimension),
        )

    def test_scaled_dot_product(self):
        self.assertEqual(
            self.attention.size(),
            (
                self.batch_size,
                self.nheads,
                self.sequence_length,
                self.dimension // self.nheads,
            ),
        )

    def test_feedforward_neural_network(self):
        embedding_layer = nn.Embedding(
            num_embeddings=self.sequence_length, embedding_dim=self.dimension
        )

        embedding = embedding_layer(
            torch.randint(
                0, self.sequence_length, (self.total_texts, self.sequence_length)
            )
        )

        position = self.positional_encoding(x=embedding)

        embedding_with_position = torch.add(embedding, position)

        dataloader = DataLoader(
            dataset=list(embedding_with_position),
            batch_size=self.batch_size,
            shuffle=True,
        )

        data = next(iter(dataloader))

        self.assertEqual(
            data.size(), (self.batch_size, self.sequence_length, self.dimension)
        )

        result = self.network(x=data)

        self.assertEqual(
            result.size(), (self.batch_size, self.sequence_length, self.dimension)
        )

    def test_layer_normalization(self):
        result = self.network(
            x=torch.randn(self.batch_size, self.sequence_length, self.dimension)
        )

        normalization = self.layernorm(x=result)

        self.assertEqual(
            normalization.size(),
            (self.batch_size, self.sequence_length, self.dimension),
        )

        self.assertIsInstance(self.layernorm, LayerNormalization)

    def test_multihead_attention_layer(self):
        x = torch.randn(self.batch_size, self.sequence_length, self.dimension)

        mask = None

        attention = self.multihead_attention(x=x, mask=mask)

        self.assertEqual(
            attention.size(), (self.batch_size, self.sequence_length, self.dimension)
        )

        embedding_layer = nn.Embedding(self.sequence_length, self.dimension)

        embedding = embedding_layer(
            torch.randint(
                0, self.sequence_length, (self.batch_size, self.sequence_length)
            )
        )

        position = self.positional_encoding(x=embedding)

        embedding_with_position = torch.add(embedding, position)

        attention = self.multihead_attention(embedding_with_position)

        self.assertEqual(
            attention.size(), (self.batch_size, self.sequence_length, self.dimension)
        )

        self.assertIsInstance(self.multihead_attention, MultiHeadAttention)
        self.assertIsInstance(embedding_layer, nn.Embedding)
        self.assertIsInstance(self.positional_encoding, PositionalEncoding)


if __name__ == "__main__":
    unittest.main()
