import sys
import torch
import unittest
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("./src/")

from positional_encoding import PositionalEncoding
from scaled_dot_product import scaled_dot_product_attention


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.total_texts = 400
        self.batch_size = 40
        self.nheads = 8
        self.sequence_length = 200
        self.dimension = 512
        self.constant = 10000

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


if __name__ == "__main__":
    unittest.main()
