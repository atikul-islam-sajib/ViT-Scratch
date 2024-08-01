import sys
import math
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None
):
    if (
        isinstance(query, torch.Tensor)
        and isinstance(key, torch.Tensor)
        and isinstance(value, torch.Tensor)
    ):
        result = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.size(-1))

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            result = torch.add(result, mask)

        attention_weight = torch.softmax(input=result, dim=-1)

        attention = torch.matmul(attention_weight, value)

        assert (
            attention.size() == query.size() == key.size() == value.size()
        ), "Sizes of inputs are not equal".capitalize()

        return attention

    else:
        raise TypeError("All inputs must be of type torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaled dot product attetion for transformer".title()
    )

    args = parser.parse_args()

    batch_size = config()["ViT"]["batch_size"]
    nheads = config()["ViT"]["nheads"]
    dimension = config()["ViT"]["dimension"]

    query = torch.randn((batch_size, nheads, 200, dimension // nheads))
    key = torch.randn((batch_size, nheads, 200, dimension // nheads))
    value = torch.randn((batch_size, nheads, 200, dimension // nheads))
    mask = torch.randint(0, 2, (batch_size, 200))

    attention = scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        mask=None,
    )

    assert attention.size() == (
        batch_size,
        nheads,
        200,
        dimension // nheads,
    ), "Sizes of inputs are not equal".capitalize()

    attention = scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        mask=mask,
    )

    assert attention.size() == (
        batch_size,
        nheads,
        200,
        dimension // nheads,
    ), "Sizes of inputs are not equal".capitalize()
