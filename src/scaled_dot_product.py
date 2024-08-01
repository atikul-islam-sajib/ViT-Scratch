import sys
import math
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


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

    query = torch.randn(())
    attention = scaled_dot_product_attention(
        query=torch.randn((40, 8, 200, 64)),
        key=torch.randn((40, 8, 200, 64)),
        value=torch.randn((40, 8, 200, 64)),
        mask=None,
    )

    attention = scaled_dot_product_attention(
        query=torch.randn((40, 8, 200, 64)),
        key=torch.randn((40, 8, 200, 64)),
        value=torch.randn((40, 8, 200, 64)),
        mask=torch.randint(0, 2, (40, 200)),
    )
