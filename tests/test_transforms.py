import pytest
from ser.transforms import flip, normalize
import torch


# def test_normalize1():
#     # input of ones should remain unchanged
#     input_img = torch.floatTensor([
#         [1, 1, 1, 1],
#         [1, 1, 1, 1]
#     ])

#     assert torch.equal(normalize()(input_img), input_img)


def test_normalize2():
    # If the input is something else, we should see the appropriate transformation applied.
    input_img = torch.FloatTensor([
        [[1, 2, 3], 
         [3, 2, 1]]
    ])

    exp_img = torch.FloatTensor([
        [[1, 3, 5], 
         [5, 3, 1]]
    ])

    assert torch.equal(normalize()(input_img), exp_img)


def test_flip():
    input_img = torch.tensor([
        [[1, 2, 3, 4],
         [5, 6, 7, 8]]
    ])

    exp_img = torch.tensor([
        [[8, 7, 6, 5],
         [4, 3, 2, 1]]
    ])

    assert torch.equal(flip()(input_img), exp_img)
