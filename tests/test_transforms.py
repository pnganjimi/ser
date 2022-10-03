from ser.transforms import flip
import torch

def test_flip():
    input_img = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])

    exp_img = torch.tensor([
        [8, 7, 6, 5],
        [4, 3, 2, 1]
    ])

    flip_transform = flip()

    assert torch.equal(flip_transform(input_img), exp_img)
