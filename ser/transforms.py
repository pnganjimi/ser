import torch
from torchvision import transforms as torch_transforms


def transforms(*stages):
    return torch_transforms.Compose(
        [
            torch_transforms.ToTensor(),
            *(stage() for stage in stages),
        ]
    )


def normalize():
    """
    Normalize a tensor to have a mean of 0.5 and a std dev of 0.5
    """
    return torch_transforms.Normalize((0.5,), (0.5,))


def flip():
    """
    Flip a tensor both vertically and horizontally
    """

    return RandomFlip()


class RandomFlip(torch_transforms.RandomVerticalFlip):
 
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """

        F = torch_transforms.functional
        if torch.rand(1) < self.p:
            return F.hflip(F.vflip(img))
        return img
