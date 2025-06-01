import torch

from basicsr.archs.iacc_arch import IACC
from basicsr.archs.nuicnet_arch import NUICNet
from basicsr.archs.uiec2net_arch import UIEC2Net
from basicsr.archs.ushape_arch import UShape


def test_IACC():
    """Test arch: IACC."""
    net = IACC().cuda()
    img = torch.rand((2, 3, 256, 256), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (2, 3, 256, 256)


def test_NUICNet():
    """Test arch: NUICNet."""
    net = NUICNet().cuda()
    img = torch.rand((2, 3, 256, 256), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (2, 3, 256, 256)


def test_UIEC2Net():
    """Test arch: UIEC2Net."""
    net = UIEC2Net().cuda()
    img = torch.rand((2, 3, 256, 256), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (2, 3, 256, 256)


def test_UShape():
    """Test arch: UShape."""
    net = UShape().cuda()
    img = torch.rand((2, 3, 256, 256), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (2, 3, 256, 256)


if __name__ == '__main__':
    test_IACC()
    test_NUICNet()
    test_UIEC2Net()
    test_UShape()
    