from .spiking_resnet1d import SpikingResNet1DBackbone
from .heads import LinearClassifierHead, AugPredHeads

__all__ = ["SpikingResNet1DBackbone", "LinearClassifierHead", "AugPredHeads"]
