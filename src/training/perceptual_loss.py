"""
Perceptual loss utilities
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

#----------------------------------------------------------------------------

class AntiAliasInterpolation2d(torch.nn.Module):
    """
    Band-limited down-sampling, that down-sample images better.
    The class create a kernel for anti-aliasing in advance and then use it for all forwards.
    """
    def __init__(self, channels, scale):
        """
        :param channels: number of channels in the input
        :param scale: how much the image should be downscaled, e.g. 0.5 for 2 times downscale.
        """
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing='ij')
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean)**2 / (2 * std**2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depth-wise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, inp, subsample=True):
        if self.scale == 1.0:
            return inp

        out = F.pad(inp, (self.ka, self.kb, self.ka, self.kb)) # pylint: disable=not-callable
        out = F.conv2d(out, weight=self.weight, groups=self.groups) # pylint: disable=not-callable
        if subsample:
            out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out

#----------------------------------------------------------------------------

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """
    def __init__(self, requires_grad=False, replace_maxpool_with_avgpool: bool=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
        if replace_maxpool_with_avgpool:
            for i, layer in enumerate(vgg_pretrained_features):
                if isinstance(layer, torch.nn.MaxPool2d):
                    assert vgg_pretrained_features[i].dilation == 1, f'Expected dilation to be 1, got {vgg_pretrained_features[i].dilation}'
                    vgg_pretrained_features[i] = torch.nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding, ceil_mode=layer.ceil_mode)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))), requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))), requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

#----------------------------------------------------------------------------

class ImagePyramid(torch.nn.Module):
    """
    Image pyramid for computing pyramid perceptual loss.
    """

    def __init__(self, scales: tuple[float], num_channels):
        super(ImagePyramid, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = torch.nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict

#----------------------------------------------------------------------------

class PerceptualPyramidLoss(torch.nn.Module):
    """
    The perceptual pyramid loss, see "First Order Motion Model for Image Animation" (Sec 3.3).
    """
    def __init__(self, scales: tuple[float]=(1,0.5,0.25), loss_weights: tuple[float]=(10, 10, 10, 10, 10), num_channels=3, replace_maxpool_with_avgpool: bool=False, downsample_to_native: bool=False):
        """
        :param scales: scales to use for image pyramid
        :param loss_weights: the weighting of different scales
        :param num_channels: number of channels in the input image
        """
        super().__init__()
        self.pyramid = ImagePyramid(scales, num_channels)
        self.vgg = Vgg19(replace_maxpool_with_avgpool=replace_maxpool_with_avgpool)
        self.loss_weights = loss_weights
        self.scales = scales
        self.downsample_to_native = downsample_to_native

    def __call__(self, gt: torch.Tensor, prediction: torch.Tensor):
        """Expects both ground-truth and predicted images (or videos) to be in the [-1, 1] range"""
        assert gt.shape == prediction.shape, f'Expected gt and prediction to have the same shape, got {gt.shape} and {prediction.shape}'

        # Normalize to the range which VGG expects.
        gt = gt * 0.5 + 0.5
        prediction = prediction * 0.5 + 0.5

        if gt.ndim == 5:
            t = gt.shape[1]
            gt = gt.flatten(0, 1) # [b * t, c, h, w]
            prediction = prediction.flatten(0, 1) # [b * t, c, h, w]
        else:
            t = None

        if self.downsample_to_native:
            gt = F.interpolate(gt, size=(224, 224), mode='area') # [b, c, 224, 224]
            prediction = F.interpolate(prediction, size=(224, 224), mode='area') # [b, c, 224, 224]

        pyramid_gt = self.pyramid(gt)
        pyramid_generated = self.pyramid(prediction)
        value_total = 0

        for scale in self.scales:
            x_vgg = self.vgg(pyramid_generated['prediction_' + str(scale)])
            y_vgg = self.vgg(pyramid_gt['prediction_' + str(scale)])

            for i, weight in enumerate(self.loss_weights):
                value = torch.abs(x_vgg[i] - y_vgg[i]).mean(dim=(1,2,3)) # [b]
                value_total += weight * value # [b]
        if t is None:
            value_total = value_total.view(-1, 1, 1, 1)
        else:
            value_total = value_total.view(-1, t, 1, 1, 1)
        return value_total # [b, t or None, 1, 1, 1]

#----------------------------------------------------------------------------
