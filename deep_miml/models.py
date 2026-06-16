import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

SUPPORTED_MODELS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnext50": models.resnext50_32x4d,
    "alexnet": models.alexnet,
}


def _build_backbone(model_name, use_pretrained=True):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Invalid model name '{model_name}'. "
            f"Supported: {list(SUPPORTED_MODELS.keys())}"
        )
    network = SUPPORTED_MODELS[model_name](pretrained=use_pretrained)
    if model_name == "alexnet":
        num_ftrs = network.classifier[6].in_features
    else:
        num_ftrs = network.fc.in_features
    feat_ext = nn.Sequential(*list(network.children())[:-1])
    return feat_ext, num_ftrs


def size_splits(tensor, split_sizes, dim=0):
    if dim < 0:
        dim += tensor.dim()

    if tensor.size(dim) != sum(split_sizes):
        raise ValueError("Sum of split sizes exceeds tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(
        tensor.narrow(int(dim), int(start), int(length))
        for start, length in zip(splits, split_sizes)
    )


class Average(nn.Module):
    def __init__(self, num_classes=2, model_name="resnet", use_pretrained=True):
        super().__init__()
        self.feat_ext, num_ftrs = _build_backbone(model_name, use_pretrained)
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, img_tensor, sizes):
        sizes = list(sizes.detach())
        feats = self.feat_ext(img_tensor)
        feats = size_splits(feats, sizes, 0)
        feats = [torch.mean(t, dim=0).view(1, -1) for t in feats]
        feats = torch.cat(feats, dim=0)
        return self.fc(feats)


class Attention(nn.Module):
    def __init__(self, num_classes=2, model_name="resnet", use_pretrained=True):
        super().__init__()
        self.D = 128
        self.K = 1
        self.num_classes = num_classes

        self.feat_ext, num_ftrs = _build_backbone(model_name, use_pretrained)

        self.attn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_ftrs, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
                )
                for _ in range(num_classes)
            ]
        )
        self.fc = nn.ModuleList(
            [nn.Linear(num_ftrs * self.K, 1) for _ in range(num_classes)]
        )

    def _attn_helper(self, h):
        M = []
        h = h.view(h.size(0), h.size(1))
        for i in range(self.num_classes):
            a = self.attn_layers[i](h)  # NxK
            a = torch.transpose(a, 1, 0)  # KxN
            a = F.softmax(a, dim=1)  # softmax over N
            m = torch.mm(a, h)  # KxL
            m = self.fc[i](m)
            M.append(m)
        return torch.cat(M, dim=1)

    def forward(self, img_tensor, sizes):
        sizes = list(sizes.detach())
        H = self.feat_ext(img_tensor)
        H = size_splits(H, sizes, 0)
        H = [self._attn_helper(bag) for bag in H]
        return torch.cat(H, dim=0)
