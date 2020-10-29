import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


def precision_recall_helper(actual, predicted, k):
    active_actual_idxs = set([i for i, e in enumerate(actual) if e == 1])
    predicted_top_k_indices = set([i[0] for i in nlargest(k, enumerate(predicted), key=lambda x: x[1])])
    intersection = active_actual_idxs.intersection(predicted_top_k_indices)
    if len(active_actual_idxs) == 0:
        return 0, 1  # precision, recall
    precision = len(intersection) / k
    recall = len(intersection) / len(active_actual_idxs)
    return round(precision, 2), round(recall, 2)


def get_avg_batch_precision_recall_at_k(actual_lists, predicted_lists, k):
    assert len(actual_lists) == len(predicted_lists)
    batch_len = len(actual_lists)
    precision = [precision_recall_helper(actual_lists[i], predicted_lists[i], k)[0] for i in range(batch_len)]
    recall = [precision_recall_helper(actual_lists[i], predicted_lists[i], k)[1] for i in range(batch_len)]
    return np.mean(precision), np.mean(recall)




class Average(nn.Module):
    def __init__(self, num_classes=2, model_name = 'resnet', use_pretrained = True):
        super(Average, self).__init__()

        if model_name == 'resnet18':
            network = models.resnet18(pretrained=use_pretrained)
            num_ftrs = network.fc.in_features

        elif model_name == 'resnet34':
            network = models.resnet34(pretrained=use_pretrained)
            num_ftrs = network.fc.in_features

        elif model_name == 'resnet50':
            network = models.resnet50(pretrained=use_pretrained)
            num_ftrs = network.fc.in_features

        elif model_name == 'resnext50':
            network = models.resnext50_32x4d(pretrained=use_pretrained)
            num_ftrs = network.fc.in_features

        elif model_name == "alexnet":
            """ Alexnet
            """
            network = models.alexnet(pretrained=use_pretrained)
            # set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = network.classifier[6].in_features
            # model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            # input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        self.feat_ext = nn.Sequential(*list(network.children())[:-1])
        # num_ftrs = network.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        del network

    @staticmethod
    def size_splits(tensor, split_sizes, dim=0):
        """Splits the tensor according to chunks of split_sizes.
        Arguments:
            tensor (Tensor): tensor to split.
            split_sizes (list(int)): sizes of chunks
            dim (int): dimension along which to split the tensor.
        """
        if dim < 0:
            dim += tensor.dim()

        dim_size = tensor.size(dim)
        if dim_size != torch.sum(torch.Tensor(split_sizes)):
            raise KeyError("Sum of split sizes exceeds tensor dim")

        splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

        return tuple(tensor.narrow(int(dim), int(start), int(length))
                     for start, length in zip(splits, split_sizes))
      
    def forward(self, img_tensor, sizes):
        sizes = list(sizes.detach())
        hazard_type_feats = self.feat_ext(img_tensor)
        # extracting tuples: splitting by sizes of each policy
        hazard_type_feats = self.size_splits(hazard_type_feats, sizes, 0)
        # taking avg of each of the tuple elements
        hazard_type_feats = [torch.mean(t, axis =0).view(1, -1) for t in hazard_type_feats]

        hazard_type_feats = torch.cat(hazard_type_feats, axis=0)

        # hazard_type_logits = torch.sigmoid(self.fc(hazard_type_feats))
        # returning logits instead of probabilities
        hazard_type_logits = self.fc(hazard_type_feats)

        return hazard_type_logits



class Attention(nn.Module):
    """
    Adding attention Layer """
    def __init__(self, num_classes=2, model_name = 'resnet', use_pretrained = True):
        super(Attention, self).__init__()

        self.D = 128
        self.K = 1 # 1 attention head
        self.num_classes = num_classes

        if model_name == 'resnet18':
            network = models.resnet18(pretrained=use_pretrained)
            num_ftrs = network.fc.in_features

        elif model_name == 'resnet34':
            network = models.resnet34(pretrained=use_pretrained)
            num_ftrs = network.fc.in_features

        elif model_name == 'resnet50':
            network = models.resnet50(pretrained=use_pretrained)
            num_ftrs = network.fc.in_features

        elif model_name == 'resnext50':
            network = models.resnext50_32x4d(pretrained=use_pretrained)
            num_ftrs = network.fc.in_features

        elif model_name == "alexnet":
            """ Alexnet
            """
            network = models.alexnet(pretrained=use_pretrained)
            # set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = network.classifier[6].in_features
            # model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            # input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        self.feat_ext = nn.Sequential(*list(network.children())[:-1])
        # num_ftrs = network.fc.in_features

        self.attn_layers = []
        self.fc = []

        for i in range(self.num_classes):

            self.attn_layers.append(
                nn.Sequential(
                    nn.Linear(num_ftrs, self.D),
                    nn.Tanh(),
                    nn.Linear(self.D, self.K)
                )

            )
            self.fc.append(
                nn.Linear(num_ftrs*self.K, 1)
            )
        del network


    @staticmethod
    def size_splits(tensor, split_sizes, dim=0):
        """Splits the tensor according to chunks of split_sizes.
        Arguments:
            tensor (Tensor): tensor to split.
            split_sizes (list(int)): sizes of chunks
            dim (int): dimension along which to split the tensor.
        """
        if dim < 0:
            dim += tensor.dim()

        dim_size = tensor.size(dim)
        if dim_size != torch.sum(torch.Tensor(split_sizes)):
            raise KeyError("Sum of split sizes exceeds tensor dim")

        splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

        return tuple(tensor.narrow(int(dim), int(start), int(length))
                     for start, length in zip(splits, split_sizes))
      



    def attn_helper(self, h):
        # 1 attention layer connecting to each category
        
        M = []
        h = h.view(h.size(0), h.size(1))
        for i in range(self.num_classes):

            a = self.attn_layers[i](h)  # NxK
            a = torch.transpose(a, 1, 0)  # KxN
            a = F.softmax(a, dim=1)  # softmax over N

            m = torch.mm(a, h)  # KxL
            m = self.fc[i](m)
            M.append(m)
        return torch.cat(M, axis=1)



    def forward(self, img_tensor, sizes):
        sizes = list(sizes.detach())
        H = self.feat_ext(img_tensor)
        # extracting tuples: splitting by sizes of each policy
        H = self.size_splits(H, sizes, 0)
        # taking avg of each of the tuple elements
        H = [self.attn_helper(bag) for bag in H]

        logits = torch.cat(H, axis=0)

        return logits


