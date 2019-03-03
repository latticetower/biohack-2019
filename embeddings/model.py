import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import Linear
#from torch.nn.modules.distance import CosineDistance


class CustomLinear(Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, \text{in\_features})` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, \text{out\_features})` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=False):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        return F.linear(self.weight, input, self.bias)

        

class BasicModel(torch.nn.Module):
    def __init__(self, ngenes, nfeatures, embedding_size=100):
        super(BasicModel, self).__init__()
        
        #Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)[S
        self.layer_e = CustomLinear(ngenes, embedding_size)
        self.layer_f = CustomLinear(nfeatures, embedding_size)
        torch.nn.init.normal(self.layer_e.weight, mean=0.0, std=0.02)
        
        torch.nn.init.normal(self.layer_f.weight, mean=0.0, std=0.02)
        
        
        #self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, e, f):
        h1 = self.layer_e(e) # 
        h2 = self.layer_f(f) #
        #h1 = torch.nn.Sigmoid()(h1)
        #h2 = torch.nn.Sigmoid()(h2)
        #h1 = torch.sum(h1, dim=1)
        #h2 = torch.sum(h2, dim=1)
        #return h1
        output = F.cosine_similarity(h1, h2, dim=0)
        #x = torch.cat([h1, h2], 1)
        #h = self.hidden_layer(x)
        #out = self.output_layer(h)
        return output