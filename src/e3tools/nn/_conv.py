import functools
from typing import Callable, Mapping, Optional, Union

import e3nn
import e3nn.o3
from openequivariance.benchmark.tpp_creation_utils import FullyConnectedTPProblem
from openequivariance import TensorProductConv, TensorProduct
import torch
from torch import nn


from ._gate import Gated
from ._interaction import LinearSelfInteraction
from ._mlp import ScalarMLP
from ._tensor_product import ExperimentalTensorProduct, SeparableTensorProduct


class Conv(nn.Module):
    """
    Equivariant convolution layer

    ref: https://arxiv.org/abs/1802.08219
    """

    def __init__(
        self,
        irreps_in: Union[str, e3nn.o3.Irreps],
        irreps_out: Union[str, e3nn.o3.Irreps],
        irreps_sh: Union[str, e3nn.o3.Irreps],
        edge_attr_dim: int,
        radial_nn: Optional[Callable[..., nn.Module]] = None,
        tensor_product: Optional[Callable[..., nn.Module]] = None,
    ):
        """
        Parameters
        ----------
        irreps_in: e3nn.o3.Irreps
            Input node feature irreps
        irreps_out: e3nn.o3.Irreps
            Ouput node feature irreps
        irreps_sh: e3nn.o3.Irreps
            Edge spherical harmonic irreps
        edge_attr_dim: int
            Dimension of scalar edge attributes to be passed to radial_nn
        radial_nn: Optional[Callable[..., nn.Module]]
            Factory function for radial nn used to generate tensor product weights.
            Should be callable as radial_nn(in_features, out_features)
            if `None` then
                ```
                functools.partial(
                    e3tools.nn.ScalarMLP,
                    hidden_features=[edge_attr_dim],
                    activation_layer=nn.SiLU,
                )
                ```
            is used.
        tensor_product: Optional[Callable[..., nn.Module]]
            Factory function for tensor product used to mix input node
            representations with edge spherical harmonics.
            Should be callable as `tensor_product(irreps_in, irreps_sh, irreps_out)`
            and return an object with `weight_numel` property defined
            If `None` then
                ```
                functools.partial(
                    e3nn.o3.FullyConnectedTensorProduct
                    shared_weights=False,
                    internal_weights=False,
                )
                ```
            is used.
        """

        super().__init__()

        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.irreps_out = e3nn.o3.Irreps(irreps_out)
        self.irreps_sh = e3nn.o3.Irreps(irreps_sh)

        if tensor_product is None:
            tensor_product_problem = FullyConnectedTPProblem(
                irreps_in1=irreps_in,
                irreps_in2=irreps_sh,
                irreps_out=irreps_out,
                shared_weights=False,
                internal_weights=False,
            )
            self.tp_conv = TensorProductConv(tensor_product_problem)
            self.tp = TensorProduct(tensor_product_problem)
        else:
            raise NotImplementedError("Temporarily Not Implemented")

        if radial_nn is None:
            radial_nn = functools.partial(
                ScalarMLP,
                hidden_features=[edge_attr_dim],
                activation_layer=nn.SiLU,
            )

        self.radial_nn = radial_nn(edge_attr_dim, tensor_product_problem.weight_numel)

    def apply_per_edge(self, node_attr_src, edge_attr, edge_sh):
        return self.tp(node_attr_src, edge_sh, self.radial_nn(edge_attr))

    def forward(self, node_attr, edge_index, edge_attr, edge_sh):
        """
        Computes the forward pass of the equivariant convolution.

        Let N be the number of nodes, and E be the number of edges

        Parameters
        ----------
        node_attr: [N, irreps_in.dim]
        edge_index: [2, E]
        edge_attr: [E, edge_attr_dim]
        edge_sh: [E, irreps_sh.dim]

        Returns
        -------
        out: [N, irreps_out.dim]
        """
        return self.tp_conv(
            node_attr, edge_sh, self.radial_nn(edge_attr), edge_index[0], edge_index[1]
        )


class SeparableConv(Conv):
    """
    Equivariant convolution layer using separable tensor product

    ref: https://arxiv.org/abs/1802.08219
    ref: https://arxiv.org/abs/2206.11990
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            tensor_product=SeparableTensorProduct,
        )


class ExperimentalConv(Conv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            tensor_product=ExperimentalTensorProduct,
        )


class ConvBlock(nn.Module):
    """
    Equivariant convolution with gated non-linearity and linear self-interaction
    """

    def __init__(
        self,
        irreps_in: Union[str, e3nn.o3.Irreps],
        irreps_out: Union[str, e3nn.o3.Irreps],
        irreps_sh: Union[str, e3nn.o3.Irreps],
        edge_attr_dim: int,
        act: Optional[Mapping[int, nn.Module]] = None,
        act_gates: Optional[Mapping[int, nn.Module]] = None,
        conv: Optional[Callable[..., nn.Module]] = None,
    ):
        """
        Parameters
        ----------
        irreps_in: e3nn.o3.Irreps
            Input node feature irreps
        irreps_out: e3nn.o3.Irreps
            Ouput node feature irreps
        irreps_sh: e3nn.o3.Irreps
            Edge spherical harmonic irreps
        edge_attr_dim: int
            Dimension of scalar edge attributes to be passed to radial_nn
        act: Mapping[int, nn.Module]
            Mapping from parity to activation module.
            If `None` defaults to `{1 : nn.LeakyReLU(), -1: nn.Tanh()}`
        act_gates: Mapping[int, nn.Module]
            Mapping from parity to activation module.
            If `None` defaults to `{1 : nn.Sigmoid(), -1: nn.Tanh()}`
        conv: Optional[Callable[..., nn.Module]] = None
            Factory function for convolution layer used for computing keys and values
        """

        super().__init__()

        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.irreps_out = e3nn.o3.Irreps(irreps_out)
        self.irreps_sh = e3nn.o3.Irreps(irreps_sh)

        if conv is None:
            conv = Conv

        wrapped_conv = functools.partial(
            conv, irreps_sh=irreps_sh, edge_attr_dim=edge_attr_dim
        )

        self.gated_conv = LinearSelfInteraction(
            Gated(
                wrapped_conv,
                irreps_in=self.irreps_in,
                irreps_out=self.irreps_out,
                act=act,
                act_gates=act_gates,
            )
        )

    def forward(
        self,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_sh: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the forward pass of the equivariant graph attention

        Let N be the number of nodes, and E be the number of edges

        Parameters
        ----------
        node_attr: [N, irreps_in.dim]
        edge_index: [2, E]
        edge_attr: [E, edge_attr_dim]
        edge_sh: [E, irreps_sh.dim]

        Returns
        -------
        out: [N, irreps_out.dim]
        """
        return self.gated_conv(node_attr, edge_index, edge_attr, edge_sh)
