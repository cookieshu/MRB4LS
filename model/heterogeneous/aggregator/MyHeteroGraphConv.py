"""Heterograph NN modules"""
from functools import partial

import torch
import torch as th
import torch.nn as nn
from dgl import DGLError


class MyHeteroGraphConv(nn.Module):

    def __init__(self, mods, aggregate="sum", use_edge_weight=False):
        super(MyHeteroGraphConv, self).__init__()
        self.mod_dict = mods
        self.use_edge_weight = use_edge_weight
        mods = {str(k): v for k, v in mods.items()}
        # Register as child modules
        self.mods = nn.ModuleDict(mods)
        # PyTorch ModuleDict doesn't have get() method, so I have to store two
        # dictionaries so that I can index with both canonical edge type and
        # edge type with the get() method.
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(
                v, "set_allow_zero_in_degree", None
            )
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

    def _get_module(self, etype):
        mod = self.mod_dict.get(etype, None)
        if mod is not None:
            return mod
        if isinstance(etype, tuple):
            # etype is canonical
            _, etype, _ = etype
            return self.mod_dict[etype]
        raise KeyError("Cannot find module with edge type %s" % etype)

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        """Forward computation

        Invoke the forward function with each module and aggregate their results.

        Parameters
        ----------
        g : DGLGraph
            Graph data.
        inputs : dict[str, Tensor] or pair of dict[str, Tensor]
            Input node features.
        mod_args : dict[str, tuple[any]], optional
            Extra positional arguments for the sub-modules.
        mod_kwargs : dict[str, dict[str, any]], optional
            Extra key-word arguments for the sub-modules.

        Returns
        -------
        dict[str, Tensor]
            Output representations for every types of nodes.
        """
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {
                    k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()
                }

            for stype, etype, dtype in g.canonical_etypes:
                if etype not in self.mod_dict.keys():
                    continue
                rel_graph = g[stype, etype, dtype]
                if self.use_edge_weight:
                    dstdata = self._get_module((stype, etype, dtype))(
                        rel_graph,
                        (src_inputs[stype], dst_inputs[dtype]),
                        edge_weight=rel_graph.edges[(stype, etype, dtype)].data['weight'].to(torch.float32)
                    )
                else:
                    dstdata = self._get_module((stype, etype, dtype))(
                        rel_graph,
                        (src_inputs[stype], dst_inputs[dtype]),
                        *mod_args.get(etype, ()),
                        **mod_kwargs.get(etype, {})
                    )
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in inputs:
                    continue
                dstdata = self._get_module((stype, etype, dtype))(
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {})
                )
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts


def _max_reduce_func(inputs, dim):
    return th.max(inputs, dim=dim)[0]


def _min_reduce_func(inputs, dim):
    return th.min(inputs, dim=dim)[0]


def _sum_reduce_func(inputs, dim):
    return th.sum(inputs, dim=dim)


def _mean_reduce_func(inputs, dim):
    return th.mean(inputs, dim=dim)


def _stack_agg_func(inputs, dsttype):  # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    return th.stack(inputs, dim=1)


def _agg_func(inputs, dsttype, fn):  # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    stacked = th.stack(inputs, dim=0)
    return fn(stacked, dim=0)


def get_aggregate_fn(agg):
    """Internal function to get the aggregation function for node data
    generated from different relations.

    Parameters
    ----------
    agg : str
        Method for aggregating node features generated by different relations.
        Allowed values are 'sum', 'max', 'min', 'mean', 'stack'.

    Returns
    -------
    callable
        Aggregator function that takes a list of tensors to aggregate
        and returns one aggregated tensor.
    """
    if agg == "sum":
        fn = _sum_reduce_func
    elif agg == "max":
        fn = _max_reduce_func
    elif agg == "min":
        fn = _min_reduce_func
    elif agg == "mean":
        fn = _mean_reduce_func
    elif agg == "stack":
        fn = None  # will not be called
    else:
        raise DGLError(
            "Invalid cross type aggregator. Must be one of "
            '"sum", "max", "min", "mean" or "stack". But got "%s"' % agg
        )
    if agg == "stack":
        return _stack_agg_func
    else:
        return partial(_agg_func, fn=fn)
