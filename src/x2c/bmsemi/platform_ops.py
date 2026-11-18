"""
Bridge to the other platform operations (the canvas modules), called directly from Python.
Uses the same functions the GUI calls, but WITHOUT the web layer (and without GUI access).
"""
from BMsemi_Model.bm_models.core.mapping import LayerDims, PEGeometry, map_layer as _map_layer
from BMsemi_Model.bm_models.models.blocks import schedule_layer, energy_estimate  # noqa: F401


def map_layer(p):
    """Same parameter names as the canvas 'core.map_layer' panel."""
    pe = PEGeometry(rows=int(p.get("pe_rows", 64)), cols=int(p.get("pe_cols", 32)))
    dims = LayerDims(**{k: int(p[k]) for k in ("K", "C", "R", "S", "Y", "X") if k in p})
    return _map_layer(dims, pe).to_dict()
