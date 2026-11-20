"""柱状图冻结前的层段拟合复算核验（ORM 适配层）。

纯比对规则见 recompute_logic.py；本模块负责取本孔记录、计算冻结依据指纹，
并把 evidence/测点交给逻辑层。核验只给结论，不改变任何业务记录。
"""
from sqlalchemy import select
from .models import (
    CoreRun, Reading, MissingInterval, Interpretation, Stratum,
    SampleInterval, Seal, Custody,
)
from .common import pack
from .layer_model import fit_layers
from . import recompute_logic as logic


def readings_basis_hash(readings):
    """当前测点依据指纹：测点集合或任一物性值变化都会变。"""
    return logic.checksum([pack(r) for r in readings])


def revision_basis_hash(s, borehole_id):
    """整孔冻结依据指纹，覆盖复算比对与冻结检查所依赖的全部当前记录。

    草稿自身字段（version/note/verification）不参与；源记录任何变化都会使旧核验过期。
    """
    parts = {}
    for key, cls in [
        ('runs', CoreRun),
        ('readings', Reading),
        ('missing', MissingInterval),
        ('samples', SampleInterval),
    ]:
        rows = s.scalars(select(cls).where(cls.borehole_id == borehole_id)).all()
        parts[key] = [pack(r) for r in rows]
    parts['strata'] = [pack(r) for r in _related_rows(s, Stratum, borehole_id)]
    parts['seals'] = [pack(r) for r in _related_rows(s, Seal, borehole_id)]
    parts['custodies'] = [pack(r) for r in _related_rows(s, Custody, borehole_id)]
    interps = s.scalars(
        select(Interpretation).where(Interpretation.borehole_id == borehole_id)
    ).all()
    parts['interpretations'] = [pack(r) for r in interps]
    return logic.checksum(parts)


def _related_rows(s, cls, borehole_id):
    """Stratum/Seal/Custody 没有直接 borehole_id，沿关联关系取本孔记录。"""
    interp_ids = s.scalars(
        select(Interpretation.id).where(Interpretation.borehole_id == borehole_id)
    ).all()
    if not interp_ids:
        return []
    if cls is Stratum:
        return s.scalars(select(Stratum).where(Stratum.interpretation_id.in_(interp_ids))).all()
    sample_ids = s.scalars(
        select(SampleInterval.id).where(SampleInterval.borehole_id == borehole_id)
    ).all()
    if not sample_ids:
        return []
    if cls is Seal:
        return s.scalars(select(Seal).where(Seal.sample_id.in_(sample_ids))).all()
    seals = s.scalars(select(Seal.id).where(Seal.sample_id.in_(sample_ids))).all()
    if not seals:
        return []
    return s.scalars(select(Custody).where(Custody.seal_id.in_(seals))).all()


def verify(s, borehole_id, confirmed_interpretation=None, readings=None):
    """按原解释所用测点复算并给出核验结论（纯读操作，不写库）。"""
    if readings is None:
        readings = s.scalars(
            select(Reading).where(Reading.borehole_id == borehole_id)
            .order_by(Reading.depth_m)
        ).all()
    current_rows = [pack(r) for r in readings]
    basis_hash = revision_basis_hash(s, borehole_id)

    interp = confirmed_interpretation
    if interp is None:
        interp = s.scalar(select(Interpretation).where(
            Interpretation.borehole_id == borehole_id,
            Interpretation.state == 'confirmed',
        ))
    if interp is None:
        return logic._hold(['本孔尚无主管确认的层段解释，无法复算冻结依据'], basis_hash)

    return logic.build_verification(
        interp.evidence or {}, current_rows, fit_layers, basis_hash, interp.id,
    )
