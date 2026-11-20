from sqlalchemy import select
from fastapi import HTTPException
from .models import (
    CoreRun, Reading, MissingInterval, Interpretation, Stratum,
    SampleInterval, Seal, Custody, ColumnRevision,
)
from .common import hole, get, pack, audit
from . import recompute_service as recompute


def create(s, p, u):
    h = hole(s, p.borehole_id)
    if s.scalar(select(ColumnRevision.id).where(
        ColumnRevision.borehole_id == h.id,
        ColumnRevision.state == 'draft',
    )):
        raise HTTPException(409, '已有待冻结版本，请继续原草稿')
    x = ColumnRevision(**p.model_dump())
    s.add(x)
    s.flush()
    # 建稿即复算：结论随草稿留档，暂缓原因页面可查
    x.verification = recompute.verify(s, h.id)
    audit(s, u, '编柱状图版本 ' + x.version + '，复算核验：' + x.verification['status'])
    return pack(x)


def revise(s, id, p, u):
    ref = get(s, ColumnRevision, id)
    h = hole(s, ref.borehole_id)
    x = get(s, ColumnRevision, id, True)
    if x.state != 'draft':
        raise HTTPException(409, '版本已冻结，不能修订')
    if p.borehole_id != h.id or p.version != x.version:
        raise HTTPException(409, '孔号和版本号固定，仅可修订编制说明')
    x.note = p.note
    # 改过的草稿不能拿旧核验放行：立即按当前源数据重新复算
    x.verification = recompute.verify(s, h.id)
    audit(s, u, '修订柱状图草稿 ' + x.version + '，重新复算核验：' + x.verification['status'])
    return pack(x)


def verify(s, id, u):
    ref = get(s, ColumnRevision, id)
    hole(s, ref.borehole_id)
    x = get(s, ColumnRevision, id, True)
    if x.state != 'draft':
        raise HTTPException(409, '版本已冻结，无需再核验')
    old = x.verification or {}
    x.verification = recompute.verify(s, x.borehole_id)
    stale = old.get('basis_hash') and old.get('basis_hash') != x.verification['basis_hash']
    audit(s, u, '复算核验柱状图 ' + x.version + '：' + x.verification['status']
          + ('（源数据已变，旧核验作废重算）' if stale else ''))
    return pack(x)


def _record_hold(s, x, reasons, savepoint, actor_email):
    """冻结不通过：回滚保存点内的检查性写入，只保留暂缓结论与审计，随外层事务提交。

    这样既不会让失败重试产生新版本，也能让暂缓原因在柱状图页面查到。
    """
    savepoint.rollback()
    x.verification = {**(x.verification or {}), 'status': 'hold',
                      'hold_reasons': reasons, 'structural_hold': True}
    audit(s, {'email': actor_email},
          '冻结暂缓留档 ' + x.version + '：' + '；'.join(reasons))


class FreezeHold(HTTPException):
    """冻结暂缓：路由层据此先提交留档，再向客户端返回 409。"""


def freeze(s, id, u):
    ref = get(s, ColumnRevision, id)
    h = hole(s, ref.borehole_id)
    x = get(s, ColumnRevision, id, True)
    if x.state != 'draft':
        raise HTTPException(409, '版本已冻结，不能重复')

    # 1) 冻结前复算。旧核验的依据指纹与当前不一致（草稿改过/源记录变过）则强制重算，
    #    旧核验永远不能放行已变化的草稿。
    sp = s.begin_nested()  # SAVEPOINT：失败只回滚检查段，留档由外层事务提交
    if not x.verification:
        x.verification = {'status': 'pending', 'hold_reasons': []}
    current_basis = recompute.revision_basis_hash(s, h.id)
    stale = (x.verification or {}).get('basis_hash') != current_basis
    if stale or x.verification.get('status') != 'passed':
        x.verification = recompute.verify(s, h.id)
    if x.verification.get('status') != 'passed':
        reasons = list(x.verification.get('hold_reasons') or ['层段拟合复算未通过，暂缓冻结'])
        _record_hold(s, x, reasons, sp, u['email'])
        raise FreezeHold(409, '层段复算未通过，暂缓冻结：' + '；'.join(reasons))

    # 2) 既有的结构性冻结检查（进尺/逐米测点或缺段/缺段说明/确认层面/样袋交接）
    try:
        runs = s.scalars(select(CoreRun).where(
            CoreRun.borehole_id == h.id).order_by(CoreRun.start_m)).all()
        edge = 0
        for r in runs:
            if abs(r.start_m - edge) > 1e-6:
                raise HTTPException(409, '进尺回次不连续，须补齐孔深记录')
            edge = r.end_m
        if abs(edge - h.total_depth) > 1e-6:
            raise HTTPException(409, '回次尚未覆盖全孔')
        points = s.scalars(select(Reading).where(
            Reading.borehole_id == h.id).order_by(Reading.depth_m)).all()
        missing = s.scalars(select(MissingInterval).where(
            MissingInterval.borehole_id == h.id)).all()
        if any(m.state != 'annotated' for m in missing):
            raise HTTPException(409, '缺失测段尚未补记原因，不得冻结')
        # 每一米测格须有实测点或明示缺段
        depth = .5
        while depth < h.total_depth:
            if not any(abs(r.depth_m - depth) < 1e-6 for r in points) and \
               not any(m.start_m <= depth < m.end_m for m in missing):
                raise HTTPException(409, '存在未标注的缺失测段 ' + str(depth) + 'm')
            depth += 1
        i = s.scalar(select(Interpretation).where(
            Interpretation.borehole_id == h.id,
            Interpretation.state == 'confirmed'))
        if not i:
            raise HTTPException(409, '须主管确认人工层段')
        samples = s.scalars(select(SampleInterval).where(
            SampleInterval.borehole_id == h.id,
            SampleInterval.state != 'cancelled')).all()
        if any(v.state != 'transferred' for v in samples):
            raise HTTPException(409, '尚有未完整封签交接样袋，不能冻结')
    except HTTPException as e:
        _record_hold(s, x, [e.detail], sp, u['email'])
        raise FreezeHold(409, e.detail) from e

    # 3) 全部通过：释放保存点，在同一草稿上落快照并冻结，重试不会产生新版本
    sp.commit()
    ids = [v.id for v in samples]
    seals = s.scalars(select(Seal).where(Seal.sample_id.in_(ids))).all() if ids else []
    sealids = [v.id for v in seals]
    cust = s.scalars(select(Custody).where(
        Custody.seal_id.in_(sealids))).all() if sealids else []
    x.snapshot = {
        'borehole': pack(h),
        'runs': [pack(v) for v in runs],
        'readings': [pack(v) for v in points],
        'missing': [pack(v) for v in missing],
        'interpretation': pack(i),
        'strata': [pack(v) for v in s.scalars(
            select(Stratum).where(Stratum.interpretation_id == i.id))],
        'samples': [pack(v) for v in samples],
        'seals': [pack(v) for v in seals],
        'custodies': [pack(v) for v in cust],
        'verification': x.verification,
    }
    x.state = 'frozen'
    x.frozen_by = u['email']
    h.state = 'frozen'
    audit(s, u, '冻结柱状图 ' + h.code + ' ' + x.version)
    return pack(x)
