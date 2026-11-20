"""服务层测试支撑：第三方库桩 + 内存假 ORM。

本机没有 fastapi/sqlalchemy/sklearn，用 sys.modules 注入最小桩，
使 revision_service / recompute_service 可在标准库下跑通冻结状态机。
"""
import sys
import types
from datetime import datetime, timezone

COLUMNS = {
    'Borehole': ['id', 'code', 'site', 'total_depth', 'cut_quota', 'state'],
    'CoreRun': ['id', 'borehole_id', 'code', 'start_m', 'end_m', 'recovered_m'],
    'Reading': ['id', 'borehole_id', 'run_id', 'depth_m', 'density',
                'susceptibility', 'velocity'],
    'MissingInterval': ['id', 'borehole_id', 'start_m', 'end_m', 'reason', 'state'],
    'Interpretation': ['id', 'borehole_id', 'state', 'evidence', 'error'],
    'Stratum': ['id', 'interpretation_id', 'start_m', 'end_m', 'lithology',
                'contact_note'],
    'SampleInterval': ['id', 'borehole_id', 'stratum_id', 'code', 'start_m',
                       'end_m', 'purpose', 'destructive', 'state'],
    'Seal': ['id', 'sample_id', 'code', 'previous_id', 'state', 'reason',
             'sealed_by', 'snapshot'],
    'Custody': ['id', 'seal_id', 'receiver', 'destination', 'condition', 'at'],
    'ColumnRevision': ['id', 'borehole_id', 'version', 'note', 'state',
                       'snapshot', 'frozen_by', 'verification'],
    'Audit': ['id', 'actor', 'action', 'at'],
}


def install_stubs():
    if 'sqlalchemy' in sys.modules:
        return
    sa = types.ModuleType('sqlalchemy')

    class _Column:
        """模型类列的占位标记，select(Cls.id) 时可反查所属表类。"""
        def __init__(self, table, name):
            self.table = table
            self.name = name

        def in_(self, values):
            return _Expr(self, 'in', list(values))

        def __eq__(self, other):
            return _Expr(self, 'eq', other)

        def __ne__(self, other):
            return _Expr(self, 'ne', other)

        def __hash__(self):
            return hash((self.table.__name__, self.name))

    class _Expr:
        def __init__(self, column, op, value):
            self.column = column
            self.op = op
            self.value = value

        def __and__(self, other):
            return _Group([self, other])

    class _Group:
        def __init__(self, exprs):
            self.exprs = exprs

        def __and__(self, other):
            return _Group(self.exprs + getattr(other, 'exprs', [other]))

    class _Select:
        def __init__(self, *entities):
            self.entities = []
            self.scalar_cols = []
            for e in entities:
                if isinstance(e, _Column):
                    self.entities.append(e.table)
                    self.scalar_cols.append(e)
                elif isinstance(e, type):
                    self.entities.append(e)
            self._where = []
            self._order = None

        def where(self, *c):
            for cnd in c:
                self._where.extend(getattr(cnd, 'exprs', [cnd]))
            return self

        def order_by(self, *c):
            self._order = c
            return self

        def with_for_update(self):
            return self

        def execution_options(self, **kw):
            return self

    sa.select = lambda *entities: _Select(*entities)
    sa._Column = _Column
    for name in ['String', 'Float', 'ForeignKey', 'JSON', 'Boolean',
                 'CheckConstraint', 'UniqueConstraint', 'Text']:
        setattr(sa, name, lambda *a, **kw: None)

    orm = types.ModuleType('sqlalchemy.orm')

    class Mapped:
        def __class_getitem__(cls, item):
            return item

    def mapped_column(*a, **kw):
        # 真实表类在 _bind 阶段用 _Column 替换，这里先给 None 占位
        return None
    orm.Mapped = Mapped
    orm.mapped_column = mapped_column

    fastapi = types.ModuleType('fastapi')

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)
    fastapi.HTTPException = HTTPException
    fastapi.APIRouter = object

    database = types.ModuleType('src.database')
    database.__path__ = []

    class Base:
        pass

    class _Engine:
        pass
    database.Base = Base
    database.engine = _Engine()
    database.Session = None  # 由 patch_session 指向 FakeSession 工厂

    common = types.ModuleType('src.common')
    common.__path__ = []

    def now():
        return datetime.now(timezone.utc).isoformat()

    def pack(x):
        out = {}
        for c in x._columns:
            v = getattr(x, c, None)
            if isinstance(v, _Column):  # 未赋值实例属性取到类级列占位
                v = None
            out[c] = v
        return out

    def get(s, cls, id, lock=False):
        for o in s.all_of(cls):
            if o.id == id:
                return o
        raise fastapi.HTTPException(404, '关联记录不存在')

    def hole(s, id):
        h = get(s, _models.Borehole, id)
        if getattr(h, 'state', None) != 'open':
            raise fastapi.HTTPException(409, '柱状图已冻结，钻孔业务记录不能改写')
        return h

    def audit(s, u, msg):
        audit_cls = _models.Audit
        audit_cls._audit_seq = getattr(audit_cls, '_audit_seq', 5000) + 1
        s.add(audit_cls(id=audit_cls._audit_seq, actor=u['email'],
                        action=msg, at=now()))

    def interval(start, end, total):
        if end <= start or end > total:
            raise fastapi.HTTPException(409, '区间终点须大于起点且不超孔深')

    def overlaps(a, b, c, d):
        return a < d - 1e-8 and b > c + 1e-8

    common.now = now
    common.pack = pack
    common.get = get
    common.hole = hole
    common.audit = audit
    common.interval = interval
    common.overlaps = overlaps

    # layer_model 的重型依赖：测试用 fit 桩替换 fit_layers，这里仅保证 import 通过
    numpy = types.ModuleType('numpy')
    sklearn = types.ModuleType('sklearn')
    preprocessing = types.ModuleType('sklearn.preprocessing')
    mixture = types.ModuleType('sklearn.mixture')
    model_selection = types.ModuleType('sklearn.model_selection')
    metrics = types.ModuleType('sklearn.metrics')
    preprocessing.StandardScaler = object
    mixture.GaussianMixture = object
    model_selection.train_test_split = lambda *a, **kw: None
    metrics.adjusted_rand_score = lambda *a, **kw: 1.0
    sys.modules.update({
        'numpy': numpy,
        'sklearn': sklearn,
        'sklearn.preprocessing': preprocessing,
        'sklearn.mixture': mixture,
        'sklearn.model_selection': model_selection,
        'sklearn.metrics': metrics,
    })

    sys.modules.update({
        'sqlalchemy': sa,
        'sqlalchemy.orm': orm,
        'fastapi': fastapi,
        'src.database': database,
        'src.common': common,
    })


install_stubs()
from src import models as _models  # noqa: E402


class _EntityBase:
    """每张表用独立子类，避免共用基类 __dict__ 导致 state 等列串扰。"""
    _seq = 1000  # 测试工厂用显式小 id，自动 id 从 1000 起避免覆盖
    _columns = ()

    def __init__(self, **kw):
        type(self)._seq += 1
        explicit_id = kw.pop('id', None)
        self.id = explicit_id if explicit_id is not None else type(self)._seq
        # 先把本类其余列按默认值落到实例，避免读到被旧实例改写的类属性
        defaults = _DEFAULTS.get(type(self).__name__, {})
        for c in self._columns:
            if c == 'id':
                continue
            setattr(self, c, defaults.get(c))
        for k, v in kw.items():
            setattr(self, k, v)

    def __repr__(self):
        return '<%s %s>' % (type(self).__name__, self.id)


# 向后兼容名称
Entity = _EntityBase


_DEFAULTS = {
    'Borehole': {'code': 'BH-T', 'site': '测试孔位', 'total_depth': 12,
                 'cut_quota': 6, 'state': 'open'},
    'Reading': {'borehole_id': 100},
    'MissingInterval': {'borehole_id': 100, 'state': 'annotated', 'reason': ''},
    'Interpretation': {'borehole_id': 100, 'state': 'confirmed',
                       'evidence': None, 'error': ''},
    'SampleInterval': {'borehole_id': 100, 'state': 'transferred',
                       'destructive': True},
    'ColumnRevision': {'state': 'draft', 'snapshot': {}, 'frozen_by': '',
                       'verification': {'status': 'pending', 'hold_reasons': []}},
    'Seal': {'state': 'intact', 'reason': '', 'previous_id': None,
             'snapshot': None, 'sealed_by': ''},
}


def _bind(name):
    base = getattr(_models, name)
    attrs = {'_columns': COLUMNS[name], '_seq': 1000}
    attrs.update({k: v for k, v in _DEFAULTS.get(name, {}).items()})
    cls = type(name, (_EntityBase, base), attrs)
    import sqlalchemy as _sa
    for col in COLUMNS[name]:
        setattr(cls, col, _sa._Column(cls, col))
    setattr(_models, name, cls)
    return cls


Borehole = _bind('Borehole')
CoreRun = _bind('CoreRun')
Reading = _bind('Reading')
MissingInterval = _bind('MissingInterval')
Interpretation = _bind('Interpretation')
Stratum = _bind('Stratum')
SampleInterval = _bind('SampleInterval')
Seal = _bind('Seal')
Custody = _bind('Custody')
ColumnRevision = _bind('ColumnRevision')
Audit = _bind('Audit')


# ---------- 假 Session ----------
class _Savepoint:
    """模拟 SAVEPOINT：rollback 原地把各对象属性恢复到保存点快照，保持对象身份。

    这样 service 中持有的 x 引用在回滚后仍指向 store 内同一对象，后续留档写回有效。
    """
    def __init__(self, session):
        import copy
        self._session = session
        self._snapshot = copy.deepcopy(session._store)

    def rollback(self):
        current = self._session._store
        # 删除保存点之后新增的对象
        for k in list(current):
            if k not in self._snapshot:
                del current[k]
        # 原地恢复既有对象属性，保留对象身份
        for k, old in self._snapshot.items():
            if k in current:
                current[k].__dict__.clear()
                current[k].__dict__.update(old.__dict__)
            else:
                current[k] = old

    def commit(self):
        pass


class FakeSession:
    # 类级指向当前 store，供 revision_service._persist_hold 的独立事务使用
    current_store = None

    def __init__(self, store=None):
        self._store = store if store is not None else FakeSession.current_store
        self.committed = 0
        self.rolled_back = 0

    @classmethod
    def begin(cls):
        return cls(cls.current_store)

    def add(self, obj):
        self._store[obj.id] = obj

    def flush(self):
        pass

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1

    def begin_nested(self):
        # 内存库无真实 SAVEPOINT：记录快照，rollback 恢复到保存点状态
        import copy
        return _Savepoint(self)

    def all_of(self, cls):
        name = cls.__name__
        return [o for o in self._store.values() if type(o).__name__ == name]

    def scalar(self, q):
        results = self._run(q)
        if not results:
            return None
        if q.scalar_cols:
            return getattr(results[0], q.scalar_cols[0].name)
        return results[0]

    def scalars(self, q):
        rows = self._run(q)
        if q.scalar_cols:
            rows = [getattr(o, q.scalar_cols[0].name) for o in rows]
        return _Scalars(rows)

    def execute(self, q):
        return None

    def _run(self, q):
        name = q.entities[0].__name__
        out = [o for o in self._store.values() if type(o).__name__ == name]
        # 近似 where 过滤：本孔记录带 borehole_id；关联表沿外键处理
        out = _apply_where(out, q)
        if q._order:
            key = q._order[0]
            col = getattr(key, 'name', None)
            if col:
                out.sort(key=lambda o: getattr(o, col, 0))
        return out

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


def _apply_where(rows, q):
    """对收集到的 _Expr 做内存求值；关联表沿外键解析。"""
    store = None
    for expr in q._where:
        col, op, val = expr.column, expr.op, expr.value
        name = col.table.__name__

        def attr(o):
            return getattr(o, col.name, None)

        if op == 'eq':
            rows = [o for o in rows if attr(o) == val]
        elif op == 'ne':
            rows = [o for o in rows if attr(o) != val]
        elif op == 'in':
            rows = [o for o in rows if attr(o) in val]
    return rows


# ---------- 测试数据工厂 ----------
def make_store(readings, evidence, *, sample_state='transferred',
               missing_annotated=True, run_end=12):
    """构造一个满足全部结构性冻结检查的本孔数据集（每次全新实体）。"""
    import copy
    store = {}

    def put(obj):
        store[obj.id] = obj
        return obj

    put(copy.deepcopy(Borehole(id=100, total_depth=12, cut_quota=6,
                                code='BH-T', state='open')))
    put(copy.deepcopy(CoreRun(id=110, borehole_id=100, code='R1', start_m=0,
                               end_m=run_end, recovered_m=11)))
    for r in readings:
        r.borehole_id = 100
        r.run_id = 110
        put(copy.deepcopy(r))
    put(copy.deepcopy(MissingInterval(
        id=120, borehole_id=100, start_m=12, end_m=13,
        reason='运输缺芯说明不少于八字',
        state='annotated' if missing_annotated else 'observed')))
    put(copy.deepcopy(Interpretation(id=130, borehole_id=100, state='confirmed',
                                     evidence=copy.deepcopy(evidence))))
    put(copy.deepcopy(Stratum(id=140, interpretation_id=130, start_m=0, end_m=12,
                               lithology='测试砂岩',
                               contact_note='人工接触面依据不少于八字')))
    put(copy.deepcopy(SampleInterval(id=150, borehole_id=100, stratum_id=140,
                                     code='CS-T1', start_m=2, end_m=3,
                                     purpose='测试切样', state=sample_state)))
    put(copy.deepcopy(Seal(id=160, sample_id=150, code='CX-T1')))
    put(copy.deepcopy(Custody(id=170, seal_id=160, receiver='实验室收样人',
                               destination='试验室',
                               condition='封签完整交接条件不少于八字', at='now')))
    return store


def readings_and_evidence(n=12):
    """确定性数据集；测试再用 fit 桩原样回显，使复算恒等通过。"""
    feats = ['density', 'susceptibility', 'velocity']
    rs = [{'id': i + 1, 'depth_m': 0.5 + i, 'run_id': 110, 'borehole_id': 100,
           'density': 2.0 + 0.1 * (i % 3),
           'susceptibility': 10.0 + i, 'velocity': 2000.0 + 50 * i}
          for i in range(n)]
    readings = [Reading(**r) for r in rs]

    def cluster_of(r):
        return min(r['depth_m'] // 4, 2) + 1

    classified = [{'reading_id': r['id'], 'depth_m': r['depth_m'],
                   'cluster': cluster_of(r), 'posterior': 0.99,
                   **{f: r[f] for f in feats}} for r in rs]
    contacts = [4.0, 8.0]
    ids = [r['id'] for r in rs]
    cut = int(len(ids) * 0.8)
    from src import recompute_logic as logic
    evidence = {
        'classified': classified,
        'suggested_contacts_m': contacts,
        'source_sha256': logic.checksum(rs),
        'train_reading_ids': ids[:cut],
        'holdout_reading_ids': ids[cut:],
        'chosen_k': 3,
    }
    return readings, evidence, rs, classified, contacts
