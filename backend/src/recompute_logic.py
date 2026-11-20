"""层段拟合复算核验的纯领域逻辑。

不依赖 ORM 与第三方库，输入全部是 dict/list，可用标准库 unittest 直接验证：
- 用原解释 evidence 记录的同一批测点重跑拟合（fit_fn 注入，生产环境为 layer_model.fit_layers）；
- 层型（成分）编号互换不算差异：划分等价按同/异组关系判定，并给出编号置换；
- 测点集合、物性值、测点归类或候选接触面变化 → hold；
- 旧解释缺复算依据 → hold，柱状图留在草稿。
"""
import hashlib
import json

CONTACT_TOL = 1e-3
REQUIRED_EVIDENCE = (
    'classified', 'suggested_contacts_m', 'source_sha256',
    'train_reading_ids', 'holdout_reading_ids', 'chosen_k',
)


def checksum(rows):
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def label_permutation(original, current):
    """原成分编号 -> 现成分编号的映射；非纯置换时返回 None。"""
    mapping = {}
    for a, b in zip(original, current):
        old, new = a['cluster'], b['cluster']
        if old in mapping and mapping[old] != new:
            return None
        mapping[old] = new
    if set(mapping.values()) != {c['cluster'] for c in current}:
        return None
    return dict(sorted(mapping.items()))


def partition_equivalent(original, current):
    """层型编号互换不算差异：同深度序列两两同/异组关系完全一致即同一划分。"""
    if len(original) != len(current):
        return False
    for i in range(len(original)):
        for j in range(i + 1, len(original)):
            if (original[i]['cluster'] == original[j]['cluster']) != \
               (current[i]['cluster'] == current[j]['cluster']):
                return False
    return True


def contacts_equal(a, b):
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= CONTACT_TOL for x, y in zip(sorted(a), sorted(b)))


def _hold(reasons, basis_hash, original_sha=None, current_sha=None):
    return {
        'status': 'hold',
        'hold_reasons': reasons,
        'basis_hash': basis_hash,
        'original_source_sha256': original_sha,
        'current_source_sha256': current_sha,
    }


def _same_partner_set(i, w, original, current):
    for j in range(len(original)):
        if i == j:
            continue
        if (original[i]['cluster'] == original[j]['cluster']) != \
           (w['cluster'] == current[j]['cluster']):
            return False
    return True


def build_verification(evidence, current_rows, fit_fn, basis_hash, interpretation_id):
    """依据原解释证据与当前测点，给出 passed/hold 结论及完整比对依据。"""
    if not evidence:
        return _hold(['旧解释缺少复算依据，须重新拟合并主管确认后再冻结'], basis_hash)

    missing = [k for k in REQUIRED_EVIDENCE if not evidence.get(k)]
    if missing:
        # 旧纸本解释等没有 GMM 复算依据：留在草稿，不得放行冻结
        return _hold(
            ['旧解释缺少复算依据（缺字段：%s），须重新拟合并主管确认后再冻结'
             % '、'.join(missing)],
            basis_hash,
        )

    original_ids = list(evidence['train_reading_ids']) + list(evidence['holdout_reading_ids'])
    current_ids = [r['id'] for r in current_rows]
    if sorted(original_ids) != sorted(current_ids):
        only_old = sorted(set(original_ids) - set(current_ids))
        only_new = sorted(set(current_ids) - set(original_ids))
        return _hold(
            ['测点归类依据变化：原解释所用测点集合与当前测点不一致'
             '（原解释独有 %s；当前新增 %s），暂缓冻结'
             % (only_old or '无', only_new or '无')],
            basis_hash, evidence['source_sha256'],
        )

    current_sha = checksum(current_rows)
    if current_sha != evidence['source_sha256']:
        return _hold(
            ['测点物性值已变化：源数据 SHA256 与原解释不一致，原核验不能放行，须重新复算'],
            basis_hash, evidence['source_sha256'], current_sha,
        )

    try:
        # 确定性算法（固定 seed）：同一输入必须复现同一结果
        fresh = fit_fn(current_rows)
    except Exception as e:  # 恒定物性等真实失败
        return _hold(['复算拟合失败：%s' % str(e)[:300]], basis_hash,
                     evidence['source_sha256'], current_sha)

    original = sorted(evidence['classified'], key=lambda z: z['reading_id'])
    current = sorted(fresh['classified'], key=lambda z: z['reading_id'])
    if [z['reading_id'] for z in original] != [z['reading_id'] for z in current]:
        return _hold(['复算测点序列与原解释不一致，暂缓冻结'], basis_hash,
                     evidence['source_sha256'], current_sha)

    permutation = label_permutation(original, current)
    equivalent = partition_equivalent(original, current)
    relabeled = bool(equivalent and permutation and
                     any(str(k) != str(v) for k, v in permutation.items()))

    reasons = []
    if not equivalent:
        drifted = [z['depth_m'] for idx, (z, w) in enumerate(zip(original, current))
                   if not _same_partner_set(idx, w, original, current)]
        reasons.append('测点归类发生变化（深度 %s m），层段划分与原解释不同，暂缓冻结'
                       % ('、'.join(str(d) for d in drifted[:10]) or '见复算明细'))

    old_contacts, new_contacts = evidence['suggested_contacts_m'], fresh['suggested_contacts_m']
    if not contacts_equal(old_contacts, new_contacts):
        reasons.append('候选接触面有变化：原 %s m → 复算 %s m，暂缓冻结'
                       % (old_contacts or '无', new_contacts or '无'))

    if relabeled:
        note = '复算划分与原解释一致，仅层型编号重排 %s，按规则编号互换不算差异' % permutation
    elif equivalent:
        note = '复算使用原解释同一批测点，测点归类与候选接触面均一致'
    else:
        note = '复算使用原解释同一批测点，源数据校验一致，但划分存在差异'

    return {
        'status': 'hold' if reasons else 'passed',
        'hold_reasons': reasons,
        'basis_hash': basis_hash,
        'interpretation_id': interpretation_id,
        'original_source_sha256': evidence['source_sha256'],
        'current_source_sha256': current_sha,
        'reading_count': len(current_rows),
        'original_chosen_k': evidence['chosen_k'],
        'recomputed_chosen_k': fresh['chosen_k'],
        'original_contacts_m': old_contacts,
        'recomputed_contacts_m': new_contacts,
        'label_permutation': permutation,
        'partition_equivalent': equivalent,
        'recomputed_bic_candidates': fresh['bic_candidates'],
        'recomputed_ari': fresh['holdout_stability_ari'],
        'note': note,
    }
