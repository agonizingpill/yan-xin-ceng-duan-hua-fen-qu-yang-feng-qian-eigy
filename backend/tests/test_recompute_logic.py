"""复算核验纯领域逻辑测试，仅用标准库：python -m unittest discover -s tests。

覆盖需求：
1) 用原解释同批测点重算；2) 层型编号互换不算差异；
3) 测点归类/候选接触面变化、测点集合或物性变化 → 暂缓；
4) 旧解释缺复算依据 → 留在草稿（hold）。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import recompute_logic as logic

FEATURES = ['density', 'susceptibility', 'velocity']


def rows(n=12):
    out = []
    for i in range(n):
        out.append({'id': i + 1, 'depth_m': 0.5 + i, 'run_id': 1, 'borehole_id': 7,
                    'density': 2.0 + 0.1 * (i % 3),
                    'susceptibility': 10.0 + i,
                    'velocity': 2000.0 + 50 * i})
    return out


def evidence_from(rows, classified, contacts, k=3, sha=None):
    ids = [r['id'] for r in rows]
    cut = int(len(ids) * 0.8)
    return {
        'classified': classified,
        'suggested_contacts_m': contacts,
        'source_sha256': sha or logic.checksum(rows),
        'train_reading_ids': ids[:cut],
        'holdout_reading_ids': ids[cut:],
        'chosen_k': k,
    }


def classify(rows, cluster_of):
    return [{'reading_id': r['id'], 'depth_m': r['depth_m'], 'cluster': cluster_of(r),
             'posterior': 0.99, **{f: r[f] for f in FEATURES}} for r in rows]


def fit_stub(rows, cluster_of, contacts, k=3):
    def _fit(_rows):
        return {
            'classified': classify(rows, cluster_of),
            'suggested_contacts_m': list(contacts),
            'chosen_k': k,
            'bic_candidates': [{'k': k, 'bic': -1.0}],
            'holdout_stability_ari': 1.0,
        }
    return _fit


def cluster_three(r):
    # 每 4 个点一层，制造两个接触面
    return min(r['depth_m'] // 4, 2) + 1


class TestComparators(unittest.TestCase):
    def test_checksum_deterministic_and_sensitive(self):
        a = rows()
        self.assertEqual(logic.checksum(a), logic.checksum([dict(x) for x in a]))
        b = [dict(x) for x in a]
        b[0]['velocity'] += 1
        self.assertNotEqual(logic.checksum(a), logic.checksum(b))
        self.assertEqual(len(logic.checksum(a)), 64)

    def test_label_permutation_identity_and_swap(self):
        original = classify(rows(), cluster_three)
        swapped = [dict(x, cluster={1: 2, 2: 1, 3: 3}[x['cluster']]) for x in original]
        self.assertEqual(logic.label_permutation(original, original),
                         {1: 1, 2: 2, 3: 3})
        self.assertEqual(logic.label_permutation(original, swapped),
                         {1: 2, 2: 1, 3: 3})

    def test_label_permutation_conflict_returns_none(self):
        original = [{'reading_id': 1, 'cluster': 1}, {'reading_id': 2, 'cluster': 1}]
        broken = [{'reading_id': 1, 'cluster': 2}, {'reading_id': 2, 'cluster': 3}]
        self.assertIsNone(logic.label_permutation(original, broken))

    def test_partition_equivalent_relabel_is_not_a_difference(self):
        original = classify(rows(), cluster_three)
        swapped = [dict(x, cluster={1: 3, 2: 2, 3: 1}[x['cluster']]) for x in original]
        self.assertTrue(logic.partition_equivalent(original, swapped))

    def test_partition_different_is_difference(self):
        original = classify(rows(), cluster_three)
        changed = [dict(x) for x in original]
        changed[0]['cluster'] = 2  # 原与第 1–4 点同层，改到第二层即破坏同/异组关系
        self.assertFalse(logic.partition_equivalent(original, changed))

    def test_partition_length_mismatch(self):
        self.assertFalse(logic.partition_equivalent(
            [{'cluster': 1}], [{'cluster': 1}, {'cluster': 2}]))

    def test_contacts_equal_tolerance_order_and_length(self):
        self.assertTrue(logic.contacts_equal([4.0, 8.0], [8.0, 4.0 + 1e-4]))
        self.assertFalse(logic.contacts_equal([4.0], [4.0, 8.0]))
        self.assertFalse(logic.contacts_equal([4.0], [4.5]))


class TestBuildVerification(unittest.TestCase):
    def setUp(self):
        self.rows = rows()
        self.cls = classify(self.rows, cluster_three)
        self.contacts = [4.0, 8.0]
        self.ev = evidence_from(self.rows, self.cls, self.contacts)

    def _run(self, ev=None, fit=None, basis='bh'):
        return logic.build_verification(
            ev if ev is not None else self.ev, self.rows,
            fit or fit_stub(self.rows, cluster_three, self.contacts),
            basis, interpretation_id=3)

    def test_identical_recompute_passes(self):
        v = self._run()
        self.assertEqual(v['status'], 'passed')
        self.assertEqual(v['hold_reasons'], [])
        self.assertEqual(v['reading_count'], len(self.rows))
        self.assertEqual(v['original_contacts_m'], v['recomputed_contacts_m'])
        self.assertTrue(v['partition_equivalent'])
        self.assertEqual(v['basis_hash'], 'bh')
        self.assertEqual(v['interpretation_id'], 3)

    def test_missing_evidence_holds_draft(self):
        # 旧纸本解释：只有说明，没有 GMM 复算依据
        v = self._run(ev={'note': '纸本区间来源'})
        self.assertEqual(v['status'], 'hold')
        self.assertTrue(any('复算依据' in x for x in v['hold_reasons']))

    def test_empty_evidence_holds(self):
        v = self._run(ev={})
        self.assertEqual(v['status'], 'hold')

    def test_reading_set_changed_holds(self):
        fewer = self.rows[:-1]
        v = logic.build_verification(
            self.ev, fewer, fit_stub(fewer, cluster_three, self.contacts), 'bh', 3)
        self.assertEqual(v['status'], 'hold')
        self.assertTrue(any('测点集合' in x for x in v['hold_reasons']))

    def test_reading_value_changed_holds_and_old_check_kept(self):
        changed = [dict(x) for x in self.rows]
        changed[2]['density'] += 0.5
        # 当前测点的 sha 与 evidence 中记录的原始 sha 不同 → 旧核验不能放行
        v = logic.build_verification(
            evidence_from(self.rows, self.cls, self.contacts), changed,
            fit_stub(changed, cluster_three, self.contacts), 'bh', 3)
        self.assertEqual(v['status'], 'hold')
        self.assertTrue(any('SHA256' in x for x in v['hold_reasons']))
        self.assertNotEqual(v['original_source_sha256'], v['current_source_sha256'])

    def test_fit_failure_holds(self):
        def boom(_):
            raise ValueError('物性缺少变化或包含非有限值')
        v = self._run(fit=boom)
        self.assertEqual(v['status'], 'hold')
        self.assertTrue(any('复算拟合失败' in x for x in v['hold_reasons']))

    def test_relabel_only_passes_and_records_permutation(self):
        swap = {1: 2, 2: 1, 3: 3}
        swapped_cls = [dict(x, cluster=swap[x['cluster']]) for x in self.cls]
        v = self._run(fit=fit_stub(self.rows,
                                   lambda r: swap[cluster_three(r)], self.contacts))
        self.assertEqual(v['status'], 'passed', v['hold_reasons'])
        self.assertTrue(v['partition_equivalent'])
        self.assertEqual(v['label_permutation'], {1: 2, 2: 1, 3: 3})
        self.assertTrue(any('编号互换不算差异' in x for x in [v['note']]))
        # 编号互换后接触面（按深度序列相邻变化计算）保持不变
        self.assertEqual(v['recomputed_contacts_m'], self.contacts)

    def test_classification_drift_holds_with_depth(self):
        def drifted(r):
            c = cluster_three(r)
            return 2 if r['id'] == 1 else c  # 第一个测点改归第二层
        v = self._run(fit=fit_stub(self.rows, drifted, [0.5, 4.0, 8.0]))
        self.assertEqual(v['status'], 'hold')
        self.assertFalse(v['partition_equivalent'])
        joined = ' '.join(v['hold_reasons'])
        self.assertTrue(any('测点归类' in x for x in v['hold_reasons']))
        self.assertIn('0.5', joined)

    def test_contacts_change_holds(self):
        v = self._run(fit=fit_stub(self.rows, cluster_three, [3.5, 8.0]))
        self.assertEqual(v['status'], 'hold')
        self.assertTrue(any('候选接触面' in x for x in v['hold_reasons']))

    def test_relabel_with_same_contacts_is_not_difference(self):
        swap = {1: 3, 2: 2, 3: 1}
        v = self._run(fit=fit_stub(self.rows,
                                   lambda r: swap[cluster_three(r)], self.contacts))
        self.assertEqual(v['status'], 'passed')

    def test_reading_sequence_mismatch_holds(self):
        # 复算返回的 reading_id 顺序集合不同
        def bad_fit(_):
            cls = classify(self.rows, cluster_three)
            cls[0]['reading_id'] = 999
            return {'classified': cls, 'suggested_contacts_m': self.contacts,
                    'chosen_k': 3, 'bic_candidates': [], 'holdout_stability_ari': 1.0}
        v = self._run(fit=bad_fit)
        self.assertEqual(v['status'], 'hold')
        self.assertTrue(any('测点序列' in x for x in v['hold_reasons']))

    def test_hold_shape_keeps_hashes(self):
        v = self._run(ev={'note': '纸本'})
        self.assertEqual(v['original_source_sha256'], None)
        self.assertEqual(v['current_source_sha256'], None)
        self.assertEqual(v['basis_hash'], 'bh')


if __name__ == '__main__':
    unittest.main()
