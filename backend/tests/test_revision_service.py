"""冻结状态机服务层测试（内存假 ORM，无第三方依赖）。

覆盖：
- 复算通过 + 结构检查通过 → 冻结成功，快照含复算依据；
- 复算暂缓 → 草稿保留、冻结抛 409、原因落档；
- 冻结失败重试不产生新版本，成功后仍是同一条草稿；
- 草稿/源数据变化使 basis_hash 改变 → 旧 passed 核验作废、强制重算并暂缓；
- 结构性检查失败（缺段未说明/样袋未交接）也在同一草稿留暂缓原因；
- 已冻结版本重复冻结被拒绝。
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import _stubs  # noqa: E402,F401  先装桩
from tests._stubs import (  # noqa: E402
    FakeSession, ColumnRevision, Borehole, Interpretation, Reading,
    make_store, readings_and_evidence,
)
from src import revision_service as revision  # noqa: E402
from src import recompute_service as recompute  # noqa: E402
from src import models  # noqa: E402

USER = {'email': 'supervisor@liheng.local', 'role': 'supervisor'}
OPERATOR = {'email': 'operator@liheng.local', 'role': 'operator'}


def fit_echo_factory(classified, contacts, k=3):
    def _fit(rows):
        return {
            'classified': [dict(x) for x in classified],
            'suggested_contacts_m': list(contacts),
            'chosen_k': k,
            'bic_candidates': [{'k': k, 'bic': -1.0}],
            'holdout_stability_ari': 1.0,
        }
    return _fit


class RevisionTestCase(unittest.TestCase):
    def setUp(self):
        # 每个用例独立构造数据，避免全局桩类/实体跨用例污染
        self.readings, self.evidence, self.rs, self.cls, self.contacts = \
            readings_and_evidence()
        self.store = make_store(self.readings, self.evidence)
        FakeSession.current_store = self.store
        self.fit = fit_echo_factory(self.cls, self.contacts)

    def session(self):
        return FakeSession(self.store)

    def make_draft(self, verification_status=None):
        with patch.object(recompute, 'fit_layers', self.fit), \
                patch.object(revision.recompute, 'fit_layers', self.fit):
            s = self.session()
            p = type('P', (), {'borehole_id': 100, 'version': 'C01',
                               'note': '成果编制说明不少于八个字',
                               'model_dump': lambda self: {
                                   'borehole_id': 100, 'version': 'C01',
                                   'note': '成果编制说明不少于八个字'}})()
            rev = revision.create(s, p, OPERATOR)
        if verification_status == 'hold':
            self.store[rev['id']].verification['status'] = 'hold'
            self.store[rev['id']].verification['hold_reasons'] = ['预置暂缓原因']
        return rev['id']

    def test_freeze_success_keeps_verification_in_snapshot(self):
        rid = self.make_draft()
        with patch.object(recompute, 'fit_layers', self.fit):
            out = revision.freeze(self.session(), rid, USER)
        self.assertEqual(out['state'], 'frozen')
        self.assertEqual(out['frozen_by'], USER['email'])
        self.assertEqual(self.store[100].state, 'frozen')
        self.assertEqual(out['snapshot']['verification']['status'], 'passed')
        self.assertEqual(len(out['snapshot']['readings']), len(self.readings))

    def test_hold_blocks_freeze_and_keeps_draft_with_reasons(self):
        rid = self.make_draft(verification_status='hold')
        with self.assertRaises(Exception) as ctx:
            revision.freeze(self.session(), rid, USER)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn('暂缓冻结', ctx.exception.detail)
        # 仍是草稿，未新增柱状图版本，暂缓原因可查（允许写审计记录）
        self.assertEqual(self.store[rid].state, 'draft')
        revisions = [o for o in self.store.values()
                     if type(o).__name__ == 'ColumnRevision']
        self.assertEqual(len(revisions), 1)
        self.assertTrue(self.store[rid].verification['hold_reasons'])

    def test_retry_failure_then_success_uses_same_revision(self):
        rid = self.make_draft(verification_status='hold')
        for _ in range(2):
            with self.assertRaises(Exception):
                revision.freeze(self.session(), rid, USER)
        drafts = [o for o in self.store.values()
                  if isinstance(o, ColumnRevision) and o.state == 'draft']
        self.assertEqual(len(drafts), 1)
        # 补正：复算可通过后，在同一草稿上冻结成功
        self.store[rid].verification = {
            'status': 'passed', 'hold_reasons': [],
            'basis_hash': recompute.revision_basis_hash(self.session(), 100)}
        with patch.object(recompute, 'fit_layers', self.fit):
            out = revision.freeze(self.session(), rid, USER)
        self.assertEqual(out['id'], rid)
        self.assertEqual(out['state'], 'frozen')
        self.assertEqual(len([o for o in self.store.values()
                              if isinstance(o, ColumnRevision)]), 1)

    def test_stale_passed_verification_is_recomputed_and_held(self):
        rid = self.make_draft()
        self.assertEqual(self.store[rid].verification['status'], 'passed')
        # 源记录变化（补一个测点）使依据指纹改变，旧核验不得放行
        new_r = Reading(id=99, borehole_id=100, run_id=110, depth_m=11.6,
                        density=2.2, susceptibility=9.0, velocity=2100.0)
        self.store[99] = new_r
        # 重算时测点集合与原解释 evidence 不一致 → hold
        with patch.object(recompute, 'fit_layers', self.fit):
            with self.assertRaises(Exception) as ctx:
                revision.freeze(self.session(), rid, USER)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(self.store[rid].state, 'draft')
        self.assertEqual(self.store[rid].verification['status'], 'hold')
        self.assertTrue(any('测点集合' in x
                            for x in self.store[rid].verification['hold_reasons']))

    def test_revise_draft_recomputes_and_invalidates_old_verification(self):
        rid = self.make_draft()
        old_basis = self.store[rid].verification['basis_hash']
        # 让重算结论变为 hold：拟合桩抛错
        def boom(_):
            raise ValueError('物性缺少变化')
        p = type('P', (), {'borehole_id': 100, 'version': 'C01',
                           'note': '修订后的编制说明仍然不少于八字'})()
        with patch.object(recompute, 'fit_layers', boom):
            out = revision.revise(self.session(), rid, p, OPERATOR)
        self.assertEqual(out['verification']['status'], 'hold')
        self.assertTrue(any('复算拟合失败' in x
                            for x in out['verification']['hold_reasons']))
        self.assertEqual(self.store[rid].state, 'draft')
        # 依据指纹不变（只改 note），但结论已由 passed 变 hold
        self.assertEqual(out['verification']['basis_hash'], old_basis)

    def test_structural_failure_records_hold_on_same_draft(self):
        rid = self.make_draft()
        # 样袋未交接
        self.store[150].state = 'planned'
        with patch.object(recompute, 'fit_layers', self.fit):
            with self.assertRaises(Exception) as ctx:
                revision.freeze(self.session(), rid, USER)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn('样袋', ctx.exception.detail)
        self.assertEqual(self.store[rid].state, 'draft')

    def test_double_freeze_rejected(self):
        rid = self.make_draft()
        with patch.object(recompute, 'fit_layers', self.fit):
            revision.freeze(self.session(), rid, USER)
            with self.assertRaises(Exception) as ctx:
                revision.freeze(self.session(), rid, USER)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn('已冻结', ctx.exception.detail)

    def test_historical_evidence_without_basis_holds(self):
        # 旧纸本解释：evidence 无复算字段
        self.store[130].evidence = {'note': '旧纸本区间来源'}
        rid = self.make_draft()
        self.assertEqual(self.store[rid].verification['status'], 'hold')
        self.assertTrue(any('复算依据' in x
                            for x in self.store[rid].verification['hold_reasons']))
        with self.assertRaises(Exception):
            revision.freeze(self.session(), rid, USER)
        self.assertEqual(self.store[rid].state, 'draft')

    def test_manual_verify_endpoint_refreshes_conclusion(self):
        rid = self.make_draft(verification_status='hold')
        with patch.object(recompute, 'fit_layers', self.fit):
            out = revision.verify(self.session(), rid, USER)
        self.assertEqual(out['verification']['status'], 'passed')
        self.assertEqual(self.store[rid].verification['status'], 'passed')

    def test_legacy_draft_with_null_verification_is_recomputed(self):
        # 旧库已存在、没有 verification 字段的草稿：冻结时强制复算，不放行
        rid = self.make_draft()
        self.store[rid].verification = None
        with patch.object(recompute, 'fit_layers', self.fit):
            out = revision.freeze(self.session(), rid, USER)
        self.assertEqual(out['state'], 'frozen')
        self.assertEqual(out['snapshot']['verification']['status'], 'passed')

    def test_failed_freeze_keeps_single_draft_with_persisted_reasons(self):
        rid = self.make_draft(verification_status='hold')
        for _ in range(3):
            with self.assertRaises(Exception):
                revision.freeze(self.session(), rid, USER)
        revisions = [o for o in self.store.values()
                     if type(o).__name__ == 'ColumnRevision']
        self.assertEqual(len(revisions), 1)
        self.assertEqual(revisions[0].state, 'draft')
        self.assertTrue(revisions[0].verification['hold_reasons'])
        self.assertEqual(revisions[0].verification['structural_hold'], True)


if __name__ == '__main__':
    unittest.main()
