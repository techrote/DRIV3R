"""Source-only and optional actual-archive regression tests.

For the complete replay set DRIV3R_MATERIAL_EPOCH_ARCHIVE to the retained ZIP;
absence is an explicit skip, never a claim that science was reproduced.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest import TestCase
from zipfile import ZipInfo

from .postmerge_archive_replay import git_blob_sha, run, validate_members


class PostmergeReplayTests(TestCase):
    def test_git_blob_identity(self):
        self.assertEqual(git_blob_sha(b'hello\n'),
                         'ce013625030ba8dba906f756967f9e9ca394464a')

    def test_unsafe_archive_rejected(self):
        for bad in ('../escape', '/absolute', 'a\\b', './x', 'x//y'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_members([ZipInfo(bad)])
        with self.assertRaises(ValueError):
            validate_members([ZipInfo('x'), ZipInfo('x')])

    def test_symlink_archive_rejected(self):
        link = ZipInfo('external_checkpoints/711-conditioned.npz')
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaisesRegex(ValueError, 'symlink'):
            validate_members([link])

    def test_missing_evidence_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            validate_members([ZipInfo('README')])

    def test_bounded_compute(self):
        for count in (0, -1, 5, True, 2.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                run(Path('/not/used.zip'), count)

    def test_external_archived_replay_when_supplied(self):
        archive = os.environ.get('DRIV3R_MATERIAL_EPOCH_ARCHIVE')
        if not archive:
            self.skipTest('External original checkpoint archive not supplied')
        result = run(Path(archive), 2)
        self.assertEqual(result['verification']['episodes'], 320)
        self.assertEqual(result['verification']['action_receipts'], 16140)
        self.assertEqual(result['verification']['probability_vectors'], 8542)
        self.assertEqual(result['native_receipts'], 0)
