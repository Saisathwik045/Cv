"""Trusted-stage reuse must never weaken validation or source immutability."""
import importlib.metadata
import json
from pathlib import Path
import tempfile
import unittest
from synthetic import make_fixture
from trustretina.data.validation import validate_dataset,file_hash,DataValidationError
from trustretina.data.stages import resume_validation
import trustretina.data.stages as stage_module


class ValidationStagesTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.config=make_fixture(self.temp.name)
        self.inventory=validate_dataset(self.config)
        self.path=Path(self.temp.name)/'stage.json'
        self.snapshot={'stage':'validated_inventory','nonce':'a'*32,'source_commit':'b'*40,
                       'training':False,'gpu_requested':False,'payload':self.inventory}
        self.path.write_text(json.dumps(self.snapshot))
        here=Path(stage_module.__file__)
        self.expected={'expected_sha256':file_hash(self.path),'expected_nonce':'a'*32,
                       'expected_source_commit':'b'*40,
                       'expected_source_hashes':{'validation.py':file_hash(here.with_name('validation.py')),
                                                'duplicates.py':file_hash(here.with_name('duplicates.py')),
                                                'config.py':file_hash(here.parents[1]/'config.py')},
                       'expected_packages':{n:importlib.metadata.version(n) for n in ('Pillow','openpyxl')}}

    def test_verified_stage_reuses_identical_inventory_after_rehashing_inputs(self):
        inventory,proof=resume_validation(self.config,self.path,**self.expected)
        self.assertEqual(inventory,self.inventory)
        self.assertTrue(proof['raw_hashes_revalidated'])

    def test_changed_raw_image_or_workbook_rejected(self):
        for path in (Path(self.config['data']['root'])/'data.xlsx',
                     Path(self.config['data']['root'])/'Training Images/0_left.png'):
            raw=path.read_bytes();path.write_bytes(b'changed')
            with self.assertRaisesRegex(DataValidationError,'changed since validation'):
                resume_validation(self.config,self.path,**self.expected)
            path.write_bytes(raw)

    def test_incomplete_or_other_execution_cannot_reuse_stage(self):
        for key,value in [('stage','image_inventory'),('nonce','c'*32),('source_commit','d'*40),('training',True)]:
            snapshot=dict(self.snapshot);snapshot[key]=value;self.path.write_text(json.dumps(snapshot))
            expected={**self.expected,'expected_sha256':file_hash(self.path)}
            with self.assertRaisesRegex(DataValidationError,'identity/stage'):
                resume_validation(self.config,self.path,**expected)
        self.path.write_text(json.dumps(self.snapshot))
        self.path.write_text(self.path.read_text()+' ')
        with self.assertRaisesRegex(DataValidationError,'checksum'):
            resume_validation(self.config,self.path,**self.expected)

    def test_changed_implementation_settings_or_dependencies_require_revalidation(self):
        options=[{**self.expected,'expected_source_hashes':{**self.expected['expected_source_hashes'],'validation.py':'0'*64}},
                 {**self.expected,'expected_packages':{'Pillow':'0','openpyxl':'0'}}]
        for expected in options:
            with self.assertRaises(DataValidationError):resume_validation(self.config,self.path,**expected)
        self.config['data']['kind']='original_odir'
        with self.assertRaisesRegex(DataValidationError,'data settings'):
            resume_validation(self.config,self.path,**self.expected)
