import io,json,sys,tempfile,time,unittest,zipfile,base64,hashlib
from pathlib import Path
from unittest.mock import MagicMock,patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from drive_resolution_execution import ObservedAPI
import resolution_experiment as c


class ExecutionTransportTests(unittest.TestCase):
    def test_missing_or_paid_or_stale_quota_never_pushes(self):
        api=MagicMock();g=ObservedAPI(api)
        with self.assertRaises(ValueError):g.kernels_push('/unused',timeout='2400',acc='NvidiaTeslaT4')
        g.last_quota=(time.time(),{'gpuQuota':{'isPayToScaleEnabled':True,'timeReserved':'0s'}})
        with self.assertRaises(ValueError):g.kernels_push('/unused',timeout='2400',acc='NvidiaTeslaT4')
        g.last_quota=(time.time()-100,{'gpuQuota':{'isPayToScaleEnabled':False,'timeReserved':'0s'}})
        with self.assertRaises(ValueError):g.kernels_push('/unused',timeout='2400',acc='NvidiaTeslaT4')
        api.kernels_push.assert_not_called()

    def test_private_source_archive_is_verified_without_running_code(self):
        from audit_resolution_execution import embedded_source
        code=b'raise RuntimeError("this fixture must never execute")';name='src/fixture.py';b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr(name,code)
        digest=hashlib.sha256(name.encode()+code).hexdigest()
        text='import base64,json\nbase64.b64decode('+repr(base64.b64encode(b.getvalue()).decode())+')\njson.loads('+repr(json.dumps({'image_size':320}))+')\n'
        with tempfile.TemporaryDirectory() as t,patch.object(c,'WORKER_FILES',[name]):
            p=Path(t)/'execute.py';p.write_text(text)
            self.assertEqual(embedded_source(p,digest,{'image_size':320}),digest)
            with self.assertRaises(ValueError):embedded_source(p,digest,{'image_size':224})
            with self.assertRaises(ValueError):embedded_source(p,'different',{'image_size':320})


if __name__=='__main__':unittest.main()
