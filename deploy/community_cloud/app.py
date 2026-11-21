"""Private Community Cloud entry point. No credentials/assets in the repository."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')


def private_setup(root, token):
    from deploy.community_cloud.recovery import private_directory
    root=private_directory(root)
    python=root/'recovery-env/bin/python'
    env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in
         ('TOKEN','SECRET','PASSWORD','CREDENTIAL','KAGGLE','GITHUB','GH_','PIP_'))}
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(root/'recovery-env')
    # Separate SDK environment: never mix its dependency lock with CPU app pins.
    install=subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','-r',
                            str(ROOT/'requirements/artifact-recovery.lock')],
                           env=env,capture_output=True,timeout=300)
    if install.returncode:raise ValueError('Recovery dependency setup failed')
    result=subprocess.run([str(python),'-B',str(ROOT/'deploy/community_cloud/recovery.py')],
                          input=json.dumps({'root':str(root),'token':token,'private_setup_confirmed':True}),text=True,
                          env=env,capture_output=True,timeout=600)
    if result.returncode or json.loads(result.stdout).get('status')!='PASS':
        raise ValueError('Bounded recovery failed')


def main():
    import streamlit as st
    try:settings=st.secrets.get('trustretina',{})
    except FileNotFoundError:settings={}
    if settings.get('owner_only_confirmed') is not True:
        st.title('TrustRetina private deployment setup')
        st.warning('Before model setup: verify Sharing → Only specific people can view this app. This setup screen is not authentication.')
        st.write('Deploy from the private source repository. Confirm owner-only platform access before adding private model credentials. Then set trustretina.owner_only_confirmed=true in Streamlit Secrets. Never put tokens or weights in GitHub or chat.')
        st.stop()
    from deploy.community_cloud.recovery import private_directory
    root=private_directory(Path.home()/'.trustretina-streamlit')
    bundle=root/'inference-v8'
    from trustretina.inference.bundle import verify_bundle
    if not bundle.exists():
        st.title('Prepare the original private model')
        st.warning('Owner-only sharing must already be active. No patient data is needed. No training, fitting or GPU jobs.')
        st.write('This explicit action retrieves only the four pinned original objects and verifies all five bundle hashes. No broad Kaggle output download, automatic retry or substitute model. The isolated recovery process receives the token via stdin; it is never passed on the command line.')
        st.caption('Exact expected payload 334,241,994 bytes; cumulative charged budget 668,483,988 bytes per private host storage lifetime, including failed reservations. A rebuild may replace ephemeral storage. Recovery is never automatic after a rebuild; click again only when authorized.')
        if st.button('Prepare verified original bundle',type='primary'):
            token=settings.get('kaggle_api_token')
            if not isinstance(token,str) or not token.strip():
                st.error('Add your own authorized Kaggle API token in Streamlit Secrets, not here or in chat.')
            else:
                try:
                    with st.spinner('Preparing isolated recovery environment and verifying the exact original bundle…'):
                        private_setup(root,token)
                    st.rerun()
                except (OSError,ValueError,RuntimeError,subprocess.SubprocessError):
                    st.error('Private setup stopped. No model is published on failure. Check source access, approved download host, disk and remaining transfer budget. No automatic retry is performed.')
        st.stop()
    try:verify_bundle(bundle)
    except (OSError,ValueError):
        st.error('Existing original bundle failed integrity checks. No overwrite, recovery or model fallback is attempted.');st.stop()
    os.environ['TRUSTRETINA_BUNDLE']=str(bundle)
    os.environ['TRUSTRETINA_HOSTED_BACKEND']='1'
    runpy.run_path(str(ROOT/'streamlit_app.py'),run_name='__main__')


if __name__=='__main__':main()
