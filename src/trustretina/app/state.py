"""Session-only uploads/results. Never cache patient input or write it to disk."""
import hashlib


def clear_patient(state):
    generation = state.get('upload_generation', 0) + 1
    state.clear()
    state['upload_generation'] = generation


def reconcile_uploads(state, left, right):
    signature = tuple(hashlib.sha256(x).hexdigest() if x is not None else None for x in (left, right))
    if state.get('pair_signature') != signature:
        for key in ('prediction', 'explanation', 'explanation_target'):
            state.pop(key, None)
        state['pair_signature'] = signature
    return signature


def reset_streamlit_session(state):
    """Purge this session's server upload/media references as well as widget state.

    Runtime APIs are covered by the pinned Streamlit version; never clear other
    sessions or the model cache. Browser/downloaded copies cannot be revoked.
    """
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    from streamlit import runtime
    ctx = get_script_run_ctx()
    if ctx is not None:
        ctx.uploaded_file_mgr.remove_session_files(ctx.session_id)
        if runtime.exists():
            manager = runtime.get_instance().media_file_mgr
            manager.clear_session_refs(ctx.session_id)
            manager.remove_orphaned_files()
    clear_patient(state)
