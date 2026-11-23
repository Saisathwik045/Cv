"""Session-only checks for declared eye positions; not an anatomical classifier."""
import re
from pathlib import PurePath


def declared_eye(filename):
    """Read an explicit left/right suffix; unknown names remain unverified."""
    stem = PurePath(str(filename or '').replace('\\', '/')).stem.casefold()
    match = re.search(r'(?:^|[_\-\s])(left|right)(?:[_\-\s]eye)?$', stem)
    if not match:
        return None, None
    patient = stem[:match.start()].rstrip('_- ')
    return match.group(1), patient or None


def pair_errors(left_name, right_name, contents, images):
    """Reject explicit filename conflicts and exact duplicates, without inferring anatomy."""
    issues = []
    left_eye, left_id = declared_eye(left_name)
    right_eye, right_id = declared_eye(right_name)
    if left_eye == 'right':
        issues.append('The file in Left eye is named as a right-eye image. Check the original source and move it to Right eye.')
    if right_eye == 'left':
        issues.append('The file in Right eye is named as a left-eye image. Check the original source and move it to Left eye.')
    if left_id and right_id and left_id != right_id:
        issues.append('The filenames declare different patient/pair IDs. Select the matching left and right files from the original source.')
    if all(item is not None for item in contents):
        duplicate = contents[0] == contents[1]
        if not duplicate and all(image is not None for image in images):
            duplicate = images[0].size == images[1].size and images[0].tobytes() == images[1].tobytes()
        if duplicate:
            issues.append('The same image was uploaded twice. Upload two distinct eye images belonging to the same patient.')
    return issues


def reconcile_confirmation(state, signature):
    """Require a fresh operator confirmation whenever the uploaded pair changes."""
    if state.get('confirmation_pair_signature') != signature:
        state.pop('pair_confirmed', None)
        state['confirmation_pair_signature'] = signature
