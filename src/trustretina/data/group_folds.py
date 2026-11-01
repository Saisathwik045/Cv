"""K-fold generalization of v3 whole-component allocation; frozen v3 stays unchanged."""
import random
from .validation import DataValidationError

ALGORITHM = "trustretina-kfold-component-rarest-label-repair-v11"


def repair_support(groups, units, settings, targets, rank):
    n_folds = len(groups)
    """Reduce support deficits with whole-component moves and no seed search.

    Never create/increase a donor deficit. If no such move exists, fail closed;
    this constructive method does not claim to solve all feasible assignments.
    """
    positive = [[sum(p['labels'][j] for p in rows) for j in range(8)] for rows in groups]
    sizes = [len(rows) for rows in groups]
    location = {p['patient_id']: k for k, rows in enumerate(groups) for p in rows}
    owner = [location[unit[0]['patient_id']] for unit in units]
    unit_pos = [[sum(p['labels'][j] for p in unit) for j in range(8)] for unit in units]
    floors = (settings['minimum_positive_patients'], settings['minimum_negative_patients'])
    moves = []
    while True:
        deficits = [(k, j, outcome) for k in range(n_folds) for j in range(8) for outcome in (0, 1)
                    if (positive[k][j] if outcome == 0 else sizes[k] - positive[k][j]) < floors[outcome]]
        if not deficits:
            return moves
        # Fixed role/class/outcome ordering, independent of any model result.
        target, label, outcome = deficits[0]
        choices = []
        for i, unit in enumerate(units):
            donor = owner[i]
            count = unit_pos[i][label] if outcome == 0 else len(unit) - unit_pos[i][label]
            if donor == target or not count:
                continue
            after_size = sizes[donor] - len(unit)
            after_pos = [positive[donor][j] - unit_pos[i][j] for j in range(8)]
            if any(after_pos[j] < min(positive[donor][j], floors[0])
                   or after_size - after_pos[j] < min(sizes[donor] - positive[donor][j], floors[1])
                   for j in range(8)):
                continue
            deviation = sum(abs(sizes[k] + (len(unit) if k == target else -len(unit) if k == donor else 0) - targets[k]) for k in range(n_folds))
            choices.append((len(unit), deviation, rank[i], i))
        if not choices:
            raise DataValidationError('No floor-preserving whole-component support repair found; no manifest published')
        i = min(choices)[-1]
        donor = owner[i]
        ids = {p['patient_id'] for p in units[i]}
        groups[donor] = [p for p in groups[donor] if p['patient_id'] not in ids]
        groups[target].extend(units[i])
        for k, sign in ((donor, -1), (target, 1)):
            sizes[k] += sign * len(units[i])
            for j in range(8):
                positive[k][j] += sign * unit_pos[i][j]
        owner[i] = target
        moves.append({'patient_ids': sorted(ids), 'from': donor, 'to': target,
                      'required_class': CLASSES[label], 'outcome': 'positive' if outcome == 0 else 'negative',
                      'reason': 'Prespecified support floor; whole component, donor support preserved; no model outcomes'})


def allocate_components(patients, patient_groups, config):
    """Pure shared allocator. Assignments are diagnostic until independently audited.

    Keeps the existing rarest-label + whole-group support-repair v3 method; no
    alternative seeds or model results are accepted as inputs.
    """
    settings = config['split']
    n_folds = len(settings['fractions'])
    if n_folds < 2 or any(f <= 0 for f in settings['fractions']) or abs(sum(settings['fractions']) - 1) > 1e-12:
        raise ValueError('At least two positive fractions summing to one required')
    n = len(patients)
    minimum_pos = settings['minimum_positive_patients']
    minimum_neg = settings['minimum_negative_patients']
    totals = [sum(p['labels'][j] for p in patients) for j in range(8)]
    if any(t < n_folds * minimum_pos or n - t < n_folds * minimum_neg for t in totals):
        raise DataValidationError('Insufficient global per-class support for all folds')
    fractions = settings['fractions']
    sizes = [int(n * f) for f in fractions]
    order = sorted(range(n_folds), key=lambda k: (-(n * fractions[k] - sizes[k]), k))
    for k in order[:n - sum(sizes)]:
        sizes[k] += 1
    if any(s < minimum_pos + minimum_neg for s in sizes):
        raise DataValidationError('Partition capacity cannot meet required positive/negative support')
    # Allocate the rarest remaining label first, choosing its greatest unfilled
    # per-partition target. No seed retries or final-evaluation outcome selection.
    by_id = {p['patient_id']: p for p in patients}
    flat = [pid for ids in patient_groups for pid in ids]
    if (len(flat) != len(set(flat)) or set(flat) != set(by_id) or any(not ids for ids in patient_groups)):
        raise DataValidationError('Components must cover each original patient exactly once')
    units = [[by_id[pid] for pid in ids] for ids in patient_groups]
    # All folds each need a component supplying each outcome. This is a
    # necessary feasibility check; the actual patient support audit remains final.
    for j in range(8):
        if (sum(any(p['labels'][j] for p in unit) for unit in units) < n_folds
                or sum(any(not p['labels'][j] for p in unit) for unit in units) < n_folds):
            raise DataValidationError('Duplicate grouping makes K-fold class support infeasible')
    remaining = set(range(len(units)))
    capacity = sizes.copy()
    need = [[total * f for total in totals] for f in fractions]
    groups = [[] for _ in range(n_folds)]
    rng = random.Random(config['seed'])
    ties = list(range(len(units)))
    rng.shuffle(ties)
    rank = {index: position for position, index in enumerate(ties)}
    while remaining:
        counts = [sum(p['labels'][j] for i in remaining for p in units[i]) for j in range(8)]
        nonzero = [j for j in range(8) if counts[j]]
        label = min(nonzero, key=lambda j: (counts[j], j)) if nonzero else None
        selected = sorted((i for i in remaining if label is None or any(p['labels'][label] for p in units[i])),
                          key=lambda i: rank[i])
        for i in selected:
            # Whole components are indivisible. Patient fraction targets are soft:
            # prefer space for the whole unit, otherwise the least overfilled target.
            available = [k for k in range(n_folds) if capacity[k] >= len(units[i])]
            if not available:
                available = [max(range(n_folds), key=lambda k: (capacity[k], -k))]
            best = max(available, key=lambda k: (need[k][label] if label is not None else 0,
                                                 capacity[k], -k))
            groups[best].extend(units[i])
            capacity[best] -= len(units[i])
            for j in range(8):
                need[best][j] -= sum(p['labels'][j] for p in units[i])
            remaining.remove(i)
    repairs = repair_support(groups, units, settings, sizes, rank)
    return groups, repairs
