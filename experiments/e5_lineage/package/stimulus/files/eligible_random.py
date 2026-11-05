"""Random control restricted to the SAME teacher-confident disagreement pool.

Quartile masks differ only by opposing strong-confidence RANK inside a
precommitted pool C={student != teacher, weak_confidence >= threshold}.
A full-data random subset of the same k conflates confidence/disagreement
filtering with margin ranking; sample the k control rows from C instead.
Seeded SHA256 ranking depends on row index and prespecified seed only,
never on hidden test truth or leaderboard outcomes.
"""
import hashlib


def eligible_random_control(candidate_indices,quota,seed):
    indices=list(candidate_indices)
    if len(set(indices))!=len(indices) or any(type(i) is not int or i<0 for i in indices):
        raise ValueError('Candidate indices must be distinct nonnegative ints')
    if type(quota) is not int or not 1<=quota<=len(indices):
        raise ValueError('Quota must be positive and <= eligible pool size')
    if type(seed) is not int:
        raise ValueError('Prespecified seed must be an integer')
    return sorted(indices,key=lambda i:hashlib.sha256(
        f'{seed}:{i}'.encode('utf-8')).digest())[:quota]


if __name__=='__main__':
    # Written reproducibility/eligibility checks only; no data/model used.
    pool=[2,4,7,11,19,27]
    a=eligible_random_control(pool,2,42)
    assert a==eligible_random_control(pool,2,42)
    assert len(a)==2 and all(i in pool for i in a)
    print('Written checks passed; no model evaluated.')
