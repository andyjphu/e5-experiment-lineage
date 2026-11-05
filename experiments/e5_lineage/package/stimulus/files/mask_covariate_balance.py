"""Unlabeled balance audit for quartile vs eligible-random mask comparisons.

Matching candidate count and eligibility does NOT match weak teacher label
class, weak-score confidence or domain mixture. Student-margin rank can
correlate with any of these; an observed quartile PGR effect could then be
mediated by weak-label quality rather than margin per se. This is an
unlabeled descriptive diagnostic, not true-label correctness or test PGR.
"""
import math


def compare_mask_balance(weak_probs, candidate_indices, arm_indices,
                         confidence_edges=(.65,.8), domains=None):
    n=len(weak_probs)
    if domains is not None and len(domains)!=n:
        raise ValueError('Optional domain labels must align to weak scores')
    if any(len(p)!=2 or not all(math.isfinite(float(v)) and 0<=v<=1 for v in p)
           or abs(sum(p)-1)>1e-4 for p in weak_probs):
        raise ValueError('Weak scores must be finite normalized binary rows')
    candidates=list(candidate_indices)
    if len(set(candidates))!=len(candidates) or not candidates:
        raise ValueError('Nonempty distinct candidate indices required')
    if any(type(i) is not int or i<0 or i>=n for i in candidates):
        raise ValueError('Invalid candidate row index')
    allowed=set(candidates)
    if any(type(i) is not int or i not in allowed for indices in arm_indices.values()
           for i in indices):
        raise ValueError('All compared arm rows must be in SAME eligible pool')
    if any(not indices or len(indices)!=len(set(indices))
           for indices in arm_indices.values()):
        raise ValueError('Each arm requires nonempty distinct indices')
    c=[max(weak_probs[i]) for i in candidates]
    mean=sum(c)/len(c)
    var=sum((v-mean)**2 for v in c)/len(c)
    edges=list(confidence_edges)
    if any(not .5<e<1 for e in edges) or edges!=sorted(set(edges)):
        raise ValueError('Confidence bin edges must be increasing in (.5,1)')
    def summarize(indices):
        scores=[max(weak_probs[i]) for i in indices]
        bins=[sum(score>=edge for edge in edges) for score in scores]
        output={'size':len(indices),
                'teacher_class_1_fraction':sum(weak_probs[i][1]>weak_probs[i][0]
                                               for i in indices)/len(indices),
                'mean_teacher_confidence':sum(scores)/len(scores),
                'teacher_confidence_bin_fractions':[
                    bins.count(j)/len(bins) for j in range(len(edges)+1)]}
        if domains is not None:
            output['domain_fractions']={str(domain):sum(domains[i]==domain
                         for i in indices)/len(indices)
                         for domain in sorted(set(domains),key=str)}
        output['confidence_standardized_mean_difference_vs_eligible']= (
            (output['mean_teacher_confidence']-mean)/math.sqrt(var)
            if var>0 else None)
        return output
    return {'eligible_pool':summarize(candidates),
            'arms':{name:summarize(list(indices)) for name,indices in arm_indices.items()}}


if __name__=='__main__':
    # Written unlabeled toy checks only, not executed on chat models/data.
    weak=[[.9,.1],[.1,.9],[.8,.2],[.2,.8]]
    r=compare_mask_balance(weak,[0,1,2,3],{'quartile':[0,1],'random':[2,3]})
    assert r['arms']['quartile']['teacher_class_1_fraction']==.5
    assert r['arms']['random']['size']==2
    print('Written checks passed; no model evaluated.')
