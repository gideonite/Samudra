"""Group results.tsv by (dataset, arm, n_train [, notes-filter]) and print mean ± std over seeds."""
import csv, collections, statistics as st, sys
rows = list(csv.DictReader(open('results.tsv'), delimiter='\t'))
g = collections.defaultdict(list)
for r in rows:
    flags = ' '.join(t for t in r['notes'].split() if '=' in t and t.split('=')[0] not in
                     ('rounds', 'steps', 'steps_per_s', 'params', 'minH', 'test_metric', 'test_onestep', 'ens_metric', 'ens_masks'))
    if float(r['wallclock_s']) > 3600: flags = 'BUDGET=%ds ' % round(float(r['wallclock_s']), -2) + flags
    g[(r['dataset'], r['n_train'], r['arm'], flags)].append((float(r['metric']), r['notes']))
for k, v in sorted(g.items()):
    m = [x for x, _ in v]
    flags = sum('MASK_COLLAPSE' in n for _, n in v)
    print(f"{k[0]:8s} n={k[1]:5s} {k[2]:14s} {k[3][:60]:60s} k={len(m)} {st.mean(m):.4f} ± {st.stdev(m) if len(m)>1 else 0:.4f}  "
          f"{[round(x,3) for x in m]}{'  COLLAPSE×%d' % flags if flags else ''}")
