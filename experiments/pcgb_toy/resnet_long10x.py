"""Analysis of the ResNet 10x (120-min) budget check: tables, trajectories, rounds/mask stats, figure.
Reads resnet/runs/*/log.jsonl whose config has budget_s==7200; 12-min reference = Phase-1 rows of resnet/results.tsv."""
import json, glob, os, csv, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); R = H + '/resnet/runs/'; O = H + '/resnet/figs/'
C = {'baseline': '#1f77b4', 'stochdepth': '#2ca02c', 'pcgb': '#d62728'}
DS = [('kolmo2d', 'Kolmo2D n=512'), ('grayscott2d', 'Gray-Scott n=512')]; ARMS = ['baseline', 'stochdepth', 'pcgb']

def load():
    runs = {}
    for f in glob.glob(R + '*/log.jsonl'):
        L = [json.loads(l) for l in open(f) if l.strip()]
        if not L or L[0].get('type') != 'config' or L[0]['budget_s'] != 7200 or L[0].get('arch') != 'resnet': continue
        fin = [x for x in L if x['type'] == 'final']
        if not fin: continue  # unfinished
        c = L[0]; runs[(c['dataset'], c['arm'], c['seed'])] = dict(
            ev=[x for x in L if x['type'] == 'eval'], rd=[x for x in L if x['type'] == 'round'], fin=fin[-1], cfg=c)
    return runs

def ref12():
    """12-min Phase-1 reference: results.tsv rows, n=512, resnet, plain arm (default flags), 720 s, commit 24d0636 or later phase-1 runs."""
    out = {}
    for r in csv.DictReader(open(H + '/resnet/results.tsv'), delimiter='\t'):
        if r['n_train'] != '512' or float(r['wallclock_s']) > 800 or 'arch=resnet' not in r['notes']: continue
        if r['dataset'] not in dict(DS) or r['arm'] not in ARMS: continue
        n = r['notes']
        if r['arm'] == 'pcgb' and ('rounds=' not in n): continue
        out.setdefault((r['dataset'], r['arm']), {}).setdefault(int(r['seed']), []).append((r['commit'], float(r['metric']), n))
    return out

def ms(v): return f"{np.mean(v):.3f} ± {np.std(v, ddof=0):.3f}"

def traj(ev):
    y = np.array([e['metric'] for e in ev]); t = np.array([e['elapsed'] / 60 for e in ev])
    return y, t

if __name__ == '__main__':
    runs = load(); ref = ref12()
    print('finished 120-min runs:', len(runs))
    for ds, title in DS:
        print(f'\n=== {title} ===')
        for arm in ARMS:
            ss = sorted(s for (d, a, s) in runs if d == ds and a == arm)
            fin = [runs[(ds, arm, s)]['fin']['metric'] for s in ss]
            if not fin: continue
            steps = [runs[(ds, arm, s)]['fin']['steps'] for s in ss]
            print(f'{arm}: 120min val {ms(fin)} (n={len(fin)}), steps {steps}')
            r12 = [v[0][1] for s, v in sorted(ref.get((ds, arm), {}).items())]
            print('   12min ref candidates:', {s: [(c, round(m, 3), n.split()[0:3]) for c, m, n in v] for s, v in sorted(ref.get((ds, arm), {}).items())})
            for s in ss:
                r = runs[(ds, arm, s)]; y, t = traj(r['ev'] + [dict(metric=r['fin']['metric'], elapsed=r['fin']['wall'])])
                print(f'   s{s} evals:', ' '.join(f'{v:.3f}' for v in y), f'| best {y.min():.3f}@{t[y.argmin()]:.0f}min final {y[-1]:.3f}')
                if arm == 'pcgb':
                    m = [x['mask'] for x in r['rd']]
                    print(f'      rounds={len(m)} mask0 frac={np.mean([k == 0 for k in m]):.2f} switches={sum(a != b for a, b in zip(m, m[1:]))} distinct={len(set(m))} first30={m[:30]}')
    # Figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, (ds, title) in zip(ax, DS):
        for arm in ARMS:
            cs = [traj(runs[k]['ev'] + [dict(metric=runs[k]['fin']['metric'], elapsed=runs[k]['fin']['wall'])]) for k in runs if k[0] == ds and k[1] == arm]
            if not cs: continue
            n = min(len(c[0]) for c in cs); x = np.mean([c[1][:n] for c in cs], 0); Y = np.array([c[0][:n] for c in cs])
            a.plot(x, Y.mean(0), '-o', ms=3, color=C[arm], label=f'{arm} (n={len(cs)})'); a.fill_between(x, Y.min(0), Y.max(0), color=C[arm], alpha=.18)
        a.set_yscale('log'); a.set_xlabel('elapsed wall-clock (min)'); a.set_ylabel('val 10-step nRMSE'); a.set_title(title + ' — ResNet, 120-min runs', fontsize=10); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(O + 'budget10x_curves.png', dpi=150); plt.close()
