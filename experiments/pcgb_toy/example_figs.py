import sys, json, glob
sys.path.insert(0, '/home/user/Samudra/experiments/pcgb_toy')
from pathlib import Path
import numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import prepare, train, horizon
torch.set_num_threads(1)
H = Path('/home/user/Samudra/experiments/pcgb_toy'); O = H / 'figs'
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13})
BLUE, RED = '#1f6fd0', '#d62728'
def load_model(rd, meta):
    cfg = json.loads((Path(rd) / 'args.json').read_text())
    m = train.build_model(meta, cfg['width'], cfg['drop_mode']); m.load_state_dict(torch.load(Path(rd) / 'model.pt')); m.eval(); return m
def run(tag): return str(H / 'runs' / tag)
def roll(m, x0, n):
    out = []; x = x0
    with torch.no_grad():
        for _ in range(n): x = m(x); out.append(x)
    return torch.stack(out)  # [n,B,C,...]
def phys(x, meta):  # denormalise
    sh = [1] * x.ndim; sh[-(x.ndim - 1) if False else 0] = 1
    return x
def load(ds):
    d = prepare.load(ds); return d['val'], d['meta']
def runs(ds):
    if ds in ('ks1d', 'kolmo2d'): return horizon.find_run(ds, 'baseline', 512, 0), horizon.find_run(ds, 'pcgb', 512, 0)
    if ds == 'grayscott2d': return run('grayscott2d_baseline_n512_s0_a23fe91_v1'), run('grayscott2d_pcgb_n512_s0_a23fe91_v1')
    return run('denoise_baseline_n512_s0_f5d18a7'), run('denoise_pcgb_n512_s0_e7d403f_v1')
TRAJ = 0
# ---------------- KS1D
val, meta = load('ks1d'); mean, std = meta['mean'].item(), meta['std'].item()
tr = val[TRAJ, :, 0] * std + mean  # [T,128]
T = 61
fig = plt.figure(figsize=(11, 7.2)); gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.45)
ax = fig.add_subplot(gs[0]); L = 22 * np.pi
im = ax.imshow(tr[:T].T, aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, (T - 1) * 2.0, 0, L], vmin=-np.abs(tr).max(), vmax=np.abs(tr).max())
ax.set_xlabel('time (thin vertical lines mark each model step; one step = 2 time units)'); ax.set_ylabel('position x'); ax.set_title('KS1D: one validation trajectory, 60 model steps (one step = 2 time units)')
for k in range(0, T, 1): ax.axvline(k * 2.0, color='k', lw=0.15, alpha=0.5)
fig.colorbar(im, ax=ax, label='u(x, t)', pad=0.01)
b, p = [load_model(r, meta) for r in runs('ks1d')]
x0 = val[TRAJ, 0:1]; tb, tp = roll(b, x0, 10)[9, 0, 0] * std + mean, roll(p, x0, 10)[9, 0, 0] * std + mean
xs = np.linspace(0, L, 128, endpoint=False); ax2 = fig.add_subplot(gs[1])
ax2.plot(xs, val[TRAJ, 10, 0] * std + mean, 'k', lw=2.5, label='truth'); ax2.plot(xs, tb, color=BLUE, lw=2, label='plain baseline'); ax2.plot(xs, tp, color=RED, lw=2, label='PCGB')
ax2.set_xlabel('position x'); ax2.set_ylabel('u'); ax2.set_title('The same trajectory, predicted 10 steps ahead from step 0 (both nets trained on 512 pairs)'); ax2.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.25), frameon=False)
fig.savefig(O / 'example_ks1d.png', dpi=140, bbox_inches='tight'); plt.close(fig)
# ---------------- Kolmo2D
val, meta = load('kolmo2d'); mean, std = meta['mean'].item(), meta['std'].item()
w = val[TRAJ, :, 0] * std + mean; lim = float(np.abs(w[:20]).max())
steps = [0, 3, 6, 9, 12]
fig, axs = plt.subplots(2, 5, figsize=(12.5, 5.8), gridspec_kw=dict(hspace=0.35, wspace=0.08))
for a, s in zip(axs[0], steps):
    im = a.imshow(w[s], cmap='RdBu_r', vmin=-lim, vmax=lim, origin='lower'); a.set_title(f'step {s}  (t = {s*0.15:.2f})'); a.axis('off')
b, p = [load_model(r, meta) for r in runs('kolmo2d')]
x0 = val[TRAJ, 0:1]; rb, rp = roll(b, x0, 10)[9, 0, 0] * std + mean, roll(p, x0, 10)[9, 0, 0] * std + mean
for a, f, t in zip(axs[1, :3], [w[10], rb, rp], ['truth, step 10', 'plain baseline, step 10', 'PCGB, step 10']):
    a.imshow(f if isinstance(f, np.ndarray) else f.numpy(), cmap='RdBu_r', vmin=-lim, vmax=lim, origin='lower'); a.set_title(t, color={'t':'k'}.get(t[0], 'k')); a.axis('off')
axs[1, 3].axis('off'); axs[1, 4].axis('off')
fig.suptitle('Kolmo2D: vorticity of one validation trajectory (top); 10-step predictions from step 0 (bottom)', y=0.98, fontsize=14)
cb = fig.colorbar(im, ax=axs, shrink=0.55, pad=0.01, label='vorticity (shared scale)')
fig.savefig(O / 'example_kolmo2d.png', dpi=140, bbox_inches='tight'); plt.close(fig)
# ---------------- Gray-Scott
val, meta = load('grayscott2d'); mean, std = meta['mean'], meta['std']
g = val[TRAJ] * std.view(1, 2, 1, 1) + mean.view(1, 2, 1, 1)  # [T,2,32,32]
late = 40
fig, axs = plt.subplots(2, 4, figsize=(12.5, 6.4), gridspec_kw=dict(hspace=0.3, wspace=0.1))
vm = [(g[:, c].min().item(), g[:, c].max().item()) for c in range(2)]
for j, (s, c) in enumerate([(0, 0), (0, 1), (late, 0), (late, 1)]):
    a = axs[0, j]; im = a.imshow(g[s, c], cmap='viridis', vmin=vm[c][0], vmax=vm[c][1], origin='lower'); a.set_title(f"{'u' if c == 0 else 'v'} at snapshot {s}"); a.axis('off'); fig.colorbar(im, ax=a, fraction=0.046)
b, p = [load_model(r, meta) for r in runs('grayscott2d')]
x0 = val[TRAJ, 0:1]; Rb, Rp = roll(b, x0, 10)[9, 0], roll(p, x0, 10)[9, 0]
tc = 1
f = [g[10, tc], (Rb[tc] * std[tc] + mean[tc]).numpy(), (Rp[tc] * std[tc] + mean[tc]).numpy()]
for a, im_, t in zip(axs[1, :3], f, ['truth (v), step 10', 'plain baseline (v), step 10', 'PCGB (v), step 10']):
    a.imshow(im_, cmap='viridis', vmin=vm[tc][0], vmax=vm[tc][1], origin='lower'); a.set_title(t); a.axis('off')
axs[1, 3].axis('off')
fig.suptitle(f'Gray-Scott: chemicals u and v of one validation trajectory (top); v predicted 10 steps ahead (bottom)', y=0.99, fontsize=14)
fig.savefig(O / 'example_grayscott2d.png', dpi=140, bbox_inches='tight'); plt.close(fig)
# ---------------- Denoise
d = prepare.load('denoise'); val, meta = d['val'], d['meta']
noisy, clean = val[:, 0], val[:, 1]
b, p = [load_model(r, meta) for r in runs('denoise')]
idx = [0, 1, 2, 3]
with torch.no_grad(): ob, op = b(noisy[idx]), p(noisy[idx])
rows = [('noisy input', noisy[idx]), ('clean target', clean[idx]), ('plain baseline', ob), ('PCGB', op)]
fig, axs = plt.subplots(4, 4, figsize=(8.5, 8.5), gridspec_kw=dict(hspace=0.12, wspace=0.05))
for r, (name, X) in enumerate(rows):
    for c in range(4):
        axs[r, c].imshow(X[c, 0].numpy(), cmap='gray', vmin=float(clean.min()), vmax=float(clean.max())); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
    axs[r, 0].set_ylabel(name, rotation=90, fontsize=12)
fig.suptitle('Denoise: four validation digits, one-step reconstruction', y=0.93, fontsize=14)
fig.savefig(O / 'example_denoise.png', dpi=140, bbox_inches='tight'); plt.close(fig)
print('ok')
