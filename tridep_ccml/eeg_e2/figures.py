"""Static research figure from the complete registered results, including failures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *


def run():
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 11})
    a = read(OUT/'engineering_adaptive_line.json')['stationary_cases']
    f = read(OUT/'engineering_fixed50.json')['stationary_cases']
    d = np.load(OUT/'failure_case104_waveforms.npz')
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5), layout='constrained')
    colors = ['#75859a', '#e3a34c', '#248b83']
    fig.suptitle('EEG 边界修复：改善明显，但未通过全部预定工程门槛', fontsize=17, fontweight='bold')
    z = ax[0, 0]
    counts = [sum(r['passed'] for r in f), sum(r['passed'] for r in a)]
    z.bar(['主方案：固定 50 Hz', '备用：估计工频'], counts, color=colors[1:])
    z.axhline(108, color='#ba4551', ls='--', lw=1.4)
    for i, count in enumerate(counts): z.text(i, count+2, f'{count}/108', ha='center', fontweight='bold')
    z.set(ylim=(0, 124), ylabel='通过的平稳合成测试数', title='A  同时满足全部门槛才计为通过')
    z.text(.98, .06, '两方案的中央瞬态测试均通过', ha='right', transform=z.transAxes, fontsize=10)
    z = ax[0, 1]
    for side, marker in [('first5', 'o'), ('last5', '^')]:
        old = np.array([r['old_nrmse'][side] for r in a])
        new = np.array([r['new_nrmse'][side] for r in a])
        z.scatter(old, new, s=20, alpha=.62, marker=marker, label='首 5 秒' if side=='first5' else '末 5 秒', color=colors[2])
    z.plot([1e-5, 100], [1e-5, 100], color=colors[0], lw=1, label='新旧相等')
    z.plot([.1, 100], [.02, 20], color='#ba4551', ls=':', lw=1.2, label='新 ≤ 旧 × 20%（旧 > 0.1）')
    z.axhline(.1, color='#444444', ls='--', lw=1)
    z.scatter([a[104]['old_nrmse']['last5']], [a[104]['new_nrmse']['last5']], s=90, facecolors='none', edgecolors='#ba4551', lw=2)
    z.set(xscale='log', yscale='log', xlim=(1e-5, 100), ylim=(1e-5, 100),
          xlabel='旧流程边界 NRMSE', ylabel='备用方案边界 NRMSE', title='B  备用方案：216 个边界的配对误差')
    z.legend(fontsize=8, loc='upper left')
    z = ax[1, 0]
    row = a[104]
    old = [row['old_nrmse'][s] for s in ['first5', 'last5']]
    new = [row['new_nrmse'][s] for s in ['first5', 'last5']]
    positions = np.arange(2)
    z.bar(positions-.19, old, .36, color=colors[0], label='旧流程')
    z.bar(positions+.19, new, .36, color=colors[2], label='备用方案')
    for x, vals in [(-.19, old), (.19, new)]:
        for i, value in enumerate(vals): z.text(i+x, value+.006, f'{value:.5f}', ha='center', fontsize=9)
    target = .2*old[1]
    z.hlines(target, .62, 1.43, colors='#ba4551', linestyles=':', linewidth=2)
    z.annotate(f'末端要求 ≤ {target:.5f}', xy=(1.4, target), xytext=(.45, .145),
               arrowprops={'arrowstyle':'->', 'color':'#ba4551'}, color='#ba4551', fontsize=10)
    z.axhline(.1, color='#444444', ls='--', lw=1)
    z.set(xticks=positions, xticklabels=['首 5 秒', '末 5 秒'], ylim=(0, .29), ylabel='NRMSE',
          title='C  唯一失败样例：无工频多频信号（编号 104）')
    z.legend(loc='upper left', fontsize=9)
    z = ax[1, 1]
    t = np.arange(59000, 60000)/200
    for key, color, label in [('old', colors[0], '旧流程'), ('adaptive_line', colors[2], '备用方案')]:
        z.plot(t, (d[key][59000:]-d['reference'][59000:])/float(d['denominator']), color=color, lw=1.1, label=label)
    z.set(xlabel='记录时间（秒）', ylabel='误差 / 保留信号 RMS', title='D  该样例末端误差相对长上下文参考')
    z.axhline(0, color='#bbbbbb', lw=.8)
    z.legend(fontsize=9)
    for z in ax.ravel():
        z.spines[['top', 'right']].set_visible(False)
        z.grid(axis='y', alpha=.15)
    fig.supxlabel('仅合成信号工程检验；108 项不是受试者样本。本轮没有生成新 EEG 缓存或训练分类器。', fontsize=10)
    REPORT.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT/'EEG_E2工程验收.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__': run()
