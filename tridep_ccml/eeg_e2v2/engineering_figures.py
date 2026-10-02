"""Whole-cohort engineering views, without labels or classification predictions."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *


def run():
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    rows=read(OUT/'input_manifest.json')['records']
    fig,ax=plt.subplots(7,6,figsize=(15,15),layout='constrained')
    for i,row in enumerate(rows):
        z=ax.flat[i]
        for name,color,label in [('old','#8994a6','旧'),('new','#168f83','新')]:
            energy=np.array(row['quality'][name]['window_energy'])
            z.plot(np.arange(60),np.sqrt(energy),color=color,lw=.9,label=label)
        z.set_title(row['subject_id'],fontsize=10);z.set_yscale('log');z.grid(alpha=.15)
        z.tick_params(labelsize=7);z.spines[['top','right']].set_visible(False)
    for i in range(len(rows),42):ax.flat[i].axis('off')
    ax.flat[0].legend(fontsize=8)
    fig.suptitle('全部37人：每5秒窗的跨导 RMS，新旧时间位置一致',fontsize=16)
    fig.supxlabel('窗口编号0–59；纵轴为原生数值尺度的 RMS（对数）；每人独立纵轴')
    fig.savefig(REPORT/'全部37人输入概览.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,5.4),layout='constrained')
    for z,side,title in zip(ax,['first5','last5'],['首5秒人工边界','末5秒人工边界']):
        old=np.array([np.median(r['pseudo_boundary']['old'][side]) for r in rows])
        new=np.array([np.median(r['pseudo_boundary']['new'][side]) for r in rows])
        z.scatter(old,new,color='#168f83',alpha=.75,s=35)
        limits=(min(old.min(),new.min())*.65,max(old.max(),new.max())*1.4)
        z.plot(limits,limits,color='#8994a6',ls='--',label='误差相同')
        for j,r in enumerate(rows):
            if new[j]>=old[j]:z.annotate(r['subject_id'],(old[j],new[j]),xytext=(5,4),textcoords='offset points',fontsize=8,color='#aa3e50')
        z.set(xscale='log',yscale='log',xlim=limits,ylim=limits,xlabel='旧处理 NRMSE',ylabel='新处理 NRMSE',title=title)
        z.grid(alpha=.15);z.legend()
    fig.suptitle('真实记录内部60–120秒截断：与完整上下文处理比较',fontsize=15)
    fig.supxlabel('每点为1人，先对19导取中位数；3个人/侧没有改善，均保留。该检验不提供真正记录外的真值。',fontsize=10)
    fig.savefig(REPORT/'真实人工边界配对.png',dpi=160);plt.close(fig)


if __name__=='__main__':run()
