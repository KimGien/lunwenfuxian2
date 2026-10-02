"""Whole-cohort spectra and observed interior differences; no labels or selection."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from .common import *


def run():
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    records=read(OUT/'input_manifest.json')['records']
    fig,axs=plt.subplots(7,6,figsize=(15,15),layout='constrained');results=[]
    for i,row in enumerate(records):
        arrays={}
        for key,descriptor in [('old',row['source']['cache']),('new',row['new_cache'])]:
            x=np.load(WORK/descriptor['path']).transpose(1,0,2).reshape(19,60000).astype(float)
            arrays[key]=x
            f,p=welch(x,fs=200,nperseg=1000,axis=-1)
            axs.flat[i].semilogy(f,np.maximum(p.mean(0),1e-20),color='#8994a6' if key=='old' else '#168f83',lw=.8)
        a,b=arrays['old'],arrays['new'];denom=np.maximum(np.sqrt(np.mean(a[:,4000:56000]**2,axis=1)),1e-12)
        error=np.sqrt(np.mean((a[:,4000:56000]-b[:,4000:56000])**2,axis=1))/denom
        results.append({'subject_id':row['subject_id'],'interior_relative_change_per_channel':error.tolist(),
                        'maximum_interior_relative_change':float(max(error))})
        axs.flat[i].set(xlim=(0,80),title=row['subject_id']);axs.flat[i].grid(alpha=.15);axs.flat[i].tick_params(labelsize=6)
    for i in range(len(records),42):axs.flat[i].axis('off')
    fig.suptitle('全部37人：全导平均功率谱（旧灰、新绿）',fontsize=17)
    fig.supxlabel('频率0–80Hz；PSD为原生数值平方/Hz，各人独立纵轴；剩余工频不因边界修复自动消失。')
    fig.savefig(REPORT/'全部37人频谱概览.png',dpi=160);plt.close(fig)
    save(OUT/'spectrum_diagnostics.json',{'rows':results,'global_max_interior_relative_change':max(r['maximum_interior_relative_change'] for r in results),
         'interior_seconds':[20,280],'labels_used':False,'diagnostic_only':True},immutable=True)
    print('SPECTRA REVIEW',max(r['maximum_interior_relative_change'] for r in results),flush=True)


if __name__=='__main__':run()
