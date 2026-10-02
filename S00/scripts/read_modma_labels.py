"""Read labels only; source workbooks remain unchanged (bundled Python)."""
import io
import zipfile
import openpyxl
from prepare import WORK, ROOT, save

def main():
    result={}
    for name,match in [('modma-dataset-2015.zip','EEG_128channels_resting_lanzhou_2015/subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx'),('modma-audio-group-6-dsai.zip','MODMA_AUDIO Datatset/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx')]:
        with zipfile.ZipFile(WORK/'dataset/modma'/name) as z:
            workbook=openpyxl.load_workbook(io.BytesIO(z.read(match)),read_only=True,data_only=True)
            rows=list(workbook['Sheet1'].values)
            assert rows[0][0]=='subject id' and rows[0][1]=='type'
            labels={}
            for row_idx,row in enumerate(rows[1:],2):
                if row[0] is None: continue
                sid=str(row[0]).zfill(8); category=row[1]
                if category not in ('MDD','HC'): raise ValueError((sid,category))
                if sid in labels: raise ValueError('Duplicate subject')
                labels[sid]={'label':int(category=='MDD'),'category':category,'row':row_idx,'sheet':'Sheet1'}
            result[name]={'archive_member':match,'labels':labels}
            workbook.close()
    save(ROOT/'manifests/modma_labels.json',result)
    print({k:len(v['labels']) for k,v in result.items()})

if __name__=='__main__': main()
