"""Extract only S00 raw modalities; preserve originals and validate ZIP CRC."""
import hashlib
from pathlib import Path
import zipfile
from prepare import ROOT, WORK, save, digest

def main():
    records=[]; archives=[]
    base=WORK/'dataset/modma'
    for name in ('modma-dataset-2015.zip','modma-audio-group-6-dsai.zip'):
        path=base/name
        print('Checking/extracting',name,flush=True)
        archives.append({'path':str(path.relative_to(WORK)),'bytes':path.stat().st_size,'sha256':digest(path),'provenance':'user_supplied_archive; original distribution URL not supplied'})
        dest=base/'raw'/path.stem
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir(): continue
                if name=='modma-dataset-2015.zip' and not info.filename.startswith('EEG_128channels_resting_lanzhou_2015/'):
                    continue
                target=(dest/info.filename).resolve()
                if not target.is_relative_to(dest.resolve()): raise ValueError('Unsafe ZIP path')
                target.parent.mkdir(parents=True,exist_ok=True)
                # Never overwrite an existing raw file; compare CRC on re-runs.
                import zlib
                if target.exists():
                    crc=0
                    with target.open('rb') as f:
                        while chunk:=f.read(1024*1024): crc=zlib.crc32(chunk,crc)
                    if target.stat().st_size!=info.file_size or crc!=info.CRC:
                        raise ValueError(f'Existing raw file mismatch: {target}')
                else:
                    part=target.with_suffix(target.suffix+'.part')
                    with z.open(info) as inp,part.open('wb') as out:
                        while chunk:=inp.read(1024*1024): out.write(chunk)
                    part.replace(target)
                records.append({'path':str(target.relative_to(WORK)),'archive':name,'member':info.filename,'bytes':target.stat().st_size,'zip_crc32':f'{info.CRC:08x}','sha256':digest(target)})
    save(ROOT/'manifests/modma_archives.json',archives)
    save(ROOT/'manifests/modma_raw_files.json',records)
    print('Extracted and verified',len(records),'files',flush=True)

if __name__=='__main__': main()
