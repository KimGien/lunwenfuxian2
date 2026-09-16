"""Publish a completion report only from a successful full transcript audit."""
from datetime import datetime,timezone
import json
from prepare import ROOT,save,digest

def main():
    audit=json.loads((ROOT/'reports/transcription_audit.json').read_text(encoding='utf-8'))
    if not audit['passed'] or audit['completed']!=1073:raise ValueError('Full transcript audit has not passed')
    multiple=sum(r['segments']>1 for r in audit['files'])
    text=f'''# B0-37 全量转写完成报告

核验时间：{audit['timestamp_utc']}。

- 37名受试者、每人29题，共1073份转写已生成并核验。
- 遗漏0，额外录音0，空文本0，来源/导出一致性错误0。
- 原始音频SHA256、固定Whisper medium模型文件SHA256、人群与五折哈希均已核对。
- 所有转写保留完整ASR分段及时间戳、拼接全文；B0-37仍采用首段文字。
- {multiple}份录音包含多个ASR分段。全文保存在对应JSON中，不覆盖基线首段选择。

## 唯一的低音量恢复例外

02010036/22原始录音长约10.43秒，原音量重试仍返回空分段。原始文件未修改；仅在内存中测试1、10、100倍音量，10倍及100倍均得到识别结果，采用较小的10倍增益结果。两档增益的部分识别文字不同，保留10倍结果原文，没有人工补写或根据诊断/分类得分选择。

这是自动ASR恢复，未经过逐字人工校对；非空不等于文字完全准确。默认转写失败和这项恢复已分别留证：

- 原空转写：archive/asr_original_empty/02010036/22.json。
- 候选与音量证据：reports/asr_empty_retry_02010036_22.json。
- 例外政策：configs/asr_recovery_policy.json。
- 采用结果：transcripts/B0-37/first_segment_code/02010036/22.json，其中recovery字段记录来源和限制。

## 结果位置与后续

- 全部结果：transcripts/B0-37/first_segment_code/<subject_id>/<question_id>.json。
- 全量核验：reports/transcription_audit.json。
- 逐题哈希：manifests/transcripts_sha256.json。

本次完成转写，没有启动分类器训练。已有37人EEG和三种子语音特征；下一步生成全量文本特征，并通过训练入口预检后再开始五折训练。B-all完整38人目录没有修改。
'''
    (ROOT/'reports/转写完成报告.md').write_bytes(text.encode('utf-8'))
    now=datetime.now(timezone.utc).isoformat()
    execution=ROOT/'reports/b0_execution_progress.json'
    old=json.loads(execution.read_text(encoding='utf-8'))
    old.update(status='awaiting_text_features',stage='transcription_complete',updated_utc=now,
        driver_active=False,transcription_audit_sha256=digest(ROOT/'reports/transcription_audit.json'))
    old['previous_driver_pid']=old.pop('pid',None);old['previous_asr_pid']=old.pop('asr_pid',None)
    if 'transcription' not in old['completed_stages']:old['completed_stages'].append('transcription')
    save(execution,old)
    preflight=ROOT/'reports/preflight.json'
    old=json.loads(preflight.read_text(encoding='utf-8'))
    old.update(timestamp_utc=now,stage='S00_B0_37_TRANSCRIPTION_COMPLETE_TEXT_FEATURES_PENDING',training_ready=False,
        transcription=json.loads((ROOT/'reports/transcription_progress.json').read_text(encoding='utf-8')),
        execution=json.loads(execution.read_text(encoding='utf-8')),
        outstanding=['Full text feature extraction; per-seed preflight; five-fold training and reporting'])
    save(preflight,old)
    inventory=ROOT/'manifests/preparation_files_sha256.json'
    entries=json.loads(inventory.read_text(encoding='utf-8'))
    for p in list((ROOT/'scripts').glob('*.py'))+list((ROOT/'configs').glob('*.json'))+[ROOT/'reports/转写完成报告.md']:
        entries[str(p.relative_to(ROOT))]=digest(p)
    save(inventory,entries)
    print('Full1073 transcript completion report written; classifier training not started.')

if __name__=='__main__':main()
