"""Read-only numerical audit of the registered engineering attempt; no new candidates."""
from __future__ import annotations
from collections import Counter
import inspect
import mne
import torch
from scipy.signal import fftconvolve
from .common import *
from .preprocessing import preprocess, extend, as_windows
from .engineering import cases, signal


def independent_chain(x):
    """MNE resampling, followed by independently applied saved FIR coefficients."""
    r = mne.io.RawArray(np.atleast_2d(x).astype(float),
                        mne.create_info(['audit'], 250, 'eeg'), verbose='ERROR')
    r.resample(200, verbose='ERROR')
    y = r.get_data()[0]
    coeff = np.load(OUT/'E0_original_filter_coefficients.npz')
    for key in ['bandpass', 'notch']:
        h = coeff[key]
        n = len(h)-1
        y = fftconvolve(np.pad(y, (n, n), mode='reflect', reflect_type='odd'), h,
                        mode='same')[n:-n]
    return y[None]


def metrics_independent(y, reference, denominator):
    return {s: float(np.linalg.norm((y-reference)[0, ix])/np.sqrt(len(ix))/denominator)
            for s, ix in [('first5', np.arange(1000)), ('last5', np.arange(59000, 60000)),
                          ('interior', np.arange(4000, 56000))]}


def run():
    reg = read(OUT/'engineering_registration.json')
    assert all(digest(WORK/p) == sha for p, sha in reg['protocol']['sources'].items())
    summaries = {}
    for method in ['fixed50', 'adaptive_line']:
        d = read(OUT/f'engineering_{method}.json')
        assert d['registration'] == desc(OUT/'engineering_registration.json')
        assert reg['registered_utc'] < d['finished_utc']
        assert len(d['stationary_cases']) == 108
        for row, cfg in zip(d['stationary_cases'], reg['protocol']['cases']):
            assert all(row[k] == v for k, v in cfg.items())
            fail = []
            if row['reference_convergence_nrmse'] > 1e-4: fail.append('reference_not_converged')
            for side in ['first5', 'last5']:
                if row['new_nrmse'][side] > .1: fail.append(side+'_absolute')
                if row['old_nrmse'][side] > .1 and row['new_nrmse'][side]/row['old_nrmse'][side] > .2:
                    fail.append(side+'_relative')
            if row['gain_relative_error_max'] > .05: fail.append('passband_gain')
            a = row['line50_attenuation_db']
            if a is not None and a < 40: fail.append('line50_attenuation')
            assert fail == row['failures'] and row['passed'] == (len(fail) == 0)
        rows = d['stationary_cases']
        total = sum(r['passed'] for r in rows)
        assert d['stationary_failed_cases'] == 108-total
        assert d['passed'] == (total == 108 and d['central_transient']['passed'])
        summaries[method] = {
            'stationary_passed': total, 'stationary_total': 108,
            'failures': dict(Counter(x for r in rows for x in r['failures'])),
            'edges_absolute_passed': sum(r['new_nrmse'][s] <= .1 for r in rows for s in ['first5', 'last5']),
            'edges_total': 216,
            'relative_eligible': sum(r['old_nrmse'][s] > .1 for r in rows for s in ['first5', 'last5']),
            'edges_worse_than_old': sum(r['new_nrmse'][s] > r['old_nrmse'][s] for r in rows for s in ['first5', 'last5']),
            'new_max_edge_nrmse': max(r['new_nrmse'][s] for r in rows for s in ['first5', 'last5']),
            'gain_relative_error_max': max(r['gain_relative_error_max'] for r in rows),
            'reference_convergence_max': max(r['reference_convergence_nrmse'] for r in rows),
            'line50_attenuation_db_min': min(r['line50_attenuation_db'] for r in rows if r['line50_attenuation_db'] is not None),
            'central_transient': d['central_transient'], 'edge_transient': d['edge_transient']}

    # Recompute a strong offset-line case, a non-integer length case, and the sole fallback failure.
    replay = []
    waveform = None
    for index in [4, 35, 104]:
        cfg = cases()[index]
        x = signal(cfg, np.arange(cfg['n'])/250)
        long = signal(cfg, np.arange(-15000, cfg['n']+15000)/250)
        reference = independent_chain(long)[:, 12000:72000]
        retained = independent_chain(signal(cfg, np.arange(-15000, cfg['n']+15000)/250, False))[:, 12000:72000]
        denom = float(np.linalg.norm(retained)/np.sqrt(retained.size))
        old = independent_chain(x)[:, :60000]
        outputs = {'old': old}
        for method in ['fixed50', 'adaptive_line']:
            new, _ = preprocess(x, method)
            outputs[method] = new
            stored = read(OUT/f'engineering_{method}.json')['stationary_cases'][index]
            for key, arr in [('old_nrmse', old), ('new_nrmse', new)]:
                e = metrics_independent(arr, reference, denom)
                assert all(np.isclose(e[s], stored[key][s], rtol=1e-5, atol=1e-9) for s in e)
            ext, _ = extend(x, method)
            manual = independent_chain(ext)[:, 6000:66000]
            maxdiff = float(np.max(abs(manual-new)))
            assert maxdiff < 1e-9
            replay.append({'case': index, 'method': method, 'manual_fir_max_abs_difference': maxdiff})
        if index == 104:
            waveform = {'reference': reference[0], **{k: v[0] for k, v in outputs.items()}, 'denominator': denom}
    np.savez_compressed(OUT/'failure_case104_waveforms.npz', **waveform)

    # Independent 19-channel shape/time-order and preservation checks on deterministic synthetic data.
    t = np.arange(75089)/250
    x = np.stack([(1+i/19)*np.sin(2*np.pi*(1+i)*t)+20*np.sin(2*np.pi*50.2*t) for i in range(19)])
    shape_checks = {}
    for method in ['fixed50', 'adaptive_line']:
        ext, _ = extend(x, method)
        assert np.array_equal(ext[:, 7500:-7500], x)
        y, _ = preprocess(x, method)
        windows = as_windows(y)
        assert windows.shape == (60, 19, 1000) and np.isfinite(windows).all()
        assert np.array_equal(windows.transpose(1, 0, 2).reshape(19, 60000), y.astype(np.float32))
        shape_checks[method] = {'recorded_samples_exact': True, 'shape': list(windows.shape), 'window_order_exact': True}

    # Phase/delay reporting is diagnostic only and cannot overturn the failed registered gate.
    phase_rows = []
    inner_t = np.arange(4000, 56000)/200
    for freq in [1, 4, 10, 20, 35]:
        t = np.arange(75000)/250
        x = (5*np.sin(2*np.pi*freq*t))[None]
        design = np.c_[np.sin(2*np.pi*freq*inner_t), np.cos(2*np.pi*freq*inner_t), np.ones(len(inner_t))]
        row = {'frequency_hz': freq}
        for method in ['old', 'fixed50', 'adaptive_line']:
            y, _ = preprocess(x, method)
            c = np.linalg.lstsq(design, y[0, 4000:56000], rcond=None)[0]
            row[method+'_phase_radians'] = float(np.arctan2(c[1], c[0]))
        phase_rows.append(row)

    # Provenance evidence is read-only; tensor contents cannot prove a training input convention.
    weights = WORK/'S00/upstream-dev/cbramod_pretrained_weights/pretrained_weights2.pth'
    ckpt = torch.load(weights, map_location='cpu', weights_only=True)
    numeric = read(P2/'unit_numeric_checks.json')
    test_raw = mne.io.RawArray(np.full((1, 500), 10e-6), mne.create_info(['audit'], 250, 'eeg'), verbose='ERROR')
    dataframe_value = float(test_raw.to_data_frame()['audit'].iloc[0])
    assert np.isclose(dataframe_value, 10)
    save(OUT/'unit_provenance.json', {
        'checked_utc': now(), 'local_checkpoint': desc(weights), 'top_level_keys': len(ckpt),
        'non_tensor_keys': [k for k, v in ckpt.items() if not torch.is_tensor(v)],
        'prior_numeric_evidence': desc(P2/'unit_numeric_checks.json'),
        'eeglab_unit_conversion_previously_verified_subjects': numeric['read_eeglab_native_times_1e_minus6_verified_subjects'],
        'mne_10_microvolt_dataframe_value_rechecked': dataframe_value,
        'physical_gain_calibration_confirmed': False, 'local_checkpoint_training_scale_verified': False,
        'upstream_main_code_observed': [
            {'url': 'https://raw.githubusercontent.com/wjq-learning/CBraMod/main/preprocessing/preprocessing_mumtaz.py', 'operation': 'raw.to_data_frame().values'},
            {'url': 'https://raw.githubusercontent.com/wjq-learning/CBraMod/main/datasets/mumtaz_dataset.py', 'operation': 'data/100'}],
        'discovered_upstream_commit': '20a57c4a905bc1a4eb5f9f156667f142cf5155b3',
        'commit_url': 'https://github.com/wjq-learning/CBraMod/commit/20a57c4a905bc1a4eb5f9f156667f142cf5155b3',
        'commit_specific_preprocessing_files_readable': False,
        'provenance_status': 'unresolved: main code is readable, commit-specific file retrieval failed, and no linkage to local fine-tuned weights exists',
        'input_scaling_changed': False,
        'local_sources': [desc(WORK/'tridep_ccml/v2/extract_eeg.py'), desc(WORK/'S00/upstream-dev/README.md')]
    })

    source_records = read(P2/'input_manifest_label_free.json')
    for rec in source_records:
        for key in ['set', 'fdt', 'mat', 'cache']:
            assert digest(WORK/rec[key]['path']) == rec[key]['sha256'], rec['subject_id']+' '+key
    history = freeze_history()
    assert history == read(OUT/'historical_artifacts_before.json')
    save(OUT/'historical_artifacts_after.json', history)
    rows = [read(p) for p in sorted((P2/'subjects').glob('*.json'))]
    save(OUT/'verification.json', {
        'verified_utc': now(), 'passed': True, 'engineering_passed': False,
        'registered_sources_unchanged': True, 'stored_gate_rows_recomputed': 216,
        'independent_fir_replay': replay, 'synthetic_channel_contract': shape_checks,
        'interior_phase_diagnostics': phase_rows, 'candidate_summary': summaries,
        'source_subjects_hash_verified': len(source_records), 'source_descriptors_verified': len(source_records)*4,
        'historical_manifest_entries_verified': sum(r['files'] for r in history.values()),
        'prior_real_duration_seconds': {'min': min(r['duration_seconds'] for r in rows), 'max': max(r['duration_seconds'] for r in rows)},
        'new_real_input_caches': 0, 'new_model_fits': 0, 'classifier_results': None,
        'scope': 'Verifies implementation/results integrity, not engineering acceptance or clinical performance.'
    })
    print(json.dumps({'verification_passed': True, 'engineering_passed': False,
                      'history_entries': sum(r['files'] for r in history.values()), 'source_subjects': len(source_records)}), flush=True)


if __name__ == '__main__':
    run()
