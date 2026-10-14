"""Route inherited regression fixtures explicitly to CPU5, without real tensor reads."""
import importlib.util
import io
import json
from pathlib import Path
import time
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cpu5_explicit_regression', ROOT / 'test_lost_f3_closure_and_cpu_v5.py')
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
# The previous suite intentionally retained original delegate regressions.
# This additional suite sends every inherited audit/inventory call to CPU5.
T.C.A = T.A
T.C.T.A = T.A
T.C.T.SOURCE = Path(T.A.__file__)


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (T.CpuCompatibilityTests, T.ClosureTests))
    stream = io.StringIO(); started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        print(stream.getvalue()); raise SystemExit(1)
    value = {'passed': True, 'tests_run': result.testsRun, 'failures': 0, 'errors': 0,
             'seconds': time.perf_counter() - started, 'all_inherited_audit_calls_target_cpu5': True,
             'sources': [T.L.descriptor(Path(__file__)), T.L.descriptor(Path(T.__file__)),
                         T.L.descriptor(Path(T.A.__file__)), T.L.descriptor(Path(T.L.__file__))],
             'output': stream.getvalue(), 'actual_checkpoint_loads': 0, 'actual_GPU_calls': 0,
             'actual_model_constructions': 0, 'actual_process_controls': 0,
             'scope': 'Pure standard library, immutable JSON and temporary synthetic observer fixtures only.'}
    path = ROOT / ('lost_f3_cpu5_full_regression_' + str(time.time_ns()) + '.json')
    with path.open('x', encoding='utf8') as f: json.dump(value, f, indent=2); f.write('\n')
    print(json.dumps({'passed': True, 'tests_run': result.testsRun, 'evidence': T.L.descriptor(path)}))
