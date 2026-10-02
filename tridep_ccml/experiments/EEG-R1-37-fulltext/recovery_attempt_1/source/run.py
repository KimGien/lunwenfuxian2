"""Run all fixed stages and verification, preserving a concrete failure record."""
from .common import *
from .train import train_all
from .complete import complete
import traceback


def main():
    verify_lock()
    try:
        train_all()
        complete()
    except Exception as error:
        previous = read_json(OUT/'status.json') if (OUT/'status.json').exists() else {}
        replace_json(OUT/'failure.json', {'error':str(error),'traceback':traceback.format_exc(),
            'previous_status':previous,'recorded_utc':now(),
            'policy':'Investigate; no dropped subjects/folds or test-driven search. If fitting was affected, rerun full paired stage.'})
        replace_json(OUT/'status.json', {**previous,'state':'paused_on_engineering_failure','error':str(error),'updated_utc':now()})
        raise


if __name__ == '__main__':
    main()
