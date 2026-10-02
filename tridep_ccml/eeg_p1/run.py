"""Run the registered train-only gate, formal comparison, replay and report."""
from __future__ import annotations
import sys
import traceback
from .common import *


class Tee:
    def __init__(self,stream,log):
        self.stream,self.log = stream,log

    def write(self,text):
        self.stream.write(text)
        self.log.write(text)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'deliverable_hashes.json').exists():
        verify_lock()
        for name,sha in read_json(OUT/'deliverable_hashes.json').items():
            require(digest(OUT/name)==sha,'Completed P1 artifact changed: '+name)
        check_history()
        print('Already complete; all frozen P1 and R1 artifacts verified.',flush=True)
        return
    with (OUT/'execution.log').open('a',encoding='utf-8') as log:
        stdout,stderr = sys.stdout,sys.stderr
        sys.stdout,sys.stderr = Tee(stdout,log),Tee(stderr,log)
        try:
            from .prepare import prepare
            from .train import train_all
            from .evaluate import evaluate_all
            from .finalize import finish
            if not (OUT/'configuration.json').exists():
                prepare()
            else:
                verify_lock()
            train_all()
            evaluate_all()
            finish()
        except BaseException:
            error = traceback.format_exc()
            print(error,flush=True)
            replace_json(OUT/'failure.json',{'time_utc':now(),'traceback':error,
                'stopped_without_adjusting_acceptance_or_outer_selection':True})
            raise
        finally:
            sys.stdout,sys.stderr = stdout,stderr


if __name__=='__main__':
    main()
