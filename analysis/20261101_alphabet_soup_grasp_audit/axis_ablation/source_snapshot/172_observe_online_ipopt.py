"""Recording-only observations around unchanged online solver calls.

No callback, settings or return value is changed. Serialization and geometry
diagnostics run after the entire maintained rollout, outside solve budgets.
"""

from dataclasses import asdict
import time

from tasktol.control.ipopt import IpoptIK, IpoptSettings
from tasktol.control.ipopt.adapter import IpoptAdapter


class OnlineIpoptObserver:
    def __init__(self):
        self.ik = None
        self.events = []
        self._active = None

    def __enter__(self):
        original_ik = IpoptIK.solve
        original_adapter = IpoptAdapter.solve
        self._original = original_ik, original_adapter
        observer = self

        def adapter(adapter, *args, **kwargs):
            if observer._active is None:
                return original_adapter(adapter, *args, **kwargs)
            wall, cpu = time.perf_counter(), time.process_time()
            result = original_adapter(adapter, *args, **kwargs)
            observer._active['adapter'] = (result, (time.perf_counter()-wall)*1000,
                                           (time.process_time()-cpu)*1000)
            return result

        def solve(ik, *args, **kwargs):
            if ik is not observer.ik:
                return original_ik(ik, *args, **kwargs)
            assert ik.settings == IpoptSettings()
            event = {}
            observer._active = event
            wall, cpu = time.perf_counter(), time.process_time()
            try:
                result = original_ik(ik, *args, **kwargs)
            finally:
                event['ik_wall_ms'] = (time.perf_counter()-wall)*1000
                event['ik_cpu_ms'] = (time.process_time()-cpu)*1000
                observer._active = None
            observer.events.append(event)
            return result

        IpoptAdapter.solve, IpoptIK.solve = adapter, solve
        return self

    def __exit__(self, *_):
        IpoptIK.solve, IpoptAdapter.solve = self._original

    def records(self, rows):
        assert len(rows) == len(self.events)
        records = []
        for index, (row, event) in enumerate(zip(rows, self.events, strict=True)):
            item = dict(index=index, ik_wall_ms=event['ik_wall_ms'], ik_cpu_ms=event['ik_cpu_ms'],
                        solver_called=row['optimizer']['solver_called'], published=not row['aborted'])
            assert ('adapter' in event) == item['solver_called']
            if 'adapter' in event:
                raw, wall, cpu = event['adapter']
                assert raw.elapsed_ms == row['optimizer']['elapsed_ms']
                item.update(status_code=raw.status_code, status=raw.status, iterations=raw.iterations,
                    adapter_wrapper_wall_ms=wall, adapter_wrapper_cpu_ms=cpu,
                    problem_solve_wall_ms=raw.elapsed_ms, raw_candidate_q_rad=raw.q.tolist(),
                    raw_candidate_validation=asdict(raw.validation), raw_optimality=raw.optimality)
            records.append(item)
        return records
