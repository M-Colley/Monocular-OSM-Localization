"""Benchmark deadline is established before spawn; no inference or labels."""
import importlib.util
from pathlib import Path
import pytest
from monocular_osm.runtime_budget import DEADLINE_ENV

spec=importlib.util.spec_from_file_location('benchmark_deadline',Path(__file__).resolve().parents[1]/'scripts/benchmark.py')
benchmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark)


@pytest.mark.parametrize('inherited,expected',[(None,8200.),('8100',8100.),('9000',8200.)])
def test_run_one_propagates_earliest_deadline_without_mutating_caller(tmp_path,monkeypatch,inherited,expected):
    captured={}
    monkeypatch.setattr(benchmark.time,'monotonic',lambda:1000.)
    class Process:
        def wait(self,timeout):captured['wait']=timeout;return 0
    def popen(argv,**kwargs):captured.update(kwargs);return Process()
    monkeypatch.setattr(benchmark.subprocess,'Popen',popen)
    env={} if inherited is None else {DEADLINE_ENV:inherited}
    before=dict(env)
    result=benchmark.run_one(['synthetic'],tmp_path/'log',env,7200.)
    assert result['rc']==0 and not result['timed_out']
    assert float(captured['env'][DEADLINE_ENV])==expected
    assert captured['wait']==expected-1000. and env==before


@pytest.mark.parametrize('timeout',[0,-1,True,float('nan'),float('inf')])
def test_invalid_timeout_refuses_before_spawn(tmp_path,monkeypatch,timeout):
    monkeypatch.setattr(benchmark.subprocess,'Popen',lambda *a,**k:pytest.fail('invalid budget spawned'))
    with pytest.raises(ValueError):benchmark.run_one(['synthetic'],tmp_path/'log',{},timeout)
