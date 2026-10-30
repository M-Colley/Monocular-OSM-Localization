"""Deadline transport/selection and real owned-process cleanup; no models/GT."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from monocular_osm import assisted_worker as worker
from monocular_osm.pipeline import PipelineConfig
from monocular_osm.runtime_budget import (
    DEADLINE_ENV, CleanupFailed, DeadlineExpired, OwnedProcessTree, RuntimeBudget, positive,
)
from monocular_osm.stages import assisted as stage
from monocular_osm.stages.context import RunContext


class Clock:
    def __init__(self): self.now = 100.
    def __call__(self): return self.now


@pytest.mark.parametrize('value', [True, False, 0, -1, float('nan'), float('inf'), '30'])
def test_budget_numbers_fail_closed(value):
    with pytest.raises(ValueError): positive(value, 'budget')


def test_absolute_deadline_is_never_reset_or_extended():
    clock = Clock()
    cfg = SimpleNamespace(runtime_budget_s=40., runtime_deadline_monotonic_s=125.,
                          runtime_finalization_reserve_s=2., runtime_cleanup_reserve_s=1.)
    budget = RuntimeBudget.from_config(cfg, environ={DEADLINE_ENV: '120'}, clock=clock)
    assert budget.deadline == 120. and budget.work_deadline == 117.
    clock.now = 118.
    with pytest.raises(DeadlineExpired): budget.require_work('expanded')
    assert budget.events[-1]['status'] == 'not_started'
    assert RuntimeBudget.from_config(SimpleNamespace(), environ={}, clock=clock) is None


@pytest.mark.parametrize('value', ['nan', 'inf', '0', '-1', 'bad'])
def test_invalid_inherited_budget_refuses(value):
    with pytest.raises(ValueError):
        RuntimeBudget.from_config(SimpleNamespace(), environ={DEADLINE_ENV: value})


def baseline():
    route = [[49. + i*.001, 8. + i*.001] for i in range(4)]
    return {'status': 'complete', 'route_latlon': route, 'route_timestamps': [0.,1.,2.,3.],
        'comparison': {'statistic': stage.STATISTIC,
            'model': {'name': 'megaloc', 'hub': 'gmberton/MegaLoc', 'weights_sha256': '1'*64},
            'query': {'indices': [0,1,2,3], 'times': [0.,1.,2.,3.], 'mode': 'warp',
                'preprocessing': {'size': [512,512]}, 'pixels_sha256': '2'*64, 'descriptors_sha256': '3'*64},
            'inner_decoder': {'actual': 'viterbi', 'fallback': False, 'settings': {'free_speed': 40}},
            'decoded_reference_cosines': [.12]*4, 'decoded_reference_latlon': route}}


def run_controller(tmp_path, *, baseline_expires=False, late_finalize=False):
    clock = Clock()
    video = tmp_path/'video.mp4'; video.write_bytes(b'synthetic video')
    output = tmp_path/'output'; output.mkdir()
    cfg = SimpleNamespace(output_dir=output, city='Synthetic', vo_start_sec=0., vo_end_sec=3., assisted_profile='v1')
    ctx = RunContext(cfg=cfg, fusion_w=None, video_path=video)
    ctx.runtime_budget = RuntimeBudget(110., 2., 1., 'test', clock(), clock)
    original = baseline()
    # Use the registered hub exactly, without constructing a model.
    from monocular_osm.kartaview_vpr import HUB_MEGALOC
    original['comparison']['model']['hub'] = HUB_MEGALOC
    calls = []
    def run_child(kind, parent):
        calls.append(kind)
        if kind == 'expanded' or baseline_expires:
            clock.now = ctx.runtime_budget.work_deadline
            event = {'kind':kind, 'reason':'child_work_deadline_expired', 'status':'timed_out',
                     'all_tracked_owned_births_gone':True}
            ctx.runtime_budget.events.append(event)
            raise DeadlineExpired(event)
        if kind == 'cues':
            clock.now += 1.
            return {'status':'failed', 'error':'synthetic cues unavailable'}
        clock.now += 1.
        payload = {**deepcopy(original), 'video_sha256':parent.result['assisted_inputs']['video']['sha256'],
            'source_sha256':parent.result['assisted_inputs']['source_sha256'], 'window':[0.,3.]}
        directory = output/'fake'/kind; directory.mkdir(parents=True)
        candidate = directory/'candidate.json'; candidate.write_text(json.dumps(payload))
        result = directory/'result.json'; result.write_text('{}')
        freeze = directory/'child_frozen.json'
        freeze.write_text(json.dumps({'scoring_has_not_run':True,
            'artifacts':{p.name:worker.file_sha256(p) for p in (candidate,result)}}))
        return {**payload, 'worker_returncode':0, **{role:{'path':str(p),'sha256':worker.file_sha256(p)}
            for role,p in [('candidate_artifact',candidate),('result_artifact',result),('frozen_artifact',freeze)]}}
    def finalize(parent):
        assert clock() < parent.runtime_budget.deadline
        if 'assisted_position' in parent.result:
            parent.result['answer'] = deepcopy(parent.result['assisted_position'])
        (output/'result.json').write_text(json.dumps(parent.result))
        clock.now += 1.
        if late_finalize:
            clock.now = parent.runtime_budget.deadline + .1
    result = stage.run_assisted(ctx, finalize=finalize, run_child=run_child)
    return result, calls, original, output, clock


def test_optional_timeout_preserves_exact_baseline_and_clock_before_hard_deadline(tmp_path):
    result, calls, original, output, clock = run_controller(tmp_path)
    assert calls == ['baseline','cues','expanded']
    assert result['answer']['route_latlon'] == original['route_latlon']
    assert result['answer']['route_t_sec'] == original['route_timestamps']
    assert result['assisted_audit']['selected_candidate'] == 'baseline'
    assert result['assisted_audit']['evidence']['expanded']['status'] == 'failed'
    assert clock() < 110.
    frozen = json.loads((output/'frozen_prediction.json').read_text())
    assert frozen['prediction_available'] is True
    assert frozen['artifacts']['assisted_runtime.json'] == worker.file_sha256(output/'assisted_runtime.json')
    from monocular_osm.assisted_policy import decide_assisted_policy
    assert [decide_assisted_policy(v) for v in result['assisted_audit']['decision_inputs']] == result['assisted_audit']['decision_trace']


def test_no_complete_baseline_publishes_explicit_decline_and_starts_no_future_work(tmp_path):
    result, calls, _, output, _ = run_controller(tmp_path, baseline_expires=True)
    assert calls == ['baseline']
    assert 'answer' not in result and result['execution']['outcome'] == 'declined'
    assert result['assisted_audit']['selected_candidate'] is None
    assert json.loads((output/'frozen_prediction.json').read_text())['prediction_available'] is False


def test_slow_mandatory_publication_is_failure_not_successful_freeze(tmp_path):
    with pytest.raises(TimeoutError,match='mandatory assisted publication exceeded'):
        run_controller(tmp_path,late_finalize=True)
    assert not (tmp_path/'output/frozen_prediction.json').exists()


def test_worker_timeout_cannot_promote_an_already_written_complete_response(tmp_path, monkeypatch):
    clock = Clock(); budget = RuntimeBudget(110.,2.,1.,'test',clock(),clock)
    video = tmp_path/'video'; video.write_bytes(b'video')
    ctx = SimpleNamespace(cfg=SimpleNamespace(output_dir=tmp_path/'output'), video_path=video,
        assisted_child_root=tmp_path/'output/children/execution', assisted_evidence={}, runtime_budget=budget,
        result={'assisted_inputs':{'video':{'sha256':worker.file_sha256(video)},'source_sha256':{}}})
    monkeypatch.setattr(worker,'_commit_headroom',lambda:9*1024**3)
    monkeypatch.setattr(worker,'child_config',lambda *args:SimpleNamespace())
    class Process:
        pid=123;returncode=-9
    class Owner:
        def __init__(self,*args,**kwargs):pass
        def wait(self,deadline):raise subprocess.TimeoutExpired('fake',1)
        def stop(self,timeout):return {'all_tracked_owned_births_gone':True,'owned_births':[{'pid':123,'create_time':1.}]}
    def popen(argv,**kwargs):
        Path(argv[-1]).with_name('response.json').write_text('{"status":"complete"}')
        return Process()
    monkeypatch.setattr(worker.subprocess,'Popen',popen)
    import monocular_osm.runtime_budget as module
    monkeypatch.setattr(module,'OwnedProcessTree',Owner)
    with pytest.raises(DeadlineExpired):worker.run_child('baseline',ctx)
    deadline=json.loads((ctx.assisted_child_root/'baseline/deadline.json').read_text())
    assert deadline['all_tracked_owned_births_gone'] and deadline['partial_output_is_selected_or_complete'] is False
    assert budget.events[-1] == deadline


def test_short_real_subprocess_cleanup_keeps_unrelated_birth_alive(tmp_path):
    ready=tmp_path/'ready'
    code=('import subprocess,sys,time,pathlib; '
          'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(20)"]); '
          f'pathlib.Path({str(ready)!r}).write_text(str(p.pid)); time.sleep(20)')
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])
    child=subprocess.Popen([sys.executable,'-c',code])
    owner=OwnedProcessTree(child)
    try:
        with pytest.raises(subprocess.TimeoutExpired):owner.wait(time.monotonic()+.7)
        assert ready.is_file(), 'synthetic owned descendant did not start'
        receipt=owner.stop(3.)
        assert receipt['all_tracked_owned_births_gone']
        assert int(ready.read_text()) in {r['pid'] for r in receipt['owned_births']}
        assert unrelated.poll() is None
    finally:
        if child.poll() is None:owner.stop(3.)
        unrelated.kill();unrelated.wait(timeout=3)


def test_recycled_descendant_is_not_signalled():
    killed=[]
    class Process:
        def __init__(self,pid):self.pid=pid
        def create_time(self):return 1. if self.pid==10 else 3.
        def children(self,recursive):return []
        def kill(self):killed.append(self.pid)
        def is_running(self):return False
    child=SimpleNamespace(pid=10,poll=lambda:0,wait=lambda **kw:0)
    owner=OwnedProcessTree(child,process_factory=Process)
    owner.births[20]=2.  # Actual PID20 now belongs to a different birth.
    assert owner.stop(.1)['all_tracked_owned_births_gone'] and killed == []


@pytest.mark.parametrize('setup_error',[OSError('synthetic birth lookup failure'),
                                       CleanupFailed('synthetic unverifiable birth')])
def test_setup_failure_stops_original_handle_and_never_accepts_response(tmp_path, monkeypatch, setup_error):
    clock=Clock()
    ctx=SimpleNamespace(cfg=SimpleNamespace(output_dir=tmp_path/'output'),video_path=tmp_path/'video',
        assisted_child_root=tmp_path/'children',assisted_evidence={},
        runtime_budget=RuntimeBudget(110.,2.,1.,'test',clock(),clock),
        result={'assisted_inputs':{'video':{'sha256':'1'*64},'source_sha256':{}}})
    monkeypatch.setattr(worker,'_commit_headroom',lambda:9*1024**3)
    monkeypatch.setattr(worker,'child_config',lambda *a:SimpleNamespace())
    stopped=[]
    process=SimpleNamespace(pid=123,poll=lambda:None,kill=lambda:stopped.append('kill'),
                            wait=lambda **kw:stopped.append('wait'))
    monkeypatch.setattr(worker.subprocess,'Popen',lambda *a,**kw:process)
    import monocular_osm.runtime_budget as module
    def bad_owner(*a,**kw):raise setup_error
    monkeypatch.setattr(module,'OwnedProcessTree',bad_owner)
    with pytest.raises(CleanupFailed):worker.run_child('baseline',ctx)
    assert stopped==['kill','wait']


def test_reused_output_without_budget_does_not_freeze_stale_runtime_receipt(tmp_path):
    stale=tmp_path/'assisted_runtime.json';stale.write_text('{"schema":"old-budget"}')
    original=stale.read_bytes()
    (tmp_path/'result.json').write_text('{"answer":null}')
    ctx=SimpleNamespace(cfg=SimpleNamespace(output_dir=tmp_path),runtime_budget=None,
        result={'assisted_inputs':{},'assisted_audit':{'reached_stages':[]}})
    stage.freeze(ctx)
    frozen=json.loads((tmp_path/'frozen_prediction.json').read_text())
    assert 'assisted_runtime.json' not in frozen['artifacts']
    assert stale.read_bytes()==original


def test_capture_failure_still_stops_bound_root():
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])
    owner=OwnedProcessTree(child)
    def fail():raise OSError('synthetic descendant capture failure')
    owner.capture=fail
    try:
        with pytest.raises(CleanupFailed,match='capture failed'):owner.stop(3.)
        assert child.poll() is not None
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=3)


def test_root_completion_cleans_observed_descendant(tmp_path):
    ready=tmp_path/'ready';release=tmp_path/'release'
    code=('import subprocess,sys,time,pathlib\n'
          'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(20)"])\n'
          f'pathlib.Path({str(ready)!r}).write_text(str(p.pid))\n'
          f'while not pathlib.Path({str(release)!r}).exists(): time.sleep(.01)\n')
    child=subprocess.Popen([sys.executable,'-c',code])
    owner=OwnedProcessTree(child)
    try:
        end=time.monotonic()+3
        while not ready.exists() and time.monotonic()<end:time.sleep(.01)
        assert ready.exists()
        owner.capture()
        descendant=int(ready.read_text())
        assert descendant in owner.births
        release.write_text('exit')
        assert owner.wait(time.monotonic()+3)==0
        assert owner.matching(descendant,owner.births[descendant]) is not None
        assert owner.stop(3.)['all_tracked_owned_births_gone']
    finally:
        if child.poll() is None or any(owner.matching(p,b) is not None for p,b in owner.births.items()):
            owner.stop(3.)


def test_cli_exposes_budget_without_changing_default(tmp_path):
    from monocular_osm.cli import build_arg_parser
    args=build_arg_parser().parse_args(['--assisted-only','--runtime-budget-s','7200'])
    assert args.runtime_budget_s == 7200 and args.runtime_finalization_reserve_s == 120
    cfg=PipelineConfig(url='',city='Synthetic',data_dir=tmp_path/'data',output_dir=tmp_path/'out')
    assert cfg.runtime_budget_s is None


def run_transport(tmp_path, monkeypatch, *, cleanup_error=None, expanded_timeout=True):
    """Real request/freeze/response transport and policy, with no inference backend."""
    import monocular_osm.runtime_budget as module
    clock=Clock();video=tmp_path/'video.mp4';video.write_bytes(b'synthetic')
    cfg=PipelineConfig(url='',city='Synthetic',data_dir=tmp_path/'data',output_dir=tmp_path/'out',
                       video_path=video,answer='assisted',vo_end_sec=3.)
    cfg.output_dir.mkdir()
    ctx=RunContext(cfg=cfg,fusion_w=None,video_path=video)
    ctx.runtime_budget=RuntimeBudget(110.,2.,1.,'test',clock(),clock)
    original=baseline()
    from monocular_osm.kartaview_vpr import HUB_MEGALOC
    original['comparison']['model']['hub']=HUB_MEGALOC
    monkeypatch.setattr(worker,'_commit_headroom',lambda:9*1024**3)
    calls=[]
    class Process:
        pid=123;returncode=0
        def __init__(self,kind):self.kind=kind
        def wait(self,**kwargs):
            if self.kind=='expanded' and expanded_timeout:
                clock.now=ctx.runtime_budget.work_deadline
                raise subprocess.TimeoutExpired('fake retrieval',0)
            clock.now+=1.
            return 0
    class Owner:
        def __init__(self,process,**kwargs):self.process=process
        def wait(self,deadline):return self.process.wait()
        def stop(self,timeout):
            if self.process.kind=='expanded' and cleanup_error is not None:
                raise cleanup_error
            self.process.returncode=-9 if self.process.kind=='expanded' else 0
            return {'all_tracked_owned_births_gone':True,'owned_births':[]}
    def popen(argv,**kwargs):
        request=json.loads(Path(argv[-1]).read_text())
        kind=request['kind'];calls.append(kind)
        if kind!='expanded':
            child_cfg=worker._from_snapshot(request['config'])
            payload=deepcopy(original) if kind=='baseline' else {'status':'failed','error':'synthetic no cues'}
            worker._finish(child_cfg,request,payload)
        return Process(kind)
    monkeypatch.setattr(module,'OwnedProcessTree',Owner)
    monkeypatch.setattr(worker.subprocess,'Popen',popen)
    def finalize(parent):
        assert clock()<parent.runtime_budget.deadline
        parent.result['answer']=deepcopy(parent.result['assisted_position'])
        (cfg.output_dir/'result.json').write_text(json.dumps(parent.result))
    result=stage.run_assisted(ctx,finalize=finalize)
    return result,calls,original,cfg,clock


def test_actual_controller_worker_transport_finalizes_after_fake_backend_timeout(tmp_path, monkeypatch):
    result,calls,original,cfg,clock=run_transport(tmp_path,monkeypatch)
    assert calls==['baseline','cues','expanded']
    assert result['answer']['route_latlon']==original['route_latlon']
    assert result['answer']['route_t_sec']==original['route_timestamps']
    assert result['assisted_audit']['evidence']['expanded']['status']=='failed'
    assert json.loads((cfg.output_dir/'frozen_prediction.json').read_text())['prediction_available']
    assert clock()<110.


@pytest.mark.parametrize('expanded_timeout',[True,False])
@pytest.mark.parametrize('failure',[OSError('synthetic kill failure'),psutil.AccessDenied(123),
                                   subprocess.TimeoutExpired('synthetic cleanup',1)])
def test_any_cleanup_error_prevents_parent_publication_even_with_complete_baseline(tmp_path,monkeypatch,failure,expanded_timeout):
    with pytest.raises(CleanupFailed,match='cleanup could not be verified'):
        run_transport(tmp_path,monkeypatch,cleanup_error=failure,expanded_timeout=expanded_timeout)
    assert list((tmp_path/'out/children').glob('*/baseline/child_frozen.json'))
    assert not (tmp_path/'out/result.json').exists()
    assert not (tmp_path/'out/frozen_prediction.json').exists()


@pytest.mark.parametrize('failure',[KeyboardInterrupt(),SystemExit(1)])
def test_cleanup_interrupt_is_not_downgraded_to_unavailable(tmp_path,monkeypatch,failure):
    with pytest.raises(type(failure)):
        run_transport(tmp_path,monkeypatch,cleanup_error=failure)
    assert not (tmp_path/'out/result.json').exists()
    assert not (tmp_path/'out/frozen_prediction.json').exists()


@pytest.mark.parametrize('step',['kill','wait'])
@pytest.mark.parametrize('failure',[OSError('synthetic root cleanup failure'),psutil.AccessDenied(123),
                                   subprocess.TimeoutExpired('root cleanup',1)])
def test_setup_fallback_kill_and_wait_errors_are_cleanup_failures(tmp_path,monkeypatch,step,failure):
    clock=Clock()
    ctx=SimpleNamespace(cfg=SimpleNamespace(output_dir=tmp_path/'output'),video_path=tmp_path/'video',
        assisted_child_root=tmp_path/'children',assisted_evidence={},
        runtime_budget=RuntimeBudget(110.,2.,1.,'test',clock(),clock),
        result={'assisted_inputs':{'video':{'sha256':'1'*64},'source_sha256':{}}})
    monkeypatch.setattr(worker,'_commit_headroom',lambda:9*1024**3)
    monkeypatch.setattr(worker,'child_config',lambda *a:SimpleNamespace())
    def cleanup_step(name):
        if name==step:raise failure
    process=SimpleNamespace(pid=123,poll=lambda:None,kill=lambda:cleanup_step('kill'),
                            wait=lambda **kw:cleanup_step('wait'))
    monkeypatch.setattr(worker.subprocess,'Popen',lambda *a,**kw:process)
    import monocular_osm.runtime_budget as module
    def bad_owner(*a,**kw):raise OSError('synthetic birth lookup failure')
    monkeypatch.setattr(module,'OwnedProcessTree',bad_owner)
    with pytest.raises(CleanupFailed,match='cleanup could not be verified'):
        worker.run_child('baseline',ctx)


def test_budgeted_parent_publishes_answer_without_optional_plot(tmp_path, monkeypatch):
    from monocular_osm import pipeline, position
    video=tmp_path/'video.mp4';video.write_bytes(b'synthetic encoded input')
    cfg=PipelineConfig(url='',city='Synthetic',data_dir=tmp_path/'data',output_dir=tmp_path/'out',
        video_path=video,answer='assisted',vo_end_sec=3.,runtime_budget_s=1000.)
    original=baseline()
    from monocular_osm.kartaview_vpr import HUB_MEGALOC
    original['comparison']['model']['hub']=HUB_MEGALOC
    original['comparison']['decoded_reference_cosines']=[.2]*4
    def child(kind,ctx):
        assert kind=='baseline'
        child_cfg=worker.child_config(cfg,kind,video)
        child_cfg.output_dir=ctx.assisted_child_root/kind;child_cfg.output_dir.mkdir(parents=True)
        return {**worker._finish(child_cfg,{'kind':kind,
            'video_sha256':ctx.result['assisted_inputs']['video']['sha256'],
            'source_sha256':ctx.result['assisted_inputs']['source_sha256']},original),'worker_returncode':0}
    plots=[]
    monkeypatch.setattr(position,'plot_answer',lambda *a,**kw:plots.append(True))
    monkeypatch.setattr(worker,'run_child',child)
    result=pipeline.run_pipeline(cfg)
    assert not plots
    assert result['answer']['route_latlon']==original['route_latlon']
    assert result['answer']['map_skipped']=='runtime_budget_preserves_mandatory_publication'
    assert (cfg.output_dir/'frozen_prediction.json').exists()
