"""Private fake-decoder proofs; never opens an actual study payload or video."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from monocular_osm import multiview_motion_inputs as m
from scripts import prepare_multiview_motion_inputs as p


class CV:
    CAP_FFMPEG = 1900
    CAP_PROP_N_THREADS = 70
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_POS_FRAMES = 1
    CAP_PROP_POS_MSEC = 0
    @staticmethod
    def setNumThreads(value):
        assert value == 2


class Capture:
    def __init__(self, *, mutate=None):
        self.position = 0
        self.released = False
        self.mutate = mutate
    def isOpened(self):
        return True
    def set(self, key, value):
        assert key == CV.CAP_PROP_POS_FRAMES
        self.position = int(value)
        return True
    def get(self, key):
        values = {CV.CAP_PROP_N_THREADS: 2, CV.CAP_PROP_FPS: 8., CV.CAP_PROP_FRAME_COUNT: 500.,
                  CV.CAP_PROP_POS_FRAMES: float(self.position), CV.CAP_PROP_POS_MSEC: self.position*125.}
        value = values[key]
        return self.mutate("get", key, value) if self.mutate else value
    def read(self):
        ordinal = self.position
        self.position += 1
        frame = np.full((4, 6, 3), ordinal % 256, np.uint8)
        return self.mutate("read", ordinal, frame) if self.mutate else (True, frame)
    def release(self):
        self.released = True


def fixture(tmp_path, monkeypatch, n=38):
    root = tmp_path.resolve()
    def write(name, value):
        path = root/name
        assert path.resolve().is_relative_to(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        m.write_json(path, value)
        return m.ref(path)
    source_root = root/"source"
    sources = {}
    for name in m.SOURCE_PATHS:
        path = source_root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# private source role\n")
        sources[name] = m.sha(path)
    monkeypatch.setattr(m, "ROOT", source_root)
    video = root/"video.private"
    video.write_bytes(b"owned fake video bytes")
    first, stride = 3, 2
    times = np.asarray([(first+r*stride+1)*.125 for r in range(n)], np.float64)
    f = 3./np.tan(np.deg2rad(70.)/2.)
    K = np.array([[f, 0., 3.], [0., f, 2.], [0., 0., 1.]], np.float64)
    pixel = hashlib.sha256()
    for r in range(n):
        frame = np.full((4, 6, 3), (first+r*stride) % 256, np.uint8)
        pixel.update(json.dumps([list(frame.shape), frame.dtype.str]).encode())
        pixel.update(frame.tobytes())
    video_binding = dict(m.ref(video), size=video.stat().st_size)
    signature = {"schema": 2, "vo_algo_version": 3, "video": {"path": str(video).lower() if __import__('os').name == 'nt' else str(video), "size": video.stat().st_size},
                 "frames_sha256": pixel.hexdigest(), "extraction": {"n_frames": n, "stride": stride, "fps": 8.,
                 "total_frames": 500, "timestamps_sha256": hashlib.sha256(times.astype('<f8').tobytes()).hexdigest()}}
    cache = root/"cache.npz"
    np.savez(cache, cache_signature=np.asarray(json.dumps(signature)), vo_algo_version=np.asarray(3, np.int64))
    clock = root/"clock.npz"
    np.savez(clock, timestamps=times, K=K)
    basis = {"native_vo": m.ref(cache), "native_clock": m.ref(clock)}
    basis.update({k: write(k+".json", {"owned": k}) for k in ("native_facts", "native_root_acceptance", "native_replay_config")})
    rows = m.windows(n, first, stride)
    for row in rows:
        row["signed_decoder_times_s"] = times[row["native_read_rows"]].tolist()
    read = {"schema": "root-fixed32-native-read-window-lineage-v1", "basis_refs": basis,
            "video_binding": video_binding, "native_dense_count": n, "first_native_video_ordinal": first,
            "original_video_ordinal_stride": stride, "last_native_video_ordinal": first+(n-1)*stride,
            "expected_native_shape": [4, 6, 3], "windows": rows, "new_pixels_decoded": False, "ground_truth_used": False,
            "full_signed_native_clock_typed_sha256": m.typed(times),
            "full_signed_native_clock_raw_F64_sha256": signature['extraction']['timestamps_sha256'],
            "full_signed_native_frames_sha256": pixel.hexdigest(), "native70_K_typed_sha256": m.typed(K)}
    read_ref = write("read.json", read)
    monkeypatch.setattr(m, "READ_LEDGER_REF", read_ref)
    study = {"schema": "root-prospective-native-multiview-window-study-v1", "native_read_ledger_ref": read_ref,
             "ground_truth_or_map_or_OCR_used": False, "new_neural_forwards_or_downloads": False,
             "policy_admission": False, "selected_answer": None,
             "frontend_source_to_promote_ref": {"path": str(source_root/'monocular_osm/multiview_motion.py'),
                                                "sha256": sources['monocular_osm/multiview_motion.py']}}
    study_ref = write("study.json", study)
    monkeypatch.setattr(m, "STUDY_REF", study_ref)
    config = {"schema": "native-multiview-input-prepare-v1", "source_hashes": sources,
              "study_registration_ref": study_ref, "native_read_ledger_ref": read_ref, "output_dir": str(root/"prepared")}
    config_ref = write("config.json", config)
    return dict(root=root, config=config, config_ref=config_ref, read=read, study=study, times=times, K=K)


def run(case, capture=None, check=lambda: None):
    capture = Capture() if capture is None else capture
    def factory(path, api, params):
        assert path == case["read"]["video_binding"]["path"] and api == CV.CAP_FFMPEG
        assert params == [CV.CAP_PROP_N_THREADS, 2]
        return capture
    result = p.execute(case["config_ref"]["path"], case["config_ref"]["sha256"],
                       capture_factory=factory, cv2_module=CV, check=check)
    return result, capture


def reseal_config(case):
    path = Path(case['config_ref']['path'])
    assert path.is_relative_to(case['root'])
    path.write_bytes(m.json_bytes(case['config']))
    case['config_ref'] = m.ref(path)


def test_whole_dense_original_read_replay_and_32_overlapping_windows(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch)
    result, cap = run(case)
    assert cap.released and result['window_count'] == 32
    assert result['full_native_proof']['dense_count'] == 38
    assert result['full_native_proof']['raw_ordinal_read_count'] == 75
    assert result['windows'][0]['native_read_rows'] == list(range(7))
    assert result['windows'][-1]['native_read_rows'] == list(range(31, 38))
    a = m.Ledger()
    arrays = m.read_prepared_window(result['windows'][9], a)
    assert arrays['frames_bgr'].shape == (7, 4, 6, 3)
    for ordinal, frame in zip(arrays['video_ordinals'], arrays['frames_bgr']):
        assert np.all(frame == int(ordinal) % 256)
    assert (Path(case['config']['output_dir'])/'manifest.json').is_file()


@pytest.mark.parametrize('defect', ['clock_zero', 'clock_changed', 'ordinal', 'shape', 'dtype', 'pixel', 'truncated', 'threads'])
def test_fake_capture_failure_never_publishes_partial_master(tmp_path, monkeypatch, defect):
    case = fixture(tmp_path, monkeypatch)
    def mutate(kind, key, value):
        if kind == 'get':
            if defect == 'clock_zero' and key == CV.CAP_PROP_POS_MSEC: return 0.
            if defect == 'clock_changed' and key == CV.CAP_PROP_POS_MSEC: return value+.001
            if defect == 'ordinal' and key == CV.CAP_PROP_POS_FRAMES: return value+1
            if defect == 'threads' and key == CV.CAP_PROP_N_THREADS: return 4.
            return value
        if defect == 'shape': return True, value[:, :-1].copy()
        if defect == 'dtype': return True, value.astype(np.float32)
        if defect == 'pixel': return True, np.zeros_like(value)
        if defect == 'truncated': return False, None
        return True, value
    cap = Capture(mutate=mutate)
    with pytest.raises(ValueError):
        run(case, cap)
    assert cap.released
    assert not (Path(case['config']['output_dir'])/'manifest.json').exists()


@pytest.mark.parametrize('defect', ['source', 'study', 'ledger', 'video'])
def test_authentication_refuses_before_npz_or_decoder(tmp_path, monkeypatch, defect):
    case = fixture(tmp_path, monkeypatch)
    if defect == 'source':
        case['config']['source_hashes'][m.SOURCE_PATHS[0]] = 'f'*64
    elif defect in ('study', 'ledger'):
        case['config']['study_registration_ref' if defect == 'study' else 'native_read_ledger_ref'] = {
            'path': str(case['root']/'unknown.json'), 'sha256': 'f'*64}
    else:
        path = Path(case['read']['video_binding']['path'])
        assert path.is_relative_to(case['root'])
        path.write_bytes(b'changed owned video')
    reseal_config(case)
    monkeypatch.setattr(m, 'load_named_npz', lambda *a, **k: pytest.fail('premature NPZ'))
    with pytest.raises(ValueError):
        run(case)
    assert not Path(case['config']['output_dir']).exists()


@pytest.mark.parametrize('which', ['source', 'window', 'membership'])
def test_last_callback_mutation_refuses_publication(tmp_path, monkeypatch, which):
    case = fixture(tmp_path, monkeypatch)
    output = Path(case['config']['output_dir'])
    changed = []
    def check():
        if output.exists() and (output/'window_31.npz').exists() and not changed:
            path = (m.ROOT/m.SOURCE_PATHS[0] if which == 'source' else output/'window_00.npz'
                    if which == 'window' else output/'intruder.json')
            assert path.resolve().is_relative_to(case['root'])
            path.write_bytes(b'private last-callback mutation')
            changed.append(path)
    with pytest.raises(ValueError):
        run(case, check=check)
    assert changed and not (output/'manifest.json').exists()


def test_compressed_oversized_last_member_refuses_before_first_header(tmp_path, monkeypatch):
    path = tmp_path/'bomb.npz'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('small.npy', b'not-read')
        z.writestr('large.npy', b'x'*100000)
    assert path.stat().st_size < 2000
    monkeypatch.setattr(m, 'MAX_NPZ', 2000)
    monkeypatch.setattr(np.lib.format, 'read_magic', lambda *a: pytest.fail('header before all ZipInfo caps'))
    with pytest.raises(ValueError, match='uncompressed'):
        m.zip_preflight(path)


@pytest.mark.parametrize('kind', ['duplicate', 'path', 'object', 'fortran', 'trailing'])
def test_npz_header_and_member_rejections(tmp_path, kind):
    path = tmp_path/'bad.npz'
    body = io.BytesIO()
    value = np.array([object()], dtype=object) if kind == 'object' else np.asfortranarray(np.ones((2, 3))) if kind == 'fortran' else np.ones(3)
    np.lib.format.write_array(body, value, allow_pickle=True)
    data = body.getvalue() + (b'x' if kind == 'trailing' else b'')
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('../a.npy' if kind == 'path' else 'a.npy', data)
        if kind == 'duplicate':
            z.writestr('a.npy', data)
    with pytest.raises(ValueError):
        m.zip_preflight(path)


def close_packet(case, manifest):
    root = case['root']
    def write(name, value):
        path = root/name
        assert path.resolve().is_relative_to(root)
        m.write_json(path, value)
        return m.ref(path)
    mr = m.ref(Path(case['config']['output_dir'])/'manifest.json')
    reg = write('registration.json', {'owned': 'external guard'})
    result = write('result.json', {'status': 'complete', 'returncode': 0, 'reason': None})
    closure = write('closure.json', {'schema': 'root-native-multiview-input-external-closure-v1',
        'prepared_manifest_ref': mr, 'registration_ref': reg, 'result_ref': result,
        'worker_births_closed': True, 'protected_inputs_unchanged': True})
    qa = {'schema': 'root-closed-native-multiview-prepared-inputs-v1',
          'status': 'ACCEPTED_NATIVE_PIXEL_CLOCK_WINDOWS_NO_GEOMETRY', 'study_ref': m.STUDY_REF,
          'native_read_ledger_ref': m.READ_LEDGER_REF, 'prepared_manifest_ref': mr,
          'producer_registration_ref': reg, 'result_ref': result, 'external_closure_ref': closure,
          'all_observed_worker_births_closed': True, 'all_protected_input_hashes_unchanged': True,
          'full_dense_native_pixels_clock_verified': True, 'no_geometry_computed': True, 'window_count': 32}
    config = dict(case['config'], schema='native-multiview-motion-probe-run-v1',
                  prepared_input_acceptance_ref=write('rootqa.json', qa))
    return config, qa


def test_closed_whole32_metadata_admission_does_not_read_npz_members(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch)
    manifest, _ = run(case)
    config, _ = close_packet(case, manifest)
    ledger = m.Ledger()
    ledger.read = case['read']
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('NPZ values during metadata admission'))
    assert m.admit_prepared(config, ledger)['windows'] == manifest['windows']


@pytest.mark.parametrize('field', ['all_observed_worker_births_closed', 'full_dense_native_pixels_clock_verified', 'window_count'])
def test_root_closure_or_complete_cohort_required_before_window_reads(tmp_path, monkeypatch, field):
    case = fixture(tmp_path, monkeypatch)
    manifest, _ = run(case)
    config, qa = close_packet(case, manifest)
    qa[field] = 31 if field == 'window_count' else False
    path = Path(config['prepared_input_acceptance_ref']['path'])
    assert path.is_relative_to(case['root'])
    path.write_bytes(m.json_bytes(qa))
    config['prepared_input_acceptance_ref'] = m.ref(path)
    ledger = m.Ledger()
    ledger.read = case['read']
    monkeypatch.setattr(m, 'zip_preflight', lambda *a, **k: pytest.fail('premature NPZ headers'))
    with pytest.raises(ValueError):
        m.admit_prepared(config, ledger)


def test_shared_json_budget_repeated_alias_and_duplicates(tmp_path, monkeypatch):
    path = tmp_path/'meta.json'
    path.write_bytes(b'{"x":1}')
    role = m.ref(path)
    ledger = m.Ledger()
    monkeypatch.setattr(m, 'TOTAL_JSON', 2*path.stat().st_size)
    assert ledger.json(role) == {'x': 1}
    assert ledger.json(dict(role, path=str(path.parent/'.'/path.name))) == {'x': 1}
    with pytest.raises(ValueError, match='budget'):
        ledger.json(role)
    with pytest.raises(ValueError, match='duplicate'):
        m.strict_json(b'{"x":1,"x":2}')


def test_unselected_original_read_pixels_are_still_verified(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch, n=300)
    selected = {r for w in case['read']['windows'] for r in w['native_read_rows']}
    missing = next(r for r in range(300) if r not in selected)
    ordinal = 3+2*missing
    def mutate(kind, key, value):
        if kind == 'read':
            if key == ordinal:
                value = value.copy()
                value[0, 0, 0] ^= np.uint8(1)
            return True, value
        return value
    with pytest.raises(ValueError, match='whole original'):
        run(case, Capture(mutate=mutate))
    output = Path(case['config']['output_dir'])
    assert len(list(output.glob('window_*.npz'))) == 32
    assert not (output/'manifest.json').exists()


def test_total_prepared_reserve_before_first_npz_write(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(m, 'MAX_PREPARED', 1)
    with pytest.raises(ValueError, match='reserve'):
        run(case)
    assert not list(Path(case['config']['output_dir']).glob('*.npz'))


@pytest.mark.parametrize('bad', [37, True, 20001])
def test_original_dense_row_eligibility(bad):
    with pytest.raises(ValueError, match='bounds'):
        m.windows(bad, 0, 4)


def test_window_replacement_refused_before_header_read(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch)
    manifest, _ = run(case)
    record = manifest['windows'][0]
    path = Path(record['array_ref']['path'])
    assert path.is_relative_to(case['root'])
    path.write_bytes(b'replaced owned window')
    monkeypatch.setattr(m, 'zip_preflight', lambda *a, **k: pytest.fail('header before expected file bytes'))
    with pytest.raises(ValueError, match='protected bytes'):
        m.read_prepared_window(record, m.Ledger())


def test_numerical_phase_never_reopens_historical_raw_assets(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch)
    manifest, _ = run(case)
    config, _ = close_packet(case, manifest)
    historical = {Path(r['path']).resolve() for r in case['read']['basis_refs'].values()}
    historical.add(Path(case['read']['video_binding']['path']).resolve())
    historical.add(Path(manifest['prepare_config_ref']['path']).resolve())
    original_open = Path.open
    def bounded_open(path, *args, **kwargs):
        assert path.absolute() not in historical, 'numerical phase reopened historical raw input'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', bounded_open)
    ledger = m.Ledger()
    ledger.read = case['read']
    accepted = m.admit_prepared(config, ledger)
    arrays = m.read_prepared_window(accepted['windows'][0], ledger)
    ledger.verify()
    assert arrays['frames_bgr'].shape == (7, 4, 6, 3)
    assert all(Path(path).absolute() not in historical for path in ledger.protected)
    assert ledger.historical_protected_refs == manifest['protected_hashes']
