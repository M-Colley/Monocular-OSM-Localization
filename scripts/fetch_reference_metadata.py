"""Plan/fetch metadata for every frozen reference ID; no images/models/labels.

Prepare freezes membership before HTTP. Fetch requires that exact plan and
source pin, preserves immutable attempts, and freezes partial unknowns too.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import quote, quote_plus
import zipfile

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from monocular_osm.reference_metadata import (PROFILE, array_sha, json_sha, parse_response,
    plan_requests, profile_record, summarize, unknown_record)

SOURCES = ("scripts/fetch_reference_metadata.py", "monocular_osm/reference_metadata.py",
    "monocular_osm/geo.py", "monocular_osm/__init__.py")


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    """No existing plan, attempt, response, result or freeze can be re-blessed."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def source_identity(commit, expected=None):
    require(Path(__file__).resolve() == (ROOT/"scripts/fetch_reference_metadata.py").resolve(), "foreign executed metadata runner")
    require(subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip() == commit
        and not subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip(), "exact clean metadata source pin required")
    result = {name: sha(ROOT/name) for name in SOURCES}
    require(expected is None or result == expected, "metadata source changed during acquisition")
    for name, module in list(sys.modules.items()):
        if name == "monocular_osm" or name.startswith("monocular_osm."):
            require(getattr(module, "__file__", None) is not None
                and Path(module.__file__).resolve().is_relative_to(ROOT), "loaded metadata module escaped pin")
    return result


def verify_freeze(path, expected):
    allowed_input_name(path)
    path = Path(path).resolve(strict=True)
    allowed_input_name(path)
    require(sha(path) == expected, "frozen manifest digest changed")
    manifest = read(path)
    for relative in manifest["artifacts"]:
        allowed_input_name(relative)  # refuse labels/saved scores before hashing their bytes
    for relative, digest in manifest["artifacts"].items():
        item = (path.parent/relative).resolve(strict=True)
        allowed_input_name(item)
        require(item.is_relative_to(path.parent) and sha(item) == digest, "frozen member changed or escaped")
    return manifest


def allowed_input_name(path):
    text = str(path).replace("\\", "/").casefold()
    basename = text.rsplit("/", 1)[-1]
    saved_score = basename.endswith((".json", ".jsonl")) and re.search(r"(?:^|[_.-])(?:scores?|scoring)(?:[_.-]|$)", basename)
    require("ground_truth" not in text and not saved_score,
            "ground_truth and saved score files are outside metadata inputs")


def artifact_hashes(root):
    """Validate all relative AND resolved names before hashing any member."""
    root = Path(root).resolve(strict=True)
    paths = [path for path in root.rglob("*") if path.is_file()]
    resolved = []
    for path in paths:
        allowed_input_name(path.relative_to(root))
        item = path.resolve(strict=True)
        allowed_input_name(item)
        require(item.is_relative_to(root), "metadata artifact escaped owned family")
        resolved.append((str(path.relative_to(root)), item))
    return {relative: sha(path) for relative, path in resolved}


def frozen_gallery(freeze, expected, relative):
    allowed_input_name(relative)
    freeze = Path(freeze).resolve(strict=True)
    manifest = verify_freeze(freeze, expected)
    require(manifest.get("status") == "complete" and manifest.get("scoring_has_not_run") is True,
            "complete original evidence frozen before scoring required")
    evidence = (freeze.parent/relative).resolve(strict=True)
    allowed_input_name(evidence)
    require(evidence.is_relative_to(freeze.parent) and relative in manifest["artifacts"]
        and sha(evidence) == manifest["artifacts"][relative], "gallery not covered by frozen membership")
    # No similarity/query arrays are read. Check ID/geotag allocation headers first.
    require(evidence.stat().st_size <= 128*1024**2, "bounded evidence artifact required")
    with zipfile.ZipFile(evidence) as archive:
        for name, maximum, kinds, size in (("ref_ids", 9750, "US", 512), ("ref_latlon", 19500, "fiu", 8)):
            require(archive.namelist().count(name+".npy") == 1, "missing/duplicate gallery array")
            with archive.open(name+".npy") as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                require(reader is not None, "unsupported gallery NPY version")
                shape, _, dtype = reader(stream)
            require(np.prod(shape, dtype=object) <= maximum and dtype.kind in kinds and dtype.itemsize <= size,
                    "gallery typed allocation budget exceeded")
    with np.load(evidence, allow_pickle=False) as arrays:
        ids, ll = arrays["ref_ids"], arrays["ref_latlon"]
    return ids, ll, {"original_freeze": str(freeze), "original_freeze_sha256": expected,
        "evidence_relative": relative, "evidence": str(evidence), "evidence_sha256": sha(evidence)}


def prepare(input_freeze, input_freeze_sha256, evidence_relative, output, expected_commit):
    sources = source_identity(expected_commit)
    ids, ll, origin = frozen_gallery(input_freeze, input_freeze_sha256, evidence_relative)
    plan = plan_requests(ids, ll, provenance=origin)
    output = Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(ROOT)
        and not output.is_relative_to(Path(input_freeze).resolve().parent), "new isolated metadata family required")
    output.mkdir(parents=True)
    for name, digest in sources.items():
        target = output/"source_snapshot"/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target); require(sha(target) == digest, "source snapshot changed")
    plan.update(source_commit=expected_commit, source_sha256=sources, prepared_utc=datetime.now(timezone.utc).isoformat())
    write_new(output/"metadata_plan.json", plan)
    source_identity(expected_commit, sources)
    frozen_gallery(input_freeze, input_freeze_sha256, evidence_relative)
    write_new(output/"plan_frozen.json", {"status": "planned", "http_has_not_run": True, "models_or_labels": False,
        "artifacts": artifact_hashes(output)})
    return {"status": "planned", "references": len(ids), "batches": len(plan["batches"]), "plan_freeze_sha256": sha(output/"plan_frozen.json")}


@dataclass
class Response:
    status: int | None
    body: bytes
    error: str | None = None
    retry_after: str | None = None


class Budget:
    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self.clock, self.sleep = clock, sleep
        self.start = clock(); self.next_at = self.start
        self.requests = self.used = self.reserved = 0
        self.stop_reason = None
        self.lock = threading.Lock()

    def stop(self, reason):
        with self.lock:
            if self.stop_reason is None:
                self.stop_reason = reason

    def admit(self):
        while True:
            with self.lock:
                now = self.clock()
                if self.stop_reason is not None:
                    return None
                if now-self.start >= PROFILE["new_request_deadline_s"] or self.requests >= PROFILE["max_requests"]:
                    self.stop_reason = "deadline_or_request_budget"; return None
                available = PROFILE["max_total_decoded_body_bytes"]-self.used-self.reserved
                if available <= 0 and self.reserved == 0:
                    self.stop_reason = "total_body_budget"; return None
                if now >= self.next_at and available > 0:
                    allowance = min(available, PROFILE["max_decoded_body_bytes"])
                    self.requests += 1; self.reserved += allowance
                    self.next_at = now+1/PROFILE["requests_per_second"]
                    return {"request_number": self.requests, "start_elapsed_s": now-self.start, "allowance": allowance}
                delay = min(1., max(self.next_at-now, .01))
            self.sleep(delay)

    def finish(self, allowance, size):
        with self.lock:
            require(0 <= size <= allowance, "transport exceeded reserved decoded-body budget")
            self.reserved -= allowance; self.used += size

    def backoff(self, seconds):
        with self.lock:
            self.next_at = max(self.next_at, self.clock()+seconds)


def safe_body(body, token):
    """Hash original decoded bytes separately; never persist echoed credentials."""
    text = body.decode("utf-8", errors="replace")
    for secret in {token, quote(token, safe=""), quote_plus(token, safe=""), json.dumps(token)[1:-1]}:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r'([?&](?:access_token|token)=)[^&\s"<>]+', r'\1[REDACTED]', text, flags=re.I)
    text = re.sub(r'("(?:access_token|authorization)"\s*:\s*")[^"]*', r'\1[REDACTED]', text, flags=re.I)
    return text.encode("utf-8")


def http_get(batch, token, allowance):
    """One bounded request. No redirects, SDK logging, credentials in URLs or validation probe."""
    import requests
    class KeepOAuth(requests.auth.AuthBase):
        def __call__(self, request):
            return request  # preserve header rather than consulting .netrc
    body = bytearray(); status = None
    try:
        with requests.Session() as session:
            with session.get(PROFILE["endpoint"], params=batch["params"],
                    headers={"Authorization": "OAuth "+token, "Accept": "application/json"}, auth=KeepOAuth(),
                    timeout=(PROFILE["connect_timeout_s"], PROFILE["read_timeout_s"]),
                    allow_redirects=False, stream=True) as response:
                status = response.status_code
                while len(body) < allowance:
                    chunk = response.raw.read(min(16384, allowance-len(body)), decode_content=True)
                    if not chunk:
                        return Response(status, bytes(body), retry_after=response.headers.get("Retry-After"))
                    if len(body)+len(chunk) > allowance:
                        body.extend(chunk[:allowance-len(body)])
                        return Response(status, bytes(body), error="body_limit")
                    body.extend(chunk)
                return Response(status, bytes(body), error="body_limit")
    except Exception as error:
        return Response(status, bytes(body), error=type(error).__name__)


def run_batches(plan, attempt_root, token, *, transport=http_get, clock=time.monotonic, sleep=time.sleep):
    """Fetch all planned IDs or freeze explicit partial unknowns after a stop.

    Body capacity is reserved BEFORE requests, so parallel readers cannot
    overrun the global decoded-byte budget. Every attempt gets new files.
    """
    require(isinstance(token, str) and bool(token.strip()), "MLY_TOKEN missing; no request sent")
    attempt_root = Path(attempt_root)
    attempt_root.mkdir(exist_ok=False)
    budget = Budget(clock, sleep)
    records = {row["reference_id"]: unknown_record(row["reference_id"]) for row in plan["references"]}

    def batch_worker(batch):
        result = {key: unknown_record(key, request_status="not_requested", reason="acquisition_stopped") for key in batch["reference_ids"]}
        attempts = []
        for attempt in range(1, PROFILE["max_attempts_per_batch"]+1):
            admission = budget.admit()
            if admission is None:
                break
            directory = attempt_root/f"batch_{batch['index']:04d}_attempt_{attempt}"
            directory.mkdir(exist_ok=False)
            # This request record is committed before entering the transport.
            write_new(directory/"request.json", {"endpoint": PROFILE["endpoint"], "params": batch["params"],
                "batch_index": batch["index"], "attempt": attempt, "admission": admission,
                "authorization": "OAuth via MLY_TOKEN; value never recorded"})
            began = clock()
            try:
                response = transport(batch, token, admission["allowance"])
                require(isinstance(response, Response) and isinstance(response.body, bytes), "invalid transport return")
                if len(response.body) > admission["allowance"]:
                    response = Response(response.status, response.body[:admission["allowance"]], error="transport_body_limit")
            except Exception as error:
                response = Response(None, b"", error=type(error).__name__)
            budget.finish(admission["allowance"], len(response.body))
            sanitized = safe_body(response.body, token)
            body_path = directory/"response.body"
            with body_path.open("xb") as stream:
                stream.write(sanitized)
            outcome = {"batch_index": batch["index"], "attempt": attempt, "status": response.status,
                "error_type": response.error, "decoded_body_bytes": len(response.body),
                "decoded_body_sha256": hashlib.sha256(response.body).hexdigest(),
                "stored_body_sha256": hashlib.sha256(sanitized).hexdigest(), "body_redacted_or_reencoded": sanitized != response.body,
                "elapsed_s": max(0., clock()-began), "request_number": admission["request_number"]}
            outcome["received_utc"] = datetime.now(timezone.utc).isoformat()
            try:
                seconds = float(response.retry_after)
                outcome["server_retry_after_s"] = seconds if np.isfinite(seconds) and seconds >= 0 else None
            except (ValueError, TypeError):
                outcome["server_retry_after_s"] = None
            if response.status in PROFILE["fatal_statuses"]:
                budget.stop("fatal_http_"+str(response.status))
            parsed = None
            if response.status == 200 and response.error is None and sanitized == response.body:
                try:
                    parsed = parse_response(response.body, batch["reference_ids"])
                except (ValueError, TypeError, KeyError) as error:
                    outcome["schema_error_type"] = type(error).__name__
            outcome["parsed_successfully"] = parsed is not None
            write_new(directory/"response.json", outcome)
            attempts.append(str(directory.relative_to(attempt_root)))
            if parsed is not None:
                return {"index": batch["index"], "success": True, "records": parsed, "attempts": attempts}
            reason = response.error or ("schema_or_redaction_error" if response.status == 200 else "http_"+str(response.status))
            result = {key: unknown_record(key, request_status="failed", reason=reason) for key in batch["reference_ids"]}
            retryable = response.error in {"ConnectTimeout", "ReadTimeout", "Timeout", "ConnectionError", "ReadTimeoutError", "ProtocolError"}
            retryable = retryable or response.error is None and response.status in PROFILE["retry_statuses"]
            if not retryable or attempt == PROFILE["max_attempts_per_batch"]:
                if response.status == 429:
                    budget.stop("rate_limit_retry_exhausted")
                break
            delay = PROFILE["retry_delay_s"]
            if response.status == 429 and response.retry_after is not None:
                try:
                    delay = max(delay, float(response.retry_after))
                except (ValueError, TypeError):
                    pass
                if not np.isfinite(delay) or delay > PROFILE["max_retry_after_s"]:
                    budget.stop("rate_limit_wait_exceeds_budget"); break
            budget.backoff(delay)
        return {"index": batch["index"], "success": False, "records": result, "attempts": attempts}

    batches = plan["batches"]
    completed = [batch_worker(batches[0])]
    records.update(completed[0]["records"])
    if not completed[0]["success"]:
        budget.stop("first_batch_compatibility_failed")
        for batch in batches[1:]:
            skipped = {key: unknown_record(key, reason=budget.stop_reason) for key in batch["reference_ids"]}
            records.update(skipped)
            completed.append({"index": batch["index"], "success": False, "records": skipped, "attempts": []})
    else:
        with ThreadPoolExecutor(max_workers=PROFILE["workers"]) as executor:
            for result in executor.map(batch_worker, batches[1:]):
                completed.append(result); records.update(result["records"])
                print(f"metadata batch {result['index']+1}/{len(batches)}: {'complete' if result['success'] else 'unknown'}", flush=True)
    return {"records": records, "batches": [{key: value for key, value in row.items() if key != "records"} for row in completed],
        "all_batches_successful": len(completed) == len(batches) and all(row["success"] for row in completed),
        "requests": budget.requests, "decoded_body_bytes": budget.used, "stop_reason": budget.stop_reason,
        "elapsed_s": clock()-budget.start, "summary": summarize(plan, records)}


def load_credential():
    token = os.environ.get("MLY_TOKEN", "").strip()
    if token:
        return token, "process_environment"
    if os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, "MLY_TOKEN")
            if isinstance(value, str) and value.strip():
                token = value.strip()
                os.environ["MLY_TOKEN"] = token  # this process only; never write registry or log value
                return token, "HKCU_Environment"
        except OSError:
            pass
    return "", "missing"


def fetch(family, plan_freeze_sha256, expected_commit):
    family = Path(family).resolve(strict=True)
    source_identity(expected_commit)
    verify_freeze(family/"plan_frozen.json", plan_freeze_sha256)
    artifact_hashes(family)  # refuse unexpected saved scores/labels before HTTP as well
    plan = read(family/"metadata_plan.json")
    require(plan["source_commit"] == expected_commit and plan["profile"] == profile_record()
        and plan["profile_sha256"] == json_sha(profile_record()), "registered source/profile mismatch")
    source_identity(expected_commit, plan["source_sha256"])
    original = plan["provenance"]
    ids, ll, provenance = frozen_gallery(original["original_freeze"], original["original_freeze_sha256"], original["evidence_relative"])
    rebuilt = plan_requests(ids, ll, provenance=provenance)
    require(all(plan[key] == value for key, value in rebuilt.items()), "registered ID membership/requests changed")
    require(not (family/"attempts").exists() and not (family/"acquisition_frozen.json").exists(), "refuse metadata acquisition resume or overwrite")
    token, credential_origin = load_credential()
    require(bool(token), "MLY_TOKEN missing; no HTTP requests sent")
    write_new(family/"acquisition_registration.json", {"plan_freeze_sha256": plan_freeze_sha256,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": expected_commit, "profile_sha256": plan["profile_sha256"],
        "credential_status": "MLY_TOKEN present; not prevalidated", "credential_origin": credential_origin, "models_or_labels": False})
    result = run_batches(plan, family/"attempts", token)
    write_new(family/"all_reference_metadata.json", result["records"])
    write_new(family/"request_ledger.json", {key: value for key, value in result.items() if key != "records"})
    source_identity(expected_commit, plan["source_sha256"])
    verify_freeze(family/"plan_frozen.json", plan_freeze_sha256)
    frozen_gallery(original["original_freeze"], original["original_freeze_sha256"], original["evidence_relative"])
    write_new(family/"acquisition_frozen.json", {"status": "complete" if result["all_batches_successful"] else "partial_unknown",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "all_ids_retained": len(result["records"]) == plan["reference_count"], "metadata_is_city_inventory": False,
        "models_or_labels": False, "torch_imported": "torch" in sys.modules, "selected_answer": None,
        "artifacts": artifact_hashes(family)})
    return {"status": "complete" if result["all_batches_successful"] else "partial_unknown", "summary": result["summary"],
        "requests": result["requests"], "stop_reason": result["stop_reason"], "freeze_sha256": sha(family/"acquisition_frozen.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--input-freeze", type=Path, required=True)
    setup.add_argument("--input-freeze-sha256", required=True)
    setup.add_argument("--evidence-relative", default="full_evidence.npz")
    setup.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("fetch")
    run.add_argument("--family", type=Path, required=True)
    run.add_argument("--plan-freeze-sha256", required=True)
    for command in (setup, run):
        command.add_argument("--expected-commit", required=True)
    values = vars(parser.parse_args(argv)); command = values.pop("command")
    if os.name == "nt":
        require(Path(values["output"] if command == "prepare" else values["family"]).resolve().drive.upper() == "E:",
                "metadata CLI requires an isolated E-drive family")
    print(json.dumps(prepare(**values) if command == "prepare" else fetch(**values)))


if __name__ == "__main__":
    main()
