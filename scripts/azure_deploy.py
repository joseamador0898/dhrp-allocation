"""End-to-end Azure ML deployment for the DHRP notebook experiment.

Provisions everything from scratch:
  1. Auth via DeviceCodeCredential (one-time browser approval)
  2. Create Azure ML workspace `dhrp-ws` in resource group `luigiboy23_-rg` (East US 2)
  3. Create AmlCompute cluster `dhrp-t4` (1x T4, scale-to-zero, idle 5 min)
  4. Build a curated environment: PyTorch + CUDA + project deps
  5. Submit a `command` job that clones the repo HEAD and runs
     `papermill notebooks/llm_dhrp_experiments.ipynb outputs/run.ipynb`
  6. Stream live logs to stdout
  7. Download the executed notebook + results/*.csv back into the local repo

Idempotent: every step checks for existing resources before creating.

Usage:
    python scripts/azure_deploy.py [--watch-only JOB_NAME]

Environment overrides (optional):
    DHRP_SUBSCRIPTION   default: bf5b14c8-093f-4ab9-a3b5-7352f72650ea
    DHRP_RESOURCE_GROUP default: luigiboy23_-rg
    DHRP_LOCATION       default: eastus2
    DHRP_WORKSPACE      default: dhrp-ws
    DHRP_COMPUTE        default: dhrp-t4
    DHRP_VM_SIZE        default: Standard_NC4as_T4_v3
    DHRP_GIT_URL        default: https://github.com/joseamador0898/dhrp-allocation.git
    DHRP_GIT_REF        default: main
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import (
    AmlCompute,
    Workspace,
    Environment,
    BuildContext,
)
from azure.ai.ml.constants import AssetTypes
from azure.identity import (
    DeviceCodeCredential,
    TokenCachePersistenceOptions,
    AuthenticationRecord,
)
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError


# -----------------------------------------------------------------------------
# Configuration (env-overridable)
# -----------------------------------------------------------------------------
SUBSCRIPTION = os.environ.get("DHRP_SUBSCRIPTION", "bf5b14c8-093f-4ab9-a3b5-7352f72650ea")
TENANT_ID = os.environ.get("DHRP_TENANT", "9c483771-7648-4b3a-bafa-7485e8b9d96b")
RESOURCE_GROUP = os.environ.get("DHRP_RESOURCE_GROUP", "luigiboy23_-rg")
LOCATION = os.environ.get("DHRP_LOCATION", "eastus2")
WORKSPACE = os.environ.get("DHRP_WORKSPACE", "dhrp-ws")
COMPUTE = os.environ.get("DHRP_COMPUTE", "dhrp-cpu2")  # fresh name; CPU fallback while T4 quota is requested
VM_SIZE = os.environ.get("DHRP_VM_SIZE", "Standard_E8s_v3")  # 8 vCPU, 64 GB RAM, ESv3 family (100 vCPU quota)
GIT_URL = os.environ.get("DHRP_GIT_URL", "https://github.com/joseamador0898/dhrp-allocation.git")
GIT_REF = os.environ.get("DHRP_GIT_REF", "main")
ENV_NAME = "dhrp-cpu2-env"  # fresh CPU env using a verified Microsoft base image
JOB_DISPLAY_NAME = f"dhrp-experiment-{int(time.time())}"
LOCAL_REPO = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[deploy] {msg}", flush=True)


# -----------------------------------------------------------------------------
# 1. Auth
# -----------------------------------------------------------------------------
def _device_code_prompt(verification_uri, user_code, expires_on):
    """Print the device-code prompt explicitly to stdout (not TTY-only)."""
    msg = (
        "\n"
        "================================================================\n"
        ">>> AZURE DEVICE CODE LOGIN <<<\n"
        f"Open this URL in any browser: {verification_uri}\n"
        f"Enter this code:              {user_code}\n"
        f"Expires at:                   {expires_on}\n"
        "================================================================\n"
    )
    print(msg, flush=True)
    sys.stdout.flush()


# Azure CLI's well-known public client ID — works on any tenant without needing
# a separate app registration.
AZURE_CLI_CLIENT_ID = "04b07795-8ddb-461a-bbee-02f9e1bf7b46"


CACHE_FILE = os.path.expanduser("~/.dhrp_azure_auth_record.json")


def _load_auth_record():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return AuthenticationRecord.deserialize(f.read())
        except Exception:
            return None
    return None


def _save_auth_record(record: AuthenticationRecord):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        f.write(record.serialize())


def get_credential() -> DeviceCodeCredential:
    cache_opts = TokenCachePersistenceOptions(
        name="dhrp-deploy", allow_unencrypted_storage=True
    )
    record = _load_auth_record()
    if record is not None:
        log("Reusing cached auth record (no device-code prompt expected)")
        return DeviceCodeCredential(
            tenant_id=TENANT_ID,
            client_id=AZURE_CLI_CLIENT_ID,
            prompt_callback=_device_code_prompt,
            cache_persistence_options=cache_opts,
            authentication_record=record,
        )
    log("Requesting device-code credential (first run)...")
    log(f"Tenant: {TENANT_ID}")
    log("A URL and code will print below. Open the URL in any browser,")
    log("paste the code, and approve. Auth is then cached and reused.")
    cred = DeviceCodeCredential(
        tenant_id=TENANT_ID,
        client_id=AZURE_CLI_CLIENT_ID,
        prompt_callback=_device_code_prompt,
        cache_persistence_options=cache_opts,
    )
    record = cred.authenticate(scopes=["https://management.azure.com/.default"])
    _save_auth_record(record)
    log(f"Auth record cached at {CACHE_FILE}")
    return cred


def get_ml_client_subscription_only(credential) -> MLClient:
    """MLClient without a workspace, used to create the workspace itself."""
    return MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION,
        resource_group_name=RESOURCE_GROUP,
    )


def get_ml_client(credential) -> MLClient:
    """MLClient bound to the target workspace."""
    return MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE,
    )


# -----------------------------------------------------------------------------
# 2. Workspace
# -----------------------------------------------------------------------------
def ensure_workspace(credential) -> None:
    sub_client = get_ml_client_subscription_only(credential)
    try:
        ws = sub_client.workspaces.get(WORKSPACE)
        log(f"Workspace already exists: {ws.name} ({ws.location})")
        return
    except ResourceNotFoundError:
        pass
    except HttpResponseError as e:
        if e.status_code != 404:
            raise

    log(f"Creating workspace '{WORKSPACE}' in {RESOURCE_GROUP}/{LOCATION}...")
    ws = Workspace(
        name=WORKSPACE,
        location=LOCATION,
        display_name="DHRP Experiment Workspace",
        description="Auto-provisioned for the DHRP paper Colab-equivalent experiment runs",
        tags={"project": "dhrp", "owner": "auto"},
    )
    poller = sub_client.workspaces.begin_create(ws)
    log("Waiting for workspace creation (3-5 min on first run)...")
    result = poller.result()
    log(f"Workspace created: {result.name}")


# -----------------------------------------------------------------------------
# 3. Compute cluster
# -----------------------------------------------------------------------------
def ensure_compute(client: MLClient) -> None:
    try:
        c = client.compute.get(COMPUTE)
        log(f"Compute cluster already exists: {c.name} ({c.size})")
        return
    except ResourceNotFoundError:
        pass

    log(f"Creating compute cluster '{COMPUTE}' ({VM_SIZE}, scale 0-1, idle 300s)...")
    cluster = AmlCompute(
        name=COMPUTE,
        type="amlcompute",
        size=VM_SIZE,
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=300,
        tier="Dedicated",
    )
    poller = client.compute.begin_create_or_update(cluster)
    result = poller.result()
    log(f"Compute cluster created: {result.name}")


# -----------------------------------------------------------------------------
# 4. Environment
# -----------------------------------------------------------------------------
def ensure_environment(client: MLClient) -> Environment:
    """Build a CPU environment with PyTorch + project deps.

    Uses an Azure ML curated registry environment as the base
    (azureml://registries/azureml/environments/sklearn-1.5/labels/latest)
    which is guaranteed to exist in the global registry. We then layer
    project deps via a conda yaml.

    The DHRP layer is small and trains comfortably on CPU; the focused
    OOS script only needs the DHRP training + classical baselines, no
    FinBERT/Qwen3.
    """
    conda_yaml = """\
name: dhrp-cpu2-env
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip=24.*
  - pip:
    - --extra-index-url https://download.pytorch.org/whl/cpu
    - torch==2.2.2+cpu
    - numpy<2
    - pandas>=2.0
    - scipy>=1.11
    - statsmodels>=0.14
    - scikit-learn>=1.4
    - cvxpy>=1.5
    - yfinance>=0.2.40
    - pandas-datareader>=0.10
    - fredapi>=0.5
    - tqdm
"""
    env_dir = LOCAL_REPO / ".azure_env_build"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "conda.yaml").write_text(conda_yaml, encoding="utf-8")

    env = Environment(
        name=ENV_NAME,
        description="DHRP CPU environment: PyTorch 2.2 CPU + finance deps (no transformers)",
        # Verified existing Microsoft AML base image (Ubuntu 22.04 with openmpi)
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        conda_file=str(env_dir / "conda.yaml"),
    )
    log(f"Registering environment '{ENV_NAME}' (image build happens on first job submission)...")
    registered = client.environments.create_or_update(env)
    log(f"Environment ready: {registered.name}:{registered.version}")
    return registered


# -----------------------------------------------------------------------------
# 5. Job submission
# -----------------------------------------------------------------------------
def submit_job(client: MLClient, env: Environment) -> str:
    """Submit a command job that clones the repo and runs the focused OOS script.

    The focused script re-runs only the cells that had the OOS-leak bug
    (notebook cells 16 ablation + 17 multi-seed). It does not need GPU,
    FinBERT, or Qwen3 — it only trains the small DHRP layer and backtests
    against the OOS-2020 hold-out window.
    """
    # Note: set -e + -x + -o pipefail (NOT -u, which would crash on unset env vars)
    # Files written to ./outputs in the working directory are auto-uploaded by Azure ML.
    cmd_script = (
        f"set -exo pipefail && "
        f"echo 'CPU info:' && nproc && cat /proc/cpuinfo | grep 'model name' | head -1 && "
        f"echo 'Memory:' && free -h && "
        f"git clone --depth 1 --branch {GIT_REF} {GIT_URL} repo && "
        f"cd repo && "
        f"git rev-parse HEAD && "
        f"mkdir -p outputs && "
        f"python scripts/azure_run_oos_focused.py 2>&1 && "
        f"echo 'Job complete. Output files in ./outputs:' && "
        f"ls -la outputs/"
    )

    job = command(
        display_name=JOB_DISPLAY_NAME,
        description="DHRP focused OOS re-run: ablation + multi-seed (CPU, no LLM)",
        compute=COMPUTE,
        environment=f"{env.name}:{env.version}",
        command=cmd_script,
        environment_variables={
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
        },
        tags={"project": "dhrp", "phase": "oos-2020-focused"},
    )

    log(f"Submitting job '{JOB_DISPLAY_NAME}'...")
    submitted = client.jobs.create_or_update(job)
    log(f"Job submitted: {submitted.name}")
    log(f"Studio URL: {submitted.studio_url}")
    return submitted.name


# -----------------------------------------------------------------------------
# 6. Log streaming
# -----------------------------------------------------------------------------
def stream_logs(client: MLClient, job_name: str) -> str:
    log(f"Streaming logs for job {job_name} (this blocks until completion)...")
    try:
        client.jobs.stream(job_name)
    except KeyboardInterrupt:
        log("Stream interrupted by user. Job continues running on Azure.")
        raise
    final = client.jobs.get(job_name)
    log(f"Job final status: {final.status}")
    return final.status


# -----------------------------------------------------------------------------
# 7. Download outputs
# -----------------------------------------------------------------------------
def download_outputs(client: MLClient, job_name: str) -> None:
    out_dir = LOCAL_REPO / "azure_outputs" / job_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Downloading job outputs to {out_dir}...")
    client.jobs.download(name=job_name, download_path=str(out_dir), output_name="outputs")
    log("Download complete. Contents:")
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out_dir)}  ({p.stat().st_size} bytes)")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-only", help="Skip provisioning; stream logs of an existing job name")
    parser.add_argument("--download-only", help="Skip everything; download outputs of an existing job name")
    parser.add_argument("--no-stream", action="store_true", help="Submit job and exit without streaming")
    args = parser.parse_args()

    credential = get_credential()

    if args.download_only:
        client = get_ml_client(credential)
        download_outputs(client, args.download_only)
        return 0

    if args.watch_only:
        client = get_ml_client(credential)
        status = stream_logs(client, args.watch_only)
        if status == "Completed":
            download_outputs(client, args.watch_only)
        return 0 if status == "Completed" else 1

    log("=" * 70)
    log("DHRP Azure ML Deployment")
    log(f"  subscription : {SUBSCRIPTION}")
    log(f"  resource grp : {RESOURCE_GROUP}")
    log(f"  location     : {LOCATION}")
    log(f"  workspace    : {WORKSPACE}")
    log(f"  compute      : {COMPUTE} ({VM_SIZE})")
    log(f"  git ref      : {GIT_URL} @ {GIT_REF}")
    log("=" * 70)

    ensure_workspace(credential)
    client = get_ml_client(credential)
    ensure_compute(client)
    env = ensure_environment(client)
    job_name = submit_job(client, env)

    log(f"\nJob name: {job_name}")
    log("Save this name. To resume streaming after disconnect:")
    log(f"  python scripts/azure_deploy.py --watch-only {job_name}")
    log("To download outputs after completion:")
    log(f"  python scripts/azure_deploy.py --download-only {job_name}")

    if args.no_stream:
        return 0

    status = stream_logs(client, job_name)
    if status == "Completed":
        download_outputs(client, job_name)
        return 0
    else:
        log(f"Job did not complete cleanly (status={status}). Check Studio URL above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
