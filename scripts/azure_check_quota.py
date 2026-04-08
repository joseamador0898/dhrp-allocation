"""Query Azure ML quota for the dhrp-ws workspace location.

Lists all VM families with their vCPU limit and current usage.
Highlights families with non-zero quota and known GPU SKUs.
"""
from __future__ import annotations
import os
import sys

from azure.ai.ml import MLClient
from azure.identity import (
    DeviceCodeCredential,
    TokenCachePersistenceOptions,
    AuthenticationRecord,
)

SUBSCRIPTION = os.environ.get("DHRP_SUBSCRIPTION", "bf5b14c8-093f-4ab9-a3b5-7352f72650ea")
TENANT_ID = os.environ.get("DHRP_TENANT", "9c483771-7648-4b3a-bafa-7485e8b9d96b")
RESOURCE_GROUP = os.environ.get("DHRP_RESOURCE_GROUP", "luigiboy23_-rg")
WORKSPACE = os.environ.get("DHRP_WORKSPACE", "dhrp-ws")
LOCATION = os.environ.get("DHRP_LOCATION", "eastus2")

AZURE_CLI_CLIENT_ID = "04b07795-8ddb-461a-bbee-02f9e1bf7b46"


def _device_code_prompt(verification_uri, user_code, expires_on):
    msg = (
        "\n================================================================\n"
        ">>> AZURE DEVICE CODE LOGIN <<<\n"
        f"Open this URL in any browser: {verification_uri}\n"
        f"Enter this code:              {user_code}\n"
        f"Expires at:                   {expires_on}\n"
        "================================================================\n"
    )
    print(msg, flush=True)


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


def make_credential():
    cache_opts = TokenCachePersistenceOptions(
        name="dhrp-deploy", allow_unencrypted_storage=True
    )
    record = _load_auth_record()
    if record is not None:
        # Reuse existing auth — silent token refresh
        cred = DeviceCodeCredential(
            tenant_id=TENANT_ID,
            client_id=AZURE_CLI_CLIENT_ID,
            prompt_callback=_device_code_prompt,
            cache_persistence_options=cache_opts,
            authentication_record=record,
        )
        return cred
    # First-time auth: prompt + persist record
    cred = DeviceCodeCredential(
        tenant_id=TENANT_ID,
        client_id=AZURE_CLI_CLIENT_ID,
        prompt_callback=_device_code_prompt,
        cache_persistence_options=cache_opts,
    )
    record = cred.authenticate(scopes=["https://management.azure.com/.default"])
    _save_auth_record(record)
    return cred


def main():
    credential = make_credential()
    client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE,
    )
    print(f"Querying quota in {LOCATION}...\n", flush=True)
    quotas = list(client.compute.list_usage(location=LOCATION))
    rows = []
    for q in quotas:
        # q.name may be a dict {'value': ..., 'localized_value': ...} or an object
        name_obj = getattr(q, "name", None)
        if isinstance(name_obj, dict):
            name = name_obj.get("localized_value") or name_obj.get("value") or "?"
        elif name_obj is not None:
            name = getattr(name_obj, "localized_value", None) or getattr(name_obj, "value", None) or str(name_obj)
        else:
            name = "?"
        limit = getattr(q, "limit", 0) or 0
        usage = getattr(q, "current_value", 0) or 0
        rows.append((str(name), limit, usage))

    rows.sort(key=lambda r: (-r[1], r[0]))
    print(f"{'Family':60s} {'Limit':>8s} {'Used':>8s}")
    print("-" * 80)
    nonzero = []
    for name, limit, usage in rows:
        marker = "  " if limit == 0 else " *"
        print(f"{marker}{name:58s} {limit:>8} {usage:>8}")
        if limit > 0:
            nonzero.append((name, limit, usage))

    print("\n--- NON-ZERO families (usable for compute) ---")
    for name, limit, usage in nonzero:
        gpu = any(g in name.lower() for g in ["gpu", "nc", "nv", "nd", "asr"])
        tag = " [GPU?]" if gpu else ""
        print(f"  {name}: {limit - usage}/{limit} vCPUs available{tag}")


if __name__ == "__main__":
    main()
