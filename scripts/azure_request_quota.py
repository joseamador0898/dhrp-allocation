"""File quota increase requests for ALL modern GPU families on this subscription.

Uses the Azure Quota REST API (Microsoft.Quota provider) which the
azure-ai-ml SDK does not expose. Iterates over the major GPU family
names and asks for the smallest viable vCPU count of each so we can
provision a 1-GPU VM:

  - T4    : standardNCASv3_T4Family   ->  4 vCPU (Standard_NC4as_T4_v3)
  - V100  : standardNCSv3Family       ->  6 vCPU (Standard_NC6s_v3)
  - A10   : StandardNVADSA10v5Family  ->  6 vCPU (Standard_NV6ads_A10_v5)
  - A100  : StandardNCADSA100v4Family -> 24 vCPU (Standard_NC24ads_A100_v4)
  - H100  : StandardNCadsH100v5Family -> 40 vCPU (Standard_NC40ads_H100_v5)

If the quota auto-approves (typical for low quota on active MSDN subs),
the new limit is reflected immediately. If a support ticket is required,
the response status indicates that and we move on to the next family.
"""
import json
import os
import sys
import time

import requests

from azure.identity import (
    DeviceCodeCredential,
    TokenCachePersistenceOptions,
    AuthenticationRecord,
)

SUBSCRIPTION = "bf5b14c8-093f-4ab9-a3b5-7352f72650ea"
TENANT_ID = "9c483771-7648-4b3a-bafa-7485e8b9d96b"
LOCATION = "eastus2"
AZURE_CLI_CLIENT_ID = "04b07795-8ddb-461a-bbee-02f9e1bf7b46"
CACHE_FILE = os.path.expanduser("~/.dhrp_azure_auth_record.json")

# All modern GPU families to request quota for. Format: (family_name, target_vcpu, sku, gpu)
QUOTA_REQUESTS = [
    # T4: underscore in usage name is rejected by Microsoft.Quota REST API as
    # InvalidResourceName; the resource ID form drops the underscore.
    ("standardNCASv3T4Family",    4,  "Standard_NC4as_T4_v3",     "1x T4 16GB"),
    ("standardNCSv3Family",       6,  "Standard_NC6s_v3",         "1x V100 16GB"),
    ("StandardNVADSA10v5Family",  6,  "Standard_NV6ads_A10_v5",   "1x A10 24GB"),
    ("StandardNCADSA100v4Family", 24, "Standard_NC24ads_A100_v4", "1x A100 80GB"),
    ("StandardNCadsH100v5Family", 40, "Standard_NC40ads_H100_v5", "1x H100 80GB"),
]


def make_credential():
    cache_opts = TokenCachePersistenceOptions(
        name="dhrp-deploy", allow_unencrypted_storage=True
    )
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            record = AuthenticationRecord.deserialize(f.read())
        return DeviceCodeCredential(
            tenant_id=TENANT_ID,
            client_id=AZURE_CLI_CLIENT_ID,
            cache_persistence_options=cache_opts,
            authentication_record=record,
        )
    cred = DeviceCodeCredential(
        tenant_id=TENANT_ID,
        client_id=AZURE_CLI_CLIENT_ID,
        cache_persistence_options=cache_opts,
    )
    record = cred.authenticate(scopes=["https://management.azure.com/.default"])
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        f.write(record.serialize())
    return cred


def get_token(cred):
    token = cred.get_token("https://management.azure.com/.default")
    return token.token


def list_all_usages(token):
    """List Microsoft.Compute usages for the location keyed by lower-case family name."""
    url = (
        f"https://management.azure.com/subscriptions/{SUBSCRIPTION}"
        f"/providers/Microsoft.Compute/locations/{LOCATION}/usages"
        f"?api-version=2024-07-01"
    )
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    r.raise_for_status()
    out = {}
    for u in r.json().get("value", []):
        name_value = (u.get("name") or {}).get("value", "")
        out[name_value.lower()] = u
    return out


def request_quota(token, family: str, target_vcpu: int):
    """File a PUT request to bump the family's vCPU limit via Microsoft.Quota."""
    scope = f"subscriptions/{SUBSCRIPTION}/providers/Microsoft.Compute/locations/{LOCATION}"
    url = (
        f"https://management.azure.com/{scope}"
        f"/providers/Microsoft.Quota/quotas/{family}?api-version=2023-02-01"
    )
    body = {
        "properties": {
            "limit": {"value": target_vcpu, "limitObjectType": "LimitValue"},
            "name": {"value": family},
            "resourceType": "dedicated",
            "unit": "Count",
        }
    }
    return requests.put(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )


def main():
    cred = make_credential()
    token = get_token(cred)

    print(f"Listing current quotas in {LOCATION}...", flush=True)
    usages = list_all_usages(token)
    print(f"  Found {len(usages)} usage entries\n", flush=True)

    summary = []
    for family, target_vcpu, sku, gpu_desc in QUOTA_REQUESTS:
        print(f"=== {family} (target {target_vcpu} vCPU = {sku}, {gpu_desc}) ===", flush=True)
        current = usages.get(family.lower())
        if current is not None:
            cur_limit = current.get("limit", 0)
            cur_used = current.get("currentValue", 0)
            print(f"  Current: limit={cur_limit}, used={cur_used}", flush=True)
            if cur_limit >= target_vcpu:
                print(f"  ALREADY SATISFIED (limit >= {target_vcpu}); skipping request", flush=True)
                summary.append((family, "satisfied", cur_limit, target_vcpu, sku))
                print()
                continue
        else:
            print(f"  (family not found in usages list; submitting request anyway)", flush=True)

        print(f"  Filing PUT to bump {family} -> {target_vcpu} vCPU...", flush=True)
        r = request_quota(token, family, target_vcpu)
        print(f"  Response status: {r.status_code}", flush=True)
        try:
            body_json = r.json()
            print(f"  Response body: {json.dumps(body_json, indent=2)[:600]}", flush=True)
        except Exception:
            body_json = {}
            print(f"  Response body (raw): {r.text[:400]}", flush=True)

        if 200 <= r.status_code < 300:
            provisioning = (
                body_json.get("properties", {}).get("provisioningState")
                or body_json.get("status")
                or "?"
            )
            if provisioning == "Succeeded":
                print(f"  AUTO-APPROVED ({provisioning})", flush=True)
                summary.append((family, "auto-approved", "?", target_vcpu, sku))
            else:
                print(f"  ACCEPTED, provisioning state = {provisioning}", flush=True)
                summary.append((family, f"accepted ({provisioning})", "?", target_vcpu, sku))
        elif r.status_code == 409:
            print(f"  ALREADY PENDING (409 conflict)", flush=True)
            summary.append((family, "already-pending", "?", target_vcpu, sku))
        elif r.status_code == 403:
            print(f"  FORBIDDEN: support ticket required", flush=True)
            summary.append((family, "needs-ticket (403)", "?", target_vcpu, sku))
        else:
            print(f"  FAILED status {r.status_code}", flush=True)
            summary.append((family, f"failed-{r.status_code}", "?", target_vcpu, sku))
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Family':40s} {'Status':25s} {'Target':>8s}  {'SKU'}")
    print("-" * 100)
    for family, status, _, target, sku in summary:
        print(f"{family:40s} {status:25s} {target:>8}  {sku}")

    auto = [s for s in summary if s[1] in ("auto-approved", "satisfied")]
    if auto:
        print(f"\n{len(auto)} families ready for use:")
        for family, _, _, target, sku in auto:
            print(f"  {sku} (family {family}, target {target} vCPU)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
