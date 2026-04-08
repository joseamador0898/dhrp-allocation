"""List ALL valid Azure ML VM sizes in eastus2 with their GPU details.

Filters and prints GPU SKUs with details so we can pick one that:
1. Actually exists in this region
2. Fits within our 6-vCPU NC/NV quota
3. Has the most powerful GPU available
"""
import os
from azure.ai.ml import MLClient
from azure.identity import (
    DeviceCodeCredential,
    TokenCachePersistenceOptions,
    AuthenticationRecord,
)

SUBSCRIPTION = "bf5b14c8-093f-4ab9-a3b5-7352f72650ea"
TENANT_ID = "9c483771-7648-4b3a-bafa-7485e8b9d96b"
RESOURCE_GROUP = "luigiboy23_-rg"
WORKSPACE = "dhrp-ws"
LOCATION = "eastus2"
AZURE_CLI_CLIENT_ID = "04b07795-8ddb-461a-bbee-02f9e1bf7b46"
CACHE_FILE = os.path.expanduser("~/.dhrp_azure_auth_record.json")


def make_credential():
    cache_opts = TokenCachePersistenceOptions(name="dhrp-deploy", allow_unencrypted_storage=True)
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            record = AuthenticationRecord.deserialize(f.read())
        return DeviceCodeCredential(
            tenant_id=TENANT_ID, client_id=AZURE_CLI_CLIENT_ID,
            cache_persistence_options=cache_opts, authentication_record=record,
        )
    cred = DeviceCodeCredential(
        tenant_id=TENANT_ID, client_id=AZURE_CLI_CLIENT_ID,
        cache_persistence_options=cache_opts,
    )
    record = cred.authenticate(scopes=["https://management.azure.com/.default"])
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        f.write(record.serialize())
    return cred


def main():
    cred = make_credential()
    client = MLClient(
        credential=cred,
        subscription_id=SUBSCRIPTION,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE,
    )

    print(f"Listing VM sizes available in {LOCATION}...\n", flush=True)
    sizes = list(client.compute.list_sizes(location=LOCATION))
    print(f"Total VM sizes returned: {len(sizes)}\n", flush=True)

    rows = []
    for s in sizes:
        name = getattr(s, "name", "?")
        family = getattr(s, "family", "?")
        vcpus = getattr(s, "v_cp_us", 0)
        ram_gb = getattr(s, "memory_gb", 0)
        gpus = getattr(s, "gpus", 0) or 0
        os_disk_gb = getattr(s, "os_vhd_size_mb", 0) // 1024
        rows.append((name, family, vcpus, ram_gb, gpus, os_disk_gb))

    # Filter to GPU-capable, sort by gpus desc then vcpus asc
    gpu_rows = [r for r in rows if r[4] > 0]
    gpu_rows.sort(key=lambda r: (-r[4], r[2]))

    print(f"=== GPU VM SIZES IN {LOCATION} ({len(gpu_rows)} total) ===")
    print(f"{'Name':38s} {'Family':30s} {'vCPU':>5s} {'RAM':>7s} {'GPUs':>5s}")
    print("-" * 95)
    for name, family, vcpus, ram, gpus, _ in gpu_rows:
        marker = " *" if vcpus <= 6 else "  "
        print(f"{marker}{name:36s} {family:30s} {vcpus:>5} {ram:>5}GB {gpus:>5}")

    print("\n  * = fits within 6-vCPU NC/NV quota\n")

    # Also show small CPU sizes within ESv3 family for fallback
    cpu_esv3 = [r for r in rows if r[4] == 0 and "esv3" in r[1].lower() and r[2] <= 16]
    cpu_esv3.sort(key=lambda r: r[2])
    print(f"=== ESv3 CPU FALLBACK ({len(cpu_esv3)} sizes) ===")
    for name, family, vcpus, ram, _, _ in cpu_esv3:
        print(f"  {name:36s} {family:30s} {vcpus:>5} {ram:>5}GB")


if __name__ == "__main__":
    main()
