# SharePoint Setup — Step-by-Step Manual

A copy-paste walkthrough for wiring an O365 SharePoint document library as Spektr's ingestion source. The syncer mirrors a single in-scope folder to local disk every few minutes; CocoIndex's existing `LocalFile` source picks it up unchanged.

This guide is the SharePoint counterpart to `s3-sqs-setup.md`. The two paths are mutually exclusive in the runtime: if `S3_BUCKET_NAME` and `S3_SQS_QUEUE_URL` are set, CocoIndex uses the AmazonS3 source; otherwise it uses LocalFile, which is where the SharePoint sync deposits its mirror.

---

## How the pieces fit together

```
SharePoint folder --(MS Graph delta)--> sharepoint-sync --(write)--> ./documents/sharepoint/
                                                                            |
                                                            (LocalFile source via CocoIndex)
                                                                            v
                                                                  ingest-live (--live)
                                                                            |
                                                                  Qdrant + Neo4j
```

The syncer polls Microsoft Graph at `sharepoint_sync_interval_seconds` (default 180s), filters delta entries to `sharepoint_root_folder_path`, downloads new/changed files into the mirror dir, and **propagates deletions** by removing the local copy. CocoIndex's `target_connector.py` then purges Qdrant points and Neo4j nodes/edges for the deleted file — full parity with the S3+SQS path.

| # | Thing | Lives on | Purpose |
|-|-|-|-|
| 1 | Azure AD app registration | Entra ID portal | Identity used by the syncer |
| 2 | Client secret | the app | The syncer's password |
| 3 | API permissions (Microsoft Graph) | the app | What the syncer can read |
| 4 | Site permission grant (Sites.Selected only) | the SharePoint site | Limits the app to a single site |
| 5 | Spektr `.env` settings | the host running the syncer | Tells Spektr how to authenticate + what to mirror |

---

## 1. Register an Azure AD app

**Console path** — Microsoft Entra admin center → **App registrations** → **New registration**.

- Name: `spektr-sharepoint-sync`
- Supported account types: **Accounts in this organizational directory only**
- Redirect URI: leave blank (we use client credentials, not browser sign-in)

Click **Register**. Copy two values from the **Overview** page:

- **Application (client) ID** → goes into `SHAREPOINT_CLIENT_ID`
- **Directory (tenant) ID** → goes into `SHAREPOINT_TENANT_ID`

## 2. Create a client secret

App page → **Certificates & secrets** → **+ New client secret**.

- Description: `spektr-sync`
- Expires: choose a value that fits your rotation policy (12 or 24 months recommended)

Click **Add**. Copy the **Value** column **immediately** — Azure shows it only once. Store it in `SHAREPOINT_CLIENT_SECRET`.

## 3. Grant API permissions

App page → **API permissions** → **+ Add a permission** → **Microsoft Graph** → **Application permissions**.

Choose ONE of the two permission models below — they are mutually exclusive in practice:

### 3a. Recommended: `Sites.Selected` (least-privilege)

Add `Sites.Selected`, then click **Grant admin consent for <tenant>**.

The app now has **no implicit access to any site**. You must explicitly grant it access to the one site you want it to read. Run this from a terminal that's logged into the Microsoft Graph PowerShell module or use the Graph Explorer:

```http
POST https://graph.microsoft.com/v1.0/sites/{site-id}/permissions
Content-Type: application/json

{
  "roles": ["read"],
  "grantedToIdentities": [
    {
      "application": {
        "id": "{client-id}",
        "displayName": "spektr-sharepoint-sync"
      }
    }
  ]
}
```

Replace `{site-id}` with the value you'll find in step 4, and `{client-id}` with `SHAREPOINT_CLIENT_ID` from step 1.

### 3b. Fallback: `Sites.Read.All`

If your tenant cannot grant `Sites.Selected` (some compliance setups disallow it), add `Sites.Read.All` instead and click **Grant admin consent**. The app can then read every site in the tenant — broader access than we want, so prefer 3a when possible.

## 4. Find `site_id` and `drive_id`

The Graph API doesn't speak SharePoint URLs natively — you have to translate them once.

### 4a. Get `site_id`

```bash
SITE_HOSTNAME="contoso.sharepoint.com"
SITE_PATH="/sites/engineering"  # the bit after the hostname, including "/sites"

curl -s -H "Authorization: Bearer $TOKEN" \
  "https://graph.microsoft.com/v1.0/sites/${SITE_HOSTNAME}:${SITE_PATH}?\$select=id,name,webUrl"
```

The response includes an `id` like `contoso.sharepoint.com,abc-1234,def-5678`. **Copy the entire string** — that's your `SHAREPOINT_SITE_ID`.

### 4b. Get `drive_id`

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://graph.microsoft.com/v1.0/sites/${SITE_ID}/drives"
```

Pick the drive that corresponds to the document library you care about (the default library is named `Documents` / `Shared Documents`). Copy its `id` into `SHAREPOINT_DRIVE_ID`.

To get a `$TOKEN` for these probe requests:

```bash
curl -s -X POST \
  -d "client_id=${SHAREPOINT_CLIENT_ID}" \
  -d "client_secret=${SHAREPOINT_CLIENT_SECRET}" \
  -d "scope=https://graph.microsoft.com/.default" \
  -d "grant_type=client_credentials" \
  "https://login.microsoftonline.com/${SHAREPOINT_TENANT_ID}/oauth2/v2.0/token" | jq -r .access_token
```

## 5. Choose the in-scope folder

Pick **one** folder inside the document library. Sibling folders at the same level are out of scope and **must not** end up in the mirror.

- Must start with `/`
- Must be a path inside the drive's root (not the library URL)
- Sibling folders are out of scope by design — moving a file out of `root_folder_path` triggers a deletion in the mirror

Examples:

```bash
SHAREPOINT_ROOT_FOLDER_PATH="/Engineering/Specs"
SHAREPOINT_ROOT_FOLDER_PATH="/Customer Docs/Public"
```

## 6. Configure `.env`

Add to `.env` (or `.env.prod`):

| Variable | Required | Example | Notes |
|-|-|-|-|
| `SHAREPOINT_TENANT_ID` | yes | `11111111-2222-3333-4444-555555555555` | Step 1 |
| `SHAREPOINT_CLIENT_ID` | yes | `aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee` | Step 1 |
| `SHAREPOINT_CLIENT_SECRET` | yes | `<secret value>` | Step 2 |
| `SHAREPOINT_SITE_ID` | yes | `contoso.sharepoint.com,abc,def` | Step 4a |
| `SHAREPOINT_DRIVE_ID` | yes | `b!abc-1234...` | Step 4b |
| `SHAREPOINT_ROOT_FOLDER_PATH` | yes | `/Engineering/Specs` | Step 5 |
| `SHAREPOINT_LOCAL_SUBDIR` | no | `sharepoint` (default) | Subdir under `local_documents_path` |
| `SHAREPOINT_SYNC_INTERVAL_SECONDS` | no | `180` (default) | Polling interval |
| `SHAREPOINT_STATE_DIR` | no | `state/sharepoint` (default) | Where the delta token + index live |

If any of the **required** vars are missing or empty, `settings.sharepoint_enabled` is `False` and the syncer refuses to start.

## 7. First-run smoke test

```bash
task sharepoint-sync-once
```

Expected:

- The process exits cleanly with code 0.
- `state/sharepoint/delta.json` exists and contains a `"delta_token"`.
- `state/sharepoint/index.sqlite` is created.
- `documents/sharepoint/<your folder layout>` contains the in-scope files.
- A sibling folder added at the same level as your in-scope folder does **not** appear under `documents/sharepoint/`.

If you see auth errors, double-check steps 1–4. The most common mistake is forgetting to **Grant admin consent** in step 3.

## 8. Wire into the live ingestion

Run two long-running processes:

```bash
task sharepoint-sync   # in one terminal, mirrors SharePoint -> ./documents/sharepoint/
task ingest-live       # in another terminal, watches ./documents/sharepoint/ via inotify
```

Drop a PDF into the SharePoint folder. Within `SHAREPOINT_SYNC_INTERVAL_SECONDS`, expect:

- The file appears under `documents/sharepoint/`.
- CocoIndex picks it up — a row appears in the Postgres tracking table.
- Qdrant gets vector points (verify with `task smoke "<query from doc>"`).
- Neo4j gets entities (verify with `task smoke-graph "<entity>"`).

In production, both processes run as the `sharepoint-sync` and `ingest-live` services in `docker-compose.prod.yml`; they share the `sharepoint_documents` named volume.

---

## Operational notes

### Tokens and refresh
The syncer uses `azure.identity.aio.ClientSecretCredential`, which transparently caches and refreshes Graph tokens (TTL ≈ 1h). You don't need to do anything beyond setting the secret.

### Throttling
Microsoft Graph throttles per-app and per-tenant. The syncer uses `httpx` with `raise_for_status`; transient `429` / `503` failures bubble up, get logged, and are retried on the next interval — Spektr's failure tracker does **not** count them against the per-file retry budget because they're at the syncer level, not the pipeline level.

### Recovering from a stuck delta token
If the syncer logs persistent `410 Gone` errors on the delta endpoint, the token has expired (this can happen after very long downtime or backend changes). Recover by:

```bash
rm state/sharepoint/delta.json
task sharepoint-sync-once
```

This forces a full re-listing. The local index keeps `(item_id → local_path, etag)` so the second pass downloads only files whose etag has actually changed.

### Resetting state
To start completely fresh:

```bash
rm -rf state/sharepoint documents/sharepoint
task sharepoint-sync-once
```

The next ingestion cycle will purge orphan vectors via `target_connector.py` (CocoIndex's deletion path) — verify with `task doctor` once the sync completes.

### Deletion semantics (parity with S3+SQS)

| In SharePoint you do… | In Qdrant + Neo4j you get… |
|-|-|
| Delete a file | The file's vectors and graph entries are purged within one sync interval. |
| Move a file out of the in-scope folder | Same as a delete. |
| Rename a file inside the in-scope folder | Old vectors purged; new vectors ingested under the new key. |
| Delete a subfolder of the in-scope folder | All descendant files purged. |
| Restore a file from the recycle bin | The file re-appears in the next delta and is re-ingested. |
