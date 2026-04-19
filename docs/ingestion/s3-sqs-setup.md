# S3 + SQS Setup

Walkthrough for wiring an AWS S3 bucket as the ingestion source with real-time SQS change events. Everything here is learned the hard way — none of it is obvious from the CocoIndex docs.

## The three things that must be true

1. **The SQS queue has an access policy** that lets `s3.amazonaws.com` publish to it.
2. **The S3 bucket has an event notification** pointing at that queue.
3. **The Spektr process has AWS creds and a region** visible to `os.environ`.

Missing any one of them, you'll see symptoms like "queue has 0 messages despite uploads", "A region must be set when sending requests to S3", or "AccessDenied on sqs:SendMessage". Each step below covers exactly one of these.

## 1. SQS queue + access policy

Create a standard SQS queue (e.g. `ragflow-ingest`) in the same region as your bucket. Then attach this access policy (SQS console → queue → **Access policy** → Edit). Replace `REGION`, `ACCOUNT_ID`, `QUEUE_NAME`, `BUCKET` with your values:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "owner",
      "Effect": "Allow",
      "Principal": {"AWS": "arn:aws:iam::ACCOUNT_ID:root"},
      "Action": "SQS:*",
      "Resource": "arn:aws:sqs:REGION:ACCOUNT_ID:QUEUE_NAME"
    },
    {
      "Sid": "s3-publish",
      "Effect": "Allow",
      "Principal": {"Service": "s3.amazonaws.com"},
      "Action": "sqs:SendMessage",
      "Resource": "arn:aws:sqs:REGION:ACCOUNT_ID:QUEUE_NAME",
      "Condition": {
        "ArnEquals": {"aws:SourceArn": "arn:aws:s3:::BUCKET"}
      }
    }
  ]
}
```

The `ArnEquals` condition scopes publishing rights to exactly one bucket — essential if multiple buckets share the account.

## 2. S3 bucket event notification

**This is a separate setting from the SQS policy.** The policy says "S3 *may* send"; the notification says "S3 *does* send". Both are required. Many people (including Claude) assume step 1 is enough — it isn't.

S3 console → your bucket → **Properties** tab → scroll to **Event notifications** → **Create event notification**:

- **Event name:** `spektr-ingest` (or whatever)
- **Event types:** tick **All object create events** (`s3:ObjectCreated:*`) and **All object removal events** (`s3:ObjectRemoved:*`)
- **Prefix / Suffix:** leave empty to catch the whole bucket
- **Destination:** SQS queue → select your queue from the dropdown

S3 will validate the queue policy on save. If the policy is missing or wrong, S3 refuses to attach the notification.

!!! warning "Existing objects don't retro-trigger"
    S3 only publishes events for PUT/DELETE operations *after* the notification is configured. Objects uploaded earlier will never arrive via SQS. Either re-upload them, or run a one-shot `task ingest` to let CocoIndex's initial scan pick them up.

## 3. Spektr configuration

In `.env`:

```bash
DOCUMENT_SOURCE=s3
S3_BUCKET_NAME=your-bucket
S3_SQS_QUEUE_URL=https://sqs.REGION.amazonaws.com/ACCOUNT_ID/QUEUE_NAME
AWS_REGION=eu-north-1
AWS_ACCESS_KEY_ID=AKIA…
AWS_SECRET_ACCESS_KEY=…
AWS_ENDPOINT_URL=
```

!!! danger "`.env` comment parsing"
    Pydantic-settings / python-dotenv treats **everything after `=`** as the value when the line has only whitespace before `#`. `AWS_ENDPOINT_URL=   # leave empty for real AWS` ends up as the literal string `"# leave empty for real AWS"`, which boto3 rejects with `Invalid endpoint`. Set the value to an empty string with nothing after it: `AWS_ENDPOINT_URL=`.

The pipeline's `run_pipeline` function also exports these into `os.environ` before `cocoindex.init()`, so CocoIndex's Rust S3 SDK (which doesn't see Pydantic Settings) has a region available. You don't have to do this by hand.

## 4. IAM permissions for the Spektr IAM user

The user whose credentials you put in `.env` needs at minimum:

| Action | Resource | Why |
|-|-|-|
| `s3:GetObject`, `s3:ListBucket` | `arn:aws:s3:::BUCKET`, `arn:aws:s3:::BUCKET/*` | read files |
| `sqs:ReceiveMessage`, `sqs:DeleteMessage`, `sqs:GetQueueAttributes` | queue ARN | consume change events |

**Optional** (only if you want Spektr to self-configure the bucket notification via boto3 instead of the console): `s3:PutBucketNotification`, `s3:GetBucketNotification`.

## 5. Smoke test

```bash
task up                       # qdrant + neo4j + postgres
task ingest-live              # pipeline watches SQS; Ctrl-C to stop
# in another shell: upload a PDF to the bucket via console or aws cli
```

Watch the pipeline output. Within a few seconds you should see:

```
Processing file: your.pdf
Using Docling HybridChunker: N chunks for your.pdf
Token usage for your.pdf: … estimated tokens
Finished file: your.pdf in Xms
RagIngestion.files (change stream): 1/1 source rows: 1 added
```

Delete the object — same flow but with `1 deleted` at the end, and a `Deleted Qdrant points for your.pdf` log from `ingestion.target_connector`.

## 6. When things go wrong

| Symptom | Root cause | Fix |
|-|-|-|
| `SQS visible: 0` after uploads | Bucket notification missing | Step 2 above |
| `AccessDenied sqs:SendMessage` | SQS policy missing | Step 1 |
| `Invalid endpoint: # leave empty…` | `.env` comment bug | Step 3 callout |
| `A region must be set when sending requests to S3` | Rust SDK doesn't see `.env` | `run_pipeline` already handles this; confirm you ran `task ingest` / `task ingest-live` not a raw `python` call |
| `1 source rows: 1 no change` despite new file | CocoIndex has stale tracking; no SQS event fired | `task doctor` → `task doctor-fix` → retry |
| Pipeline killed with exit 137 | OOM on a huge picture-heavy PDF | Temporarily `GRAPH_ENABLED=false IMAGE_EMBED_STRATEGY=none task ingest`; it's idempotent |

See [CocoIndex Pipeline](cocoindex.md) for the flow internals, [Ingestion Failure Semantics](../operations/atomicity.md) for the re-raise + poison-pill contract.
