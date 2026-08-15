# S3 + SQS Setup — Step-by-Step Manual

A copy-paste walkthrough for wiring an AWS S3 bucket as the ingestion source with real-time SQS change events. Everything here is learned the hard way — none of it is obvious from the CocoIndex docs.

This guide gives you **two parallel paths** for every step:

- **CLI path** — `aws` commands you can paste into a terminal
- **Console path** — where to click in the AWS web console

Pick whichever you're comfortable with. The CLI path is faster and reproducible; the console is friendlier when you're learning.

---

## How the pieces fit together

```
S3 bucket --(event notification)--> SQS queue <--(long-poll)-- Spektr ingestion
     ^                                 |                           |
     |                            (DLQ on N retries)               |
     +--------------(catch-up scan: list changed objects)----------+
```

!!! note "SQS is a trigger, not a transport"
    CocoIndex v1's `amazon_s3` connector is scan-only — it has no built-in SQS support. Spektr's `ingestion/sqs_trigger.py` long-polls the queue and, on an event, debounces and then runs one ordinary catch-up scan. The file bytes always come from S3, never from the message body, and only objects that actually changed are downloaded. See [CocoIndex Pipeline](cocoindex.md#s3-sqs-as-a-trigger).

    The queue is therefore **optional**: without `S3_SQS_QUEUE_URL`, live mode falls back to sweeping every `S3_FULL_SCAN_INTERVAL_HOURS` (default 24). Everything below is about getting change latency down to seconds.

You wire up **four independent things**, each with its own policy in a different AWS console. People get confused because nothing tells you up front that all four are required.

| # | Thing | Policy lives on | Purpose |
|-|-|-|-|
| 1 | SQS queue (+ DLQ) | itself | Receive S3 events, hold them until Spektr consumes |
| 2 | SQS access policy | the queue | Lets `s3.amazonaws.com` publish to the queue |
| 3 | S3 event notification | the bucket | Tells S3 to actually emit events |
| 4 | IAM user/role | itself | Lets Spektr read S3 + consume SQS |

Missing any one and you'll see "queue has 0 messages despite uploads", "AccessDenied on sqs:SendMessage", or "A region must be set". Each step below covers exactly one of these.

---

## Naming convention

Use `{project}-{env}-{purpose}`. Lowercase-kebab for S3/SQS/IAM users (matches AWS norms), PascalCase for IAM policies/roles.

| Resource | Example name | Notes |
|-|-|-|
| S3 bucket | `spektr-prod-documents` | Globally unique. Add `-eu` if multi-region. |
| S3 bucket (staging) | `spektr-staging-documents` | One bucket per env. |
| SQS queue | `spektr-prod-ingest-events` | "events" makes intent obvious |
| SQS DLQ | `spektr-prod-ingest-events-dlq` | DLQ is always `<source>-dlq` |
| IAM user | `spektr-prod-ingest` | One user per env |
| IAM policy | `SpektrProdIngestPolicy` | PascalCase — AWS convention for IAM |
| Event notification | `spektr-ingest-objectchange` | Lives inside the bucket; name is local |

---

## Step 0 — Set shell variables (used by every CLI block below)

Pick a region. **Bucket and queue must be in the same region.** Don't mix.

```bash
export AWS_REGION=eu-north-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export PROJECT=spektr
export ENV=prod

export BUCKET=${PROJECT}-${ENV}-documents
export QUEUE=${PROJECT}-${ENV}-ingest-events
export DLQ=${QUEUE}-dlq
export IAM_USER=${PROJECT}-${ENV}-ingest
export IAM_POLICY=SpektrProdIngestPolicy
```

Sanity check:

```bash
echo "Region:  $AWS_REGION"
echo "Account: $ACCOUNT_ID"
echo "Bucket:  $BUCKET"
echo "Queue:   $QUEUE"
echo "DLQ:     $DLQ"
```

---

## Step 1 — Create the SQS queues (DLQ first, then source)

The DLQ must exist before the source queue can reference it.

### CLI

```bash
# 1a. Create DLQ
aws sqs create-queue \
  --queue-name "$DLQ" \
  --region "$AWS_REGION" \
  --attributes MessageRetentionPeriod=1209600
# 1209600 = 14 days

# Capture DLQ ARN for the redrive policy
export DLQ_ARN=$(aws sqs get-queue-attributes \
  --queue-url "https://sqs.${AWS_REGION}.amazonaws.com/${ACCOUNT_ID}/${DLQ}" \
  --attribute-names QueueArn \
  --query 'Attributes.QueueArn' --output text \
  --region "$AWS_REGION")
echo "DLQ ARN: $DLQ_ARN"

# 1b. Create source queue with redrive policy + sane defaults
aws sqs create-queue \
  --queue-name "$QUEUE" \
  --region "$AWS_REGION" \
  --attributes "{
    \"VisibilityTimeout\": \"300\",
    \"MessageRetentionPeriod\": \"345600\",
    \"RedrivePolicy\": \"{\\\"deadLetterTargetArn\\\":\\\"${DLQ_ARN}\\\",\\\"maxReceiveCount\\\":\\\"5\\\"}\"
  }"

export QUEUE_URL="https://sqs.${AWS_REGION}.amazonaws.com/${ACCOUNT_ID}/${QUEUE}"
export QUEUE_ARN="arn:aws:sqs:${AWS_REGION}:${ACCOUNT_ID}:${QUEUE}"
echo "Queue URL: $QUEUE_URL"
echo "Queue ARN: $QUEUE_ARN"
```

**Why these numbers:**

- `VisibilityTimeout=300` (5 min) — longer than the slowest single-file ingest. Too short and Spektr re-processes a file mid-flight.
- `MessageRetentionPeriod=345600` (4 days) — gives you a long weekend to fix outages without losing events.
- `maxReceiveCount=5` — five retries before a poison message goes to the DLQ.
- DLQ retention `14 days` — enough to investigate during the work week.

### Console

SQS console → **Create queue**:

- Type: **Standard**
- Name: `spektr-prod-ingest-events-dlq`
- Message retention: `14 days`
- Everything else: defaults
- **Create queue**

Now create the source queue:

- Type: **Standard**
- Name: `spektr-prod-ingest-events`
- Visibility timeout: `5 minutes`
- Message retention: `4 days`
- Scroll to **Dead-letter queue** → Enable → pick the DLQ → Maximum receives: `5`
- **Create queue**

---

## Step 2 — Attach the SQS access policy (let S3 publish)

This is the policy on the **source queue** (not the DLQ). It says "S3 *may* send to me, but only on behalf of one specific bucket".

### CLI

```bash
cat > /tmp/sqs-access-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "owner",
      "Effect": "Allow",
      "Principal": {"AWS": "arn:aws:iam::${ACCOUNT_ID}:root"},
      "Action": "SQS:*",
      "Resource": "${QUEUE_ARN}"
    },
    {
      "Sid": "s3-publish",
      "Effect": "Allow",
      "Principal": {"Service": "s3.amazonaws.com"},
      "Action": "sqs:SendMessage",
      "Resource": "${QUEUE_ARN}",
      "Condition": {
        "ArnEquals": {"aws:SourceArn": "arn:aws:s3:::${BUCKET}"}
      }
    }
  ]
}
EOF

# SetQueueAttributes wants the policy as a JSON-encoded string inside JSON.
aws sqs set-queue-attributes \
  --queue-url "$QUEUE_URL" \
  --region "$AWS_REGION" \
  --attributes "Policy=$(cat /tmp/sqs-access-policy.json | jq -c . | jq -Rs .)"
```

The `ArnEquals` condition scopes publishing rights to exactly one bucket — essential if multiple buckets share the account.

### Console

SQS console → source queue → **Access policy** tab → **Edit** → paste the JSON above (with `${...}` already substituted). Save.

---

## Step 3 — Create the S3 bucket

### CLI

```bash
# us-east-1 is special and must NOT receive --create-bucket-configuration
if [ "$AWS_REGION" = "us-east-1" ]; then
  aws s3api create-bucket --bucket "$BUCKET" --region "$AWS_REGION"
else
  aws s3api create-bucket \
    --bucket "$BUCKET" --region "$AWS_REGION" \
    --create-bucket-configuration LocationConstraint="$AWS_REGION"
fi

# Recommended hardening
aws s3api put-public-access-block \
  --bucket "$BUCKET" \
  --public-access-block-configuration '{
    "BlockPublicAcls": true, "IgnorePublicAcls": true,
    "BlockPublicPolicy": true, "RestrictPublicBuckets": true
  }'

aws s3api put-bucket-versioning \
  --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled

aws s3api put-bucket-encryption \
  --bucket "$BUCKET" \
  --server-side-encryption-configuration '{
    "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
  }'
```

### Console

S3 console → **Create bucket**:

- Name: `spektr-prod-documents`
- Region: same as your queue
- Block all public access: **on** (default)
- Bucket versioning: **Enable**
- Default encryption: **SSE-S3 (AES-256)**
- **Create bucket**

---

## Step 4 — Add the S3 event notification

**This is a separate setting from the SQS policy.** The policy from Step 2 says "S3 *may* send"; this notification says "S3 *does* send". Both are required. Many people (including Claude) assume Step 2 is enough — it isn't.

### CLI

```bash
cat > /tmp/s3-notification.json <<EOF
{
  "QueueConfigurations": [
    {
      "Id": "spektr-ingest-objectchange",
      "QueueArn": "${QUEUE_ARN}",
      "Events": [
        "s3:ObjectCreated:*",
        "s3:ObjectRemoved:*"
      ]
    }
  ]
}
EOF

aws s3api put-bucket-notification-configuration \
  --bucket "$BUCKET" \
  --notification-configuration file:///tmp/s3-notification.json
```

S3 validates the queue policy on save. If Step 2 is missing or wrong, this command fails with `Unable to validate the following destination configurations`. That's the diagnostic — fix the SQS policy and retry.

### Console

S3 console → bucket → **Properties** tab → scroll to **Event notifications** → **Create event notification**:

- Event name: `spektr-ingest-objectchange`
- Event types: tick **All object create events** (`s3:ObjectCreated:*`) and **All object removal events** (`s3:ObjectRemoved:*`)
- Prefix / Suffix: leave empty to catch the whole bucket
- Destination: **SQS queue** → select your queue from the dropdown
- **Save**

!!! warning "Existing objects don't retro-trigger"
    S3 only publishes events for PUT/DELETE operations *after* the notification is configured. Objects uploaded earlier will never arrive via SQS — but they don't have to: the daemon's startup sweep and its periodic interval sweep both list the bucket, so pre-existing objects are picked up anyway. A one-shot `task ingest` does the same thing immediately.

!!! note "Suffix filters"
    S3 filter rules use OR logic for suffix filters within a single configuration, and you can only have one filter set per notification. Don't try to be clever here — Spektr's `included_patterns` setting handles file-type matching downstream. Leave the S3 filter empty unless you want S3-side cost savings.

---

## Step 5 — Create the IAM user for Spektr

Spektr needs read on S3 + consume on SQS. Nothing else.

### CLI

```bash
# 5a. Create the policy
cat > /tmp/spektr-ingest-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadBucket",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::${BUCKET}",
        "arn:aws:s3:::${BUCKET}/*"
      ]
    },
    {
      "Sid": "ConsumeQueue",
      "Effect": "Allow",
      "Action": [
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:GetQueueAttributes",
        "sqs:ChangeMessageVisibility"
      ],
      "Resource": "${QUEUE_ARN}"
    }
  ]
}
EOF

aws iam create-policy \
  --policy-name "$IAM_POLICY" \
  --policy-document file:///tmp/spektr-ingest-policy.json

export POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${IAM_POLICY}"

# 5b. Create the user and attach the policy
aws iam create-user --user-name "$IAM_USER"
aws iam attach-user-policy --user-name "$IAM_USER" --policy-arn "$POLICY_ARN"

# 5c. Generate an access key (this prints the secret ONCE — store it now)
aws iam create-access-key --user-name "$IAM_USER"
```

### Console

IAM console → **Policies** → **Create policy** → JSON tab → paste the policy above (with `${...}` substituted) → name it `SpektrProdIngestPolicy` → Create.

IAM console → **Users** → **Create user** → name `spektr-prod-ingest` → don't enable console access → next → **Attach policies directly** → tick `SpektrProdIngestPolicy` → Create.

User → **Security credentials** tab → **Create access key** → "Application running outside AWS" → copy both the key and secret immediately (you won't see the secret again).

!!! tip "EC2/ECS/EKS hosting"
    If Spektr runs on AWS, prefer an **IAM role** over an IAM user — same policy, attached to the instance/task/pod. No long-lived secrets in `.env`.

---

## Step 6 — Configure Spektr's `.env`

```bash
DOCUMENT_SOURCE=s3
S3_BUCKET_NAME=spektr-prod-documents
S3_PREFIX=                    # optional: restrict the scan to a key prefix
S3_SQS_QUEUE_URL=https://sqs.eu-north-1.amazonaws.com/111122223333/spektr-prod-ingest-events
S3_SQS_DEBOUNCE_SECONDS=5     # coalesce an event burst into a single scan
S3_FULL_SCAN_INTERVAL_HOURS=24  # safety-net sweep for missed/expired events
AWS_REGION=eu-north-1
AWS_ACCESS_KEY_ID=AKIA…
AWS_SECRET_ACCESS_KEY=…
AWS_ENDPOINT_URL=
```

!!! danger "`.env` comment parsing"
    Pydantic-settings / python-dotenv treats **everything after `=`** as the value when the line has only whitespace before `#`. `AWS_ENDPOINT_URL=   # leave empty for real AWS` ends up as the literal string `"# leave empty for real AWS"`, which boto3 rejects with `Invalid endpoint`. Set the value to an empty string with nothing after it: `AWS_ENDPOINT_URL=`.

These are read from `Settings` and passed explicitly to the `aiobotocore` S3 and SQS clients — each falling back to the default boto3 credential chain when left empty. Nothing is exported into `os.environ`.

---

## Step 7 — Smoke test

```bash
task up                       # qdrant + neo4j
task ingest-live              # daemon: startup sweep, then long-polls SQS; Ctrl-C to stop
```

The daemon logs `SQS trigger active on <queue-url> (debounce 5.0s, sweep every 24.0h)` once it's watching.

In another shell, upload a PDF:

```bash
aws s3 cp tests/fixtures/sample.pdf s3://spektr-prod-documents/sample.pdf
```

Within a few seconds you should see:

```
SQS trigger: 1 event(s) coalesced
Processing file: sample.pdf
Using Docling HybridChunker: N chunks for sample.pdf
Finished file: sample.pdf in Xms
Update finished: 1 added, 0 reprocessed, 0 unchanged, 0 deleted, 0 errors
```

Delete the object — same flow but with `1 deleted` at the end, plus a `Source file removed, cleaning up graph data for sample.pdf` log from `ingestion.graph_target`. The Qdrant points are deleted by CocoIndex itself, by point id.

Messages are only removed from the queue after an update that reported **zero** errors, so a failed file replays its event rather than dropping it.

You can also peek at the queue directly:

```bash
aws sqs get-queue-attributes \
  --queue-url "$QUEUE_URL" \
  --attribute-names ApproximateNumberOfMessages \
  --region "$AWS_REGION"
```

---

## When things go wrong

| Symptom | Root cause | Fix |
|-|-|-|
| `SQS visible: 0` after uploads | Bucket notification missing | Step 4 |
| `AccessDenied sqs:SendMessage` | SQS policy missing/wrong | Step 2 |
| `put-bucket-notification-configuration` returns `Unable to validate destination` | Same as above — S3 tested the SQS policy and it failed | Step 2 |
| `Invalid endpoint: # leave empty…` | `.env` comment bug | Step 6 callout |
| `NoRegionError` / credential errors | `AWS_REGION` or the key pair is empty **and** no ambient credential chain is available | Set `AWS_REGION` (+ keys, or use an instance role) in `.env` — they're passed straight to the client |
| Nothing happens for hours after an upload | `S3_SQS_QUEUE_URL` is empty, so the only trigger is the interval sweep | Set the queue URL, or lower `S3_FULL_SCAN_INTERVAL_HOURS` |
| `0 added, N unchanged` despite a new file | The scan ran but the object is memoized as unchanged | `task doctor`; force with `task ingest -- --full-reprocess` |
| Pipeline killed with exit 137 | OOM on a huge picture-heavy PDF | Temporarily `GRAPH_ENABLED=false IMAGE_EMBED_STRATEGY=none task ingest`; it's idempotent |
| Messages piling up in DLQ | A specific file keeps poisoning the pipeline | Inspect a DLQ message body to find the S3 key, then look at logs for that key |

---

## Tearing it down

If you want to delete the whole stack later:

```bash
aws s3 rm "s3://${BUCKET}" --recursive
aws s3api delete-bucket --bucket "$BUCKET"
aws sqs delete-queue --queue-url "$QUEUE_URL"
aws sqs delete-queue --queue-url "https://sqs.${AWS_REGION}.amazonaws.com/${ACCOUNT_ID}/${DLQ}"
aws iam detach-user-policy --user-name "$IAM_USER" --policy-arn "$POLICY_ARN"
aws iam list-access-keys --user-name "$IAM_USER" \
  --query 'AccessKeyMetadata[].AccessKeyId' --output text \
  | xargs -n1 -I{} aws iam delete-access-key --user-name "$IAM_USER" --access-key-id {}
aws iam delete-user --user-name "$IAM_USER"
aws iam delete-policy --policy-arn "$POLICY_ARN"
```

DLQ deletion may need a 60-second wait after the source queue is deleted.

---

## See also

- [CocoIndex Pipeline](cocoindex.md) — how the polled events become Qdrant/Neo4j writes
- [Ingestion Failure Semantics](../operations/atomicity.md) — re-raise + poison-pill contract
- [AWS Setup (reference)](../deployment/aws-setup.md) — same content as a flat reference, plus a LocalStack section for running this whole flow on your laptop
