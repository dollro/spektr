# AWS Setup

This guide is a flat **reference** for AWS S3 + SQS configuration. If you want a step-by-step tutorial with copy-paste blocks for both AWS CLI and the console, plus naming conventions and a DLQ, see [S3 + SQS Setup — Step-by-Step Manual](../ingestion/s3-sqs-setup.md).

## Architecture

```
S3 Bucket → Event Notification → SQS Queue → CocoIndex Pipeline
```

S3 sends event notifications (ObjectCreated, ObjectRemoved) to an SQS queue.
The CocoIndex pipeline polls SQS for new/deleted files and processes them incrementally.

## Prerequisites

- AWS CLI configured with appropriate credentials
- An S3 bucket for document storage
- An SQS queue for event notifications

## IAM Permissions

The pipeline requires an IAM user/role with these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::YOUR_BUCKET",
        "arn:aws:s3:::YOUR_BUCKET/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:GetQueueAttributes"
      ],
      "Resource": "arn:aws:sqs:REGION:ACCOUNT_ID:YOUR_QUEUE"
    }
  ]
}
```

## SQS Queue Policy

The SQS queue must allow S3 to send messages:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "s3.amazonaws.com"
      },
      "Action": "sqs:SendMessage",
      "Resource": "arn:aws:sqs:REGION:ACCOUNT_ID:YOUR_QUEUE",
      "Condition": {
        "ArnEquals": {
          "aws:SourceArn": "arn:aws:s3:::YOUR_BUCKET"
        }
      }
    }
  ]
}
```

## S3 Event Notification Configuration

Configure S3 to send notifications to SQS for object events:

```bash
aws s3api put-bucket-notification-configuration \
  --bucket YOUR_BUCKET \
  --notification-configuration '{
    "QueueConfigurations": [
      {
        "QueueArn": "arn:aws:sqs:REGION:ACCOUNT_ID:YOUR_QUEUE",
        "Events": [
          "s3:ObjectCreated:*",
          "s3:ObjectRemoved:*"
        ],
        "Filter": {
          "Key": {
            "FilterRules": [
              {"Name": "suffix", "Value": ".pdf"},
              {"Name": "suffix", "Value": ".txt"},
              {"Name": "suffix", "Value": ".md"},
              {"Name": "suffix", "Value": ".png"},
              {"Name": "suffix", "Value": ".jpg"}
            ]
          }
        }
      }
    ]
  }'
```

Note: S3 filter rules use OR logic for suffix filters within a single configuration.
For comprehensive filtering, the pipeline's `included_patterns` setting handles additional file type matching.

## Environment Variables

Add to your `.env` (see [Environment Variables](../configuration/environment.md#aws-when-document_sources3) for full reference):

```bash
DOCUMENT_SOURCE=s3
S3_BUCKET_NAME=your-bucket-name
S3_SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/your-queue
AWS_REGION=us-east-1

# AWS credentials (if not using instance profile/role)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

Set `DOCUMENT_SOURCE=s3` to enable S3 ingestion. Both `S3_BUCKET_NAME` and `S3_SQS_QUEUE_URL` are required when `DOCUMENT_SOURCE=s3`; the pipeline fails fast at startup if either is missing. Without `DOCUMENT_SOURCE=s3`, the pipeline ignores S3 settings even when populated.

## LocalStack (Local Development)

For local development without AWS, use LocalStack:

```bash
# Add to docker-compose.yml
localstack:
  image: localstack/localstack:3.0
  ports:
    - "4566:4566"
  environment:
    - SERVICES=s3,sqs
    - DEFAULT_REGION=us-east-1

# Create resources
awslocal s3 mb s3://test-bucket
awslocal sqs create-queue --queue-name test-queue

# Configure event notification
awslocal s3api put-bucket-notification-configuration \
  --bucket test-bucket \
  --notification-configuration '{
    "QueueConfigurations": [{
      "QueueArn": "arn:aws:sqs:us-east-1:000000000000:test-queue",
      "Events": ["s3:ObjectCreated:*", "s3:ObjectRemoved:*"]
    }]
  }'

# Upload test file
awslocal s3 cp tests/fixtures/sample.txt s3://test-bucket/sample.txt
```

Set environment for LocalStack:

```bash
DOCUMENT_SOURCE=s3
S3_BUCKET_NAME=test-bucket
S3_SQS_QUEUE_URL=http://localhost:4566/000000000000/test-queue
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
AWS_ENDPOINT_URL=http://localhost:4566
```

## Verifying the Setup

1. Upload a test file to S3
2. Check SQS for the notification message
3. Run the pipeline and verify processing

```bash
# Upload
aws s3 cp tests/fixtures/sample.txt s3://YOUR_BUCKET/sample.txt

# Check SQS (should see a message)
aws sqs receive-message --queue-url YOUR_QUEUE_URL

# Run pipeline
uv run python -m ingestion.pipeline
```
