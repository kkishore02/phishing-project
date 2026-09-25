# Safe Demo Walkthrough

This repository is designed only for authorised security-awareness training.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Use the included synthetic example data

The repository contains:
- `targets.csv` with fictional `example.com` users
- `templates/password_reset.json` with a clearly labelled training template

## 3. Run in dry-run mode

```bash
python phishing-simulation-framework.py --authorize --domain example.com --dry-run
```

When prompted, confirm authorisation only in an environment where you have permission.

The framework deliberately does **not** implement real outbound email sending. The dry-run creates a results record demonstrating template selection, scope validation, personalisation and tracking identifiers without sending messages.

## What this demonstrates

- Scope and authorisation controls
- Approved-domain validation
- Safe synthetic target data
- Template selection
- Input validation
- Logging
- Results generation
- Security-awareness workflow design

## Portfolio integrity

All included targets and domains are synthetic examples. This project is for defensive education and authorised awareness testing only.
