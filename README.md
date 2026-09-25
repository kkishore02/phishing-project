# Security Awareness Phishing Simulation Framework

An educational Python framework for **authorised security-awareness training and controlled cybersecurity testing**.

## Overview

This project demonstrates how a phishing simulation can be structured in an authorised environment while enforcing safeguards such as approved-domain checks and explicit simulation authorisation.

It is designed as a cybersecurity learning and portfolio project, not for unsolicited or unauthorised campaigns.

## Key Features

- Explicit simulation-authorisation control
- Approved-domain validation
- Target data loading from CSV
- Template loading from JSON
- Template selection based on role, department and interests
- Optional NLP-based similarity scoring
- Personalised security-awareness templates
- Campaign and tracking identifiers
- Logging and results collection

## Technologies

- Python
- NLTK
- NumPy
- Pandas
- Scikit-learn

## Workflow

```text
Authorised Test Scope
        |
        v
Approved-Domain Check
        |
        v
Load Training Targets
        |
        v
Template Selection
        |
        v
Optional NLP Matching
        |
        v
Controlled Awareness Simulation
        |
        v
Logging & Results
```

## Safety Controls

The framework includes controls intended to keep usage within an authorised training environment:

- `simulation_authorized` must be explicitly enabled.
- Target domains are checked against an authorised-domain list.
- Targets outside the approved scope are skipped.
- The source code includes an educational-use notice.

Only use the project with systems, domains and participants for which you have explicit permission.

## Installation

Clone the repository:

```bash
git clone https://github.com/kkishore02/phishing-project.git
cd phishing-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Skills Demonstrated

- Python automation
- Security-awareness concepts
- Defensive phishing simulation
- Input validation
- Logging
- NLP / text similarity
- Ethical scoping and authorisation controls

## Future Improvements

- Add unit tests
- Add sample redacted training data
- Add example safe templates
- Generate analyst-friendly summary reports
- Map awareness scenarios to common social-engineering techniques
- Add a simple dashboard for authorised training results

## Disclaimer

This project is for education, security awareness and authorised cybersecurity testing only. Do not use it against individuals, organisations, accounts or systems without explicit permission.

## Author

**Kishore Bandi**  
Cybersecurity MSc Graduate | Aspiring SOC Analyst
