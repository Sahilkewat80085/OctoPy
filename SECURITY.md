# Security Policy

## Supported Versions

We actively monitor and patch security vulnerabilities. The table below lists the versions of **OctoPy** that receive security updates:

| Version | Supported |
| ------- | --------- |
| 1.0.x   | Yes       |
| < 1.0.0 | No        |

## Reporting a Vulnerability

We take the security of our users and package seriously. If you identify a security vulnerability, please **do not open a public issue**. Instead, report it privately through the following protocol:

1. Send an email to the maintainer at **kkewat315@gmail.com** with the details.
2. Include a detailed description of the vulnerability, steps to reproduce, and any proof of concept (PoC).

### Expected Response

*   **Triage**: Within 48 hours of receipt.
*   **Remediation**: We aim to release a patch or advisory within 14 days for high/critical vulnerabilities.
*   **Credit**: We will credit reporters in release changelogs and security advisories unless they prefer anonymity.

## Deserialization Safety Guidelines

As **OctoPy** uses standard serialization engines (like `pickle` and `joblib` inside the `explain` and `report` modules), please adhere to the following safety precautions:

1. Never deserialize untrusted files.
2. Restrict environment privileges when executing ML model reports.
