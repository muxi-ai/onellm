# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| Latest  | :white_check_mark: |
| < Latest | :x:               |

Only the latest released version of OneLLM receives security updates.

## Reporting a Vulnerability

If you discover a security vulnerability in OneLLM, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please report vulnerabilities via one of the following methods:

- **Email:** [security@aroussi.com](mailto:security@aroussi.com)
- **GitHub Security Advisories:** [Report a vulnerability](https://github.com/muxi-ai/onellm/security/advisories/new)

### What to Include

When reporting a vulnerability, please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact of the vulnerability
- Any suggested remediation (if applicable)

### Disclosure Timeline

- **Acknowledgement:** We will acknowledge receipt of your report within 48 hours.
- **Assessment:** We will assess the vulnerability and determine its impact within 7 days.
- **Fix:** We aim to release a fix within 30 days for critical vulnerabilities, and within 90 days for lower-severity issues.
- **Disclosure:** We will coordinate public disclosure with the reporter after the fix is released.

### What to Expect

- You will receive a response acknowledging your report within 48 hours.
- We will work with you to understand and validate the issue.
- We will keep you informed of our progress toward a fix.
- We will credit you in the security advisory (unless you prefer to remain anonymous).

## Security Best Practices for Users

- Always use the latest version of OneLLM.
- Never commit API keys or secrets to version control.
- Use environment variables or secure secret management for provider API keys.
- Review the [configuration documentation](./docs/configuration.md) for secure setup guidance.
