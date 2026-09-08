# Security policy

This repository is intended to become public. Treat it as public now: never rely
on repository visibility as protection for a credential, endpoint, dataset, or
machine detail.

## Never commit

- API keys, tokens, passwords, SSH private keys, certificates, cookies, or
  credential-bearing connection strings;
- cloud credentials, provider configuration, `.env` files, or local override
  files;
- private hostnames, IP addresses, user names, mount paths, or SSH configuration;
- raw logs, profiler traces, prompts, or request captures that may contain any of
  the above.

Do not put secrets in source code, notebooks, shell history copied into Markdown,
test fixtures, screenshots, issue comments, pull-request descriptions, or commit
messages. A value that has been redacted in a report must not be recoverable from
its filename, surrounding context, or generated artifact metadata.

## Configuration model

Tracked configuration describes **non-sensitive experiment parameters** such as
batch size, dtype, model identifier, prompt length, and concurrency. Store it in
an exercise's `configs/` directory.

Sensitive values must be injected at runtime through environment variables or the
platform's secret store. Machine-specific settings go in ignored files under
`config/local/` or a locally managed `.env` file. Code that needs a credential
must fail with a clear missing-variable error; it must not have a credential-like
default value.

Use a placeholder such as `<set-in-environment>` in documentation. Never copy an
actual credential into an example, even temporarily.

## Local checks

The repository includes a Gitleaks pre-commit hook in
`.pre-commit-config.yaml`. Install and use it before the first commit:

```bash
brew install gitleaks pre-commit
pre-commit install
pre-commit run --all-files
gitleaks git --redact
```

The hook is an early warning, not permission to commit sensitive data. Do not
bypass it. If a detector produces a false positive, investigate it and use a
minimal, documented rule in `.gitleaks.toml`; never broadly allowlist a file type
or directory.

## CI controls

GitHub Actions scans commit changes on pushes and pull requests, and full history
on scheduled and manual runs. The workflow pins third-party actions to full commit SHAs and grants
only read access to repository contents. Keep the scan required on the default
branch once GitHub branch protection is enabled.

Before opening the repository publicly, enable these GitHub repository settings:

1. Secret scanning and push protection.
2. Dependabot alerts and security updates.
3. Branch protection for `main`, requiring the security workflow and a pull
   request review for future changes.
4. Least-privilege GitHub Actions permissions; do not enable write tokens by
   default.

## Suspected exposure

Assume any committed secret is compromised, including one later deleted.

1. Revoke or rotate the credential immediately.
2. Stop using it and identify every system it could access.
3. Remove it from the working tree and all affected generated artifacts.
4. Scan history, then rewrite history if the repository has not already been
   broadly cloned; rotation remains mandatory either way.
5. Record the remediation privately. Do not disclose the secret in an issue or
   report.

## Public-release gate

Run the full-history scan after all development branches are merged and before
changing visibility. Review [docs/public-release-checklist.md](docs/public-release-checklist.md)
and resolve every finding before the visibility change.
