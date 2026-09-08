# Public-release checklist

Complete this checklist immediately before making the repository public. The
private phase is not a trusted boundary: this is a final audit, not the first
security review.

## Repository and history

- [ ] `gitleaks git --redact` reports no unreviewed findings on the complete
  history, not only the working tree.
- [ ] Inspect `git log --all --name-status` for unexpected files, including
  deleted files and abandoned branches/tags that may be pushed.
- [ ] Search tracked content for credential-bearing patterns and private machine
  identifiers; inspect generated charts, notebooks, and screenshots manually.
- [ ] Verify no model weights, raw traces, raw prompts, datasets, or large logs
  are tracked unintentionally.
- [ ] Confirm `.gitignore`, `.gitleaks.toml`, the pre-commit hook, and the CI
  workflow are present on the default branch.

## GitHub controls

- [ ] Enable GitHub secret scanning and push protection before changing
  visibility.
- [ ] Enable Dependabot alerts and security updates.
- [ ] Protect `main`: require pull requests, require the secret-scan check, and
  restrict direct pushes.
- [ ] Review Actions permissions and third-party action pins; retain
  `contents: read` unless a workflow demonstrates a need for more.
- [ ] Verify any deployed tokens are repository/environment secrets, not values
  in Actions workflow files.

## Content review

- [ ] Reports describe hardware generically where precise identifiers would be
  sensitive; remove private hostnames, user names, filesystem paths, and account
  IDs.
- [ ] Configuration examples contain placeholders, never live values.
- [ ] Reproduction instructions work without access to a private network or
  personal infrastructure.
- [ ] Copyright, licenses, and model/dataset terms have been reviewed for every
  committed external asset.

## If anything is found

Stop the visibility change. Rotate exposed credentials first, then remediate the
history and re-run this checklist. Removing a secret from the latest commit alone
does not make it safe.
