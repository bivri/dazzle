# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/bivri/dazzle/security/advisories/new),
or by email to support@bivri.app. Do not open a public issue for a vulnerability.

Include the affected version or commit, the steps to reproduce, and the impact you observed. We
reply to the reporter and agree a disclosure date once a fix is ready.

## Scope

In scope: the `dazzle` package in this repository, its release artifacts on GitHub Releases, and
the weights and model card published at
[bivri/dazzle-generator](https://huggingface.co/bivri/dazzle-generator).

A transformation filter or a training recipe that reduces the protective pattern's effect is a
limit of the method, not a vulnerability. Open a regular issue with the evaluation run that shows
it; [docs/evaluation.md](docs/evaluation.md) describes how effects are measured.

## Supported versions

Fixes go into the latest release only.
