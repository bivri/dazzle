# 0001. Fresh public history under Apache-2.0

Status: accepted, 2026-10-08

## Context

The project is published on GitHub with generator weights on the Hugging Face Hub. Public release
requires a self-contained repository free of internal infrastructure references and restricted to
permissive open-source licenses.

## Decision

- This repository maintains a clean, self-contained public history.
- Code and published weights are licensed under Apache-2.0.
- Runtime code, weights and assets are Apache-2.0, MIT, BSD or CC0 only. Models used only for
  evaluation are downloaded at run time and never redistributed; non-commercial models are not
  used, not even for evaluation.

## Consequences

Contributions are accepted under Apache-2.0. Third-party notices go in `NOTICE`. The SDXL VAE (MIT)
may be a training objective; OpenRAIL SDXL UNets and CC-BY CSD may only evaluate.
