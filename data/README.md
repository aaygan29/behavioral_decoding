# data/

Nothing in here is committed except this file and the `.gitkeep` markers. See
`.gitignore`.

- `raw/`       downloaded datasets, never modified in place
- `external/`  market outcomes and other data pulled from outside the lab
- `interim/`   partially processed, safe to delete and regenerate
- `processed/` extracted feature blocks ready for the loaders

Two rules worth keeping.

**Raw stays raw.** If a script needs to change something in `raw/`, it writes to
`interim/` instead. That way a corrupted pipeline is always recoverable without
re-downloading.

**Nothing identifiable gets committed.** Neuroimaging files can carry participant
identifiers, and defacing is not always applied upstream. The `.gitignore`
blocks the common formats, but it is a backstop, not a substitute for checking.

Candidate datasets and access conditions: [`../docs/data_sources.md`](../docs/data_sources.md)
