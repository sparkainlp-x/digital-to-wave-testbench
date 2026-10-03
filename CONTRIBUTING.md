# Contributing to Digital to Wave Testbench

Thank you for helping! Please read the [Code of Conduct](CODE_OF_CONDUCT.md) first. Report security issues privately as described in [SECURITY.md](SECURITY.md), not as public issues.

## Ground rules

- **Synthetic data only.** Never commit private, proprietary or personal data, correspondence or drafts. The `private/` folder is git-ignored for local notes.
- **No physical claims.** Numbers produced here describe a dimensionless software model. Do not present them as evidence about lab-grown diamond, spin waves, phonons, GHz hardware, biological signals or any physical channel. Label evidence as SYNTHETIC, REPORTED, TARGET or UNRUN.
- **Hash-locked modules stay unchanged.** `digital_to_wave.py` and `robustness_benchmark_v2.py` … `robustness_benchmark_v5_timing.py` are locked by [`MODULES.sha256`](MODULES.sha256). New experiments go in new files (for example `robustness_benchmark_v6_*.py`) with their own conditions fixed before the full run, a new master seed and their own tests.
- **Real measured numbers only.** Any number in the README must come from a command in this repository that anyone can rerun (state the command and seed). Report losses and null results as honestly as wins.
- **License.** The project is AGPL-3.0-only (see [LICENSE](LICENSE) and [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)). By contributing, you agree that your contribution is licensed under the same terms.

## Development setup

```bash
python -m pip install -e ".[test]"
python -m pip install ruff
```

## Before you open a pull request

1. **Tests:** `python -m pytest` must pass (CI runs Python 3.11–3.13).
2. **Lint:** `ruff check .` must be clean.
3. **Locks and artifacts:** `sha256sum -c MODULES.sha256` must pass. If you regenerate artifacts with `tools/reproduce_all.sh`, the committed CSV/PNG files must stay byte-identical unless the change is intentional and explained.
4. **Docs:** update the README and `CHANGELOG.md` (under *Unreleased*) for any user-visible change.

Commit messages should say what changed and why. Keep pull requests focused.
