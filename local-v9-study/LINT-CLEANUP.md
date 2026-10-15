# Local-only lint cleanup record

Prepared against repository commit `8090573f0ce059bbb6fa1ca556f68715538a8f3a` on 2026-10-15. This record accompanies a local working-tree cleanup only; no commit, push, or public repository write was made.

## Scope and behavior

- `study.py`: wrapped long expressions and report strings without changing their evaluated content; replaced `timezone.utc` with its equivalent `datetime.UTC` alias.
- `tests/test_study.py`: sorted imports and used `importlib.import_module("study")` after the existing local path setup to avoid a module-level import-order violation. Test assertions were not changed.
- The frozen design/configuration, vendored compatibility source files, disclosed limitations, and published numerical/result artifacts were not edited.
- The pre/post generated report text is byte-for-byte identical. Reduced deterministic study outputs were compared with runtime/timestamp and lock-provenance metadata excluded.

## Checksum impact

The original frozen lock recorded these SHA-256 values before the lint cleanup:

| File | Original frozen SHA-256 | Local lint-clean SHA-256 |
|---|---|---|
| `study.py` | `ed5a25935def65024c18099fff7b6a714ad4fb97c0c40339e6f344c2afd751e4` | `f761c13d8df20d41ad67f559a5bb94e79f782328e1377ff077b105d3ac129d5c` |
| `tests/test_study.py` | `367bb0e1d5ef8fbac182ad7e1e700c678399986dd50fbcafc712b3a37e8beb1e` | `0478523c6a98bc9658d7f3170f2fe623eeeee30e5641e16979085ad5b917e769` |

`lock.json` and `BUNDLE-SHA256SUMS` are updated locally so lock verification and bundle integrity checks reflect the style-clean files. This changes the checked source bytes and local published-bundle manifest; it does not rewrite the original result artifacts or claim that the cleaned source produced the already-published full-run results. The original `BUNDLE-SHA256SUMS` file had SHA-256 `203a2e3275d5196d1bc36c9d8552b6942fcf1f751c7af25d39988e7ec0e46e47`; `results/SHA256SUMS` remains unchanged at `276c6399ce5551c22c37b86240e7829d77232bb0078a16c2d8123d4abecf8f68`.

No full study was rerun. Validation is limited to repository lint, the six reduced v9 tests, lock verification, and checksum/report comparisons.
