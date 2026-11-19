# Image tests: known inconsistencies

These behaviours are currently **not specified** by the project contract, so they are
deliberately not asserted as correct in the test suite. They are recorded here so that a
future decision (and the matching production change) can update the tests on purpose.

1. `--quality` help text says `default: 95`, while the implementation uses `90`.
   The contract "help must describe actual behaviour" is unambiguous, so it is covered by an
   addressed `xfail(strict=True)` in `tests/test_cli.py`
   (`test_quality_help_matches_implemented_default`).
2. Transparency to BMP: `crop`, `resize` and `rotate` accept RGBA input for `.bmp` and
   silently lose the alpha channel (Pillow writes RGB on save), while `annotate` rejects
   alpha input for BMP with `ValueError`. Whether BMP should be rejected or accepted is
   undecided.
3. `--quality` with a non-JPEG output raises `ValueError` instead of a warning. A comment in
   `annotate.py` questions this choice, so the contract is not fixed. The four tests that
   assert this current behaviour are explicitly marked as characterizing, not as correct.
4. `sheet` creates/reuses the output directory (`exist_ok=True`) and publishes sheets
   sequentially. If publishing a later sheet fails, earlier sheets stay published
   (`Could not publish ... Already published: [...]`). Not asserted as correct; only the
   no-overwrite guarantee for an individual sheet is tested.
5. `image info` accepts multi-frame images (metadata inspection), while all transforming
   tools reject them. Treated as intended, not asserted either way.
6. `crop`, `resize` and `rotate` reject mode `1` (bilevel) output to JPEG, although current
   Pillow can encode it. Conservative validation, contract undecided.
