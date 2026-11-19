# Image tests: known inconsistencies

These behaviours are currently **not specified** by the project contract, so they are
deliberately not asserted as correct in the test suite. They are recorded here so that a
future decision (and the matching production change) can update the tests on purpose.

1. Transparency to BMP: `crop`, `resize` and `rotate` accept RGBA input for `.bmp` and
   silently lose the alpha channel (Pillow writes RGB on save), while `annotate` rejects
   alpha input for BMP with `ValueError`. Whether BMP should be rejected or accepted is
   undecided.
2. `--quality` with a non-JPEG output raises `ValueError` instead of a warning. A comment in
   `annotate.py` questions this choice, so the contract is not fixed. The four tests that
   assert this current behaviour are explicitly marked as characterizing, not as correct.
3. `sheet` creates/reuses the output directory (`exist_ok=True`) and publishes sheets
   sequentially. If publishing a later sheet fails, earlier sheets stay published
   (`Could not publish ... Already published: [...]`). A regression test characterizes
   that partial result and the temporary-artifact cleanup; it does not assert that the
   partial publication itself is correct.
4. `image info` accepts multi-frame images (metadata inspection), while all transforming
   tools reject them. Treated as intended, not asserted either way.
5. `crop`, `resize` and `rotate` reject mode `1` (bilevel) output to JPEG, although current
   Pillow can encode it. Conservative validation, contract undecided.
