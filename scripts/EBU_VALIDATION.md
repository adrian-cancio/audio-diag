# Official EBU v05 local validation

This is internal technical testing, not EBU certification. No official audio or
measurements belong in this repository. Obtain the dataset directly from EBU:
https://tech.ebu.ch/publications/ebu_loudness_test_set

© EBU. Use is governed by the official terms:
https://tech.ebu.ch/files/live/sites/tech/files/shared/testmaterial/use%20of%20EBU%20AUDIO%20test%20sequences.pdf

## Run

From repository root, with numpy and system ffmpeg/ffprobe available:

```sh
python3 -m scripts.validate_ebu --dataset /external/ebu/v05 --output /external/ebu/results/run
python3 -m unittest tests.test_ebu_runner tests.test_profiles tests.test_controlled
```

The sequential runner requires exactly 70 WAVs, resets analysis per file, persists
its complete ledger after every file, and writes raw analyzer JSON, Markdown
renders and their SHA256 hashes. The initial ledger includes pending rows, so an
interrupted run remains explicitly incomplete. Restarting reruns all files; it
does not currently skip completed work. The runner requires a new output directory
outside the dataset, repository and protected music library for each audit.
No DSP changes or tolerance adjustments are performed.

## Expected-value authority

- EBU Tech 3341, November 2023 (editorial revision of 2016), §2.9 Table 1:
  https://tech.ebu.ch/docs/tech/tech3341.pdf
- EBU Tech 3342, November 2023, §4 Table 1:
  https://tech.ebu.ch/docs/tech/tech3342.pdf

3341 cases 1/2 check I and M/S maxima; these maxima do not independently prove
constant time-series ballistics. Cases 3–8 check I; 10 checks maximum S and 13
maximum M. Cases 9/12 require constant S/M readings after 3/1 seconds, which the
public aggregate API cannot verify. Cases 11/14 require successive live readings;
these are marked unsupported, not failed or passed from a global maximum.
Cases 15–23 check true peak with asymmetric −0.4/+0.2 dB tolerance, not loudness.
Both multichannel versions of case 6 are unsupported because production analysis
downmixes. Shared programmes check both I and LRA. 3342 cases 1–6 use ±1 LU.
The three extra sine files and reference-listening noise have no Table 1
conformance tolerance and are measured but not counted as normative passes.

Tech 3342 §5 mentions at least 1.5 seconds of following silence in its reference
file-based implementation. This audit preserves full original files and baseline
behavior without appending samples. Any LRA discrepancy must be investigated
with that boundary convention in mind; the external ffmpeg cross-check is not an
expected-value authority. It runs on failed cases and all LRA cases, with stderr
and exact command saved externally.

All comparisons use unrounded values and the stated inclusive bounds. Unknown,
unsupported, error, pending and not-applicable cases are distinct from passes.
The 70-row count and unique filenames are asserted on every ledger write.

## Interpreting the baseline audit

The local audit of baseline `553ffe5` recorded 70 file rows: 60 passed,
6 unsupported and 4 auxiliary (`not_applicable`). There are 72 check definitions:
66 numeric comparisons were executed and passed; 6 checks on unsupported rows
were not executed. Thus 72 is not the number of tested comparisons. The two
successive-reading definitions each represent an expected sequence, not 20
individually executed comparisons.

This is bounded evidence against published expected values, not certification
or proof of full conformance. The production numpy meter is separate from
ffmpeg's meter; agreement with that external implementation is a cross-check,
not an independent derivation of correctness. The official dataset is supplied
locally; the runner does not download it or verify an official archive manifest.
Input hashes and render hashes identify this run, but do not establish dataset
authenticity. The ledger records the Git HEAD, not dirty-tree contents or tool
versions; retain those separately when reproducing an audit.
