# Validated LSC development checkpoint, 19 November 2026

This checkpoint records device-input LAMMPS preparation, optimized forward/reverse charge transfer, and the factorized second-layer reverse convolution.

Validation: 95 selected H100 pytest cases and 18 timing/control accuracy checks passed in job 7497925. The same stage3 water model, one H100, 300 K, 1 bar and 0.5 fs were used for timings. At 10,368 atoms: FP32 27.4835 ms/step (1.57185 ns/day); mixed 27.9676 ms/step (1.54464 ns/day).

Detailed evidence: `.lsc-reference-logs/local-reverse-factorized-20261119/REPORT.md` and `audit.json`. Frozen development artifacts and source bundle: `.lsc-reference-logs/checkpoint-20261119`. Binary SHA-256: `5135fe4baef074676e5e23472d2473e979b9f1f9319b480039d81fc7154d0cf2`.

The new basis/counter investigation is separate. Stable remains pinned to release 2762b9f; this checkpoint does not promote or rebuild it.
