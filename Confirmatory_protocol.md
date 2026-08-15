# Frozen confirmatory protocol

## Project

**Working title:** Mapping Analysis-Model Sensitivity in Binary Black
Hole Parameter Estimation with GWTC-5.0

**Primary model comparison:**  
`C00:IMRPhenomXPHM-SpinTaylor` versus `C00:IMRPhenomXPNR`

**Freeze date:** 2026-08-01

## 1. Separation of development and confirmation

The following 17 events form the frozen development set and may not
enter the confirmatory analysis:

- GW240612_081540
- GW240908_082628
- GW241011_233834
- GW241230_233618
- GW250114_082203
- GW240920_124024
- GW241127_061008
- GW241129_021832
- GW240513_183302
- GW240615_113620
- GW241116_151753
- GW240621_195059
- GW241225_082815
- GW241130_034908
- GW240920_073424
- GW240919_061559
- GW241229_155844

All metric choices, metadata tiers, screening rules, and diagnostic
plots were developed using those events. Confirmatory conclusions must
use previously untouched events only.

## 2. Confirmatory sampling frame

The GWTC-5.0 summary table contains 104 events with released XPHM and
XPNR analyses. After removing the 17 development events, the untouched
sampling frame contains **87 events**.

The input ranking file has SHA-256:

`7acedfbd0d2406bf6628ee39fb7fcaed4a80c940807fafd73c1eafe0671753b4`

## 3. Stratified random selection

Two catalog quantities are used for stratification:

1. `maximum_screening_shift`, the largest summary-level normalized
   XPHM-XPNR median displacement among source-frame chirp mass,
   effective spin, and luminosity distance;
2. catalog network matched-filter SNR.

Each quantity is divided into tertiles using the untouched 87-event
pool.

Screening-score tertile boundaries:

- minimum: 0.009817758943
- low/medium boundary: 0.051968060957
- medium/high boundary: 0.085201565692
- maximum: 0.156739476596

SNR tertile boundaries:

- minimum: 7.507138371500
- low/medium boundary: 9.589740516146
- medium/high boundary: 11.454639276034
- maximum: 31.946599616728

The crossing of three score strata and three SNR strata gives nine
cells. Two events are selected randomly from each cell, producing an
18-event primary sample.

Randomization uses NumPy `default_rng` with seed `20260801`.
Events are sorted alphabetically within each stratum before
randomization.

## 4. Frozen primary sample

| download_batch   | event           | score_stratum   | snr_stratum   |   maximum_screening_shift |   network_matched_filter_snr |
|:-----------------|:----------------|:----------------|:--------------|--------------------------:|-----------------------------:|
| A                | GW240525_031210 | high            | low           |                 0.0949183 |                      8.0172  |
| A                | GW240414_054515 | high            | medium        |                 0.0875788 |                     10.6655  |
| A                | GW240629_145256 | low             | high          |                 0.0503991 |                     12.2751  |
| A                | GW241101_220523 | low             | medium        |                 0.0444748 |                     10.5596  |
| A                | GW240922_142106 | medium          | high          |                 0.0620376 |                     12.3515  |
| A                | GW240531_075248 | medium          | low           |                 0.0597211 |                      8.58787 |
| B                | GW240428_225440 | high            | high          |                 0.111644  |                     15.637   |
| B                | GW250109_010541 | high            | medium        |                 0.0937745 |                     11.1823  |
| B                | GW240413_022019 | low             | high          |                 0.0334003 |                     17.3284  |
| B                | GW240627_131622 | low             | low           |                 0.0490975 |                      9.2253  |
| B                | GW240426_031451 | medium          | low           |                 0.0546451 |                      9.0682  |
| B                | GW241109_033317 | medium          | medium        |                 0.0619935 |                     10.3929  |
| C                | GW241230_084504 | high            | high          |                 0.103859  |                     11.7806  |
| C                | GW240526_093944 | high            | low           |                 0.103494  |                      8.46613 |
| C                | GW250101_011205 | low             | low           |                 0.0370683 |                      9.07858 |
| C                | GW250108_152221 | low             | medium        |                 0.0338575 |                     10.3103  |
| C                | GW241114_235258 | medium          | high          |                 0.0594743 |                     12.1508  |
| C                | GW240825_055146 | medium          | medium        |                 0.0615335 |                      9.70977 |

The A, B, and C labels are operational download batches only. All 18
events are already selected. Results from one batch may not be used to
alter or stop later batches.

## 5. Eligibility and replacement

The strict primary analysis requires a **Tier A** XPHM-XPNR pair:

- detector data and detector set match;
- PSDs and channels match;
- detector and waveform frequency settings match;
- likelihood and marginalization settings match;
- priors match, apart from negligible serialization differences;
- sampler settings and seeds match;
- reference frame and cosmology match;
- only waveform-model and model-specific waveform-argument fields may
  differ.

Tier B and Tier C events are retained for labeled secondary or
descriptive tables but do not count toward the target of 18 strict
events.

A missing, corrupt, parameter-incomplete, Tier B, or Tier C primary
event is replaced by the next unused event in the **same score-by-SNR
stratum** from `confirmatory_randomization_queue.csv`. Replacement may
not depend on the observed posterior-distance metrics.

The first reserve in each stratum is:

| event           | score_stratum   | snr_stratum   |   selection_order_within_stratum |
|:----------------|:----------------|:--------------|---------------------------------:|
| GW240705_053215 | high            | high          |                                3 |
| GW241007_082943 | high            | low           |                                3 |
| GW240630_101703 | high            | medium        |                                3 |
| GW241109_115924 | low             | high          |                                3 |
| GW240908_125134 | low             | low           |                                3 |
| GW240601_231004 | low             | medium        |                                3 |
| GW240923_204006 | medium          | high          |                                3 |
| GW240420_175625 | medium          | low           |                                3 |
| GW240519_012815 | medium          | medium        |                                3 |

## 6. Posterior parameters

Primary scientific parameters:

- detector-frame `chirp_mass`;
- `mass_ratio`;
- `chi_eff`;
- `luminosity_distance`.

Screening-validation parameters:

- `chirp_mass_source`;
- `chi_eff`;
- `luminosity_distance`.

No posterior weights are applied unless a released analysis explicitly
provides nonuniform weights. Any such case must be documented before
the event is analyzed.

## 7. Frozen metrics

For every event, model pair, and parameter:

1. **Normalized Wasserstein distance**

   `W1 / mean(XPHM 90% width, XPNR 90% width)`

2. **Jensen-Shannon divergence**

   Base-2 JSD from 60 equal-width bins spanning the pooled sample range.

3. **Standardized median displacement**

   Absolute median difference divided by
   `sqrt(var_XPHM + var_XPNR)`.

4. **90% interval overlap**

   Intersection length divided by union length.

5. **Log width ratio**

   `log(width_XPHM / width_XPNR)`.

6. **Finite-sampling noise floor**

   One hundred random half-splits for each event, model, and parameter.
   The pair threshold is the larger of the two model-specific 95th
   percentiles. Split seed: `20260802`.

The code and definitions may be corrected only for an identified
implementation error. Any correction must be documented and applied to
the entire confirmatory sample.

## 8. Confirmatory endpoints and hypotheses

### Primary endpoint

For each strict event, calculate the median normalized Wasserstein
distance across the four primary scientific parameters. Call this
`event_median_NW1`.

### Primary hypothesis H1

Higher catalog network matched-filter SNR is associated with larger
`event_median_NW1`.

Test:

- Spearman rank correlation;
- one-sided alternative `rho > 0`;
- 100,000 event-label permutations;
- permutation seed `20260803`;
- significance level `alpha = 0.05`.

This is the sole primary confirmatory hypothesis.

### Secondary confirmatory hypothesis H2

The catalog `maximum_screening_shift` is positively associated with the
maximum full-posterior normalized Wasserstein distance across
`chirp_mass_source`, `chi_eff`, and `luminosity_distance`.

Test:

- one-sided Spearman permutation test;
- 100,000 permutations.

### Secondary confirmatory hypothesis H3

Within events, `chi_eff` normalized Wasserstein distance is greater than
`luminosity_distance` normalized Wasserstein distance.

Test:

- one-sided Wilcoxon signed-rank test.

H2 and H3 are adjusted together using Holm's method at familywise
`alpha = 0.05`.

## 9. Descriptive and exploratory analyses

The following do not determine confirmatory success:

- event and parameter fractions exceeding finite-sampling noise;
- medians and bootstrap confidence intervals by parameter;
- parameter-specific screening correlations;
- regression of event-level sensitivity on log SNR, catalog screening
  score, total source-frame mass, and detector count;
- Tier B, Tier C, SEOBNRv5PHM, or NRSur7dq4 comparisons;
- individual event case studies.

They must be labeled secondary or exploratory.

## 10. Missing-data and stopping rules

- No imputation of posterior parameters.
- An event lacking any of the four primary parameters does not count
  toward the strict 18-event target and triggers same-stratum
  replacement.
- No event is removed because its model difference is small, large, or
  contrary to the hypotheses.
- Downloading occurs in batches for storage convenience only.
- The confirmatory analysis is not evaluated until all 18 strict events
  are available or the prespecified within-stratum queues are exhausted.
- Any deviation is recorded in a protocol-deviation table.

## 11. Batch A

The first operational batch contains:

| event           | score_stratum   | snr_stratum   |   maximum_screening_shift |   network_matched_filter_snr |
|:----------------|:----------------|:--------------|--------------------------:|-----------------------------:|
| GW240525_031210 | high            | low           |                 0.0949183 |                      8.0172  |
| GW240414_054515 | high            | medium        |                 0.0875788 |                     10.6655  |
| GW240629_145256 | low             | high          |                 0.0503991 |                     12.2751  |
| GW241101_220523 | low             | medium        |                 0.0444748 |                     10.5596  |
| GW240922_142106 | medium          | high          |                 0.0620376 |                     12.3515  |
| GW240531_075248 | medium          | low           |                 0.0597211 |                      8.58787 |
