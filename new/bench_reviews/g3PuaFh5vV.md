## Summary

This paper investigates whether decoding MEG data in source-reconstructed brain space (voxels) rather than sensor space enables practical advantages: spatial inductive biases, domain-specific augmentations, interpretability, zero-shot cross-dataset generalization, and multi-dataset harmonization. The work makes a genuine attempt to answer an important practical question in non-invasive neural decoding—whether source space can serve as a common representation across datasets and subjects. The paper honestly reports that source space does not clearly beat sensor space under matched MLP models, and demonstrates modest but non-trivial improvements when using a source-space CNN with cross-dataset transfer. The strongest novelty claim is that zero-shot evaluation across datasets with different sensor configurations is structurally impossible in sensor space but feasible in source space, which the paper validates experimentally.

## Strengths

- **Novel structural capability for cross-dataset evaluation**: Section 6 and Table 6 demonstrate that CNNs trained on Schoffelen achieve above-chance balanced accuracy (52.7–55.7%) on Armeni subjects without any fine-tuning—a capability structurally impossible for fixed-domain sensor-space models due to incompatible sensor configurations. This is the paper's most important positive result.

- **Intellectual honesty in internal comparisons**: Section 3 and Table 3 directly report that under matched MLPs, sensor space slightly outperforms source space in both single-subject (67.4±0.7 vs 66.8±0.6) and inter-subject (54.0±0.4 vs 53.5±0.5) settings. The paper does not hide this inconvenient result, which strengthens its credibility.

- **Source space enables spatial architectures**: Section 3.1 and Table 4 show that the source-space CNN (54.5±0.3) outperforms all tested sensor-space and source-space MLPs, as well as GATs on both domains. The CNN works despite zero-padding voxels outside the brain, suggesting the spatial grid structure is being exploited—a genuine contribution to understanding what source space makes *possible* architecturally.

- **Careful experimental design**: The paper uses temporal extrapolation splits (sessions as train/val/test) rather than random splits, correctly noting that random splits cause leakage and inflated performance (Section 3, footnote 2). The preprocessing pipeline is systematically ablated (Tables 1–2) with transparent reporting of parameter choices.

- **Cross-dataset learning shows promise**: Table 8 demonstrates that pooling Armeni subjects into Schoffelen training improves CNN performance on all three Armeni test subjects (e.g., 55.7→59.3 on subject 003 with 100% probability of improvement across seeds), supporting the feasibility of source-space data harmonization.

## Weaknesses

### Fatal

None. The paper's core claims are overstated relative to evidence, and the experiments have notable limitations, but the study does demonstrate a coherent, non-trivial finding: source space is a viable common representation that enables certain architectural and transfer tricks, albeit with modest gains.

### Major

- **Headline claims substantially exceed what the experiments support.** The Abstract announces "zero-shot generalisation between datasets" and "data harmonisation" as demonstrated capabilities, but the evidence is narrow and asymmetric. Table 6 shows transfer works in only one direction (Schoffelen→Armeni) with modest accuracy (52.7–55.7 on a binary task), while the reverse direction fails at chance (48.2–51.2). Table 8's harmonization result primarily benefits from having seen the test subjects' data during training—an expected result, not a broad harmonization demonstration. Section 7's claim that combining datasets improves held-out performance on the *other* dataset (Table 7) is only 0.3 percentage points (54.5→54.8). The abstract and introduction repeatedly frame results as general advantages of source space for non-invasive decoding, but the data support at best: "source space enables one direction of modest cross-dataset transfer on a single binary task and small gains from pooled training." This scope inflation undermines the paper's credibility.

- **The experimental setup is too narrow to support its broad framing.** Almost all results are on a single binary task (heard-speech detection vs. silence) using single time slices without temporal context (Section 3, "All models were given single time slices as inputs instead of short windows"). The authors chose this task because it is "easier" (Section 2) and "requires far less temporal context" (Section 3). This makes the comparison cleaner but severely limits what can be concluded about "non-invasive neural decoding" broadly. A representation that helps on instantaneous speech-vs-silence detection may not help on temporally structured decoding tasks (phoneme, word classification) that dominate the motivating literature in the Introduction. The paper lacks evidence that source-space advantages persist on more complex or temporally dependent tasks.

### Minor

- **The interpretability claim is underdelivered empirically.** The Abstract and Introduction list "better interpretability" as a headline contribution, but Section 5's region-masking experiment (Figure 3) mostly shows that masking individual regions has small effect, with the paper admitting "no simple consistent trend is seen across subjects" and leaving "understanding this is left for future work." At best, masking Temporal Pole and Insular Cortex cause ~3% drops, which roughly aligns with known speech-processing neuroanatomy, but this is exploratory evidence, not a demonstration of meaningful interpretability. The interpretability claim is partially addressed by the paper's conceptual argument (brain regions are more meaningful than sensor numbers), but the empirical section does not substantively advance it.

- **Source space's advantage is architecture-dependent rather than representation-inherent, and the paper does not fully disentangle this.** The fact that source-space CNNs win while source-space MLPs underperform source-space MLPs suggests the benefit comes from architectural exploitability, not raw representation quality. The paper notes this (Section 3 vs Section 3.1) but does not analyze *why* the CNN succeeds when the MLP does not, nor does it cleanly separate the contributions of source reconstruction, anatomical morphing, template alignment, zero-padding geometry, and positional embeddings to the observed gains.

- **The sensor-space structured baseline is weak (GAT), weakening the "source enables spatial inductive bias" claim.** Table 4 shows sensor-space GAT (53.3±0.7) underperforms even the sensor-space MLP (54.0±0.4). This could be due to suboptimal GAT implementation, architecture mismatch (GATs are notoriously hard to train), or the graph construction being a poor fit for sensors. Without a competitive sensor-space structured baseline, the paper's claim that source space uniquely enables spatial architectures is not fully convincing.

### Trivial

- The cross-dataset evaluation setups are asymmetric: Armeni subjects are morphed into the template brain while Schoffelen subjects are morphed into subject-specific Armeni anatomy (Section 6). The paper does not analyze whether this asymmetry affects performance.

- Many reported improvements are modest in absolute balanced accuracy (often <0.5 percentage points), and the paper does not discuss whether these margins are practically meaningful for downstream decoding applications.

## Nice-to-Haves

- Testing source-space decoding on a temporally richer task (e.g., phoneme classification) would strengthen the paper's relevance to the broader neural decoding literature.
- Analyzing why source-space CNNs succeed when source-space MLPs do not would clarify the paper's actual contribution.
- Stronger sensor-space structured baselines (e.g., different graph architectures or attention models) would help isolate whether source space specifically enables spatial architectures or whether a better sensor-space model would close the gap.
- Disentangling contributions from source reconstruction, morphing, and template alignment would deepen understanding of what drives the cross-subject/generalization benefits.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Overclaim of zero-shot/harmonization claims (Harsh Critic #1)**: *Retained, not removed* — this is a substantive weakness.

- **Source space doesn't beat sensor space (Harsh Critic #2)**: The reviewer frames this as a major weakness, but the paper itself reports this honestly in Table 3. The real issue is not that source space "loses" but that the *claims exceed the evidence*. Retained as a minor point about scope inflation rather than as a separate weakness.

- **All results on a single binary task with single time slices (Harsh Critic #3)**: *Retained as Major weakness* — this is a valid and substantive concern about the gap between the scope of claims and the scope of evidence.

- **Interpretability is a null result (Harsh Critic #4)**: *Retained* — this is a valid concern.

- **Preprocessing tuned on a single proxy task (Section 2 tuning)**: The reviewer's concern about overfitting the preprocessing pipeline to a narrow task is partially valid but overstated. The paper's ablations (Tables 1–2) show the pipeline choices are relatively robust to reasonable alternatives. The paper also overrides some logistic regression findings. WEAKENED to a minor point.

- **"The paper's empirical case for source space is weak"**: The paper makes its case that source space enables *architectural* advantages, not that it inherently produces better representations. The reviewer conflates these two claims. The paper *does* show that source space enables CNNs to outperform MLPs—this is the paper's actual contribution. WEAKENED.

- **Missing external baselines**: The paper explicitly notes it uses internal baselines due to the custom task and data splits (Section 3, footnote 3). This is standard practice in this emerging field. MOVED TO NICE-TO-HAVE.

- **Formatting/style nitpicks about notation inconsistencies and section structure**: REMOVED — these are minor presentation issues.

## Novel Insights

The paper's most interesting contribution is structural rather than performance-based: it demonstrates that source-reconstructed MEG data can serve as a domain-agnostic input representation that enables zero-shot evaluation across datasets with fundamentally different sensor configurations—a capability that sensor-space models cannot provide without retraining or reconfiguring architecture. This is important because the field currently lacks standardized benchmarks, and each new dataset requires building models from scratch. The paper honestly shows that source space does not inherently improve on sensor space for raw decoding accuracy, but positions its contribution around what source space *permits* (architectures, augmentations, cross-dataset pipelines) rather than claiming raw superiority—a distinction the paper itself draws but the abstract does not adequately reflect. This reframing—from "source space is better" to "source space enables capabilities sensor space structurally cannot"—is the paper's genuine insight.

## Suggestions

- Revise the Abstract, Introduction, and Discussion to more conservatively frame the cross-dataset and harmonization results, specifying the asymmetric, modest, and single-task-limited nature of the evidence rather than implying broad zero-shot generalization capability.

- Add at least one experiment on a temporally richer task (e.g., phoneme or word classification) to establish whether source-space advantages persist beyond instantaneous speech-vs-silence detection.

- Deepen the analysis of why the source-space CNN outperforms MLPs while the GAT fails in both domains—this analysis would clarify the paper's actual contribution about architectural exploitability vs. representation quality.

- Include a competitive sensor-space structured baseline (e.g., a well-tuned GAT variant or spatial attention model) to more convincingly demonstrate that source space uniquely enables spatial inductive biases.

## Score and Decision

**Calibration anchors compared against:**

- **cRmmIdqvZp.md** (scores 5, 8, 6): A paper whose claims about alignment and modality superiority were overstated relative to evidence. Humans gave mixed scores (reject consensus). This paper has similar scope-inflation issues but is more honest about its limitations and has a structurally novel cross-dataset contribution — the paper under review is slightly stronger.

- **IAFStwZPNu.md** (scores 6, 5, 6): A MEG speech decoding paper using self-supervised learning. Humans gave borderline scores and it was ultimately withdrawn. Its weaknesses (low absolute performance, narrow evaluation) are similar to this paper's. The paper under review is arguably more novel in its cross-dataset capability claim.

- **HsB1sQvXML.md** (5,3,3,3,5): A paper with good real-world focus but narrow scope and limited generalization claims. Scored low due to scope limitations — but this paper is better-positioned and more honest.

- **YZRgB5DnXw.md** (5,6,5,5): Overclaimed improvements on small LLMs with narrow evaluation — scored middling/reject. Similar pattern of headline claims exceeding evidence.

The paper under review occupies a borderline position. Its strengths (novel cross-dataset capability, intellectual honesty about source not being inherently better, careful experimental design) are real and substantive. Its weaknesses (overclaimed headline results, narrow experimental scope, modest effect sizes) are also real but not fatal — the paper does demonstrate something meaningful. The main issue is tone and framing: if the paper were more conservative about its claims, it would read more favorably.

This aligns with the 5.5–6 range from the calibration set — borderline papers with genuine contributions but scope-inflated claims receive mixed-to-marginal scores. This paper is slightly stronger than the MEG paper that was withdrawn because its cross-dataset result is structurally novel and better supported; it is weaker than papers that fully back their claims with evidence (7+).

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>