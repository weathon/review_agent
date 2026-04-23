Now I have all the information I need. Let me compile the final review.

## Summary

SimE proposes a class-incremental learning framework that fine-tunes CLIP image encoder with adapters only on the first task's base classes, then freezes all weights and computes class prototypes (mean feature centroids) for all subsequent tasks. The paper also introduces Multi-Adapter (multiple adapter sub-modules within transformer blocks) and reports a "remarkable phenomenon" that more intra-block adapter connections do not always improve incremental learning performance.

## Strengths

- **Strong empirical results under certain configurations**: SimE achieves high accuracy on CIFAR-100 and TinyImageNet benchmarks, substantially outperforming prior methods in Table 1. Table 2's first row (0 parameters) also provides a useful frozen-CLIP baseline showing that adapter fine-tuning on base classes contributes approximately +6.6% Last accuracy over frozen CLIP + prototypes on ViT-B/16 CIFAR-100 10 steps (77.10% vs. 70.08%).

- **Transparent design disclosure**: Figure 1(C) clearly shows that only Task 1 involves training and subsequent tasks only compute prototypes, allowing readers to understand the method's actual operation despite the broader framing.

- **Systematic CLIP backbone comparison**: Tables 3–4 provide controlled comparisons across pre-training datasets (WIT-400M, LAION-400M, LAION-2B, DataComp-1B, CommonPool-1B) and ViT architectures (ViT-B/16, ViT-B/32, ViT-L/14), offering practical guidance for practitioners.

## Weaknesses

### Fatal

None.

### Major

- **SimE does not meaningfully perform incremental learning after Task 1, undermining the paper's core framing.** Section 3.1 and Figure 1(C) explicitly state that after Task 1, all weights are frozen and only prototypes (class centroids) are computed for new classes. The model never updates any parameters on tasks 2–T. While the paper frames this as an incremental learning contribution ("learns new tasks while preserving previously acquired knowledge"), no learning of representations occurs after the first task—only nearest-centroid matching on a fixed feature space. This makes comparisons with methods that actually confront the stability–plasticity tradeoff (ZSCL, LwF-VKD, CoOp) fundamentally asymmetric. SimE trivially avoids catastrophic forgetting by never updating, which is not a methodological advance but a design constraint that eliminates the core challenge of incremental learning.

- **The main comparison in Table 1 appears to use a stronger backbone (ViT-L/14 + LAION-2B) for SimE while comparison methods use ViT-B/16 + WIT-400M, making accuracy gains largely attributable to backbone strength rather than the method.** The table caption states "the remaining methods use CLIP ViT-B/16 as the backbone, where † indicates the result based on the CLIP ViT-L/14 pre-trained on Laion-2B," yet the SimE(Ours) row shows 91.66%/86.03% on CIFAR-100 10 steps—far exceeding what Table 2's ViT-B/16 results show (best: 85.94%/77.10%, which exactly matches ZSCL's ViT-B/16 result). Table 4 shows ViT-L/14 achieving 88.79%/81.44% with WIT-400M, and Table 3 shows LAION-2B + ViT-B/16 at 88.34%/81.33%. The Table 1 SimE numbers are only achievable by combining LAION-2B + ViT-L/14, a configuration not shared by any comparison method. This means on equal backbones (ViT-B/16), SimE matches rather than surpasses ZSCL. The paper does not clearly communicate this critical fact.

- **Efficiency claims are artifacts of not training after Task 1, not of a methodological advance.** The abstract celebrates "only thousands of parameters and no memory bank," but these gains directly result from never updating the model after Task 1. A method that trains on one task is trivially more parameter-efficient and GPU-efficient than methods that train on every task. The parameter count claim also misleads by counting only adapter parameters while the frozen CLIP backbone (tens of millions of parameters) is required at inference. Figure 4(b)'s GPU comparison is similarly misleading—SimE uses less GPU time because it does less training.

### Minor

- **The Multi-Adapter "remarkable phenomenon" is weakly supported by sub-1% differences with no error bars or significance tests.** Table 2 shows that at 10 steps, the best config (AdaptMLP+AdaptAtten, 1.19M) achieves 77.10% Last vs. the full 3.57M config at 76.51%—a 0.59% gap easily attributable to noise. At 50 steps the trend reverses (75.16% vs. 73.77%), but the differences remain small. No standard deviations, confidence intervals, or significance tests are reported. Without repeated runs or statistical analysis, calling this a "remarkable phenomenon" is overclaimed. Furthermore, since the model is frozen after Task 1, this is really about how well different adapter configurations fine-tuned on base classes generalize to unseen classes—a transfer learning question, not an incremental learning one.

- **Equation 3 misrepresents ViT block processing.** The equation E(x) = Σ_i^B (g_i(φ_i, f_i(θ_i, x_i)) + d_i(η_i, x_i)) sums outputs across all blocks as if each block independently processes the input. In a standard ViT, blocks are sequential: block i takes the output of block i−1, and the final output is the output of the last block, not the sum. This notation error compounds in Equation 7's double summation. While this likely does not affect the implementation, it undermines confidence in the formal analysis and suggests insufficient care in the mathematical presentation.

- **The systematic CLIP study (Tables 3–4) confirms the unsurprising finding that larger models and larger pre-training datasets improve performance.** While useful for practitioners, the recommendation to use ViT-L/14 on LAION-2B is not a novel insight and does not constitute a research contribution.

## Nice-to-Haves

- Comparison with SimpleCIL/APER (which proposes the same frozen-PTM + prototype-classifier approach) would properly contextualize SimE's contribution and avoid redundancy.
- Experiments with few or dissimilar base classes would stress-test whether the approach works when the base class feature space doesn't generalize well to incremental classes.
- Allowing adapter training on later tasks (with a forgetting mitigation strategy) could demonstrate whether SimE's simplicity is a ceiling or a floor for performance.
- Feature space visualizations (e.g., t-SNE) showing how adapter fine-tuning on base classes shapes representations for incremental classes would add insight.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Equation 3 suggests the authors may not have a clear model of the architecture they are modifying"** (Harsh Critic): This overclaims the severity of the notation issue. The equation is poorly written but does not imply the authors don't understand their own architecture—the implementation likely uses standard ViT sequential processing. The notation issue is minor, not evidence of architectural misunderstanding.

- **"Missing comparison against frozen CLIP + prototype classification without adapter fine-tuning"** (Harsh Critic): This baseline IS present in Table 2's first row (0 parameters: 79.69/70.08), though it is buried in the ablation rather than highlighted as a main comparison. The critic's claim that this baseline is entirely missing is factually incorrect.

- **"Missing comparison against full fine-tuning on base classes + prototype classification"** (Harsh Critic): While this would be informative, it's a nice-to-have rather than a critical missing baseline, as full fine-tuning risks destroying CLIP's pre-trained features and is not a fair comparison for an efficiency-focused method.

- **"Abstract cherry-picks numbers (9.6%, 5.3%)"** (Harsh Critic): While the specific numbers may be selected from favorable configurations, the improvements are large across all configurations, so cherry-picking specific numbers is a minor presentation issue rather than a substantive problem.

- **"Parameter count claim is misleading because it only counts adapter parameters"** (Harsh Critic and Strength Finder): While technically the paper says "thousands of parameters," it's clear from context this refers to trainable parameters. The inference cost of the full CLIP backbone is a separate concern. This is a minor overclaim, not a major one.

## Novel Insights

The review reveals that SimE's claimed large accuracy improvements over prior CIL methods in Table 1 are largely attributable to using a significantly stronger backbone (ViT-L/14 on LAION-2B) than the comparison methods (ViT-B/16 on WIT-400M), not to the method itself. On the same backbone, Table 2 shows SimE matching rather than surpassing ZSCL. This backbone asymmetry, combined with the trivial avoidance of catastrophic forgetting (by never updating), means SimE's actual contribution reduces to: "fine-tuning CLIP adapters on base classes plus prototype classification is simple and effective"—a finding that closely parallels the already-published SimpleCIL work, which is not cited.

## Suggestions

- Clearly state which backbone and pre-training dataset is used for each result in Table 1. Report SimE results with the same backbone (ViT-B/16 + WIT-400M) as comparison methods to isolate the method's contribution from backbone strength.
- Cite and compare with SimpleCIL/APER, which proposes the same frozen-PTM + prototype-classifier paradigm.
- Report standard deviations across multiple runs for the Multi-Adapter ablation to establish whether the non-monotonic behavior is statistically significant.
- Rewrite Equation 3 to correctly represent sequential ViT block processing rather than summing across blocks.

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| SimpleCIL/APER | mrRbIcyouU | 4.75 (Reject) | Same core idea (frozen PTM + prototype classifier). SimE adds adapter fine-tuning on base classes but doesn't cite this work. SimE has similar comparability concerns (prototypes ≈ stored data). SimE is marginally better due to adapter addition but worse due to unfair backbone comparison. |
| "How far without finetuning" | H6XYCIlZdo | 3.0 (Reject) | Even simpler frozen-feature approach. SimE is more complete but shares the same core limitation. |
| SD-LoRA | 5U1rlpX68A | 7.5 (Oral) | Actually trains incrementally with novel decoupled LoRA, has theoretical analysis. Far more sophisticated than SimE. |
| SEED | sSyytcewxe | 7.0 (Poster) | Selective expert training for CIL, genuine incremental learning. Much more novel. |
| OVOR | FbuyDzZTPt | 6.0 (Poster) | Learnable prompts + VOR, actually trains on each task, simpler than SD-LoRA but still genuinely incremental. |
| MetaAdapter | 88hh5GtLBJ | 5.4 (Reject) | Adapter + meta-learning for FSCIL, updates on new tasks. More sophisticated than SimE. |
| CIL self-training integration | 10fsmnw6aD | 2.5 (Reject) | Simple combination of existing ideas with poor novelty. SimE is somewhat above this level. |

SimE's core idea (frozen backbone + prototype classification) closely parallels SimpleCIL (4.75, rejected), and the paper doesn't cite it. The main accuracy gains in Table 1 appear attributable to using a much stronger backbone rather than the method itself. On the same backbone (ViT-B/16), SimE matches rather than surpasses ZSCL. The Multi-Adapter "phenomenon" is weakly supported. The efficiency claims are trivial consequences of not training after Task 1. The paper does provide useful empirical guidance on CLIP backbones, but this is not novel enough to carry the paper. SimE is slightly below SimpleCIL in terms of contribution honesty (SimpleCIL at least clearly labeled itself as a baseline and proposed new benchmarks), making it appropriate to score somewhat below 4.75.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>