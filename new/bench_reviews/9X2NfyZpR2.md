Now I have enough context. Let me carefully verify the paper's key claims and synthesize the final review.

## Summary

TbLTA proposes the first weakly-supervised framework for dense long-term action anticipation (LTA), training exclusively from video transcripts (ordered action lists without temporal boundaries) rather than dense frame-level annotations. The method combines a transformer encoder-decoder architecture with an ATBA-based temporal alignment module to generate pseudo-labels, CTC loss for transcript-level consistency, cross-modal attention for grounding video features with transcript semantics, and a CRF loss for enforcing temporal coherence in future predictions. Experiments on Breakfast, 50Salads, and EGTEA show competitive results with some fully supervised baselines, particularly on Breakfast.

## Strengths

- **Novel and important problem formulation**: TbLTA is the first approach to address dense LTA under purely transcript-level weak supervision, tackling a genuine scalability bottleneck. Prior LTA work has exclusively required dense frame-level annotations, and the only weakly-supervised prior (Zhang et al., 2021) still used temporally localized annotations. This is a meaningful contribution to the LTA research landscape.

- **Competitive deterministic results on Breakfast**: The deterministic TbLTA achieves 29.03% average MoC on Breakfast, genuinely competitive with supervised baselines (ActFusion: 28.45%, FUTR: 26.59%). At 30% observation, deterministic TbLTA outperforms all supervised baselines (40.28 vs 35.79 MoC). This demonstrates that transcript-level semantic structure carries substantial predictive power for procedural activities.

- **Well-structured multi-component architecture with informative ablations**: The combination of CTC, cross-modal attention, CRF, and duration priors forms a coherent pipeline. Ablations in Tables 3-4 show each component contributes non-trivially: cross-modal attention yields ~5.7 MoC improvement on Breakfast, CRF is critical for longer horizons (~5.3 drop on 50Salads), and CTC stabilizes pseudo-labels.

- **Self-supervised duration estimation**: The affinity-based duration loss using momentum-based class-wise priors is a clever mechanism that avoids requiring temporal ground truth while providing meaningful duration regularization.

## Weaknesses

### Major:

- **The "stochastic" protocol underpinning the strongest results is not adequately explained.** Table 1 reports "TbLTA* – Mean" and "TbLTA* – Top1" under a stochastic protocol, and the Top-1 numbers are the ones most prominently highlighted (e.g., Breakfast avg 37.15 vs deterministic 29.03). However, the methodology section never clearly defines how stochastic sampling works: what is sampled, how many samples, how "Mean" and "Top1" are computed, or which probabilistic mechanism generates the diverse futures. The paper references "the stochastic protocol of Abu Farha & Gall (2019) in the supp. mat." without further detail in the main text, and the CRF (which the paper calls a "stochastic variant") is standard discriminative NLL training decoded via Viterbi at inference. The lack of specification means the headline results cannot be independently verified from the main paper, and the contrast with "deterministic" supervised baselines in the same table is misleading when the stochastic protocol selects the best sample from multiple draws.

- **The claim of "pure transcript-only supervision" obscures a heavy dependency on the ATBA alignment module.** The paper's core framing—"trained exclusively from transcripts"—while technically accurate (transcripts are the only human annotation), underplays that TbLTA relies on ATBA (Xu & Zheng, 2024), a sophisticated existing weakly-supervised TAS method, to produce the frame-level pseudo-labels that drive all downstream supervision. ATBA is treated as a black box, and no ablation replaces it with simpler alignment strategies (e.g., CTC-only, uniform expansion) to isolate whether TbLTA's specific architectural contributions (cross-attention, CRF, duration loss) genuinely add value, or whether the performance stems primarily from ATBA's pseudo-label quality. This matters because the central claim is that "transcripts alone suffice for dense LTA"—what the evidence actually shows is that "transcripts + a strong transcript-based TAS method suffice for dense LTA."

- **CRF supervision target Y_LTA is ambiguously specified.** In Section 3.2.3, the CRF loss is described with "Y_LTA the target anticipate transcript," and the training objective uses the "negative log-likelihood of the ground-truth anticipation sequence." However, the paper's weak supervision setting provides only a flat transcript Y without boundaries. While ATBA partitions Y into Y_obs and Y_future, the paper does not explain how this label-level sub-transcript becomes T_pred frame-level targets for the CRF. If pseudo-labels Ŷ_pred serve as targets, the phrase "ground-truth" is a misnomer, and the CRF loss trains against its own alignment module's noisy outputs—a regime whose implications are not discussed.

- **Performance gap on 50Salads undermines generalizability claims.** Deterministic TbLTA achieves only 20.92% average MoC on 50Salads vs. supervised ActFusion's 28.39%—a gap of ~7.5 points. Even the stochastic Top-1 (28.51%) barely matches ActFusion. The paper attributes this to "denser action distributions and frequent transitions" but does not analyze the failure modes or propose targeted improvements, leaving the reader uncertain whether transcript-only supervision is viable beyond highly procedural datasets with strong regularities.

### Minor:

- **The cross-attention mask M construction is underspecified.** The paper defines M ∈ {0,1}^{N×T} as restricting "each action a_i to a temporal neighborhood around its predicted occurrence" but does not specify the neighborhood size, whether it is fixed or adaptive, or how it interacts with uncertain pseudo-labels. This affects reproducibility of the key cross-modal mechanism.

- **No analysis of pseudo-label quality or failure modes.** The entire pipeline supervises both TAS and LTA heads with ATBA pseudo-labels, yet their alignment accuracy, boundary precision, and error distribution are never measured. Understanding whether LTA errors originate from the alignment module or the anticipation decoder is crucial for guiding future improvements.

- **EGTEA evaluation is narrow.** The paper restricts EGTEA to verb-only prediction and compares against only two supervised baselines (Timeception, Anticipatr), where TbLTA falls substantially behind (65.37 vs 76.80 mAP). The claim that the method "proves competitive on rare classes" is stated but not shown in Table 2 with a per-frequency breakdown for TbLTA.

- **No transcript-only oracle baseline.** A simple baseline that predicts the most common transcript for a given activity class would establish how much performance is attributable to transcript regularities versus visual anticipation. Its absence makes it hard to calibrate the contribution of the model's visual processing.

### Trivial:

- The section numbering for losses (3.2.3 → 3.2.2 → 3.2.1) appears in reverse order, which may confuse readers.

## Nice-to-Haves

- Analysis of pseudo-label quality (frame-level accuracy of ATBA alignment) and its correlation with final LTA performance, which would diagnose whether alignment or anticipation is the bottleneck.
- A shuffled-transcript ablation to quantify how much the model relies on transcript ordering structure vs. learned visual features.
- Comparison on EGTEA with full verb-noun action classes rather than verbs only.
- Runtime/computational cost comparisons with supervised baselines.

## Removed Points

- **Claim that the stochastic variant is "not a defined method" and "not supported by the architecture"**: The stochastic evaluation protocol (sampling multiple futures from the CRF distribution, taking best/mean) is a well-established paradigm from Abu Farha & Gall (2019). While its explanation in the main text is insufficient, the mechanism itself (sampling from the learned CRF model) is standard and plausible. The issue is one of presentation clarity, not of the method being undefined or impossible. → *Retained as a Major weakness about inadequate explanation, but removed the claim that the method is fundamentally unsupported.*

- **Demand for comparison with PALM (Kim et al., 2024)**: The reviewer suggests comparing with PALM, a language-based anticipation method. PALM operates in a fundamentally different setting (using LLMs with in-context learning, predicting symbolic sequences without dense frame-level outputs) and is not directly comparable to the dense frame-level LTA task defined in this paper. This comparison would be inappropriate. → *Removed as out-of-scope.*

- **Demand for confidence intervals / statistical tests**: Single-run evaluation is standard practice in the LTA/TAS community on these benchmarks. Requesting statistical significance tests for established benchmarks with standard splits is not the community norm. → *Removed as not standard in this setting.*

- **Demand for end-to-end visual feature learning vs. pre-extracted I3D features**: Using pre-extracted I3D features is the standard protocol on Breakfast, 50Salads, and EGTEA. Criticizing this is asking the paper to deviate from community standards. → *Removed as out-of-scope.*

- **"Not even a paper" / fundamental invalidation claims**: The paper presents a coherent method, clearly defined problem, and extensive experiments. The harsh critic's suggestion that the paper's core claims are fundamentally invalid is disproportionate. The contributions are real; the issues are about overclaiming and insufficient clarity. → *Removed.*

- **Concern about training with full video at training time but partial at inference**: This is standard practice in LTA (also used by FUTR, ActFusion) and is explicitly acknowledged by the authors ("Following previous work Gong et al. (2024), we segment the full video during training"). → *Removed as the paper already addresses this.*

- **Lack of comparison under the same weakly-supervised setting**: The only prior weakly/semi-supervised LTA method (WS-DA, Zhang et al. 2021) uses a fundamentally different form of weak supervision (semi-supervised with observed-frame labels), so a direct apples-to-apples comparison is impossible. The paper acknowledges this. → *Removed as unfair comparison concern—the asymmetry actually favors the baseline.*

## Novel Insights

The most interesting empirical finding is that transcript-based weak supervision can match or exceed fully supervised LTA on Breakfast at 30% observation (40.28 vs 35.79 deterministic MoC). This suggests that for highly procedural activities, the strong prior provided by transcript ordering may partially compensate for the lack of frame-level annotations—transcripts encode "what happens next" structure that dense labels for the observation window alone may not capture as effectively. However, the 50Salads results reveal this advantage collapses for less structured activities, pointing to a clear boundary condition: transcript-based supervision is most effective when activities have strong sequential regularities (a finding that parallels insights from the TAS literature but has not been demonstrated for LTA).

## Suggestions

- **Clearly define the stochastic evaluation protocol in the main text**: Specify the number of samples, the sampling procedure from the CRF, and how Top-1/Mean are computed. Make stochastic and deterministic results visually distinct in tables (e.g., separate panels) and explicitly state which protocol is the primary one.
- **Add an ATBA replacement ablation**: Replace ATBA with a simpler alignment (e.g., CTC-only or uniform expansion) to isolate the performance contribution of TbLTA's own components vs. ATBA's pseudo-label quality. This would substantially strengthen the "transcripts alone" claim.
- **Clarify the CRF training target**: Explicitly state whether Y_LTA in the CRF loss refers to the transcript sub-sequence Y_future or the frame-level pseudo-labels Ŷ_pred, and explain how the label-to-frame mapping works.
- **Tone down claims of competitiveness with fully supervised methods**: Report deterministic results as the primary comparison and present stochastic results separately with clear caveats about the sampling advantage of Top-1 selection.

## Score and Decision

Calibration against retrieved papers:
- **AntGPT** (LTA, accepted poster): scores 5/8/6/6, novel LLM-based framing for LTA with strong results. TbLTA has a comparable novelty of problem formulation but weaker methodological clarity.
- **Action Sequence Augmentation** (accepted poster): scores 6/6/8/6, clear contribution with good experiments. TbLTA has a more significant novelty gap but comparable experimental depth.
- **Actions-to-Action** (rejected): scores 5/3/3/6/5, incremental novelty with overclaims. TbLTA has a more genuine novelty claim but similar issues with presentation and overclaiming.
- **Fine-grained Separation for Point-Level TAL** (rejected): scores 5/3/6/3, incremental on top of LACP with pseudo-labels. TbLTA similarly depends on ATBA pseudo-labels but tackles a genuinely new problem setting.

TbLTA represents a meaningful first step for an important and underexplored setting (transcript-only LTA), with real competitive results on Breakfast. However, the combination of (1) an inadequately explained stochastic protocol supporting the strongest results, (2) overclaiming about "pure transcript supervision" that obscures heavy ATBA dependency, (3) an ambiguous CRF target specification, and (4) a significant generalizability gap on 50Salads, collectively prevent these results from being fully convincing as presented. The paper would benefit substantially from revisions addressing the presentation issues and adding ATBA ablations.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>