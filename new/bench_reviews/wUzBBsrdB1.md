Now I have a thorough understanding of the paper and the reviews. Let me synthesize my final review.

## Summary

This paper investigates how the L0 hyperparameter (average number of active latents per token) affects the quality of features learned by Sparse Autoencoders. Using toy models with ground-truth features, the authors demonstrate that too-low L0 causes SAEs to "cheat" by mixing correlated features (feature hedging), and that such mixed-feature SAEs can achieve better reconstruction than ground-truth SAEs at the same L0—making sparsity–reconstruction tradeoff plots misleading. Too-high L0 also harms features via degenerate solutions. The authors propose decoder pairwise cosine similarity (c_dec) as a proxy metric for identifying appropriate L0, validated in both toy models and on Gemma-2-2b and Llama-3.2-1b.

## Strengths

- **Important and timely conceptual contribution.** The demonstration that MSE loss incentivizes feature hedging at low L0—that a "ground-truth" SAE achieves worse reconstruction than a corrupted SAE at sub-true L0—is a genuinely counterintuitive and significant finding. It directly challenges a widely held assumption in the SAE community that lower L0 is uniformly fine as long as reconstruction is acceptable. This is a clear and valuable insight.

- **Clean, well-designed toy model experiments.** The toy setups with orthogonal features and controllable firing probabilities make the feature-mixing phenomenon directly visible and intuitively understandable. The positive/negative correlation experiments (Figure 2 vs 3) cleanly demonstrate the direction of mixing. The ground-truth SAE construction enables comparisons that would be impossible in real LLMs.

- **Practical diagnostic metric.** The c_dec metric (pairwise decoder cosine similarity) is simple to compute from any trained SAE, requires no ground truth, and shows clear minima at the true L0 in toy models and at the "elbow" near peak sparse probing performance in LLMs. Even with its limitations acknowledged by the authors, it provides a useful practical tool for detecting clearly-too-low L0 regimes.

- **Honest acknowledgment of limitations.** The paper explicitly states (Section 6): "the metric can sometimes remain nearly flat for a wide range of L0" and "we do not view this as a perfect guide." Section 4.2 notes that "there is no reason why every latent has the same firing threshold, so there is likely a range of L0s where some latents are firing more than they ideally should while other latents are firing less than they ideally should."

## Weaknesses

### Major

- **The "true L0" / "correct L0" framing overclaims relative to the evidence.** The abstract states: "if L0 is not set correctly, the SAE fails to disentangle the underlying features of the LLM" and "L0 must be set correctly to train SAEs with correct features." In toy models, there is a well-defined true L0 by construction. In LLMs, however, the evidence only supports the existence of an *approximate region* where metrics look better—not a unique correct value. The paper's own LLM results show this: (1) Gemma-2-2b layer 5 has a "long shallow region" in c_dec where identifying a unique optimum is ambiguous; (2) JumpReLU and BatchTopK SAEs trained on the same data give different c_dec minima (~200 vs ~250–300) and different optimal L0 for sparse probing. The authors acknowledge a range of plausible L0s in Section 4.2 but the abstract, introduction, and section headings persistently frame L0 as having a single correct value. This mismatch between framing and evidence is consequential: it shapes how readers understand the contribution.

- **The claim that sparsity–reconstruction tradeoff plots are "not a sound method of evaluating SAEs" is overstated.** The paper demonstrates a valid and important specific failure mode: at too-low L0, reconstruction-based evaluation can prefer a corrupted SAE over a ground-truth one. However, this shows that sparsity–reconstruction plots can be *misleading under specific conditions* (capacity below the true feature count), not that the entire evaluation paradigm is "unsound." The paper's own ground-truth SAE would dominate in reconstruction at or near the true L0, so the failure is specifically in the too-low-L0 regime. Sparsity–reconstruction curves are typically used to compare SAEs across a range of L0 values and often supplemented with other metrics. The paper does not analyze whether multi-criteria usage of these plots avoids the identified pitfall.

- **The claim that "most commonly used SAEs have an L0 that is too low" is under-supported.** The abstract and discussion make this a headline finding, but the supporting evidence is: (a) toy models showing low L0 is harmful (valid but not directly about existing SAEs), (b) two small models at a few layers with one SAE width, and (c) "a cursory search of open source SAEs on Neuronpedia" (Section 6/Appendix A.13) that reports L0 values but no evaluation of those SAEs' quality (c_dec, sparse probing, or interpretability). The jump from "many SAEs use L0<100" to "they are too low" is not demonstrated, as the optimal L0 likely varies by model size, layer depth, SAE width, and training distribution.

- **Limited LLM experimental coverage.** All LLM experiments use a single SAE width (h=32768) on two small models (Gemma-2-2b at layers 5 and 12; Llama-3.2-1b at one layer), trained on 500M tokens from The Pile. It is unclear how well c_dec minima and their correspondence with sparse probing generalize to larger models, different layers, different widths, or different training data. Since the paper's practical implications target SAE practitioners working with diverse models, this thin coverage is a meaningful gap.

### Minor

- **c_dec lacks a precise, automated selection criterion.** The paper uses an informal "elbow" heuristic to identify the right L0 from c_dec curves. Different readers could draw different elbows, especially given the acknowledged "nearly flat" regions. A reproducible, automated criterion would greatly improve the metric's practical utility (acknowledged in Section 6 and Appendix A.11 as future work).

- **No qualitative interpretability evidence in LLMs.** While c_dec and sparse probing provide quantitative proxies for monosemanticity, the paper provides no inspection of individual latents at different L0s in LLMs to verify that the claimed mixing phenomenon actually manifests. Showing concrete examples of latents degrading into polysemantic mixtures at low L0 (as done beautifully in toy models, e.g., Figure 1) would significantly strengthen the LLM story.

- **Theoretical justification of c_dec deferred to appendix.** The central proposed metric's formal justification is in Appendix A.6, with only intuition given in the main text. Since c_dec is a core contribution, more of the theoretical argument—including its assumptions and potential failure modes—deserves visibility in the main body.

## Nice-to-Haves

- Validating c_dec on larger models (7B+) and across more layers and SAE widths to establish generality.
- Comparing c_dec-guided L0 selection against alternative approaches (e.g., MDL-SAEs, AFA-SAEs) that also aim to address L0 selection.
- Developing an automated procedure for reading c_dec curves rather than the informal "elbow" heuristic.
- Per-latent case studies showing feature mixing in real LLM SAEs at different L0s.
- Error bars or variance estimates across random seeds for k-sparse probing results.
- Investigating how c_dec interacts with SAE width (h), since width and L0 jointly determine effective capacity.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Not yet released" / availability concerns about cited methods**: Reviewers questioned whether JumpReLU SAEs (from Anthropic) and BatchTopK SAEs are properly available. The paper cites them appropriately and they are assumed to exist per review rules. Removed.

- **Demanding reproducibility details (hyperparameters, training logs, seeds)**: The paper provides code in supplementary materials and training details in appendices. Nitpicks about specific training configurations are not substantive. Removed.

- **Formatting and presentation nitpicks**: The harsh critic's section-by-section notes include many formatting comments. These are irrelevant to the contribution. Removed.

- **Demand for user studies or causal intervention experiments**: The paper is an empirical study of SAE training properties, not a human-subjects evaluation. Requesting user studies is scope creep. Moved to nice-to-have.

- **Strawman reading of the sparsity-reconstruction claim**: Some interpretations read the paper as claiming reconstruction is *never* useful, but the paper specifically argues it's misleading at too-low L0. The actual overclaim is about declaring the entire paradigm "unsound" rather than identifying a specific failure mode. Kept the softened version in Major weaknesses.

- **Demanding proofs for what is an empirical paper**: The paper primarily contributes empirical findings with toy and LLM experiments. Demanding rigorous theoretical proofs for c_dec beyond what's in the appendix is beyond the paper's scope. Moved theoretical extension to nice-to-have.

- **Weaknesses from human finder about similar SAE papers that don't directly apply**: Several human-finder-sourced weaknesses (e.g., from "SAGE" or "feature absorption" reviews about single model evaluation) were considered but only kept the ones that are substantively relevant to this paper's claims.

## Novel Insights

The paper's most novel insight is that MSE-based training at low L0 actively *incentivizes* feature hedging—not just tolerates it—because mixing correlated features reduces reconstruction error relative to the ground-truth decomposition. This reframes low-L0 SAE pathology from "slightly suboptimal" to "actively misleading," since standard evaluation metrics would prefer the corrupted solution. The observation that JumpReLU SAEs "stick" near a stable L0 across a range of sparsity coefficients (Figure 7) hints that per-latent adaptive thresholds may naturally mitigate the low-L0 problem, though this is not fully explored. The Section 4.2 observation that L0 can be simultaneously too low for some latents and too high for others is also noteworthy and could reshape how practitioners think about SAE sparsity design.

## Suggestions

- Reframe the "true L0" / "correct L0" language. Replace with "an appropriate range of L0 values" or "the region where c_dec is minimized and features are most monosemantic." Acknowledge explicitly that the optimal L0 likely depends on SAE architecture, width, and layer. The abstract and introduction should reflect this nuanced framing.

- Soften the claim about sparsity–reconstruction plots. Instead of calling them "not a sound method," state that they "can be misleading when L0 is below the level needed to represent the underlying features, because MSE loss incentivizes feature mixing." This is still a strong and important claim, just not a blanket dismissal.

- Back up the "most SAEs have too low L0" claim with actual evaluation. Either systematically evaluate existing SAEs (e.g., Gemma Scope) on c_dec and sparse probing, or qualify the claim as a hypothesis for future work rather than a finding.

- Add at least one qualitative example from an LLM SAE showing feature mixing at low L0 versus cleaner latents at the c_dec-identified L0. This would connect the toy model intuition directly to real models.

## Score and Decision

Calibration: I compared against several SAE papers from human reviews:
- **"SAEs Do Not Find Canonical Units"** (scores 6,8,6,8, avg ~7): A conceptual contribution with some empirical validation. Accepted as poster. This paper has a comparable conceptual contribution (new diagnostic metric, new understanding of L0 pathology) but with more overclaiming.
- **"A is for Absorption"** (scores 8,8,6,8, avg ~7.5): Identified a real phenomenon (feature absorption) in a limited setting, similar profile. This paper was actually rejected, suggesting the field holds even interesting SAE analysis papers to a high bar for completeness.
- **"Scaling and evaluating SAEs"** (scores 10,8,10,10,3, avg ~8.2, oral): Major contribution (TopK SAEs + scaling laws). Far beyond this paper's scope.
- **"Enhancing NN Interpretability"** (scores 3,5,3,5, avg ~4): Weak contribution, no real interpretability evidence. This paper is significantly better.
- **"Decomposing Dark Matter"** (scores 3,3,5,3, avg ~3.5): Unclear definitions, hard to follow. This paper is better.

This paper makes a genuine and important conceptual contribution (low-L0 feature hedging, sparsity–reconstruction can be misleading), with clean toy model evidence and a practical diagnostic metric. However, it significantly overclaims: the "true L0" uniqueness claim is not supported for LLMs; the dismissal of sparsity–reconstruction evaluation goes beyond what the evidence shows; and "most SAEs are too low" is a cursory observation rather than a demonstrated finding. These are not minor phrasing issues—they are woven into the paper's central narrative. The limited LLM experimental scope further weakens the external validity of the claims. Compared to "SAEs Do Not Find Canonical Units" (poster, ~7), this paper has comparable conceptual novelty but weaker empirical support and stronger overclaiming, placing it below that bar.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>