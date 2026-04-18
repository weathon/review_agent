Now I have a good understanding of similar papers and their scores. Let me synthesize the review.

Key calibration points:
- Causal representation learning papers with flawed theoretical claims and synthetic evaluations tend to score 3-5 (OatZMyMuIo: 5,5,3,3; fHZ04oyEed: 5,3,3,3; y2qZhSTtzU: 3,3,3; JttlL9xosQ: 3,3,5,5)
- The front-door adjustment paper (wFf9m4v7oC) with similar theoretical issues got 5,6,6,6 → poster
- Papers with strong OOD generalization + PLMs (d6H4RBi7RH: 8,8,5 → accept poster)

The paper under review has:
- A novel idea (using PLM pre-trained vs. fine-tuned as paired environments for front-door adjustment)
- But serious theoretical issues with Theorem 2 and the front-door derivation
- Strong assumptions that are unvalidated
- Narrow, synthetic evaluation
- Missing standard DG baselines

This aligns more with the lower-scoring causal papers. Let me write the review.</think>## Summary

This paper proposes Causal Transfer Learning (CTL), a method that uses pre-trained and fine-tuned language model representations as "paired environments" to identify causal features via invariant representation learning, then applies a front-door causal adjustment using token-level local features as mediators for out-of-domain (OOD) generalization from single-domain data. The approach is motivated by do-calculus and the front-door criterion, with formal identification results (Theorems 1 and 2), and is evaluated on semi-synthetic and constructed "real-world" sentiment analysis tasks.

## Strengths

- **Novel and creative framing**: The idea of treating pre-trained and fine-tuned PLM representations as paired observations from different "environments" for causal identification (Assumption 2) is inventive. This sidesteps the common requirement for multi-domain data by leveraging the PLM itself as an augmentation source, which is practically appealing.

- **Principled ablation design**: The CTL-N, CTL-C, and CTL-Φ variants allow clear disentanglement of the roles of causal features, local features, and the front-door adjustment. The consistent pattern—CTL-Φ collapses under shift while CTL remains stable—provides interpretable evidence about what drives the method's behavior (Tables 1 and 2).

- **Consistent empirical improvements**: CTL outperforms SFT, SWA, and WISE across all OOD shift magnitudes, with the margin widening as the shift intensifies (e.g., OOD 10% F1 of 58.40 vs. 49.24 for SFT on Yelp; 49.22 vs. 37.78 on the real-world task).

- **Well-structured theoretical presentation**: Sections 3–4 lay out the causal motivation (non-transportability of P(Y|X), identifiability conditions) transparently, making the assumptions explicit and the logical flow clear.

## Weaknesses

### Fatal

- **The front-door identification formula (Theorem 2, Eq. 1) is incorrect or insufficiently justified, undermining the paper's central claim.** The theorem claims that P(y|do(x)) equals the sum in Eq. (1), but the proof has multiple gaps: (a) The step P(y|do(s,c)) = P(y|do(c)) via Rule 3 of do-calculus requires that S has no effect on Y given C, which must be established from the causal graph but is not argued rigorously—especially problematic since Φ, which descends from R₁ (which depends on S₁), is present. (b) The transition to the front-door expression ∑_{Φ'} P(y|Φ',c)P(Φ') is not a standard application of the front-door criterion. In a canonical front-door adjustment, P(y|do(x)) = ∑_m P(m|x) ∑_{x'} P(y|x',m)P(x'), but Eq. (1) conditions only on Φ' and c (not x') in the first term, and the factorization P(Φ') = ∑_{x'} P(Φ'|x')P(x') is asserted without justification under the assumed SCM with confounders U, S, and σ. (c) The algorithmic implementation does not match Eq. (1): Algorithm 2 shuffles Φ within a mini-batch to approximate Φ' and computes predictions from these samples, but never explicitly estimates P(x') or performs the marginalization over x'. The correspondence between the theoretical estimator and the implemented shuffling procedure is simply not established. Since the paper's core narrative—computing a principled causal estimator P(y|do(x)) via front-door adjustment—rests on this theorem, its unsoundness fundamentally undermines the claimed theoretical contribution. At best, the method is a heuristic ensemble/denoising trick that happens to work empirically.

### Major

- **Key identification assumptions are extremely strong and empirically unvalidated.** The entire framework depends on four assumptions, of which Assumptions 2 and 4 are particularly demanding: Assumption 2 requires that pre-trained (R₀) and fine-tuned (R₁) representations share the same causal factor C but differ only in spurious factors. Fine-tuning can substantially modify semantic representations, not just "spurious" ones—no probing or perturbation analysis is provided to validate this. Assumption 4 requires that C fully mediates Φ→Y, which is architecture-dependent (patch splitting + MLP) and essentially unverifiable. The paper provides no empirical diagnostic—e.g., probing whether C is invariant across environments, or whether Φ contains label-predictive information beyond C. Without such validation, it is unclear whether the causal story corresponds to reality or whether the method is, as the harsh critic observes, "a heuristic that benefits from mixing pretrained and fine-tuned representations and from a kind of feature dropout via shuffling Φ."

- **The evaluation is narrowly scoped to artificially constructed distribution shifts.** Both "semi-synthetic" and "real-world" experiments inject spurious correlations by inserting literal tokens (stopwords or "amazon.xxx"/"yelp.yyy" strings) into input text. This makes the spurious features trivially local, exactly matching the architecture's patch-based Φ construction. The paper acknowledges this limitation ("PLMs are already highly resilient to perturbations in text inputs, and introducing spurious information at the input level requires significant effort"), but still claims "superior generalizability ... compared to existing approaches" and "robust causal representation learning." The evaluation does not test naturally occurring distribution shifts (e.g., topic, genre, or register shifts), where spurious features are more diffuse and semantic. The "real-world" experiment's label is misleading given it uses the same synthetic injection mechanism.

- **Missing relevant baselines for domain generalization and causal/invariant learning.** The paper compares against SFT0, SFT, SWA, and WISE—generic robustness baselines—but not against established single-domain DG or causal methods such as IRM, GroupDRO, CORAL, or other causality-inspired baselines that address the same problem setting. Without these comparisons, it is unclear whether CTL's gains come from genuine causal identification or from simple regularization effects (e.g., ensembling frozen/fine-tuned representations or shuffling local features).

### Minor

- **The label-conditioned sampling in Section 5.1 is insufficiently motivated.** The paper samples x̃ from P̂ conditioned on its label y for training, stating this leads to "more stable performance," but no justification or ablation is provided for why this resampling is necessary or how it connects to the causal framework.

- **The entropy maximization in Eq. (3) lacks explicit derivation from Theorem 1 / Von Kügelgen et al. (2021).** While intuitively reasonable (preventing collapse), the balance between alignment and entropy terms is not analyzed, and no sensitivity study is provided for how this tradeoff affects the learned representation C.

- **CTL underperforms SFT in-distribution.** In Table 2 (real-world), CTL achieves ID F1 of 90.59 vs. SFT's 91.39—a small but consistent cost of the causal adjustment for in-domain performance, which should be discussed more explicitly.

### Trivial

- The paper uses "real-world experiment" (Section 6.2) for the Amazon/Yelp dataset with injected strings, which is more accurately described as "semi-synthetic with domain-indicator injection."

## Nice-to-Haves

- Qualitative analysis (e.g., attention maps, token-level probes, or t-SNE visualizations) showing what C vs. Φ actually encode, which would validate or at least provide evidence for Assumptions 2 and 4.
- Evaluation on at least one natural OOD NLP benchmark (e.g., MultiNLI→HANS, CivilComments), which would significantly strengthen claims about broad applicability.
- Ablations on the number of patches (currently 10), the inference sample size K, and the balance between alignment and entropy in Eq. (3).
- Comparison with standard DG baselines (IRM, GroupDRO, CORAL) to contextualize CTL's contribution relative to the broader invariant learning literature.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Claim that SWA/WISE hyperparameters are unfair (WISE using default 0.5 mixing):* The paper states it used a default of 0.5, and this is the method's standard configuration. Per the rules, if baselines are configured in their standard way (potentially favoring the baseline), this is fine, not a criticism of the paper.

- *Concerns about code availability or reproducibility of Theorem 1 from Von Kügelgen et al. (2021):* The paper cites this result; per the rules, we assume cited works exist and are available. Whether Theorem 1's conditions perfectly match the PLM setting is a substantive concern captured in the main review.

- *Request for confidence intervals or error bars beyond 5 runs:* For benchmarks of this type (single-run evaluation with random seeds), reporting mean over 5 runs is standard practice. Demanding more statistical rigor is a nice-to-have, not a weakness.

## Novel Insights

The paper introduces a genuinely interesting heuristic—leveraging the pre-trained vs. fine-tuned representations of a PLM as "paired environments" for invariant representation learning and using token-level local features as mediators in a front-door-style adjustment. However, the gap between the formal causal claims (front-door identification, do-calculus derivation) and what the algorithm actually computes (feature alignment + batch-level shuffling) is severe. The method's empirical success on simple, engineered spurious correlations likely stems from the regularizing effect of ensembling two representations and decorrelating local features via shuffling, rather than from computing a valid causal estimand. This decomposition of what works (representation ensembling + feature shuffling as regularization) versus what is claimed (front-door causal adjustment) is the most important insight for the community.

## Suggestions

- Either rigorously prove Theorem 2 (including all conditional independence assertions and the validity of the factorization) or tone down the claim to "inspired by front-door adjustment" rather than "computing P(y|do(x))".
- Validate Assumptions 2 and 4 by probing whether C is invariant across distribution shifts and whether Φ captures spurious vs. causal features (e.g., via controlled perturbation experiments).
- Add standard DG baselines (IRM, GroupDRO) and evaluate on at least one natural distribution shift benchmark.
- Rename Section 6.2 from "real-world experiments" to something more accurate, given the synthetic injection mechanism.

## Score and Decision

**Calibration:** I compared against papers at similar points on the quality spectrum:

- **OatZMyMuIo** (Causal Representation Learning for Cross-Domain Predictions): Similar causal invariance claims, questionable identifiability, synthetic evaluation. Scores: 5,5,3,3 → **Rejected**.
- **fHZ04oyEed** (Representation Learning from Interventional Data): Only ERM baselines, weak connection between theory and method. Scores: 5,3,3,3 → **Rejected**.
- **JttlL9xosQ** (RetroTune: Mitigating Spurious Features): Heuristic design without theoretical guarantees, limited synthetic evaluation, missing baselines. Scores: 3,3,5,5 → **Rejected**.
- **wFf9m4v7oC** (Front-Door Adjustment with iVAE): Front-door theory with identifiable VAE, but gap between theory and implementation. Scores: 5,6,6,6 → **Accepted (poster)**.
- **d6H4RBi7RH** (Spurious Feature Diversification for OOD): Strong theoretical + empirical contribution on PLM-based OOD. Scores: 8,8,5 → **Accepted (poster)**.

This paper shares core weaknesses with the rejected causal papers (flawed theoretical derivation, unvalidated strong assumptions, narrow synthetic evaluation, missing standard baselines) while having a creative and interesting core idea. The fatal gap in Theorem 2 puts it below the front-door iVAE paper (which had correct theory, albeit with an implementation gap), and the narrow evaluation puts it well below papers like d6H4RBi7RH that have stronger empirical grounding.

MY FINAL SCORE: <pineapple>3.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>