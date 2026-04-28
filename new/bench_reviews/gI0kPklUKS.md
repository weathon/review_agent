## Summary
This paper proposes bilinear MLPs as an architecture enabling weight-based mechanistic interpretability via tensor eigendecomposition, demonstrating interpretable eigenvectors on image classification tasks (MNIST/Fashion-MNIST) and extending the analysis to small language models using SAE-derived feature bases. The method reveals low-rank structure in bilinear layer weights, enables adversarial mask construction without forward passes, and identifies a sentiment negation circuit in a 6-layer transformer.

## Strengths
- **Novel mathematical framework with concrete demonstrations**: The eigendecomposition of bilinear interaction matrices (Section 2-3) provides a rigorous, fully equivalent reformulation of bilinear MLP computations. Section 4.3 demonstrates this can recover the exact ground-truth labeling function (target image similarity) from weights alone without dataset knowledge—a stronger result than typical correlational interpretability methods.
- **Compelling image classification results**: Top eigenvectors for MNIST digits correspond to semantically meaningful stroke components (Figure 2, 3), and the adversarial mask construction (Section 4.4) achieves significant misclassification rates without optimization or forward passes, demonstrating causal utility in shallow networks.
- **Evidence of low-rank structure in LLMs**: Section 5.2 shows output feature activations in bilinear transformers are well-approximated by top 2 eigenvectors with average correlations exceeding 0.75 (Figure 9), suggesting the method scales beyond toy tasks.

## Weaknesses

### Fatal
None

### Major
- **Tension between "weights alone" claims and SAE dependency**: The Abstract claims circuits are identified "directly from the weights alone" and enables interpretability "without using inputs," but Section 5 explicitly requires SAEs trained on activation data to define interpretable input/output feature bases for LLMs. Without SAE projection, eigenvectors exist in the raw residual stream space, which is known to be polysemantic and uninterpretable. The Limitations section (Section 6) acknowledges this ("in deeper models, we rely on features derived from sparse autoencoders"), but this admission contradicts the stronger Abstract/Introduction claims. This undermines the central thesis that bilinear MLPs bypass the primary bottleneck of current interpretability (feature discovery).
- **No causal validation for LLM circuits**: While Section 4.4 demonstrates causal utility via adversarial masks on MNIST, Section 5's sentiment negation circuit lacks equivalent intervention experiments. The circuit is validated primarily through correlation metrics (~0.66-0.76 between low-rank approximation and feature activation, Figure 9), with no ablation showing that zeroing top eigenvectors destroys the circuit's function. Without causal intervention in the LLM setting, the "circuit discovery" claim remains correlational for the setting where interpretability is most needed.

### Minor
- **Limited scaling evidence for "drop-in replacement" claim**: The Abstract positions bilinear layers as an "interpretable drop-in replacement for current activation functions," but experiments use only 6-layer TinyStories and 12/16-layer FineWeb models. Section 2 acknowledges "marginally worse loss when keeping data constant" compared to SwiGLU, with no evidence at 1B+ parameter scales where interpretability is most critical. If interpretability comes at the cost of state-of-the-art performance, the architecture is a specialized research tool rather than a viable replacement.
- **Symmetrization may obscure asymmetric structure**: Section 2 symmetrizes the interaction matrix ($B_{aij} = \frac{1}{2}(w_{ai}v_{aj} + w_{aj}v_{ai})$), merging the distinct roles of W and V matrices. In GLUs, W and V often learn asymmetric roles (e.g., gate vs. value), but the paper does not analyze whether they diverge during training or what interpretability information is lost by symmetrizing.

### Trivial
- **Figure 8 circuit visualization could be clearer**: The sentiment negation circuit diagram would benefit from explicit labeling of eigenvalue magnitudes and clearer indication of which eigenvectors correspond to which feature clusters.

## Nice-to-Haves
- Add causal ablation experiments for the LLM sentiment circuit (e.g., zero top eigenvectors and measure drop in "not-good" feature function) to strengthen mechanistic claims.
- Include W vs. V asymmetry analysis before symmetrization to justify whether the symmetric form loses interpretable structure.
- Provide performance comparisons at larger scales (1B+ parameters) to better evaluate the "drop-in replacement" claim's practical relevance.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Harsh Critic's "Contradiction Between Core Claim and Methodology"**: This is a valid weakness and is kept in Major weaknesses.
- **Harsh Critic's "Viability as Drop-in Replacement"**: This is a valid weakness and is kept in Minor weaknesses (scaled down since the paper does acknowledge performance tradeoffs).
- **Harsh Critic's "Lack of Causal Validation for LLM Circuits"**: This is a valid weakness and is kept in Major weaknesses.
- **Harsh Critic's point about W vs. V asymmetry**: This is a valid minor weakness and is kept.
- **Strength Finder's "Recovery of ground-truth computation"**: Kept as a strength with evidence.
- **Strength Finder's "Visual semantic alignment"**: Kept as a strength with evidence.
- **Strength Finder's "Viability of weight-based analysis in language models"**: Kept but tempered—the correlation evidence is real but doesn't fully support "weight-based interpretability" without SAE dependency.
- **Generic strengths about "important problem" or "interesting question"**: Removed per instructions.

## Novel Insights
The paper's core insight—that bilinear MLPs admit exact eigendecomposition of interaction matrices, enabling weight-only analysis in shallow networks—is genuinely novel within mechanistic interpretability. The MNIST adversarial mask construction (Section 4.4) demonstrates a practical capability distinct from gradient-based attacks: masks are constructed analytically from eigenvector pseudoinverses without optimization. However, the LLM extension does not fundamentally bypass activation-based feature discovery; it provides an analytic map *between* SAE features rather than discovering features from weights alone. This hybrid approach (weight analysis + activation-derived bases) is still valuable but less revolutionary than the Abstract suggests.

## Suggestions
- Revise the Abstract and Introduction to accurately frame the LLM methodology as hybrid (weight decomposition + SAE feature bases) rather than "weights alone." The current phrasing sets incorrect expectations.
- Add at least one causal ablation experiment for the sentiment circuit (e.g., ablate top eigenvectors and measure specific functional degradation) to elevate the LLM claims from correlational to mechanistic.
- Include a discussion comparing the bilinear eigendecomposition approach to transcoder-based circuit discovery (Dunefsky et al., 2024) on the same tasks to quantify the claimed "better robustness" from weight-grounded analysis.

## Calibration Anchors
I retrieved and compared against the following papers:

**High-scoring anchors (avg ≥ 6):**
- `/home/wg25r/review_agent/human_reviews_2026/J4GYMiE3JT.md` (6.50, Accept Poster): Introduces susceptibilities framework with novel mathematical formulation, evaluated on 3M-parameter transformer. Similar to this paper in having strong theory + small-scale experiments. This paper has stronger empirical demonstrations (adversarial masks, ground-truth recovery).
- `/home/wg25r/review_agent/human_reviews_2026/pdNaYcApbz.md` (6.00, Accept Poster): Discusses bilinear relational structure in transformers for model editing. Similar mathematical novelty but on synthetic tasks only. This paper has more concrete interpretability demonstrations.
- `/home/wg25r/review_agent/human_reviews_2026/6As4wfTB77.md` (6.00, Accept Poster): Proposes weight diff interpretation with acknowledged preliminary nature. Similar overclaim/scope tension but accepted. This paper has stronger empirical results.
- `/home/wg25r/review_agent/human_reviews_2026/u6JLh0BO5h.md` (7.00, Accept Poster): Jet Expansions for LLM decomposition with thorough experiments. Higher score due to more comprehensive evaluation.

**Medium-scoring anchors (avg ~5-5.5):**
- `/home/wg25r/review_agent/human_reviews_2026/W5BPGXR9jf.md` (5.33, Accept Poster): NerVE uses eigenspectrum dynamics for MLP analysis on small models. Very similar scope/limitations. This paper has more concrete demonstrations (adversarial masks vs. metrics).
- `/home/wg25r/review_agent/human_reviews_2026/2ytWHufA5j.md` (6.00, Reject): SVD weight decomposition analysis with extensive experiments but rejected for lacking actionable insights. This paper has more actionable demonstrations.
- `/home/wg25r/review_agent/human_reviews_2026/wqwtDpPeEf.md` (5.50, Reject): Scalability claims insufficiently supported. Similar overclaim issue.
- `/home/wg25r/review_agent/human_reviews_2026/2Jyb1yu3nN.md` (4.67, Accept): Circuit discovery beyond activations but relies on activation sampling. Similar methodology tension.

**Low-scoring anchors (avg ≤ 4):**
- `/home/wg25r/review_agent/human_reviews_2026/C5Dgtmk7ho.md` (3.00, Reject): Claims "causal" explanations but provides only correlational weight-based visualization without causal validation. This paper has actual causal validation in MNIST (adversarial masks), distinguishing it from this anchor.
- `/home/wg25r/review_agent/human_reviews_2026/TTVHWgCqBH.md` (4.00, Reject): Circuit learning without explicit causal assumption validation. This paper has stronger grounding in MNIST.

**Scoring rationale**: This paper scores higher than C5Dgtmk7ho (3.0) because it has genuine causal validation in MNIST (adversarial masks successfully cause misclassification). It scores similarly to W5BPGXR9jf (5.33) and pdNaYcApbz (6.0)—novel mathematical framework with small-scale validation and some overclaim issues. The MNIST results are stronger than most calibration anchors, but the LLM overclaims are a real weakness. Positioned at 5.5, reflecting strong core method + compelling image results tempered by LLM claim/scope mismatch and lack of causal validation at scale.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>