=== CALIBRATION EXAMPLE 61 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title accuracy:** Accurately captures the core focus (LLMs, semantic categorization, evolution via iterated learning, compression/IB). 
- **Abstract clarity:** Clearly states the problem, method (English color naming + IICLL paradigm), and key findings. The qualification that only Gemini 2.0 robustly recapitulates the human IB tradeoff range prevents overgeneralization.
- **Unsupported claims:** The claim that LLMs exhibit a "human-like inductive bias toward IB-efficiency" is empirically supported for specific models, but the abstract slightly undersells the strong model-dependency revealed later. The transition from empirical observation to "same fundamental principle" is a conceptual leap that is acceptable for an abstract but should remain carefully bounded in the main text.

### Introduction & Motivation
- **Problem motivation & gap:** Well-motivated at the intersection of cognitive science and AI. The gap (LLMs are not trained with IB objectives, yet humans optimize it; do LLMs develop similar systems?) is clearly articulated. The choice of color naming is justified by the unique availability of cross-linguistic (WCS) and cultural evolution (Xu et al., 2013) datasets.
- **Contributions:** Clearly listed and match the content. The introduction of IICLL as a prompt-based workflow for simulating cultural transmission is a recognizable and useful extension of prior I-ICL work.
- **Claim calibration:** The introduction appropriately frames the work as an empirical investigation rather than a theoretical proof. One area to strengthen: the motivation is heavily cognitive-science oriented. For ICLR, explicitly linking why IB efficiency in LLMs matters for broader ML challenges (e.g., efficient representation learning, prompt engineering, or alignment with human conceptual boundaries) would strengthen the venue fit.

### Method / Approach
- **Description & reproducibility:** The IB framework and color-naming protocols are clearly linked to prior established code (Zaslavsky et al., 2018). Model list, prompting strategies, and IICLL procedure (Appendix G) are sufficiently detailed for replication.
- **Key assumptions & justification:** 
  - The IB model assumes Gaussian distributions over CIELAB space for meanings. This is standard in the color literature and well-justified.
  - The core methodological leap is treating iterative prompt sampling (IICLL) as a proxy for cultural transmission. While acknowledged as a simulation, the assumption that in-context learning approximates human iterated learning lacks formal grounding. In humans, learning involves parameter-like updates (schema adaptation), whereas ICL relies entirely on attention over a fixed context window without weight updates or consolidation. This should be explicitly discussed as a conceptual limitation.
- **Logical gaps / technical concerns:** 
  - **Stochasticity in IB computation:** The IB metric (Eq. 1, 2, 3) fundamentally depends on the conditional distribution $q(w|m)$. The paper mentions using "mode maps" and log-probability scoring for constrained decoding, but it is unclear whether the mutual information $I_q(M;W)$ and $I_q(W;U)$ were computed from the full predictive distributions or from discretized, deterministic mode assignments. If the latter, complexity is artificially underestimated, and accuracy ignores uncertainty. Clarification on whether full softmax/probability distributions were used to compute MI is critical.
  - **Context window artifact:** The sliding window of 10 previous interactions (Appendix K) is used to mimic short-term memory. While practical, this introduces a known state-dependency into the production process that may drive convergence independently of inductive bias. An ablation isolating window effects (partially done in App K) should explicitly report how degeneracy rates vary with window size across models.

### Experiments & Results
- **Testing claims:** The dual-experiment design appropriately separates surface alignment (English naming) from underlying inductive bias (IICLL). The results logically support the conclusion that larger, instruction-tuned models better approximate human systems.
- **Baselines & fairness:** The Nearest-Neighbor baseline (Appendix M) is a strong control showing that structure isn't just geometric partitioning. However, missing are standard clustering baselines (e.g., K-Means, GMM, or explicit IB optimization on the 330 chips) run under identical generational sampling conditions. These would strengthen the claim that LLM convergence reflects a specific *inductive bias* rather than just optimization pressure toward compact geometric clusters.
- **Ablations:** Window size (Appendix K) and input representation (sRGB vs CIELAB, App C; hinting sRGB, App F) are covered. Missing: sensitivity to generation temperature and top-p. Given the stochastic decoding (temp 0.6, top-p 0.9 noted in App J), how does temperature variation affect the trajectory toward IB optimality vs. degeneracy?
- **Statistical rigor:** The human IL literature (Xu et al., 2013) reports results across 4 replications per condition. The paper mentions "chains" but does not specify the number of random seeds or independent runs per LLM per condition. For ICLR standards, reporting variance across seeds with error bars or statistical tests on the trajectory endpoints is necessary to establish reliability, especially given the observed divergence between Gemini and other large models (Qwen, Gemma, Llama).
- **Results vs. claims:** Findings are generally well-supported and not cherry-picked; the paper openly reports which models fail to align. The claim that models are "not merely mimicking" training data is supported by the random initialization in IICLL, though it is worth acknowledging that LLM priors over color terms (from pretraining) could still implicitly shape early generalized responses even without direct exposure to the random labels.

### Writing & Clarity
- **Clarity of explanation:** The pipeline from IB theory (Section 2.2) to experimental setup (Section 3) is logically structured. The distinction between English naming (static evaluation) and IICLL (dynamic evolution) is clear.
- **Figures & tables:** Mode maps (e.g., Figure 9, Figure 13) effectively visualize category boundaries. The IB complexity-accuracy plots (Figure 2a, Figure 3) are standard and informative. The IICLL trajectory plots (referenced as Fig. 4 and Appendix figures) need to clearly display variance across runs if multiple seeds were used. Table 1 is comprehensive. Overall, figures and tables serve the arguments well.

### Limitations & Broader Impact
- **Acknowledged limitations:** The authors explicitly note the narrow domain (color), unclear origins of the bias (data vs. architecture vs. tuning), and the need to integrate communicative pressures.
- **Missed fundamental limitations:** 
  - As noted, the methodological gap between ICL and true iterated learning (parameter updates vs. context recall) needs deeper discussion. Human IL chains benefit from cumulative schema refinement; LLM IICLL chains are stateless between generations except for the sampled training set. This limits claims about "cultural evolution."
  - The IB computation relies on a fixed, discrete stimulus space (330 WCS chips). How findings scale to continuous, high-dimensional semantic spaces remains unknown and should be flagged.
- **Broader impact / societal implications:** Not discussed. For ICLR, a brief paragraph on implications would be valuable: e.g., understanding how LLMs compress concepts can inform efficient finetuning, bias mitigation in categorical judgments, or designing models that align with human conceptual boundaries for safer human-AI interaction.

### Overall Assessment
This paper presents a well-motivated, cognitively grounded investigation into whether LLMs exhibit human-like IB-efficient semantic structures. The experimental scope (39 models, dual-task design, rigorous use of established IB metrics) is a significant strength, and the finding that only the most capable frontier model robustly recapitulates human IB tradeoffs is intriguing. However, to meet ICLR's rigor standards, the authors must address two key methodological concerns: (1) clarify whether IB mutual information is computed from full predictive distributions or discretized outputs, as this fundamentally affects complexity/accuracy measurements, and (2) strengthen statistical reporting by providing multi-seed results with variance/error bars for the IICLL trajectories. Additionally, a more explicit discussion of the conceptual gap between in-context learning and parameter-based cultural transmission, alongside stronger ML-oriented motivation, would elevate the work. Despite these concerns, the core contribution is novel, the empirical evidence is broadly consistent with the claims, and the framework offers a valuable methodology for studying LLM inductive biases. With clarification on stochasticity handling and statistical robustness, this work stands as a strong candidate for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper investigates whether large language models develop human-like, Information Bottleneck (IB)-efficient semantic categorization systems, using color naming as a primary testbed. Through a comprehensive evaluation of 39 models on English color naming and a novel Iterated In-Context Language Learning (IICLL) paradigm simulating cultural transmission, the authors demonstrate that while many models struggle to capture human baselines, larger instruction-tuned models—particularly Gemini 2.0—exhibit a human-like inductive bias toward IB efficiency, evolving random pseudo-color systems into efficient, human-aligned category structures over generations.

### Strengths
1. **Comprehensive and Systematic Model Evaluation:** The study benchmarks 39 models across six families, varying size, instruction-tuning status, and modality (Table 1, Appendix D). This scale provides a robust landscape of current LLM capabilities and clearly identifies instruction-tuning and scale as key drivers of English-alignment and IB efficiency.
2. **Strong Cognitive Science Integration & Methodological Novelty:** The extension of prior iterated learning paradigms to an Iterated In-Context Language Learning (IICLL) framework (Section 2.3, Fig 1c) is a clear conceptual contribution. By closely mirroring human behavioral experiments (Xu et al., 2013; Lindsey & Brown, 2014) and applying the IB framework with standard metrics (efficiency loss, NID alignment), the work enables rigorous, quantitative human-LLM comparisons.
3. **Robust Validation and Baseline Controls:** The authors strengthen their empirical claims through targeted ablations and controls. Appendix C tests alternative input spaces (CIELAB vs. sRGB), Appendix F analyzes training checkpoints (Olmo 2) to isolate pretraining vs. instruction-tuning effects, Appendix L confirms smaller models fail to sustain the process, and Appendix M introduces a Nearest-Neighbor baseline to rule out trivial feature clustering as an explanation for the observed IB convergence.
4. **High Reproducibility Standards:** The reproducibility statement is exemplary for an ICLR submission. The authors provide explicit model IDs, full prompt templates (Appendix J), decoding parameters (e.g., temperature, top-p, log-prob constrained decoding), dataset sources, and public code repositories, enabling straightforward replication.

### Weaknesses
1. **Under-specified Generalization Claims:** The abstract and conclusion claim the findings extend to domains beyond color, noting initial evidence of structured category development in a different domain. However, the body and appendices provide no methodology, datasets, or quantitative results for this secondary domain. This leaves the generalizability of the IICLL framework and IB-bias claim empirically unsupported in the current draft.
2. **Mechanistic Opacity Regarding "Inductive Bias":** While the empirical trajectories strongly suggest convergence toward IB efficiency, the paper does not investigate the underlying mechanisms driving this bias. It remains unclear whether this stems from pretraining data statistics, instruction-tuning objectives, architectural properties of in-context learning, or a combination thereof. The Olmo trajectory analysis is correlational and does not isolate causal factors.
3. **Heuristic-Dependent IICLL Protocol:** The cultural evolution simulation relies on several unvalidated design choices, such as a fixed sliding context window of 10 interactions (Appendix K) and specific sanity-check thresholds. While the authors show a 0-window degrades performance, there is no systematic sensitivity analysis for window size, training set sample size (fixed at 6×vocabulary), or temperature scaling, making it difficult to disentangle model capability from prompt engineering heuristics.
4. **Reliance on sRGB Inputs Limits Perceptual Grounding Claims:** Appendix C reveals that models perform poorly when prompted with perceptually uniform CIELAB coordinates, despite CIELAB being the standard in color vision research. This suggests LLMs may be leveraging sRGB-to-text statistical correlations from web data rather than learning perceptually grounded categorical boundaries, which weakens claims about alignment with human perceptual semantics.

### Novelty & Significance
**Novelty:** The work introduces a novel methodological paradigm (IICLL) for probing LLM inductive biases in semantic system formation and applies it systematically across a large model zoo. While IB-based analyses of emergent communication exist, applying them to iterated cultural transmission in modern LLMs, complete with human baselines and NN controls, represents a meaningful advance at the intersection of cognitive science and AI.
**Significance:** This paper aligns well with ICLR’s focus on understanding model behavior, emergent capabilities, and human-AI alignment. By demonstrating that IB-efficient categorization can emerge in LLMs through cultural transmission-like pressures, it bridges theoretical linguistics/cognition with practical LLM evaluation. The findings have direct implications for assessing whether LLMs internalize efficient compression principles naturally, which is critical for designing models that communicate effectively with humans and adapt to novel semantic environments.

### Suggestions for Improvement
1. **Substantiate or Temper Generalization Claims:** Either include a full experimental section (setup, dataset, IB metrics, results) for the promised non-color domain, or explicitly state that cross-domain validation is out of scope and frame the finding as a hypothesis for future work.
2. **Conduct Mechanistic Probing:** Move beyond correlational training dynamics. Analyze how attention patterns or hidden representations shift across IICLL generations, or run controlled experiments (e.g., training base models with/without specific instruction-tuning data subsets) to isolate the source of the IB bias.
3. **Systematic Hyperparameter Sensitivity Analysis:** Run ablations over the IICLL context window size (e.g., 0, 5, 10, 20), training sample size, and decoding temperature to demonstrate that the observed IB convergence is robust to protocol variations and not an artifact of a specific heuristic setup.
4. **Address the sRGB/CIELAB Discrepancy:** Discuss why models fail on CIELAB inputs despite its perceptual uniformity. Propose and test a straightforward mitigation (e.g., fine-tuning on CIELAB-aligned data, or explicitly evaluating Vision-Language Models that natively process color patches) to strengthen claims about perceptual grounding and reduce the risk of the task merely measuring web-scraped coordinate memorization.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add standard geometric clustering baselines (e.g., K-Means, GMMs in CIELAB/sRGB) to the IICLL analysis, because without them, you cannot rule out that Gemini’s high IB-efficiency is merely trivial spatial partitioning rather than a meaningful semantic inductive bias.
2. Run IICLL with randomized or bijectively remapped sRGB coordinate inputs to test if models rely on numerical token proximity instead of learned perceptual topology; without this control, the claim that LLMs evolve "perceptually grounded" categories remains unconvincing.
3. Report results across multiple independent seeds and replications with variance/error bounds for each chain, because single-trajectory IICLL runs in stochastic autoregressive models cannot substantiate a systematic inductive bias over random sampling artifacts.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify how the fixed sliding context window and prompt template mechanically enforce category regularization, because failing to separate finite ICL memory limits from genuine cultural transmission invalidates the interpretation of IICLL as an inductive bias probe.
2. Validate whether the human-centric IB assumptions (Gaussian perceptual noise and specific communicative priors) actually align with how LLMs represent uncertainty and meaning; if LLM internal priors diverge from these assumptions, the reported efficiency metrics become theoretically unmotivated.
3. Provide ablations on decoding strategies (e.g., varying temperature, top-p, and API-controlled constraints vs. open-weight sampling) because high alignment scores driven by rigid instruction-following heuristics or aggressive sampling constraints do not reflect true semantic capacity.

### Visualizations & Case Studies
1. Provide side-by-side visualizations of the WCS grid color boundaries at start, intermediate, and final generations for both converging and collapsing chains to demonstrate whether high IB scores correspond to smooth human-plausible boundaries or arbitrary, metrically efficient but perceptually implausible partitions.

### Obvious Next Steps
1. Formalize and include full quantitative metrics for the non-color domain IICLL experiment referenced in the introduction and conclusion; making broad claims about cross-domain applicability without empirical data or figures weakens the paper’s scope and must be backed by actual results before publication.

# Final Consolidated Review
## Summary
This paper investigates whether large language models develop human-aligned, Information Bottleneck (IB)-efficient semantic categorization systems. By benchmarking 39 models on English color naming and introducing a novel Iterated In-Context Language Learning (IICLL) paradigm to simulate cultural transmission, the authors demonstrate that only frontier, instruction-tuned models (notably Gemini 2.0) evolve random pseudo-color systems into structured, IB-efficient categories that align with human linguistic data.

## Strengths
- **Methodological novelty at the AI-Cognition intersection:** The IICLL framework successfully adapts rigorously validated human iterated learning paradigms to probe LLM inductive biases via prompt-based workflows, enabling direct quantitative comparison between human and machine category evolution.
- **Comprehensive and systematic empirical landscape:** The evaluation spans 39 models across six families, systematically varying scale, instruction-tuning status, and modality. This thoroughly establishes that model size and instruction-tuning are primary drivers of IB-efficiency and English-alignment.
- **Strong empirical controls and baselines:** The inclusion of a Nearest-Neighbor baseline effectively rules out trivial geometric clustering as the sole driver of IB convergence. Additional ablations on context window size and CIELAB vs. sRGB inputs demonstrate careful methodological hygiene and highlight boundary conditions of model capabilities.

## Weaknesses
- **Insufficient statistical rigor for stochastic generation:** The IICLL paradigm relies on autoregressive decoding with inherent stochasticity, yet the paper does not report the number of independent seeds/runs per condition, nor does it provide variance or error bars on trajectory endpoints. Without multi-seed statistical reporting, claims of a systematic inductive bias toward IB efficiency cannot be rigorously distinguished from high-variance sampling artifacts, failing to meet standard ML reproducibility and reliability expectations.
- **Overstated claims of perceptual grounding:** The reliance on sRGB coordinates severely limits the claim that LLMs align with human *perceptual* semantics. Appendix C explicitly shows that using perceptually uniform CIELAB inputs degrades performance across all models, strongly suggesting that LLMs are leveraging web-scraped sRGB-text statistical correlations rather than learning perceptually grounded categorical boundaries. This undermines the core cognitive science motivation of the paper.
- **Unsubstantiated cross-domain generalization:** The abstract, introduction, and conclusion repeatedly suggest that the findings and IICLL framework apply to "other semantic domains." However, the manuscript provides zero methodology, datasets, or quantitative results for any domain outside of color. Making broad generalization claims without empirical backing is a significant overreach that weakens the paper's scientific rigor.
- **Conceptual mismatch between ICL and cultural transmission:** The paper frames IICLL as simulating cultural evolution, but in-context learning operates over a static attention window without parameter updates, consolidation, or schema refinement. The fundamental architectural gap between stateless prompt sampling and human iterative learning is not sufficiently discussed as a limitation, leaving the precise mechanistic origins of the observed bias (e.g., pretraining data priors vs. genuine inductive preferences) opaque.

## Nice-to-Haves
- Include standard geometric clustering baselines (e.g., K-Means, GMMs in CIELAB/sRGB) in the main text to further isolate the contribution of LLM architectural inductive biases versus simple spatial partitioning algorithms.
- Provide visualizations of category boundaries at intermediate generations alongside final outputs to illustrate the evolutionary trajectory and differentiation between converging versus collapsing chains.
- Run ablations on decoding hyperparameters (temperature, top-p) to demonstrate that observed IB convergence is robust to sampling variations and not an artifact of constrained generation settings.

## Novel Insights
The paper provides compelling evidence that IB efficiency—a foundational principle in human semantic organization—can emerge in frontier LLMs purely through iterated cultural transmission-like pressures, without explicit optimization for compression. This suggests that advanced in-context capabilities coupled with instruction-tuning may unlock latent inductive biases optimal for semantic compression, bridging information-theoretic linguistics with modern AI and indicating that human-aligned categorization is not merely memorized, but actively reconstructed through simulated cultural evolution.

## Potentially Missed Related Work
- None identified. The literature review on IB theory, iterated learning, and LLM representation of perception is comprehensive and appropriately contextualized.

## Suggestions
1. **Implement rigorous statistical reporting:** Re-run IICLL trajectories across multiple independent random seeds (minimum 3-5 per condition) and report mean trajectories with variance/error bounds. Statistical tests on endpoint metrics are necessary to claim systematic inductive bias.
2. **Temper or substantiate cross-domain claims:** Either include a full experimental section with quantitative IB metrics for a second semantic domain (e.g., household objects or kinship terms) or explicitly reframe the cross-domain statements as hypotheses for future work to avoid overgeneralization.
3. **Clarify IB metric computation and sampling constraints:** Explicitly state whether mutual information $I_q(M;W)$ and $I_q(W;U)$ are computed from full predictive probability distributions or discretized mode assignments. Discuss how constrained decoding and fixed temperature settings might artificially cap estimated complexity, and provide sensitivity analyses where possible.
4. **Deepen the mechanistic discussion of ICL vs. human learning:** Add a dedicated limitations subsection analyzing how the static nature of in-context learning differs from parameter-based schema refinement in human iterated learning. Investigate whether the observed bias stems from explicit instruction-tuning data, architectural attention patterns, or pretraining token co-occurrence statistics, moving beyond correlational training checkpoint analysis.

# Actual Human Scores
Individual reviewer scores: [4.0, 8.0, 6.0]
Average score: 6.0
Binary outcome: Accept
