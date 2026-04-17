Now I have sufficient calibration data. Let me synthesize my review.

Key calibration anchors:
- "When LLMs Play the Telephone Game" (cultural transmission in LLMs): scores 3,5,8,8 → Accept Poster
- "Do LLMs Exhibit Human-like Response Biases?": scores 5,5,5,8,8 → Reject
- "Geometry of Categorical and Hierarchical Concepts in LLMs": scores 5,6,8,8 → Accept Oral
- "Computational Framework for Color Vision": scores 6,8,8,10 → Accept Oral
- "LLMs Assume People Are More Rational": scores 5,6,6,8 → Accept Poster

Now let me carefully analyze the harsh critic's points against the paper.

**Checking harsh critic point 1 (core causal claim not established):** This is a serious point but needs nuance. The paper does make strong claims about "inductive bias toward IB-efficiency" and "same fundamental principle." The IICLL paradigm uses pseudo-words and doesn't explicitly say stimuli are colors, which is a genuine attempt at control. However, the WCS grid structure itself could be recognizable. The paper does acknowledge in the Discussion that "the precise origins of the bias we observe in LLMs toward efficiency are unclear." Still, the abstract and title make very strong claims. This is a valid *major* weakness about overclaiming, but I should note the paper does try to address this with the IICLL design (pseudo-words, no explicit color identification). The question is whether those controls are sufficient.

**Checking harsh critic point 2 (IB-efficiency metrics and null models):** The paper includes a rotation analysis (Appendix H) and a feature-based clustering baseline (Appendix M). The harsh critic seems to understate these. However, these baselines are indeed relegated to appendices and not thoroughly discussed. The rotation analysis is mentioned in §4.2. The lack of comparison with simple clustering algorithms (like k-means) in the IICLL setup is a genuine gap. This is a valid concern but partially addressed.

**Checking harsh critic point 3 (training-data contamination):** This is a real concern. The WCS data and IB color naming analyses are publicly available and widely discussed. The paper uses pseudo-words and doesn't explicitly identify stimuli as colors in IICLL, but the numeric structure and grid patterns could still be recognizable. However, the paper's design is explicitly intended to address this: using pseudo-words rather than real color terms is a meaningful control. The question is whether it's sufficient. This is a valid concern but not a fatal one—the paper acknowledges this implicitly through its design choices.

**Checking harsh critic point 4 (IL theory mismatch):** This is valid but overblown. The paper doesn't claim IICLL perfectly implements the Bayesian IL framework. It uses IL as an *inspiration* and *motivation*, acknowledging differences. The paper says this "enables a direct comparison" but doesn't claim mathematical equivalence. Still, the theoretical connection should be more carefully qualified. This is a moderate concern.

**Checking harsh critic point 5 (Shepard circles weakness):** This is valid. The Shepard circles experiment is clearly preliminary—one model, one k value, no IB analysis. The paper acknowledges this: "An important direction for future work is to test whether this emergent structure also supports greater IB-efficiency." But the paper does use it to claim potential domain generality, which is overreaching.

Now for the neutral reviewer's points and human finder's points:

**Single model dependency:** Valid - the strongest claim (full IB tradeoff range) rests on Gemini 2.0 alone. The paper should be more explicit about this.

**Mechanistic understanding:** Valid but partially acknowledged in the Discussion.

**IICLL vs. ILL mismatch:** Valid concern about prompt sensitivity and deterministic chains.

**Limited models for IICLL:** Only 4 models tested; the paper acknowledges smaller models struggle.

**Statistical uncertainty:** Valid concern about lack of error bars/variance.

Let me also check some specific claims in the harsh review against the paper:

- The paper does include a rotation analysis (§4.2, p.8) and references Appendices H and M.
- The paper does note "the precise origins of the bias we observe in LLMs toward efficiency are unclear" (Discussion).
- The paper does acknowledge smaller models are relegated to Appendix L.
- The Shepard circle section explicitly says it's "preliminary."

Now, I need to assess: is this paper fundamentally flawed (fatal) or does it make genuine contributions with overclaiming issues?

The paper makes several genuine contributions:
1. Comprehensive evaluation of 39 LLMs on color naming
2. Novel IICLL paradigm
3. Interesting finding that many LLMs struggle with English color naming
4. Finding that instruction-tuning matters more than scale
5. Replication of a cognitive science paradigm with LLMs

The overclaiming is real but doesn't invalidate the empirical work. The paper would be stronger with more tempered claims, more baselines, and more models in the IICLL condition.

Compared to calibration papers:
- "Telephone Game" (similar cultural transmission in LLMs, similar overclaiming concerns, weaker empirical grounding): Accept Poster (avg ~6)
- "Do LLMs Exhibit Human-like Response Biases?" (LLM-human comparison, limited models, overclaiming): Reject (avg ~6.2)
- "Geometry of Categorical Concepts" (stronger theory, novel contribution): Accept Oral (avg ~6.75)

This paper has stronger empirical grounding and theoretical framework than "Telephone Game" and stronger novelty than "Response Biases." It's somewhere between 5.5-6.5 based on calibration. The contribution is real but overclaimed, and there are significant methodological concerns that partially but not fully undermine the core claims.

I'll give it a 5.5, leaning toward acceptance as a poster paper but with substantial caveats.</think>## Summary

This paper investigates whether LLMs develop human-aligned semantic categorization systems by applying the Information Bottleneck (IB) framework and iterated learning paradigms to color categorization. Across two studies—an English color-naming task with 39 LLMs and a novel Iterated In-Context Language Learning (IICLL) paradigm—the authors find that larger instruction-tuned models achieve better English-alignment and IB-efficiency, and that LLMs iteratively restructure random systems toward greater IB-efficiency. However, only Gemini 2.0 recapitulates the full range of near-optimal IB tradeoffs observed in humans.

## Strengths

- **Strong interdisciplinary theoretical framing.** The paper rigorously combines the IB framework (Zaslavsky et al., 2018) with iterated learning (Griffiths & Kalish, 2007), both well-established in cognitive science, providing a principled quantitative lens rather than ad hoc metrics. This enables precise comparison of LLM category systems against an optimality bound.

- **Comprehensive model evaluation in the naming study.** Testing 39 models across 6 families with systematic variation in size, instruction-tuning, and modality enables meaningful analysis. The finding that many state-of-the-art LLMs struggle to recapitulate the English color naming system (Section 4.1, Figure 2) is striking and counterintuitive, and the analysis showing instruction-tuning matters more than scale alone is informative for both NLP and cognitive science.

- **Novel IICLL paradigm.** Extending Zhu & Griffiths (2024)'s I-ICL to iterated in-context *language* learning is a genuine methodological contribution that closely replicates human experimental conditions (Xu et al., 2013), enabling direct human–model comparison. The design choice to use pseudo-words and not explicitly identify stimuli as colors is a meaningful attempt to address memorization concerns.

- **Empirically interesting IICLL results.** The finding that LLM chains converge toward IB-efficient systems within ~4 generations, with trajectories climbing in complexity toward the IB bound and then evolving alongside it (Figure 3), is a genuinely novel and intriguing empirical phenomenon worth documenting, regardless of interpretive disputes.

- **Rotation analysis and baselines.** The rotation analysis (Appendix H) and feature-based clustering baseline (Appendix M) provide evidence that Gemini's systems are non-trivially structured, going beyond purely correlational observations.

## Weaknesses

### Major

- **Overclaiming about "inductive bias toward IB-efficiency" as a causal principle.** The paper's central interpretive claim—that LLMs exhibit "a human-like inductive bias toward IB-efficiency" and that "the same fundamental principle" underlies human and LLM semantic organization—is stronger than the evidence warrants. Convergence toward IB-efficient systems under IICLL is compatible with multiple explanations: (a) genuine IB-specific inductive bias, (b) generic clustering/smoothing tendencies of next-token prediction over structured inputs, (c) memorization of color-naming literature and WCS data in training corpora, or (d) pattern-completion on in-context examples. The paper uses pseudo-words and does not name colors in IICLL, which addresses (c) partially, but the WCS grid's numerical structure and the well-known IB color analyses (cited in the paper, publicly available) remain exploitable cues. The paper acknowledges in the Discussion that "the precise origins of the bias we observe in LLMs toward efficiency are unclear," but the abstract, title, and framing throughout make strong principled claims ("same fundamental principle," "emergence of human-aligned categorization") that this acknowledgment does not sufficiently temper. This does not invalidate the results, but it means the paper's contribution should be read as documenting an interesting phenomenon rather than establishing a shared underlying principle.

- **Core finding rests heavily on a single model.** The paper's most distinctive claim—that LLMs can recapitulate the full range of near-optimal IB tradeoffs observed in humans—is supported almost exclusively by Gemini 2.0. The other three models (Gemma 3 27B, Qwen 2.5 32B, Llama 3.3 70B) converge to low-complexity solutions, which is a qualitatively different and arguably less human-like outcome. The paper underplays this asymmetry; the abstract claims "LLMs iteratively restructure" (implying generality), and the title generalizes to "LLMs," while the strongest result is model-specific. The smaller models relegated to Appendix L further show that this is not a generic LLM property but one contingent on very strong in-context capabilities.

- **Insufficient null models and baselines for the IICLL results.** While rotation analysis (Appendix H) and a feature-based clustering baseline (Appendix M) are included, they are relegated to appendices and not rigorously discussed. A simple k-means iteration over CIELAB space, or other non-LLM clustering algorithms applied to the same iterated learning protocol, would serve as critical baselines. Without them, it remains unclear whether "convergence toward IB-efficiency" is a special property of LLMs or an expected outcome of any smoothing/compression process applied iteratively. The observation that many points on the IB plane have some "nearest IB system" partly by construction further underscores the need for stronger baselines.

### Minor

- **IICLL departs from classical iterated learning assumptions.** The paper motivates IICLL using the Griffiths & Kalish (2007) convergence theorem, which requires ideal Bayesian agents with shared priors and stationarity—conditions explicitly violated in the LLM setup. While the paper does not claim mathematical equivalence, the theoretical framing leans heavily on IL theory to justify interpreting IICLL convergence as revealing "inductive bias" in the IL sense. This connection should be further qualified, as IICLL dynamics may reflect in-context pattern extrapolation rather than a prior over languages.

- **Shepard circles extension is preliminary.** Only Gemini is tested at one k value, and no IB-efficiency analysis is conducted for this domain. The paper acknowledges this is "preliminary" but still uses it to support potential "domain-general" conclusions (Section 4.3, Discussion), which overreaches.

- **Training-data contamination is not adequately discussed.** The WCS dataset, IB color naming analyses, and Xu et al. (2013) iterated learning study are publicly available and likely present in training corpora. While the pseudo-word design mitigates direct memorization, the paper should explicitly acknowledge this confound and discuss how it might affect interpretation, rather than treating the IICLL design as sufficient to rule out training data effects.

- **Statistical uncertainty is not reported.** The IICLL results show trajectories for a limited number of chains, but no confidence intervals, standard errors, or variance estimates across multiple runs are provided for the efficiency and alignment metrics. This makes it difficult to assess the robustness of the reported differences between models.

### Trivial

- None substantive.

## Nice-to-Haves

- **Ablation or controlled comparison to isolate what drives Gemini's advantage.** The paper attributes Gemini's superior IICLL performance to "the strongest in-context capabilities" but does not provide quantitative evidence linking ICL benchmark scores to IB-efficiency outcomes. Correlating standard ICL metrics with IICLL performance across models would substantiate this.

- **Analysis of where LLMs' color systems diverge from English.** The aggregate NID alignment scores are informative, but breakdowns by region of color space (e.g., blue-green boundary, warm colors) would reveal whether errors reflect systematic perceptual/linguistic gaps or random noise, and would connect better to the cognitive science literature.

- **Additional frontier models in IICLL (e.g., GPT-4o, Claude).** With only one model achieving the full IB range, testing additional models would clarify whether the finding is idiosyncratic or general.

- **Discussion of faster convergence in LLMs.** The paper notes LLM chains converge in ~4 generations versus longer for humans, but does not explore whether this difference has implications for the claimed "human-like" nature of the bias.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Reproducibility concerns about closed-source models.** The harsh critic raised the availability of models like Gemini 2.0 and closed-source APIs. Per the rules, models cited in the paper are assumed to exist and be available; this is not a valid weakness.

- **Demand for novel secret meaning spaces.** The harsh critic demanded "novel, secret meaning spaces and efficiency metrics never appearing in the training corpus." While this would strengthen the paper, it demands a fundamentally different experimental design outside the paper's stated scope, which is about color categorization specifically because of uniquely available human data. This is a nice-to-have at best.

- **Demand that the paper provide a mechanistic account of why the IB bias emerges.** The paper explicitly acknowledges this as future work ("the precise origins of the bias we observe in LLMs toward efficiency are unclear, and investigating this is another important direction for future work"). While deeper mechanistic understanding would strengthen the paper, demanding it goes beyond the paper's empirical scope.

- **Demand for full Bayesian IL validation in LLMs.** The harsh critic demands that IICLL satisfy all Griffiths & Kalish assumptions. The paper uses IL as a methodological inspiration, not as a formal model, and the empirical observations (regularization across generations) stand regardless of whether the theoretical guarantees transfer. Overstating the theoretical connection is a valid concern, but demanding formal equivalence is unreasonable.

- **Formatting and prompt specification nitpicks.** Concerns about undisclosed temperature settings, handling of invalid outputs, etc., are implementation details that do not affect the core claims and are standard reproducibility nits.

- **Missing related works.** Per the rules, concerns about missing citations cannot be verified and are removed.

## Novel Insights

The finding that many state-of-the-art LLMs struggle with the English color naming system—despite massive training data and billions of parameters—is genuinely surprising and underdiscussed in the broader conversation about LLM capabilities. The observation that some models (e.g., Olmo 2 32B inst., Qwen 2.5 VL 7B inst.) produce systems resembling low-resource WCS languages rather than English is particularly intriguing, suggesting that the alignment landscape for grounded semantic categories may be far more heterogeneous across models than previously assumed. The IICLL finding that chains initially climb in complexity toward the IB bound before evolving alongside it is also a novel dynamical pattern worth noting—it differs from the common expectation that iterated learning simply minimizes complexity.

## Suggestions

- **Reframe the central claim from causal/principled to descriptive/phenomenological.** Replace "human-aligned semantic categories can emerge in LLMs via the same fundamental principle" with more cautious language such as "LLMs can produce category systems that approximate IB-efficient structures under iterated learning conditions." This preserves the genuine contribution without overclaiming shared underlying principles.

- **Add simple clustering baselines in IICLL.** Run k-means or spectral clustering in the perceptual space as an iterated learning agent to establish whether IB-convergence is achievable without an LLM. This would clarify what is special (if anything) about the LLM's inductive structure.

- **Be explicit about model-specificity.** The abstract and title should make clear that the full IB tradeoff range is demonstrated for only one model. Frame the contribution as: (a) comprehensive naming analysis showing wide variation, (b) IICLL as a paradigm, and (c) evidence that at least one frontier model can recapitulate human-like IB trajectories.

- **Report variance across chains.** Add error bars or ensemble statistics for IICLL trajectories to establish robustness.

## Score and Decision

**Calibration comparison:**

- "When LLMs Play the Telephone Game" (cultural transmission in LLMs, similar scope, similar concerns about limited models and overclaiming): Accept Poster, avg ~6
- "Do LLMs Exhibit Human-like Response Biases?" (LLM-human comparison, limited models, overclaiming): Reject, avg ~6.2
- "Geometry of Categorical and Hierarchical Concepts in LLMs" (strong theory, novel formalization, empirical validation): Accept Oral, avg ~6.75
- "Computational Framework for Color Vision" (novel framework, empirical + theoretical): Accept Oral, avg ~8

This paper has stronger theoretical grounding and empirical breadth than "Telephone Game" or "Response Biases," but its overclaiming is more severe relative to the evidence (claiming shared fundamental principles based on one model's IICLL performance). The empirical contribution is real and interesting, the IICLL paradigm is novel, and the naming study across 39 models is thorough—these are genuine contributions. However, the interpretive framework significantly outpaces the evidence. The paper belongs in the "interesting empirical work undermined by overclaiming" category, similar to "Telephone Game" but with somewhat stronger methodology.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>