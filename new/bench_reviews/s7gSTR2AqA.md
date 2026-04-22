## Summary

This paper investigates whether LLMs develop human-aligned semantic categories guided by the Information Bottleneck (IB) efficiency principle, despite not being trained for this objective. Using color categorization as a testbed, the authors (1) conduct an English color-naming study across 39 LLMs showing that larger instruction-tuned models achieve better IB-efficiency, and (2) introduce Iterated In-Context Language Learning (IICLL) to simulate cultural transmission of pseudo color-naming systems, finding that LLMs iteratively restructure initially random systems toward greater IB-efficiency—with only Gemini 2.0 recapitulating the full range of near-optimal IB tradeoffs observed in humans.

## Strengths

- **The IICLL paradigm is a creative and promising methodological contribution.** Extending Zhu & Griffiths (2024)'s I-ICL to iterated language learning using pseudo color terms and withholding any indication that stimuli are colors (referring only to "features"; Figure 1c) ensures that convergence toward IB-efficient systems reflects genuine inductive structure rather than retrieval of training data patterns. This design directly supports the paper's central claim about bias versus memorization.

- **Comprehensive cross-model evaluation revealing systematic factors.** Testing 39 models across 6 families with systematic variation in size, instruction-tuning, and modality (Table 1, Appendix D) goes well beyond prior work. This breadth enables the finding that instruction-tuning and scale jointly drive English-alignment and IB-efficiency (Figure 2c), and that instruction-tuning alone is insufficient (e.g., Llama 3.3 70B inst. still performs poorly; Section 4.1).

- **The IB framework provides principled, quantitative evaluation.** Using efficiency loss and NID-based alignment (Section 3) allows meaningful comparison between LLM outputs and human systems on theoretically grounded metrics, substantially better than ad-hoc similarity measures.

- **Quantitative evidence of progressive structuring over IICLL generations.** Figure 4 provides three converging quantitative measures—increasing efficiency (4a), IB-alignment (4b), and WCS-alignment (4c)—demonstrating that restructuring toward human-like systems is systematic and progressive.

- **Rotation analysis confirming non-trivial structure for Gemini.** The hue-rotation analysis (Section 4.2, Figure 11 in Appendix H) shows that rotating the color-label mapping significantly decreases efficiency and alignment for Gemini, ruling out trivial or artifact-driven efficiency for the strongest model.

- **Discovery of non-English but WCS-like category systems.** The finding that some models (Olmo 2 32B inst., Qwen 2.5 VL 7B inst.) produce systems resembling low-resource WCS languages rather than English (Section 4.1, Figure 9 in Appendix E) enriches the interpretation of what "human-aligned" means and is a genuinely interesting finding.

- **Input modality and representation analyses revealing human-LLM differences.** The minimal pair analysis of text vs. image input (Figure 8, Appendix E) and the CIELAB vs. sRGB comparison (Section 4.1) reveal that visual input doesn't help larger models and perceptually-aligned coordinates hurt all models—negative results that are informative and expose fundamental differences in how LLMs and humans represent color.

- **Close replication of established human experimental paradigms.** By directly replicating Lindsey & Brown (2014) for naming and Xu et al. (2013) for iterated learning with the same stimuli and evaluation metrics, the paper enables principled comparison between LLM and human behavior.

## Weaknesses

### Fatal
None.

### Major

- **The central claim that LLMs exhibit IB-efficiency "via the same fundamental principle" as humans is overclaimed relative to the evidence.** The paper states in the abstract and conclusion that findings "demonstrate how human-aligned semantic categories can emerge in LLMs via the same fundamental principle that underlies semantic efficiency in humans." However, the evidence establishes that IICLL produces systems that *are* IB-efficient, not that IB-efficiency is the *driving principle* causing their emergence. A general regularization or simplification bias—present in any system that prefers fewer categories and smoother boundaries—would produce compressible solutions that are *incidentally* IB-efficient, because simple, regular partitions tend to be compressible. The rotation analysis (Appendix H) was designed to address this concern but is "less conclusive" for 3 of 4 models (Section 4.2). The paper does not discuss or rule out this alternative explanation anywhere in the main text or discussion. Without distinguishing convergence toward IB-optimal solutions *specifically* from convergence toward solutions that score well on IB metrics because they are simple, the claim about the "same fundamental principle" is an overreach. A more precise claim would be that frontier LLMs can produce IB-efficient categorizations under iterated transmission, with the mechanism behind this convergence meriting further investigation.

- **The headline result (full IB-frontier recapitulation) is model-specific, but the framing applies it to "LLMs" broadly.** The abstract states "We find that akin to humans, LLMs iteratively restructure initially random systems towards greater IB-efficiency" before qualifying the Gemini-specific result. The other three tested models (Gemma 3 27B, Qwen 2.5 32B, Llama 3.3 70B) converge to low-complexity solutions—a qualitatively different outcome. The paper acknowledges this in the body ("only a model with strongest in-context capabilities"), but the abstract and conclusion frame findings as demonstrating a property of LLMs generally. The IICLL experiments also exclude smaller models that produce "degenerate" systems (Appendix L), further narrowing the scope. The paper should frame this more precisely as a finding about the relationship between in-context learning capability and IB-efficiency, rather than a general property of LLMs.

### Minor

- **The Shepard circles experiment (Section 4.3) provides only preliminary qualitative support for generalization beyond color.** It tests only one model (Gemini), one condition (k=4), provides no IB-efficiency analysis, no human comparison data, and only qualitative assessment of "increasingly compact" partitions. The paper explicitly acknowledges these limitations, calling it "preliminary" and noting that "testing whether this emergent structure also supports greater IB-efficiency" is "an important direction for future work"—so the paper is appropriately cautious in its framing. However, the section still occupies space in a results section and implies generalization that it cannot quantitatively support.

- **The initial complexity increase pattern in IICLL trajectories is unexplained.** Figure 3 shows trajectories initially climbing in complexity before descending along the IB bound. The paper calls this "striking" but does not analyze whether this is an artifact of the prompt structure, the initialization, or a genuine feature of the model's learning dynamics. A brief ablation or discussion of what drives this pattern would strengthen interpretation.

- **The theoretical grounding for IICLL revealing "inductive biases" is weaker than the language implies.** The iterated learning framework (Griffiths & Kalish, 2007) shows convergence to the prior for *Bayesian* learners who share priors and likelihoods. LLMs are not Bayesian learners in this sense, and the paper provides no argument for why IICLL should reveal inductive biases analogously. The paper acknowledges this implicitly by treating IICLL as a "method" rather than deriving theoretical guarantees, but the repeated use of "inductive bias" language implies a stronger theoretical connection than is justified.

### Trivial
None.

## Nice-to-Haves

- An ablation distinguishing IB-efficiency from general regularization would significantly strengthen the core interpretive claim—for instance, testing IICLL with randomly shuffled stimulus-feature mappings that destroy the perceptual structure IB depends on, or testing in a domain where IB-optimal solutions are *not* the simplest ones.

- Testing more frontier models (GPT-4o, Claude, etc.) in IICLL would clarify whether the full-IB-frontier recapitulation is Gemini-specific or a capability emerging at a certain scale.

- Comparing IICLL with simple algorithmic baselines (k-means, Gaussian mixture models) under the same iterated transmission protocol beyond the feature-based clustering baseline in Appendix M (which is only for Gemini) would establish whether LLM behavior is genuinely novel.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic's claim that CIELAB vs sRGB mismatch "undermines the claim that LLMs and humans share the 'same fundamental principle.'"** This misreads the paper. The CIELAB finding reveals a *difference* in representation format between LLMs and humans, which the paper appropriately highlights as a "key difference." The IB framework evaluates efficiency in perceptual (CIELAB) space regardless of input format—the fact that LLMs achieve IB-efficiency despite receiving sRGB input is actually notable. This doesn't undermine the efficiency finding; it provides an informative contrast about representation grounding.

- **Harsh Critic's demand for the Shepard circles experiment to be "removed or substantially strengthened."** The paper explicitly frames this section as preliminary with acknowledged limitations. Removing it would eliminate the only cross-domain evidence, however weak. It is better evaluated as a minor weakness than as something that should be cut.

- **Harsh Critic's concern about "IICLL with randomly shuffled stimulus-feature mappings" as a missing critical experiment.** This is a valid suggestion for future work, but treating it as a critical missing experiment overstates its necessity. The rotation analysis already partially addresses the concern about structure dependence, and the pseudo-term design already prevents memorization. This is a nice-to-have, not a requirement.

- **Strength Finder's claim about "Shepard circles generalization beyond color" as a supporting strength.** This conflicts with the verified weakness that Section 4.3 provides only qualitative, single-model evidence with no IB analysis. The weakness wins—this is downgraded.

- **Harsh Critic's section-by-section note about the "feature-based clustering baseline in Appendix M but only for Gemini."** The paper does include this baseline comparison for the strongest model. Requesting it for all models is a reasonable nice-to-have but not a core flaw.

## Novel Insights

The paper reveals an interesting asymmetry between LLMs and humans in the iterated learning setting: all four tested LLMs converge toward IB-efficient solutions, but three are trapped at low complexity, unable to sustain higher-complexity efficient systems that humans readily produce. This suggests that the capacity for *efficient compression at multiple complexity scales*—not just compression itself—may be the key differentiator between frontier and non-frontier LLMs, and between LLMs and humans. This nuanced finding, that IB-efficiency is necessary but not sufficient for recapitulating the full human range, is more insightful than a simple "LLMs are/aren't human-like" conclusion.

## Suggestions

- Revise the abstract and conclusion to replace "the same fundamental principle" with language that accurately reflects the evidence—e.g., "consistent with the IB-efficiency principle observed in humans" or "along the same IB-efficiency dimension"—while preserving the genuine contributions about emergence and convergence.

- Add a brief discussion in Section 5 addressing the alternative explanation of general regularization/simplification bias and why the current evidence does or does not distinguish it from genuine IB-efficiency as a driving principle. Even acknowledging this as an open question would substantially improve the paper's intellectual honesty.

- Consider scaling the Shepard circles experiment to at least one additional model or adding quantitative IB analysis before publication, or reframe Section 4.3 as a "preliminary exploration" subsection rather than a full results section.

## Score and Decision

**Calibration anchors:**

| Paper | Score | Topic Comparison |
|-------|-------|-----------------|
| eiC4BKypf1 | 8.0 | LLMs as cognitive models—clean methodology, well-supported claims, careful framing |
| bNt7oajl2a | 8.0 | LLM inductive reasoning—creative paradigm, thorough evaluation, appropriate caveats |
| wPMRwmytZe | 7.6 | Progressive distillation as iterated learning—novel connection, well-supported |
| HC0msxE3sf | 6.0 | Lewis signaling game as beta-VAE—creative IB/cultural evolution connection, some overclaim |
| CfdPELywGN | 5.2 | Cognitive maps for LLM extrapolation—overclaimed human-like properties beyond narrow domain |
| UXCfRU2Qs4 | 4.25 | LLMs as windows on psychopathology—fundamental circularity and overclaimed proxy |
| fI6TkT050a | 2.5 | Piaget for LLM evaluation—unjustified framework application, weak methodology |

This paper has genuinely strong methodology (IICLL paradigm, 39-model evaluation, careful experimental design with pseudo-terms) and real empirical contributions. It is clearly above the low-scoring papers (which have fundamental methodological failures) and above the medium-low papers with circularity or fundamental overclaims (4.25-5.2). It sits in the same range as HC0msxE3sf (6.0), which also connects cognitive science frameworks to AI with creative methodology and some overclaim. The current paper has broader empirical scope (39 vs. few models) but a more significant overclaim issue ("same fundamental principle"). It falls below the high-scoring papers (7+) which have cleaner claims and more appropriate framing. The gap between the evidence and the central claim is the main factor keeping this below 7.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>