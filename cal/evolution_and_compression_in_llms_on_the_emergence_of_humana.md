=== CALIBRATION EXAMPLE 61 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is broadly accurate and signals the two main themes: evolutionary/iterated learning and compression/IB in LLM color categorization. “Emergence of human-aligned categorization” is supported only for a subset of models and tasks, so it is a bit stronger than the evidence warrants.
- The abstract clearly states the problem, the two studies, and the headline result. It is a strong abstract in that sense.
- However, it overstates the generality of the conclusion: “LLMs are capable of evolving efficient human-aligned semantic systems” and “human-aligned semantic categories can emerge in LLMs via the same fundamental principle” read as broader than what is shown. The paper’s own results indicate that this is model-dependent, with only Gemini 2.0 recapitulating the widest human-like tradeoffs, while other models converge to low-complexity solutions.
- The abstract also implies a clean causal interpretation of “human-like inductive bias toward IB-efficiency.” The evidence is suggestive, but the paper does not fully rule out alternative explanations tied to prompt sensitivity, decoding method, or task-specific artifacts.

### Introduction & Motivation
- The motivation is compelling and well situated in the literature on semantic efficiency, color naming, and iterated learning. The gap is clearly framed: LLMs are powerful but not explicitly trained for IB efficiency, so do they nevertheless exhibit human-like semantic compression?
- The contributions are stated clearly: English color naming across many models, iterated in-context language learning (IICLL), and an extension to another domain. This is a coherent and interesting package.
- The introduction does somewhat over-claim at the end. It moves from “capable of structuring meaning according to the same principles” to “demonstrate how human-aligned semantic categories can emerge” and finally to “same fundamental principle that underlies semantic efficiency in humans.” That is stronger than what the evidence can support for ICLR-level rigor, because the empirical support is concentrated in a single domain and heavily mediated by one very capable model.
- A key limitation of the framing is that it does not sufficiently distinguish between recovering human-like category structure from prompts and evidence for an internal inductive bias. This distinction matters, especially at ICLR, where reviewers will want careful causal claims.

### Method / Approach
- The overall method is interesting and mostly reproducible at a high level: 39 models for English naming, then IICLL with a subset of stronger models, evaluated against IB and human color systems.
- The paper does a good job describing the IB framework in Appendix A and the IICLL workflow in Appendix G. The use of Zaslavsky et al. (2018) and Xu et al. (2013) as anchors is appropriate.
- There are, however, important methodological ambiguities and some logical gaps:
  - **Efficiency loss definition**: the main text says efficiency loss is the minimum deviation from optimality, but the exact operationalization is not crisply stated in the main paper. The paper should make it clearer how \(\epsilon\) is computed over \(\beta\), whether it is normalized, and how sensitive it is to the discretization of the IB curve.
  - **NID as semantic alignment**: using 1 - NID is reasonable, but the paper does not explain why NID is the right metric for the specific geometry of color category systems. Since color terms can differ in boundary placement and granularity, it would help to justify why NID captures the aspects of similarity the authors care about better than alternatives.
  - **LLM naming task**: the use of constrained generation / logprob scoring is sensible, but the paper does not fully spell out whether all models were evaluated comparably across base, instruct-tuned, text, and multimodal settings. For some models, prompt format and decoding can strongly affect outputs; that makes cross-model comparisons fragile.
  - **IICLL procedure**: the paper aims to simulate iterated learning, but the procedure departs in nontrivial ways from human IL. The prompt includes a sliding history window of 10 previous interactions, the full training set is reused in a way that may preserve more information than typical human experiments, and the model is evaluated on the same full stimulus space after each generation. These choices may be necessary for LLM prompting, but they are also consequential and should be more carefully justified as approximations rather than faithful replications.
  - **Sanity checks / condition handling**: Appendix G notes that the \(k=14\) condition was unstable and often could not retrieve the required 84 examples, yet results for that condition are still emphasized in the main text and used in the nearest-neighbor comparison. This raises concern that the hardest condition may be confounded by prompt-context capacity rather than the category-learning phenomenon itself.
  - **Causal interpretation**: the method is not sufficient to conclude that models have an “inductive bias toward IB-efficiency” in the same sense as humans. The iterated prompting setup could reflect local pattern completion, optimization under constrained decoding, or artifacts of the training examples rather than a stable prior over languages.
- For the theoretical side, the IB derivation is standard and appears correct in spirit. I do not see a mathematical error in the core equations, but the paper’s claims depend more on experimental operationalization than on proof.

### Experiments & Results
- The experiments are well matched to the paper’s core claims in a broad sense: one study tests alignment with English color naming, the other tests whether models can move toward efficient systems under cultural transmission pressure.
- The dataset choices are appropriate: WCS, Lindsey & Brown (2014), and Xu et al. (2013) are the canonical references for this topic.
- The biggest issue is that the evidence is not always as strong as the narrative suggests:
  - **Model selection in IICLL**: the main IICLL experiments restrict to models that already performed well in the English naming task. That is defensible as a focused follow-up, but it also weakens the inference that the effect generalizes across LLMs. The paper should be explicit that the iterated-learning result is conditional on a model’s strong initial color competence.
  - **Missing ablations**: several ablations would materially change the conclusions:
    - varying the sliding window size in IICLL more systematically in the main results;
    - testing whether the effect persists without the full training set retained in the prompt during generation;
    - comparing against a simpler prompt-based clustering or prototype learner baseline more directly than the nearest-neighbor baseline in Appendix M;
    - evaluating whether the effect holds with alternative constrained decoding strategies.
  - **Statistical reporting**: the paper appears to report aggregate comparisons but does not clearly provide error bars, confidence intervals, or significance tests in the main narrative. Given the strong claims about one frontier model outperforming others, uncertainty estimates are important.
  - **Cherry-picking risk**: the paper emphasizes Gemini 2.0 as uniquely strong, but it is not fully clear whether its performance is robust across prompt variants, decoding settings, and task formulations. The CIELAB appendix already shows meaningful sensitivity to input representation, which suggests the results may be less stable than the main story implies.
  - **Baseline adequacy**: the nearest-neighbor baseline in Appendix M is not fully convincing as a ceiling or null model. It is a useful sanity check, but it does not rule out richer non-IB heuristics, especially given the categorical structure and the limited data regime.
  - **Interpretation of “higher than human IL trajectories”**: the claim that LLM trajectories exceed human IL trajectories on the IB bound is interesting but needs careful contextualization. Higher IB efficiency does not automatically imply “more human-like” if the system collapses toward a degenerate low-complexity regime or if the comparison does not control for communicative functionality.
- The results are promising and likely genuine, but the strongest claims would benefit from more statistical rigor and robustness checks.

### Writing & Clarity
- The paper is generally well organized: the distinction between English naming and IICLL is clear, and the connection to the IB framework is motivated throughout.
- The main clarity issue is that some central methodological choices are easy to miss or understand only after reading the appendices. In particular:
  - the operational difference between text-only and multimodal prompts;
  - how the IB efficiency loss is computed;
  - why the sliding window is included and how it changes the analogy to human iterated learning;
  - why \(k=14\) is included despite instability.
- The figures and tables seem to be intended to communicate the key claims, but from the provided text it is hard to assess the visual clarity in detail. Still, the paper would benefit from clearer main-text summaries of what each figure establishes, especially for Figures 2–4 and Figure 13.
- Overall the paper is understandable, but some of the methodological logic is buried too deeply in appendices for a contribution that depends on careful interpretation.

### Limitations & Broader Impact
- The limitations section is partial and misses some central ones.
- The authors acknowledge future work on adding communication pressure and extending to more languages/domains. That is good, but several key limitations are underdeveloped:
  - **Domain specificity**: the evidence is overwhelmingly from color, a domain with unusually rich structure and a well-known efficiency landscape. The paper should be more cautious about extrapolating to “semantic categories” broadly.
  - **Prompt dependence**: the results depend on a specific prompting and constrained-generation setup. This is a serious limitation for any claim about latent inductive bias.
  - **Model dependence**: the strongest effect appears in one model family and one frontier model. That makes the conclusion less about LLMs in general and more about capability thresholds in current frontier systems.
  - **Training-data confound**: the paper does not fully disentangle whether the observed structure emerges from pretraining data, instruction tuning, multimodal alignment, or in-context reasoning capacity.
  - **Safety / societal impact**: there is little discussion of broader implications beyond a positive framing. While this is not a high-risk paper, the authors could still note that “human-aligned” category systems may not transfer across cultures, tasks, or contexts, and that over-interpreting such alignment could mislead users about model groundedness.
- I would not expect a major societal-risk discussion here, but a sharper limitations section would strengthen the paper substantially.

### Overall Assessment
This is an interesting and ambitious paper with a real conceptual contribution: it brings together Information Bottleneck theory, color categorization, and iterated in-context learning to probe whether LLMs can converge toward efficient, human-like semantic systems. The empirical story is promising, especially for strong instruction-tuned models and particularly Gemini 2.0. However, at ICLR’s standard, the paper currently feels somewhat over-claimed relative to the evidence: the strongest conclusions rest on a narrow domain, a highly specialized prompting setup, limited robustness ablations, and a model-dependent result that may not generalize across LLMs. I think the paper has the ingredients of a strong contribution, but it needs a more cautious interpretation and stronger experimental robustness to fully support the headline claims.

# Neutral Reviewer
## Balanced Review

### Summary
This paper studies whether large language models can develop human-aligned semantic category systems under principles of information-theoretic efficiency. It evaluates 39 models on English color naming and introduces Iterated In-Context Language Learning (IICLL) to simulate cultural transmission of pseudo-color systems, arguing that some frontier models—especially Gemini 2.0—can evolve toward near-optimal Information Bottleneck tradeoffs similar to human languages.

### Strengths
1. **Strong cognitive-science framing with a well-motivated testbed.**  
   The paper is grounded in established theories of color categorization, Information Bottleneck efficiency, and iterated learning, and color is an appropriate domain because of the rich human datasets (e.g., WCS, Lindsey & Brown, Xu et al.) used for comparison.

2. **Broad empirical sweep across many models and model families.**  
   Evaluating 39 models spanning base vs. instruction-tuned, text-only vs. multimodal, and different sizes is a genuine strength. This goes beyond a single-model case study and gives a more systematic picture of which properties correlate with better human alignment.

3. **Interesting methodological contribution: IICLL.**  
   Extending iterated in-context learning to iterated in-context language learning is a novel and potentially useful experimental paradigm. It is a concrete way to probe whether models can acquire structural biases beyond direct memorization of training data.

4. **Clear connection between alignment and efficiency.**  
   The paper does not only compare labels to human languages, but also places model outputs on the IB complexity–accuracy plane, which is more theoretically informative than surface similarity alone. This makes the analysis more substantive than a simple benchmark matching exercise.

5. **Includes sanity checks and baselines.**  
   The appendix reports a nearest-neighbor baseline and small-model IICLL failures, which helps support the claim that the observed behavior in Gemini is not trivial. The Olmo training-trajectory analysis is also a useful diagnostic for when alignment emerges during training.

### Weaknesses
1. **Main causal claim is stronger than the evidence supports.**  
   The paper argues that LLMs exhibit a “human-like inductive bias toward IB-efficiency,” but the evidence is largely observational and model-specific. Since IICLL depends heavily on prompting, model capability, and the chosen decoding procedure, it is difficult to separate true inductive bias from prompt sensitivity, instruction-following competence, or artifacts of generation constraints.

2. **Limited control over confounds in the IICLL setup.**  
   The method adds a 10-turn sliding history, uses constrained decoding, and in some cases requires model-specific prompting/representation choices. These design decisions likely influence stability and category formation, but the paper does not convincingly disentangle their effects from the hypothesized efficiency bias.

3. **The strongest result is concentrated in one frontier model.**  
   The headline claim that only Gemini 2.0 recapitulates the human range of near-optimal tradeoffs suggests the phenomenon may not be broadly shared across LLMs. That makes the paper more of a case study about a particular model class than evidence for a general property of LLMs.

4. **Evaluation is somewhat narrow and domain-specific.**  
   Color is an excellent cognitive testbed, but the claim in the title and abstract suggests broader implications for “human-aligned categorization.” The paper gives only preliminary evidence outside color, so the generality claim currently exceeds the demonstrated scope.

5. **Reproducibility is partial.**  
   The paper provides prompts, model IDs, and code links, which is good. However, some key experimental details remain under-specified or hard to reproduce exactly: API-dependent behavior, constrained generation settings, chain sampling procedures, and potentially nondeterministic multimodal/model-version behavior. The statement that generated LLM data is “available upon request” is weaker than fully releasing outputs.

6. **Significance is somewhat limited by the field’s expectations for ICLR.**  
   ICLR typically values technical novelty, algorithmic or methodological rigor, and broadly useful machine learning insights. This paper is interesting and interdisciplinary, but much of the analysis is tied to a specific cognitive-science application rather than a general ML method or a broadly reusable modeling advance.

### Novelty & Significance
**Novelty:** Moderate to good. The IICLL paradigm is the most novel component, and the cross-comparison of many LLMs under IB and iterated-learning lenses is not something I have seen done in this exact form. That said, the work is largely an application/extension of existing frameworks (IB, iterated learning, and color naming) rather than a fundamentally new machine learning approach.

**Clarity:** Generally good at the level of motivation and high-level narrative, but weaker in operational details. The paper is conceptually coherent, though the iterative procedure and evaluation choices would benefit from more explicit algorithmic presentation and ablation structure.

**Reproducibility:** Fair but not ideal. The paper is unusually transparent for an empirically heavy study in that it provides model lists, prompts, and code links. Still, API-dependent models and generation settings reduce exact reproducibility, and the core IICLL results would be stronger with more complete release of outputs and a more standardized evaluation pipeline.

**Significance:** Moderate. The paper is likely of interest to the cognitive-science/LLM interface community and may inspire follow-up work on learned category systems and cultural transmission in models. For ICLR’s acceptance bar, however, the contribution feels somewhat specialized and the main claims are not yet supported by enough causal analysis or generality to be clearly compelling at the conference level.

### Suggestions for Improvement
1. **Add stronger ablations for the IICLL pipeline.**  
   Test the impact of the sliding history window, constrained decoding method, training-set size, and prompt wording separately. This would clarify whether the emergence of efficiency depends on the proposed inductive bias or on a specific experimental recipe.

2. **Strengthen causal interpretation with control experiments.**  
   Compare against simpler baselines beyond nearest neighbors, such as k-means-like clustering, random but memory-preserving transmission, or model variants with shuffled/neutral prompts. This would help isolate what is special about frontier LLMs.

3. **Quantify variance and robustness more explicitly.**  
   Report confidence intervals or bootstrap distributions over chains, prompts, and random seeds for the key metrics. The paper makes strong claims about one model outperforming others, so robustness statistics are essential.

4. **Clarify the relationship between English alignment and IB efficiency.**  
   It would help to explicitly separate “matching English labels,” “matching English category boundaries,” and “lying near the IB bound.” Right now these are sometimes discussed together, but they are not the same property.

5. **Broaden the analysis beyond color or temper the scope of the claim.**  
   If possible, add at least one more semantic domain with comparable structure, or else narrow the title/claims to avoid overgeneralizing from a single domain.

6. **Provide a more formal algorithmic description of IICLL.**  
   A step-by-step pseudocode block with all sampling, prompting, and stopping criteria would make the method much easier to reproduce and evaluate.

7. **Release full generated outputs and evaluation scripts.**  
   Given the dependence on model versions and API behavior, a public dump of all prompts, outputs, and postprocessing code would materially improve reproducibility and trust in the results.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add strong nontrivial baselines for both tasks, especially prompting-only baselines that preserve the same decoding protocol but replace the LLM with simpler learners (e.g., k-means / nearest-neighbor / majority-cluster / Bayesian category learner). Without these, the claim that frontier LLMs exhibit a special bias toward IB-efficient category learning is not convincing.

2. Add a causal ablation of the IICLL setup: remove the sliding history window, vary the number of in-context examples, and vary the random resampling schedule across generations. The current result could be an artifact of prompt memory and repeated exposure rather than an intrinsic inductive bias.

3. Add direct comparisons to the most relevant prior LLM color-naming baselines, especially Marjieh et al. (2024) under matched prompting and decoding. Without head-to-head evaluation, the paper’s claim of broad improvement across model families is not sufficiently grounded for ICLR.

4. Add human-matched analyses on the exact Xu et al. chain statistics, not just final-generation summaries. ICLR reviewers will expect evidence that the full distribution over trajectories, not only endpoint similarity, reproduces the human cultural-evolution pattern.

5. Add robustness across more than one perceptual representation and stimulus encoding protocol, including a consistent multimodal-vs-text comparison under matched input information. The current dependence on sRGB vs CIELAB suggests the core result may be encoding-sensitive, which weakens the general claim about learned semantic alignment.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify whether “IB-efficiency” is genuinely emergent or just a byproduct of collapsing to fewer labels. The paper needs analysis separating compression from trivial degenerate solutions, because otherwise low-complexity convergence can be mistaken for meaningful efficiency.

2. Analyze model behavior as a function of pretraining stage, instruction tuning, and model size with proper controls. Right now the paper asserts these factors matter, but it does not disentangle whether gains come from scale, instruction-following, multimodal training, or decoding differences.

3. Provide statistical uncertainty for all key comparisons across chains, models, and conditions. ICLR expects significance or confidence intervals for claims like “Gemini is the only model that recapitulates the wide range of near-optimal tradeoffs,” especially given likely high variance across stochastic generations.

4. Examine whether the observed human-alignment depends on the specific English lexicon of 14 basic terms. Without sensitivity analysis to label set choice, the reported alignment may partly reflect the evaluation vocabulary rather than genuine semantic structure.

5. Test whether the apparent inductive bias transfers beyond color with a parallel quantitative evaluation, not only a qualitative example. The paper currently overgeneralizes from one domain; a second domain needs the same IB and alignment metrics to support the broader claim.

### Visualizations & Case Studies
1. Show full trajectory plots for many chains with human reference overlays, not just selected examples. This is necessary to see whether the model truly converges toward the human regime or only occasionally lands there.

2. Add failure-case visualizations for models that do poorly: where they collapse, fragment categories, or become unstable across generations. This would expose whether the method works reliably or only under cherry-picked conditions.

3. Include side-by-side mode maps of human, WCS, and model systems at matched complexity levels. Without these, it is hard to judge whether the “alignment” is structural or just coarse partition similarity.

4. Show per-generation label stability and label permutation behavior in IICLL. If labels are unstable but the metric still improves, the result may reflect metric artifacts rather than genuine category evolution.

5. Add a case study of the best and worst prompts under the same model, showing how prompt phrasing or input encoding changes the learned categories. This would reveal whether the claimed bias is robust or prompt-engineered.

### Obvious Next Steps
1. Run matched ablations that isolate whether the effect depends on in-context memory, multimodal input, or instruction tuning. This should have been in the paper because the main claim hinges on the mechanism behind the effect.

2. Replace the exploratory “one model works best” narrative with a predictive scaling analysis. ICLR would expect evidence that performance changes smoothly with model capability rather than an anecdotal frontier-model exception.

3. Evaluate the same IICLL procedure under additional semantic domains with established human baselines. The paper’s central claim is too broad to rest on color alone.

# Final Consolidated Review
## Summary
This paper asks whether large language models can develop human-aligned semantic categories that reflect the information bottleneck (IB) efficiency observed in human color naming. It studies 39 models on an English color-naming task and introduces Iterated In-Context Language Learning (IICLL) to probe whether models can restructure initially random color systems toward more efficient, human-like tradeoffs over generations. The core empirical story is interesting, but the strongest conclusions are much narrower than the rhetoric suggests.

## Strengths
- The paper is grounded in a coherent and well-established theoretical framework: IB-based semantic efficiency, color categorization, and iterated learning are all highly appropriate for the question being asked, and the use of WCS/Lindsey & Brown/Xu et al. makes the comparison meaningful.
- The breadth of the model sweep is a real strength. Evaluating 39 models across families, sizes, and tuning regimes gives a more informative picture than a single-model case study, and the observation that larger instruction-tuned models tend to do better is plausible and useful.
- IICLL is a genuinely novel experimental adaptation. Extending iterated in-context learning to iterated in-context language learning is a nice idea, and the appendix shows the authors put nontrivial effort into reproducing the cultural transmission setup as closely as prompting-based evaluation allows.
- The paper does include some useful sanity checks and diagnostics, such as the Olmo training trajectory, the smaller-model IICLL failures, and the nearest-neighbor baseline. These help show the effect is not completely trivial.

## Weaknesses
- The central causal claim is overstated. The paper repeatedly suggests that LLMs have a human-like inductive bias toward IB-efficient semantic systems, but the evidence is still largely correlational and heavily model-/prompt-dependent. The setup cannot cleanly separate genuine inductive bias from instruction-following ability, constrained decoding artifacts, or quirks of the transmission protocol.
- The IICLL result is fragile and only clearly strong for one frontier model. Gemini 2.0 is doing most of the heavy lifting in the headline story, while many other models collapse to low-complexity or otherwise weak solutions. That makes the contribution feel more like a capability case study than evidence for a general property of LLMs.
- The iterative-learning protocol has important confounds that are not resolved: the sliding history window, repeated exposure to the full stimulus set, and the constrained prompting/decoding scheme all materially shape the outcome. Without stronger ablations, it is hard to know whether the observed evolution reflects an intrinsic bias or simply the mechanics of the prompt setup.
- The scope of the claims exceeds the evidence. The paper is almost entirely about color, a domain with unusually rich structure and unusually strong prior work. The title and conclusion talk about “human-aligned categorization” in general, but the paper only provides a narrow and preliminary hint beyond color.

## Nice-to-Haves
- A fuller ablation suite for IICLL: vary the sliding window, the number of in-context examples, the resampling scheme, and the decoding strategy.
- More direct head-to-head comparisons with prior LLM color-naming work under matched prompting, especially Marjieh et al. (2024).
- Clearer reporting of uncertainty: confidence intervals or bootstrap variation over chains, prompts, and seeds would make the strong claims more credible.
- A more formal pseudocode description of IICLL and a fuller release of generated outputs would improve reproducibility.

## Novel Insights
The most interesting substantive insight is that the paper is not merely showing that LLMs can “name colors like humans,” but that some models may organize color categories in a way that lands near the same compression–accuracy frontier that has been argued to shape human semantic systems. That said, the evidence suggests a capability threshold rather than a broadly distributed cognitive bias: the phenomenon appears strongest in a single frontier model, and the iterated-learning behavior is sensitive to prompt design and context handling. So the real novelty is less “LLMs naturally evolve human-like categories” and more “under carefully engineered prompting conditions, some frontier models can be induced to behave as if they were optimizing a human-relevant compression tradeoff.”

## Potentially Missed Related Work
- Marjieh et al. (2024) — directly relevant prior work on LLM color naming and human sensory judgments; should be more explicitly compared under matched prompting/decoding.
- Ren et al. (2024) — iterated learning perspective on language model evolution; relevant as a close conceptual predecessor for the cultural-transmission framing.
- Carlsson et al. (2024) — relevant for cultural evolution, iterated learning, and information-theoretic color naming in agents.
- Abdou et al. (2021) and Patel & Pavlick (2022) — relevant for prior work recovering perceptual structure from language models.
- None identified beyond those if the authors want to keep the paper tightly focused on the color/IB/IL literature.

## Suggestions
- Reframe the main claim more cautiously: emphasize that the paper demonstrates a model-dependent capacity for IB-like evolution under constrained prompting, rather than a general human-like inductive bias across LLMs.
- Add the most important ablations first: remove the sliding window, vary context size, and compare against stronger nontrivial baselines under the same decoding protocol.
- Report uncertainty across chains and random seeds for every headline plot, especially the Gemini-vs-others comparisons.
- If space permits, add one more semantic domain with the same IB and alignment metrics; otherwise explicitly narrow the title and conclusion to color naming.

# Actual Human Scores
Individual reviewer scores: [4.0, 8.0, 6.0]
Average score: 6.0
Binary outcome: Accept
