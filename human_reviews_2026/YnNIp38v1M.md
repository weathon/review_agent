# CIMemories: A Compositional Benchmark For Contextual Integrity In LLMs

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 8

## Abstract
Large Language Models (LLMs) increasingly use persistent memory from past interactions to enhance personalization and task performance. However, this memory introduces critical risks when sensitive information is revealed in inappropriate contexts. We present CIMemories, a benchmark for evaluating whether LLMs appropriately control information flow from memory based on task context. CIMemories uses synthetic user profiles with over 100 attributes per user, paired with diverse task contexts in which each attribute may be essential for some tasks but inappropriate for others. Our evaluation reveals that frontier models exhibit up to 69% attribute-level violations (leaking information inappropriately), with lower violation rates often coming at the cost of task utility. Violations accumulate across both tasks and runs: as usage increases from 1 to 40 tasks, GPT-5’s violations rise from 0.1% to 9.6%, reaching 25.1% when the same prompt is executed 5 times, revealing arbitrary and unstable behavior in which models leak different attributes for identical prompts. Privacy-conscious prompting does not solve this—models overgeneralize, sharing everything or nothing rather than making nuanced, context-dependent decisions. These findings reveal fundamental limitations that require contextually aware reasoning capabilities, not just better prompting or scaling. Code is available at https://github.com/facebookresearch/CIMemories.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces CIMemories, a benchmark for testing whether memory-augmented LLMs respect Contextual Integrity such that LLMs reveal stored user facts to third parties only when appropriate, while still being helpful. It composes synthetic user profiles and social contexts so the same attribute can be required in one setting but inappropriate in another, and evaluates models with two complementary measures: violation (leaking “not-to-share” facts) and completeness (sharing what’s needed). Experiments on frontier models reveal a clear privacy–utility trade-off and accumulating leaks across tasks, with scaling and “reasoning” prompts offering only modest relief. The contribution is a clear formalisation of Contextual Integrity, a controllable compositional dataset, and an empirical study that identifies where current assistants fail and how mitigation shifts the trade-offs.

### Strengths
* The paper applies Contextual Integrity (CI) to memory-augmented LLMs in a compositional setting, formalising a benchmark where the same attributes of a user profile can be appropriate in one context but inappropriate in another. The ideas of flexible memory composition and multi-task composition per user are clearly specified and expand the scope beyond single-instance evaluations. The two scores, Violation@n and Completeness, provide a simple but representative evaluation of the privacy–utility trade-off in persistent LLM memory

* The methodology is rigorous and systematic. The paper describes the data-generation pipeline, the evaluation setup, and uses quantitative and qualitative analyses across multiple models and configurations. Results include consistent patterns (e.g., violation/completeness trade-off, scaling saturation, and domain-level “granularity” errors), supported by tables/figures and concrete violation excerpts.

* The presentation is clear. The benchmark workflow, task formation, and the REVEAL judge are explicitly laid out, with prompt templates and an overview figure that maps profiles, contexts, metrics, and the judge’s role. This makes it straightforward to understand what counts as an explicit reveal and how metrics are computed

* The work addresses a timely risk in persistent memory: disclosing the information in the wrong place. By quantifying violations and completeness, showing accumulation over tasks/generations, and analyzing scaling, reasoning, and conservative prompts, it offers actionable insights for deployment and motivates future mitigation work.

### Weaknesses
* Reliance on LLM-Generated Ground Truth and Judges: The benchmark’s contextual integrity labels are entirely generated using closed-source LLMs (GPT-5 personas) and judged by another LLM (DeepSeek-R1) that only flags explicit disclosures. This creates a potential circular evaluation loop, the same class of models(GPT-5) being tested also defines the “ground truth.” To strengthen validity, the authors can include a small-scale human validation study to measure inter-annotator agreement and human–LLM alignment.

* Limited Coverage and Cultural Bias in Labeling: The dataset includes only 10 user profiles and further filters out all attribute–context pairs where privacy personas disagree (entropy > 0), resulting in potential selection bias toward “easy-to-label” cases. Moreover, the benchmark grounds its contextual-integrity labels in Westin’s U.S. privacy personas and a set of U.S.-centric contexts (e.g., HR departments, IRS agents, USCIS officers). Because contextual-integrity norms vary across cultures, this narrow framing limits the benchmark’s representativeness and generalizability. Expanding the dataset with cross-cultural personas, diverse contexts, and non-U.S. human raters would both mitigate coverage bias and improve external validity for global adoption.

* Minor Presentation Issues: A dangling “?” in § 5.2.2.

### Questions
* What proportion of attribute–context pairs survive the unanimity filter across Westin personas (entropy = 0), and how are discarded pairs distributed across domains?

* Could you run a small human validation study to measure inter-annotator agreement and human–LLM alignment on “share” vs “private” labels? This would help verify that LLM-generated CI labels reflect plausible human norms.

* Since CI norms vary culturally, and current personas and contexts are U.S.-centric (e.g., IRS, USCIS), would you consider adding cross-cultural personas and context to evaluate whether violation/completeness patterns shift in future work?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposed a new benchmark focusing on evaluating how LLMs correctly leverages in-context memories, particularly user profiles. The problem is that, based on different tasks, the LLM should reveal certain user information but not others. The paper curated a new benchmark which features two key innovations: (1) flexible memory composition; (2) multi-task composition; The results show that recent LLMs still struggle with many violations on user privacy and shows a trade-off between task completeness and contextual integrety

### Strengths
- The problem definition and motivation is very clear and under-explored in the community
- The paper writing and techical details are clear; The analysis on the violation vs completeness trade-off, and the impact of model size and thinking is very helpful for further understanding the challenge

### Weaknesses
- The discussion on how to mitigate the problem is too weak. It is ideal to have an initial improved baseline based on the insights from the benchmarking analysis; these actionable insights are most helpful to the community; for example, which type of reasoning may benefit most to reduce violation while keeping completeness and general performance?
- Without a quantative comparison with prior contextual privacy benchmarks, it is unclear whether the CIMemories benchmark is testing similar skills or is actually revealing some new challenges. Concretely, it would be good to add columns in Table 1 reflecting performance in prior privacy related benchmarks such as CI-Bench, to show if there is a strong correlation between the performances.
- Minor: The use of color in Figure 4, 5 can be improved considering red-green color blind readers

### Questions
- Figure 5 (b) seems to show that reasoning is a promising direction in addressing this tradeoff; can the authors provide more details on how the reasoning is performed in the experiments

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces CIMemories, a benchmark for evaluating whether memory-augmented LLMs appropriately control information flow based on social context, based on Contextual Integrity theory. The benchmark features synthetic user profiles with ~147 attributes each across 9 domains (finance, health, employment, etc.), paired with ~45 task contexts per user. It evaluates 8 models and find violation rates of 14-69%.  It also demonstrates that violations accumulate over time as users engage in more tasks, and that current mitigation strategies (scaling, prompting) provide limited relief.

### Strengths
S1. Timely and critical benchmark. Most of the companies are deploying Memory-augmented LLMs, but prior benchmarks don't capture the compositional nature of contextual privacy.

S2: The formalization in Section 3 clearly connects CI theory to measurable metrics, with appropriate violation and completeness perspectives.

S3. Analysis is done properly. The granularity failure finding (models identify right domain but over-share details) and the domain-wise breakdown in Figure 3 provides interpretable insights.

### Weaknesses
W1. Benchmark scale is one of the major concern that I have. With only 10 profiles and no statistical testing, the results lack the rigor needed for definitive conclusions. 

W2. Synthetic data. While acknowledged, this is a fundamental limitation. Real users have complex, inconsistent preferences that synthetic profiles cannot capture. 

W3. Requiring unanimous agreement across all 3 personas discards many valid scenarios. This may bias toward only "obvious" privacy violations. Real privacy often involves legitimate disagreement, which is excluded and the paper doesn't report what percentage of attribute-context pairs were discarded.

### Questions
Q1: 10 profiles is very less in my pov to make result significant. I get the cost but can it be increased for the open-source models where inference is cheaper? 

Q2. Synthetic data might not capture complex real user preference. Real LLM users have maybe 20-50 memories after a few months. 147 attributes per profile seems to high. Why so many? Any plan to collect real user data?

Q3. What percentage of attribute-context pairs were discarded?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces CIMemories, a benchmark for evaluating whether memory‑augmented LLM assistants respect contextual integrity. The dataset uses synthetic user profiles (∼147 attributes per profile) and curated social contexts (∼46 per profile)

Each attribute–context pair is labeled (necessary to share vs. inappropriate to share) by sampling multiple “privacy personas” (Westin’s fundamentalist/pragmatist/unconcerned) from a strong LLM and retaining only unanimous labels; evaluation measures (i) Violation@n:worst‑case attribute leakage over n samples in contexts where the attribute should not be shared and (ii) Completeness—fraction of necessary attributes conveyed in contexts where they should be shared.

The paper further analyzes domain‑level failure modes (“granularity failure”), scaling effects, prompting defenses, and how both multi‑task composition and memory composition exacerbate leakage.

### Strengths
- Prior CI‑style evaluations (e.g., ConfAIde, PrivacyLens, CI‑Bench, LLM‑CI) largely focus on single‑shot vignettes or agent trajectories without rich, persistent user memory. CIMemories squarely targets that gap.
- The worst‑case attribute‑level Violation@n coupled with task‑level Completeness makes the privacy–utility trade‑off explicit; the visual breakdown in Figure 3 (p.6) convincingly illustrates “granularity failure” (right domain, wrong detail).
- Results span multiple model families and sizes and include simple defenses (prompting) and test‑time “reasoning” variants, yielding actionable observations (e.g., light reasoning sometimes lowers violations with minimal completeness loss)
- Well‑motivated by the trend from retrieval‑based memory (RAG/MemoryBank/MemGPT) toward long‑context, “prefix the memories” assistants, and shows those settings remain privacy‑fragile

### Weaknesses
- The “ground‑truth” labels (“necessary” vs. “inappropriate”) are produced by GPT‑5 with persona prompts, and then other LLMs are evaluated against those labels. Even with personas, this can encode the teacher model’s normative and stylistic biases. The paper argues LLMs are conservative vs. humans when labeling sensitive content, but no human audit is provided to calibrate false positives/negatives of the labeler on a subset. A 5–10% human‑labeled slice would materially increase credibility.
- Westin personas are U.S.‑centric and decades old; their priors may not reflect contemporary or cross‑cultural norms. The paper uses Westin‑based priors to mix persona votes but does not test sensitivity to those priors. A cross‑cultural variant or at least a sensitivity analysis is warranted.
- The headline “violations increase with more tasks/generations” is partly tautological because Violation@n is worst‑case per attribute over more trials. This is informative for risk, but the paper should also report the per‑turn hazard rate (probability of first leakage at turn t) and time‑to‑leak distributions to separate inherent risk from simple exposure.

### Questions
- Reference Missing In line 376. please fix
- Each context has ~7 “necessary” vs. ~84 “not‑to‑share” attributes, so a model that is slightly verbose can accumulate many apparent violations. Completeness is an average, whereas Violation@n is a worst‑case max across tasks and these aggregations are not symmetric. Consider reporting AU‑Privacy–Utility curves, pareto fronts or a balanced score.
- Reveal detection is too strict in one dimension and too lax in another. Could you use multiple judges in order to make it more robust. This under‑counts partial leaks (e.g., “my antidepressant dosage increased last month”) and leaks via implicature (e.g., “after the DUI class …”)
- Can you provide results under shared decoding parameters across all models? Right now, defaults differ by vendor and may be optimized for safety/verbosity differently.
- Given the heavy class imbalance (∼6.7 necessary vs. ∼83.7 not‑to‑share per context; Table 2), how would your conclusions change under a balanced per‑context metric (macro‑averaging) and a per‑turn hazard‑rate view rather than worst‑case Violation@n?

### Soundness
3

### Presentation
3

### Contribution
4
