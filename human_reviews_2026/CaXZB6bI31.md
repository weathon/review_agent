# Figma2Code: Automating Multimodal Design to Code in the Wild

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Front-end development constitutes a substantial portion of software engineering, yet converting design mockups into production-ready *User Interface* (UI) code remains tedious and time-costly.  
While recent work has explored automating this process with *Multimodal Large Language Models* (MLLMs), existing approaches typically rely solely on design images. As a result, they must infer complex UI details from images alone, often leading to degraded results.  
In real-world development workflows, however, design mockups are usually delivered as Figma files—a widely used tool for front-end design—that embed rich multimodal information (e.g., metadata and assets) essential for generating high-quality UI.  
To bridge this gap, we introduce Figma2Code, a new task that generalizes *design-to-code* into a multimodal setting and aims to automate *design-to-code*  in the wild.  
Specifically, we collect paired design images and their corresponding metadata files from the Figma community. We then apply a series of processing operations, including rule-based filtering, human and MLLM-based annotation and screening, and metadata refinement. This process yields 3,055 samples, from which designers curate a balanced dataset of 213 high-quality cases.  
Using this dataset, we benchmark ten state-of-the-art open-source and proprietary MLLMs. Our results show that while proprietary models achieve superior visual fidelity, they remain limited in layout responsiveness and code maintainability.  
Further experiments across modalities and ablation studies corroborate this limitation, partly due to models’ tendency to directly map primitive visual attributes from Figma metadata.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper constructs a new benchmark called Figma2Code for the task of converting visual front-end designs into code implementations. 

The main difference as compared to prior works is that prior works only considered image inputs (like screenshots of the webpages) while Figma2Code inputs also include images, assets, and metadata, such that the model can possibly reconstruct the full UI from the input. 

The raw data are crawled from the Figma user community with multiple stages of filtering and additional metadata refinement. The final benchmark contains 213 processed examples. 

The evaluation metrics cover three aspects, including visual fidelity, layout responsiveness, and code maintainability. The authors benchmarked various vision-language foundation models as well as various scaffolds including a simple F2CAgent that does better than naive prompting baselines.

### Strengths
- I think including real Figma data as the input is a valuable contribution to the community. And I'm reasonably convinced that the authors have produced a high-quality benchmark (despite the final benchmark being a bit small in size). 

- The F2CAgent seems to be a simple but effective agent scaffold for this task. 

- Quite thorough experiments.

### Weaknesses
- I think you should try to include a few actual examples in the main paper to illustrate both what the benchmark examples look like and also how good the model generations are (I think this would be much more informative than your current Figure 6). 

- I'm a little worried about the set of evaluation metrics that you have selected. For example, from Table 2, it looks like these metrics don't necessarily agree with each other (even for metrics in the same category). How do you reconcile this? It'd be nice to have a small-scale human eval to see how well these metrics correlate with human judgement (especially the visual fidelity metrics). 

- Do you have more in-depth explanations for why "current MLLMs still struggle to balance visual fidelity with code quality"? Do you think there's a way to finetune base models to get better at both? 

- While I recognize the value of this new benchmark, I also recognize that the contribution could be limited in terms of novelty. I'm therefore leaning towards a borderline score.

### Questions
- What are you doing with the 2,842 auxiliary samples? Are there finetuning experiments on them?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a benchmark to evaluate UI code generation reflecting enterprise workflows given Figma artifacts (images and structured metadata) rather than screenshots alone. Across multiple MLLMs, the authors show that models can leverage both image and metadata to get high visual fidelity but still suffer from code responsiveness and maintainability.

### Strengths
1. Strong problem formulation. Moving beyond screenshots-only to include metadata that is a part of enterprise UI design process makes the study highly relevant. The benchmark evaluates on axes like code responsiveness and maintainability in addition to visual fidelity.
2. Dataset creation shows good coverage with thorough automatic filtering and human-in-the-loop checks.
3. The metrics introduced for each axis are clear and reproducible.
4. Experiments and ablations are informative. Results show that image + metadata improves visual fidelity. Ablations clearly indicate geometry/hierarchy drive responsiveness and metadata encourages brittle code affecting maintainability. Tables 2 and 3 sufficiently show these patterns across multiple MLLMs.

### Weaknesses
1. RUR and APR metrics don’t guarantee cross-device behavior as code-level proxies can miss real rendered issues like overflow/overlap at narrow viewports, failed wrapping/reflow, aspect-ratio distortions among others. Similarly, STR and AVU might only be superficial and ignore semantic correctness and accessibility (e.g.,  heading hierarchy), overlook architectural signals like specificity, duplication and component reuse.
2. The MLLMs considered are predominantly closed source. Adding more open-source models can improve reproducibility. Robustness of findings to prompt variations and per-model instruction following quirks (if any; especially with open-source models) and how they are normalized is unclear. 
3. Correlation of the proposed code quality metrics with developer insights and judgements is missing.

### Questions
1. Can code maintainability shortcomings be addressed with explicit and detailed instructions? How about few-shot samples? Generally, how robust are the models to system prompt perturbations?
2. Are there any human studies that tie proposed metrics to developer experience?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces FIGMA2CODE, a realistic multimodal design-to-code task and benchmark built from public Figma community files. Each example pairs a rendered screenshot with cleaned JSON metadata (hierarchy, geometry, styles) and linked assets (icons/images). From ~2.1k files and ~30k pages, the authors curate a 213-sample benchmark for MLLM evaluation.

They evaluate 10 state-of-the-art MLLMs with reference-free metrics for visual fidelity (VES via DINOv2; MAE), responsiveness (RUR, APR), and maintainability (STR, AVU). Results show proprietary models lead in visual similarity but lag on responsiveness/maintainability, whereas some open models produce cleaner, more responsive code. An agentic baseline (F2CAGENT) improves fidelity when using both image and metadata, and an ablation over five metadata components quantifies their differing effects (styles/assets critical for fidelity; geometry/hierarchy for responsiveness).

### Strengths
- The paper reframes design-to-code as multimodal (image + metadata + assets) rather than image-only; clearly motivated by real Figma workflows

- The paper shows significant effort in data curation and data quality assurance, including heuristic filters, CLIP-based dedup (0.95), expert screening, metadata pruning, asset unification, resulting in self-contained samples.

- Evaluation metrics go beyond fidelity to responsiveness and maintainability; definitions are explicit and implementable.

- Clear, actionable finding: e.g., metadata boosts fidelity but encourages rigid/absolute layouts; multimodality partly mitigates.

### Weaknesses
- While the paper presents a valuable MLLM benchmarking resource, the significance of the Figma2Code task is unclear given the existing commercialized figma-to-code solutions (e.g., Figma’s native export/Dev Mode plugins and third-party tools like Locofy/TeleportHQ). Further discussion is needed to justify how MLLMs may benefit front-end designers beyond existing solutions.

- VES + MAE lack user-centric validation; perceptual ranking or human rater studies would strengthen claims. Both responsiveness and code quality metrics are rather hard-coded and can be “optimized” without genuinely high-quality webpages or maintainable code (e.g., inserting semantic tags mechanically). Calibrate with expert code ratings or lints.

- The core benchmark (213) is relatively small for training-free conclusions and may not capture long-tail UI patterns despite stratified sampling. Recommend expanding the dataset, construct multiple subsets with varying difficulty levels, or reporting variance across multiple random subsets.

- While protocols are unified, randomness of inference-time sampling and absence of statistical significance leave uncertainty about small deltas.

### Questions
- Can the authors perform human ratings and preferences as auxiliary metrics? How well does each of the metrics (VES, MAE, RUR, APR) align with empirical human preferences?

- Can the authors provide any "smarter" or more comprehensive evaluations of code quality in addition to the two rule-based maintainability metrics? Any correlation with linter-based or expert maintainability ratings?

- Can the authors report some ablation studies on the proposed agentic workflow?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Existing methods generate code solely from images, overlooking that design mockups are typically delivered as Figma files—a widely used front-end design tool. This paper introduces a new task, FIGMA2CODE, pushing design-to-code research beyond image-only approaches toward a multimodal, industry-relevant setting. It is the first to establish a systematic evaluation framework that assesses not only visual fidelity but also code quality in a multimodal context.

### Strengths
1. The paper is clearly written.

2. The paper presents novelty and comprehensive experiments.

### Weaknesses
1. The Method section requires a more concrete and transparent description, especially within the Metadata Refinement subsection. It would be beneficial to include specific figures, examples, or case studies that illustrate the data-cleaning pipeline in practice. For example, showing how noisy examples are identified, filtered, or corrected would help readers better understand the robustness and reliability of the dataset construction process.

2. The definition of “difficulty” remains unclear. Lines 254–255 suggest that complexity is determined by the interface complexity, but it would strengthen the paper to provide concrete examples or ablation studies that demonstrate how tasks differ across varying difficulty levels. Furthermore, the relationship between task difficulty and model difficulty (as opposed to human preference or perception) should be clarified. It would be helpful to quantify how “difficult” tasks challenge the model’s reasoning or generation capabilities, ideally supported by empirical evidence.

3. A core concern lies in the conceptual positioning of the benchmark. Benchmarks are typically designed to evaluate the ability boundaries of large models under consistent and unconditional setups. However, in this work, the proposed benchmark incorporates conditional inputs intended to assist the model in generating UI code. This design choice seems to blend evaluation with task guidance, potentially compromising the benchmark’s intended diagnostic purpose. The paper would benefit from a clearer justification for this approach—explaining whether the goal is to measure raw capability or conditional adaptability—and from additional discussion on how this aligns with broader benchmarking principles in the LLM community.

### Questions
1. The authors state (Line 240) that the benchmark dataset was constructed via stratified sampling followed by expert selection, with the goal of maintaining balance across key dimensions such as platform, complexity, and content category. However, it remains unclear how this process quantitatively guarantees diversity and coverage. Specifically, how were the strata defined and weighted, and what measures (e.g., entropy, coverage ratio, distribution analysis) were used to verify that the final dataset indeed represents a broad and balanced distribution? Including a concrete analysis or visualization (e.g., distribution plots or summary statistics) would substantially strengthen this claim.

2. The current evaluation setup seems to rely on a single run per model, which may underestimate model performance due to stochastic generation variability. Have the authors considered adopting pass@k metrics (e.g., pass@1, pass@5, pass@10) as done in standard code generation or reasoning benchmarks? Such an evaluation would provide a more robust and statistically grounded assessment of model capabilities, particularly for generative tasks.

### Soundness
2

### Presentation
3

### Contribution
2
