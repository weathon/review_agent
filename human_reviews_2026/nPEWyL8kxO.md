# CoT-Self-Instruct: Building high-quality synthetic prompts  data  for reasoning and non-reasoning tasks

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
We propose CoT-Self-Instruct, a synthetic data generation method that instructs LLMs to first reason and plan via Chain-of-Thought (CoT) based on given seed tasks, and then generate a new synthetic example of similar quality and complexity. This is followed by a filtering step to select high-quality data using automatic metrics, which are then used for LLM training. In verifiable reasoning, our synthetic data significantly outperforms existing training datasets, such as s1k and OpenMathReasoning, when evaluated on MATH500, AMC23, AIME24, and GPQA-Diamond. For non-verifiable instruction-following tasks, our method surpasses the performance of both human and standard Self-Instruct training data on the AlpacaEval 2.0 and Arena-Hard benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CoT-Self-Instruct for generating high-quality synthetic data for reasoning and open-ended instruction following tasks. The approach consists of 2 main stages: Synthetic Instruction Creation with Chain-of-Thought (CoT) and Synthetic Instruction Curation. CoT-Self-Instruct data outperforms existing training datasets when evaluated on mathematical benchmarks. For open-ended tasks, CoT-Self-Instruct + RIP improves DPO and online-DPO performance.

### Strengths
This paper introduces a novel synthetic data generation method using Chain of Thought (CoT) Reasoning. While Self-Instruct and CoT are well-known and widely used, their integration for both instruction and target generation creates a simple yet effective method for synthetic data generation. Empirically, CoT-Self-Instruct improves performance in a consistent way, and filtering typically improves results despite reducing the quantity.

### Weaknesses
1. Limited generalisation across datasets: The evaluation focuses mainly on math (MATH, AMC, AIME) and only one multiple-choice dataset (GPQA-Diamond). Evaluation on more diverse domains such as TheoremQA [1] or broader benchmarks like MMLU [2] (or MMLU-Redux [3], if computational complexity occurs) would better demonstrate generalisability.

2. Limited model diversity in evaluation: All reasoning experiments use only the Qwen3 family for generation and training, while instruction-following uses only Llama 3.1. Evaluation across additional model families (e.g., Gemma, Mistral and others) is needed to demonstrate the generalisability of the method.

3. Missing statistical significance: The paper reports averages over 16 seeds but provides no standard errors or confidence intervals.

4. Limited baseline comparisons: Comparing only against s1k and OpenMathReasoning is insufficient. Recent datasets like DeepScaleR-Preview-Dataset [4] and NuminaMath [5], and others, should be included.


[1] Chen, Wenhu, et al. "Theoremqa: A theorem-driven question answering dataset." arXiv preprint arXiv:2305.12524 (2023).

[2] Hendrycks, Dan, et al. "Measuring massive multitask language understanding." arXiv preprint arXiv:2009.03300 (2020).

[3] Gema, Aryo Pradipta, et al. "Are we done with mmlu?." arXiv preprint arXiv:2406.04127 (2024).

[4] Luo, Michael, et al. "Deepscaler: Surpassing o1-preview with a 1.5 b model by scaling rl." Notion Blog (2025).

[5] Li, Jia, et al. "Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions." Hugging Face repository 13.9 (2024): 9.

### Questions
1. Are there any plots that show the number of data vs accuracy? Since there seems to be convergence towards accuracy on datasets.

2. Is the method cross-generalized towards other models except Qwen3-4B and Llama-3.1? It is relatively unconvincing with evaluating only 2 models.

3. What are the computational costs (e.g. GPU hours) used for evaluating? As it may be computationally expensive.

4. Is the model only capable of handling math or other cross-domain question-answering questions?

For others, see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The work aims to design a data synthesis method that ensures both the quality and effectiveness of synthetic data. The proposed framework leverages the reasoning ability of LLMs to plan and generate more challenging samples, combined with self-filtering mechanisms to improve data quality.

### Strengths
The work provides a detailed baseline for synthetic data generation and incorporates several widely used techniques, including reasoning with LLMs, reward model based data filtering, and consistency-based filtering. The experimental framework is well-structured, and the motivation of using reasoning to enhance data synthesis is reasonable.

### Weaknesses
1. The paper strongly emphasizes the importance of Chain-of-Thought (CoT) reasoning (line 50–52), but this point has already been well-established in prior work.
2. The proposed Answer-Consistency mechanism (line 207–215) is essentially a minor variant of Self-Consistency, a widely adopted heuristic. While useful, it does not provide a novel theoretical or methodological insight, as its upper bound and effectiveness are not guaranteed.
3. The use of a Reward Model for data filtering (line 244–246) is also a conventional technique and does not constitute a substantive methodological innovation.
4. In experiments, the paper uses s1k as the seed dataset and employs Qwen3 as the data synthesizer. However, the comparison against s1k and OpenMathReasoning is not fair, as Qwen3-4B already achieves strong performance (84.8 on MATH500 and 25.0 on AIME). After synthesis (line 380–394), the resulting model does not surpass the Qwen3-4B Instruct model, which weakens the method’s empirical impact.
5. The experimental evaluation is insufficiently comprehensive: it without compares with other existing approaches. Datasets such as MATH, AIME, AlpacaEval 2, and ArenaHard all have rich baselines that should be included for a more evaluation.

### Questions
1. Expand the experimental comparison to include existing strong baselines (e.g., methods from the AlpacaEval leaderboard, GRPO and its variants in MATH-like benchmarks).
2. The Related Work section lacks comparative analysis with methods closely related to the proposed approach. Adding such comparisons would help clarify the novelty and positioning of this work.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a technically sound and empirically strong contribution to the ongoing discussion on synthetic data generation for LLM post-training, which extends Self-Instruct by embedding explicit reasoning and quality self-verification, offering a principled mechanism for both data creation and data curation. Experimental results cover multiple datasets, models (Qwen3-4B, LLaMA3-8B), and training paradigms (GRPO, DPO, online DPO). However, while the work is solidly executed, it somewhat overlaps conceptually with many recent works that uses self-improving paradigm, e.g., Self-Consistency (Prasad et al., 2024) and Self-Rewarding LM (Yuan et al., 2024), and could benefit from deeper theoretical justification for why CoT-based generation yields systematically higher-quality data beyond empirical observation.

### Strengths
1. Studying a key bottleneck in LLM training: generating diverse, verifiable, and high-quality synthetic data without human supervision. The CoT reasoning to controllable data generation is conceptually clear and timely given the trend toward self-improving LLMs.

2. Comprehensive quantitative evidence (Tables 1–2) demonstrates consistent superiority over baselines across both reasoning and instruction-following domains. The ablation tables (Tables 4–11) systematically isolate the effects of CoT, Answer-Consistency, and RIP filtering, confirming robustness across data scales and models.

### Weaknesses
1. The paper attributes gains to “reasoning before generation” but provides no formal analysis or metrics quantifying how CoT reasoning changes the distributional properties or entropy of generated instructions.

2. All reasoning experiments rely on Qwen3-4B variants; no evidence is provided that CoT-Self-Instruct data generalizes to larger LMs.

3. The paper omits qualitative examples of rejected vs. accepted synthetic data under Answer-Consistency and RIP. And RIP filtering’s dependence on Athene-RM and INF-ORM reward models may bias the resulting instruction pool toward their value functions. I question the practicability of this method.

4. The synthetic dataset remains closed.

### Questions
1. For Answer-Consistency, it’s stated that examples are kept if the LLM’s majority-vote answer matches the CoT-generated target, but the exact threshold (K, majority ratio, tie-breaking) is not given.

2. For RIP, the paper says the lowest RM score among responses represents the sample’s “quality,” yet the biased choice of that aggregation (min vs. mean or percentile) is not justified.

3. It’s also unclear how topics are balanced after per-category sampling, or whether filtering introduces domain bias.

4. The generation parameters (temperature/top-p) differ slightly across models, but it is not shown whether these affect data diversity or downstream results.

5. The paper does not specify compute or model call cost for generating and filtering 10k examples.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes CoT-Self-Instruct, a synthetic-data generation and filtering pipeline that augments Self-Instruct by inserting a CoT reasoning phase before synthetic instruction creation.
Each generated sample includes step-by-step reasoning (for verifiable tasks) or a structured “plan” (for general tasks).
Two automatic curation filters are applied: Answer-Consistency (for verifiable reasoning) and Rejecting Instruction Preferences (RIP) (for non-verifiable tasks).
Models trained on CoT-Self-Instruct data outperform Self-Instruct, s1k, and OpenMathReasoning on reasoning benchmarks (MATH500, AIME 24, AMC 23, GPQA-Diamond) and also yield higher win-rates on AlpacaEval 2.0 and Arena-Hard for general instruction following.

The paper claims that reasoning-guided synthetic data yields better downstream reasoning and instruction-following ability than prior synthetic or human-annotated datasets.

### Strengths
- **Clear conceptual pipeline.** The two-stage creation + curation design is straightforward and reproducible.
Prompts (Figures 2–3) are well-documented and easy to adapt.

- Experiments span both reasoning (GRPO) and instruction-following (DPO) regimes across multiple datasets, with ablations for filter types (Self-Consistency, RIP, Answer-Consistency).

- The method yields approx. 10 pp improvement over Self-Instruct and approx. 13 pp over s1k baselines in reasoning tasks, with consistent upward trends across filters (Tables 1, 4, 5).

- Readable writing and thorough related work. The paper contextualizes itself well among Self-Instruct, Evol-Instruct, RIP, and Self-Consistency PO.

### Weaknesses
- The filter accepts examples when the generated answer matches a majority of model-sampled answers, assuming majority approximates correctness.
Without a symbolic or human ground-truth check, this could reinforce systematic reasoning errors. A qualitative study would strengthen claims.

- The approach merges established components, e.g., CoT prompting, Self-Instruct, and RIP, into one pipeline.
Integration is valuable, but the conceptual advance is modest relative to prior work.

The paper says that using CoT makes the synthetic data better, but it never really proves that part. The tests they did ("NoSolve" and "Short CoT") mix up a few things , like reasoning steps, how long the text is, and how complex the words are. So we can’t tell if it’s the actual reasoning that helps or just the longer and richer text. Without checking that carefully, it’s hard to say for sure that CoT is the real reason for the improvement.

### Questions
1) Can you report standard deviations or CIs for Tables 1 and 2 to confirm significance?

2) Have you tested CoT-Self-Instruct on non-math reasoning datasets (e.g., BoolQ, StrategyQA)?

3) Will the authors release full GRPO/DPO training configs to enable replication?

### Soundness
3

### Presentation
3

### Contribution
2
