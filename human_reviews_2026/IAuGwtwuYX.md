# ReasoningShield: Safety Detection over Reasoning Traces of Large Reasoning Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Large Reasoning Models (LRMs) leverage transparent reasoning traces, known as Chain-of-Thoughts (CoTs), to break down complex problems into intermediate steps and derive final answers. However, these reasoning traces introduce unique safety challenges: harmful content can be embedded in intermediate steps even when final answers appear benign. Existing moderation tools, designed to handle generated answers, struggle to effectively detect hidden risks within CoTs. 
To address these challenges, we introduce *ReasoningShield*, a lightweight yet robust framework for moderating CoTs in LRMs. Our key contributions include: (1) formalizing the task of CoT moderation with a multi-level taxonomy of 10 risk categories across 3 safety levels, (2) creating the first CoT moderation benchmark which contains 9.2K pairs of queries and reasoning traces, including a 7K-sample training set annotated via a human-AI framework and a rigorously curated 2.2K human-annotated test set, and (3) proposing a specialized framework tailored for complex reasoning tasks, which utilizes a structured stepwise analysis paradigm and a strategic two-stage training pipeline to capture risk propagation and boundary ambiguity. Experiments show that ReasoningShield achieves state-of-the-art performance, outperforming task-specific tools like LlamaGuard-4 by 35.6\% and general-purpose commercial models like GPT-4o by 15.8\% on benchmarks, while also generalizing effectively across diverse reasoning paradigms, tasks, and unseen scenarios. All resources are released at https://anonymous.4open.science/r/ReasoningShield.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This submission addresses the problem of moderating chains-of-thought (CoTs) produced by large reasoning models (LRMs), i.e., detecting safety risks within CoTs. The submission describes several contributions in this direction: 1) a taxonomy of risks, adapted from previous works; 2) a ReasoningShield-Train dataset for training moderation models, labelled by three annotation models with a human expert fallback; 3) a ReasoningShield-Test dataset, labelled by human experts only and with safety datasets and LRMs not included in ReasoningShield-Train; 4) 3B- and 1B-parameter ReasoningShield moderation models trained using supervised fine-tuning and direct preference optimization. The experiments presented in the main paper mainly show that ReasoningShield clearly outperforms existing moderation tools and generalizes to LRMs and datasets not seen in training.

### Strengths
- Appears to be the first to comprehensively tackle the problem of CoT moderation, contributing training and test datasets, a training procedure, and two moderation models.
- ReasoningShield's CoT moderation performance (shown in Table 2) is strong and clearly better than the baselines, particularly prompting of much larger LLMs (e.g. Gemma-3-27B or larger vs. 3B or 1B for ReasoningShield).
- I think the evaluation is well-designed with separation of in-distribution (used for training ReasoningShield) safety datasets and LRMs, and out-of-distribution datasets and LRMs.

### Weaknesses
1. A missing class of baselines: As discussed in the introduction, existing moderation tools are challenged by the length and stepwise nature of CoTs. It seems that these challenges could be overcome by also applying existing tools in a stepwise manner, i.e., applying them to each reasoning step in turn and then drawing a conclusion from the results in some way. How much better would this perform? I acknowledge that the computational cost would increase significantly.
1. The number of expert annotators (3) seems low for labelling the ReasoningShield-Test dataset (and similarly the data for the pilot study in Section 3.2).
1. Clarity could be improved as I found quite a few unclear sentences. Please see the questions below, particularly question 4 and onward.
1. The claim of "enhanced explainability" is only supported by three examples given in Figures 6-8, which are furthermore not in the main paper. It would be good to discuss one of these in detail in the main text. 
1. Less important: Since results are reported as the average of five runs, standard deviations could also be reported.

Minor:
- Line 241, "each of the ten primary categories is further refined into 42 subcategories": I think the authors mean 42 subcategories in total, not for each primary category.
- Line 250: I would not say that the "Potentially Harmful" label resolves ambiguity, but rather it accommodates ambiguity.
- Lines 348-349: Is $\mathcal{L}$ the set of three safety levels? I do not think it was defined previously.
- Eqs. (3) and (4): Should the subscript $T$ be $y_{CoT}$?
- Lines 361-362: I do not see how this contrastive learning aligns with human standards. As I understand, it aligns with the majority of the three annotation models.
- Lines 371-372: "Baseline Moderation Methods" may be a better heading for this paragraph.

### Questions
In addition to questions requesting clarification, please also respond to the first question about related work.

1. Related work:
    1. Lines 102-103 state that Zhou et al. (2025) find that reasoning steps hide unsafe content. What method(s) did they use to obtain this finding, and how do they relate to ReasoningShield?
    1. I am not completely convinced that CoT monitoring (lines 109-110) is out of scope, for example because "Deception & Misinformation" is one of the risk categories in the proposed risk taxonomy. I have a similar question of what methods do these CoT monitoring works use and how they relate.
1. How are reasoning traces segmented into steps $t_1, t_2, ...$? 
1. Table 1: 
    - For Claude-Sonnet-3.7, why is the F1$_{CoT}$ column mostly zero?
    - Does "OpenAI Moderation" correspond to GPT-4o mentioned in lines 193-194?
1. In line 297, reproducibility is mentioned as a reason for choosing open-source LRMS for ReasoningShield-Train. What is the reproducibility benefit here?
1. Line 307: What does "requirements for consequence-focused judgment" mean?
1. Questions about agreement among the annotation models:
    1. Does consensus mean agreement in terms of both the risk category and the safety level?
    1. Why the term "hard negatives"?
    1. What is the difference between partial consensus ($\leq 2$) and single-vote cases (since $1 < 2$)?
    1. What is meant by "model consistency" in "97% model consistency"? 
1. Line 360: How is the scoring function $f_{\phi}$ related to the model? 
1. Lines 373-374: What does "raw text detection limitations" mean?
1. Line 434: Does "open-source-derived" refer to open-source LRMs, "OSS" in Table 2?
1. Lines 455-456: What do "evaluation guidance" and "analysis process" refer to exactly?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies safety risks in large reasoning models (LRMs)when chain-of-thought (CoT) is revealed, and proposes ReasoningShield, a safety monitor designed to detect harmful content embedded in reasoning traces. The authors construct a new dataset (9.2K CoT training samples, 2.2K test samples) with multi-level safety annotations, and train a detector via SFT + DPO. Experiments show that the method outperforms prior safety monitors, including LlamaGuard and GPT-4o, in detecting covert harmful reasoning steps. The approach

### Strengths
1. The problem motivation is timely. The paper clearly demonstrates that releasing CoT increases safety risks and that existing safety monitors fail to capture covert malicious reasoning.
2. A reasonably sized dataset with structured safety labels is provided, which may benefit future research on LRM safety and CoT oversight.
3. The approach is simple, inference-efficient, and improves smaller model performance, which is important for realistic deployment scenarios.
4. Empirical results cover several safety benchmarks and model types. Ablations validate the importance of prompt design and DPO training.
5. The paper is clearly written and presents compelling examples illustrating real safety failure modes in CoT.

### Weaknesses
1. Innovation is relatively incremental. The method primarily applies standard SFT + DPO pipelines to a new dataset rather than proposing new architectures or fundamentally novel mechanisms for CoT safety.
2. Dataset construction and annotation quality are insufficiently detailed. The paper lacks breakdowns on annotator consistency, error rates, and quality control procedures.
3. Limited exploration of scaling behavior. The study focuses on smaller models; results on larger safety monitors (e.g., 70B-scale) would strengthen generality.
4. Lack of adversarial or adaptive attack evaluation. It is unclear whether the method remains reliable when attackers intentionally obfuscate harmful reasoning.
5. Some claims regarding “comprehensive coverage” feel overstated given limited analysis on multi-turn interaction, longer reasoning chains, or domain-specific harms.

### Questions
1. Clarify contribution relative to existing safety monitors (e.g., LlamaGuard) beyond dataset + fine-tuning; explicitly highlight differences in handling sequential reasoning structures.
2. Provide detailed annotation protocol, inter-annotator agreement, and dataset quality diagnostics.
3. Include adversarial tests targeting reasoning-chain obfuscation to demonstrate robustness.
4. Evaluate more model scales or provide discussion on scalability and model distillation strategies.
5. Consider adding qualitative failure cases to better illustrate remaining challenges.

Note: I am currently leaning toward a borderline rejection. However, I am open to increasing my score if the concerns outlined above are adequately addressed during the rebuttal phase.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper targets the safety gap in LRMs, where harmful content may appear inside reasoning traces even if final answers look benign. It formalizes CoT moderation, builds a multi-level taxonomy, releases a dataset, and proposes ReasoningShield, a lightweight moderation model trained with a two-stage SFT→DPO pipeline. Experiments report SOTA F1 on CoT moderation and strong OOD generalization, outperforming LlamaGuard-4 and GPT-4o while remaining efficient enough for resource-constrained deployment.

### Strengths
1. The paper convincingly motivates why answer-only moderation fails, and why stepwise CoT analysis is required; several mainstream LRMs are shown to degrade on CoT moderation vs. answer moderation. 

2.  ReasoningShield-3B reaches ≈91.8% F1, beating LlamaGuard-4 by ~35% and GPT-4o by ~15%; a 1B variant also performs competitively. The two-stage training on small backbones is easy to replicate and practical for deployment. This efficient and accurate detection tool has brought great application value to fields such as online content moderation and reinforcement learning.

### Weaknesses
1. Although the dataset mixes sources and shows OOD gains, more naturally occurring product logs or red-teaming traces (with appropriate approvals) would better validate real-world robustness beyond curated constructs. 

2. Results are strong aggregate-wise, but more category-wise error analysis would guide deployment boundaries. 

3. While the paper demonstrates significant practical value, it lacks theoretical analysis and higher-level methodologies, making it difficult to further generalize the experiences presented.

### Questions
1. The paper focuses on reasoning-trace safety, how well does ReasoningShield generalize beyond CoT text to agent reasoning graphs, planning traces, or tool-use trajectories? 

2. Reasoning traces often evolve from safe to unsafe or vice versa. Has the model been tested for drift detection—i.e., identifying when the reasoning starts deviating into unsafe territory?


3. With methods like RLVR becoming mainstream, can ReasoningShield act as a reward model within reinforcement alignment loops? For example, could its detection scores be integrated into PPO/RLHF-style training for online safety feedback, effectively turning moderation into an inner-loop verifier rather than a post-hoc classifier?

4. Can ReasoningShield expose a category/level-specific threshold to control over-flagging vs. misses (precision–recall curves per class; safe/potential/harmful)? This would help match heterogeneous org policies and risk tolerances.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents ReasoningShield, a framework for moderating the reasoning traces generated by large reasoning models. It highlights that harmful content can appear in the reasoning process even when the final answer looks safe, meaning traditional moderation fails to detect such risks. ReasoningShield formalizes CoT moderation with a structured taxonomy of risk types and safety levels, and introduces a large annotated dataset of reasoning traces. The model uses a two-stage training process combining structured reasoning analysis and preference optimization to improve reliability. Experiments show that ReasoningShield provides more accurate, interpretable, and generalizable moderation of reasoning traces compared to existing moderation systems, demonstrating strong robustness across unseen tasks and models.

### Strengths
1. This paper is well-written and clearly presents the dataset description, construction process, and experiments.

2. To improve dataset quality and label reliability, the authors effectively employ methods such as multi-model annotation and human–AI collaboration.

3. Developing a small model that performs competitively in CoT moderation and maintains consistent performance across reasoning traces from various models makes it highly practical and useful.

4. The model demonstrates robust generalization, showing consistent performance not only on the dataset introduced in this paper but also across other benchmarks.

### Weaknesses
The dataset and its construction process introduced by the authors are quite impressive; however, there are several issues to point out regarding the experimental setup and analysis of the results.

1. In the pilot study for Table 1, how is the label distribution of the 800-sample dataset? Are the labels well-balanced? Since the F1 score depends on the label distribution, it is important to ensure balance in the dataset.

2. Compared to other models, the F1CoT metric of Claude-Sonnet-3.7 is remarkably low (near zero) in Table 1. Since the authors assume that each model has distinct reasoning traces, they should explain how the reasoning trace of Claude-Sonnet-3.7 differs from others and why it produces such a unique result.

3. While it is encouraging that the small model achieves strong reasoning moderation performance with limited data and simple training, the proposed method itself is not particularly novel.

4. In Table 2, the authors mention that they used Llamaguard’s system prompt for the prompted LLM. However, they should also present results using the ReasoningShield model’s system prompt. As seen in Appendix J.1, ReasoningShield’s system prompt is highly detailed and provides extensive response-generation guidelines, whereas the Llamaguard prompt shown in Appendix J.4 is much simpler. The authors must include this experiment and compare the results.

5. In Table 2, ReasoningShield and Llamaguard perform a three-way classification, and the “potentially harmful” category is merged with “unsafe” for computing the F1 score. In contrast, the prompted LLM performs a purely binary classification task. This difference leads to unequal random baselines: for ReasoningShield and Llamaguard, the random chance of predicting “unsafe” is 2/3 and “safe” is 1/3, whereas for the prompted LLM, both classes have a 1/2 chance.
Since the F1 score considers both precision and recall, this discrepancy introduces unfairness between the two approaches. The authors must provide a justification for this.



+) Typo
L193 WWe->We 
L426 cgeneralization->generalization

### Questions
Questions are in listed in weakness section.

### Soundness
2

### Presentation
3

### Contribution
3
