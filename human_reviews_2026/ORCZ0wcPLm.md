# MM-HELIX: Boosting Multimodal Long-Chain Reflective Reasoning with Holistic Platform and Adaptive Hybrid Policy Optimization

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
While current Multimodal Large Language Models (MLLMs) have demonstrated proficiency in reasoning tasks such as mathematics and logic,  their capacity for long-chain reflective reasoning, a prerequisite for solving complex real-world problems, remains largely underexplored.  In this work, we first conduct an extensive empirical investigation to evaluate this capability. Leveraging a carefully designed data synthesis engine, we construct MM-HELIX, a multimodal benchmark consisting 1,260 samples of 42 challenging synthetic tasks that require iterative thinking and backtracking.  Empirical results on this benchmark reveal that existing MLLMs exhibit significant performance deficits in long-chain reflective reasoning.  To address this limitation, we generate post-training data and further explore learning paradigms for exploiting such data.  We first develop the Step-Elicited Response Generation pipeline to create MM-HELIX-100K, a large-scale dataset of 100k high-quality, reflective reasoning traces for instruction-tuning stage. Given that standard Reinforcement Learning fails on complex tasks due to sparse reward signals and catastrophic forgetting after Supervised Fine-Tuning,  we propose Adaptive Hybrid Policy Optimization (AHPO), a novel training strategy that dynamically unifies offline supervision and online optimization into a single stage. This strategy enables the model to learn from expert data when rewards are sparse and conduct independent exploration once proficient.  When applied to the Qwen2.5-VL-7B baseline, our method achieves a +18.6\% accuracy improvement on MM-HELIX benchmark and demonstrates strong generalization with a +5.7\% average performance gain on general mathematic and logic tasks. Our work demonstrate that reflective reasoning in MLLMs can be effectively learned and generalized, paving the way for developing more capable MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents MM-HELIX, a novel multimodal benchmark designed to evaluate MLLMs on long-chain, reflective reasoning tasks that require iteration and backtracking. MM-HELIX consists of 42 diverse, programmatically generated tasks across four domains—algorithms, graphs, puzzles, and games—spanning five difficulty levels. Through evaluation on this benchmark, the authors identify key shortcomings in existing MLLMs and propose a training pipeline that features: 1. SERG for generating large-scale reflective chain-of-thought data, and 2. AHPO for integrating off-policy SFT and on-policy RL. Empirical results demonstrate significant improvements on both MM-HELIX and standard mathematical and logic benchmarks, highlighting strong generalizability.

### Strengths
**1. Originality:** The work makes notable contributions through (1) the MM-HELIX benchmark targeting multimodal long reasoning, (2) the SERG pipeline combining rule-based anchors with LLM refinement for large-scale CoT data generation, and (3) AHPO's dynamic integration of off-policy SFT and on-policy RL to address reward sparsity and catastrophic forgetting.

**2. Quality:** The technical execution is strong, with SERG demonstrating efficiency and quality, rigorous AHPO formulation (Section 3.3), and comprehensive experiments showing substantial improvements on MM-HELIX and generalization to math/logic benchmarks (Table 2). The ablations (Figures 6-7) effectively validate the design choices.

**3. Clarity:** The paper is well-structured with clear problem formulation and effective visualizations (Figures 1-3) that illustrate the benchmark design and system architecture. Additionally, the motivation for training methods that elicit extended reasoning from open-source MLLMs is well established.

### Weaknesses
**1. Benchmark Synthetic Nature / Coverage:** Although MM-HELIX procedurally generates tasks and spans many categories, all tasks appear synthetic and rule-based (see Figure 2 and example in Figure 3). There is no clear discussion or experiment detailing how well the benchmark reflects the “real-world” complexity or ambiguity of practical reasoning. Similarly, the tasks' diversity outside logic/math settings is not addressed. This may limit the external validity of the results.

**2. Limited Evaluation Metrics:** The evaluation in Table 1 and the Appendix relies almost exclusively on raw accuracy, with limited discussion of more nuanced aspects such as error typologies or qualitative analysis beyond completion rates. Given that the benchmark aims to probe "reflective" capabilities—a term used throughout the paper—a deeper analysis of response length, backtracking frequency, and self-correction behaviors would strengthen the evaluation and provide insights into how models engage in iterative reasoning.

**3. Equation Clarity and Notational Issues:** Although the main equations in Section 3.3 are written out, some aspects (but not limited to) lack precision for reproducibility:
Equation defining $N$ could be more clearly defined as number of roll-outs, and the use of $\operatorname{CLIP}$ could be further detailed.
The loss unification assumes normalization factor $Z$, but this is undefined and never discussed again.
The citation in line 305 should be added.
The off-policy and on-policy learning is uncleared whether they are applied to the same training data or different data point.

**4. Ablations Analysis Limited:**
There is also no ablation analysis on hyperparameters introduced in Equation 5, despite their potential importance.

### Questions
1. How are the difficulty levels in the benchmark determined? Is there a principled methodology or validation process?

2. Performance Disparities Across Domains: Open-source models perform substantially worse on Games and Puzzles compared to Algorithms and Paths (Table 1 and Appendix). What are the failure patterns underlying these disparities? Are they primarily due to reasoning capability limitations, context length constraints, instruction-following difficulties, or other factors? A more fine-grained error analysis could provide valuable insights.

3. In SERG, the initial step provides logically sound trajectories. However, beyond oracle trajectories that directly reach the correct solution, could trajectories that incorporate rethinking, reflection, or error handling be more valuable for developing the model's generalizable reasoning capabilities? At the second step, what specific prompts are used to transform rule-based trajectories into more natural, comprehensive reasoning processes with reflective steps? How is diversity in reasoning behaviors (mentioned above) controlled/encouraged during this refinement stage? 

4. Are on-policy and off-policy learning applied to the same data points or separate sets? How is the number of samples per training step determined? Are there ablation studies on key hyperparameters, such as the ratio of on-policy to off-policy samples or the success rate threshold?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces MM-HELIX, a multimodal benchmark of 42 tasks (algorithms, graphs, puzzles, games) with 1,260 procedurally generated instances designed to test long-chain reflective reasoning in MLLMs. It further proposes a Step-Elicited Response Generation (SERG) pipeline to build MM-HELIX-100K, a dataset of ~100k verified reasoning traces, and a training algorithm Adaptive Hybrid Policy Optimization (AHPO) that adaptively mixes off-policy supervision with on-policy exploration via a reward-gated coefficient. On MM-HELIX, state-of-the-art models underperform (e.g., strong open-source models <35% acc.; large proprietary models ≤58.1% multimodal), while AHPO on Qwen2.5-VL-7B improves accuracy by +18.6 points and yields +5.7 points average gains on external math/logic benchmarks. The work argues that reflective reasoning in MLLMs is learnable and transferable.

### Strengths
- Well-posed evaluation target. MM-HELIX isolates long-chain reflective reasoning with verifiable tasks, hierarchical difficulty, and balanced sampling (1,260 instances across 42 tasks), enabling controlled diagnosis
- SERG uses rule-based scaffolds refined by a strong LLM, then filtered by verifiers to produce 100k reflective traces with far lower time/cost than naive rollouts and with better downstream utility
- AHPO’s reward-gated mixing of off-policy expert loss and on-policy GRPO addresses reward sparsity and SFT-forgetting. The authors show the learning curves of AHPO and demonstrate consistent gains against different baseline algorithms (GRPO/LUFFY/SFT)

### Weaknesses
- Equation (6) describes ξ’s activation by a success-rate threshold \hat{R}, but the exact estimator (win-dow size, grouping), threshold choice, and sensitivity are not fully ablated. Providing an ablation over \hat{R} and these different parameters would be helpful to get a deeper understanding of the algorithm.

### Questions
- I'd love to see some abaltion on the success rate threshold and how would that impact the perfomrance
- For generation pipeline, Did you evaluate sample efficiency when using fewer than 22k traces? Is there a particular reason for the "22k" number? Would be interesting to see the scaling behavior of the generated samples (i.e., if 5k SERG vs. 5k rule-based, how would the performance comparison look like).

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
4

### Summary
This paper addresses the weakness of MLLMs in long-chain reflective reasoning. It introduces MM-HELIX, a benchmark of 42 complex multimodal tasks and 1260 samples across difficulty levels. The authors also build MM-HELIX-100K, a dataset of 100k verified reflective reasoning traces generated via the SERG pipeline. They also propose AHPO, a training algorithm that adaptively balances expert supervision and reinforcement learning to improve reasoning stability and exploration.

### Strengths
1. The paper presents a substantial amount of work, including both a benchmark and a dataset, as well as a proposed improvement method. It makes a meaningful contribution to research on MLLM reasoning tasks.
2. Based on the proposed dataset and methodology, the paper achieves significant improvements across various visual reasoning benchmarks. The proposed AHPO effectively alleviates the issue of insufficient high-quality rollouts when the model faces challenging tasks. Compared to previous hybrid method LUFFY, AHPO demonstrates a considerable performance gain.

### Weaknesses
My main concern is that the paper lacks many important implementation details, including those related to the benchmark, dataset, and ablation studies of the proposed method. These missing parts may confuse readers and make the work difficult to reproduce.

Benchmark:
1. What are the text and image settings during evaluation, and how are image-based tasks converted into textual form?
2. What are the criteria for difficulty categorization? The paper only briefly mentions this. Is the classification done manually?
3. What are the implementation details of the automated verifier?

Dataset:
1. The paper lacks detailed information on data construction, including prompts and heuristic rules.
2. There is no statistical breakdown of the proportions of different task types within the dataset.

Experiments and Analysis:
1. In Section A.8.2, the improvement on Graph-type tasks is smaller than on other types, but there is no accompanying analysis.
2. The base model used for training should be explicitly stated in the experimental setup, even if readers can infer it from Table 2.
3. In Figures 6 and 7, why is only static-AHPO compared with other algorithms, while Figure 7 also includes a non-static version(AHPO). Does static-AHPO mean the activation coefficient = 0?
4. In Appendix A.2 (Training Setting), why are only 22k samples used for SFT instead of the full dataset? Is there any comparison showing the relationship between data volume and performance, and how were these 22k samples selected?
5. Why is math data included during the RL stage? Was the AHPO model in Table 2 also trained with math data? If so, do benchmarks such as MathVision and MathVerse fall into the in-domain category? If that’s the case, does the claimed generalization ability still hold? Also, is there any comparison between models trained with and without math data?
6. [Minor] The paper lacks ablation studies on different success rate thresholds and activation coefficient.
7. [Minor] Is there any comparison showing how the CoT length in MM-HELIX-100K affects reasoning performance?

Typo:
Line 305–306: “following()”

Others:
[Minor] I suggest that the authors move Sections A.5–A.7 to the end of the appendix, so that readers don’t have to scroll extensively or risk overlooking the subsequent content. Alternatively, at the very least, an appendix table of contents should be added.

If the authors address most of my concerns, I would be willing to raise my score.

### Questions
See weaknesses

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the deficiency of current Multimodal Large Language Models (MLLMs) in long-chain reflective reasoning. The authors make three main contributions: (1) MM-HELIX, a benchmark consisting of 1,260 samples across 42 challenging tasks designed to evaluate multimodal reflective reasoning capabilities; (2) MM-HELIX-100K, a large-scale dataset of 100k high-quality reflective Chain-of-Thought (CoT) traces generated via the Step-Elicited Response Generation (SERG) pipeline; and (3) Adaptive Hybrid Policy Optimization (AHPO), a novel training algorithm that dynamically integrates off-policy expert guidance with on-policy exploration. When applied to Qwen2.5-VL-7B, their method achieves +18.6% accuracy improvement on MM-HELIX and +5.7% average performance gain on general mathematics and logic tasks, demonstrating effective learning and generalization of reflective reasoning capabilities.

### Strengths
- Novel focus on end-to-end multimodal reflective reasoning, going beyond existing benchmarks that use multiple-choice formats
- Rigorous benchmark construction with automated verification ensuring correctness
- Well-structured paper with clear methodology sections
- Practical improvements on both in-domain and out-of-domain tasks

### Weaknesses
- Missing analysis of under what conditions AHPO would fail or succeed
- Limited error analysis showing what types of mistakes the model makes
- Training mix includes both MM-HELIX-100K and MMK12; unclear how much each contributes
- No analysis of diversity in generated CoT traces (might all be similar)

### Questions
- Can you ablate training on MM-HELIX-100K only vs. MMK12 only to understand each dataset's contribution?
- Can AHPO work with smaller base models (e.g., 1B-3B parameters)?

### Soundness
3

### Presentation
3

### Contribution
3
