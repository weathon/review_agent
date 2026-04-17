# Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 2

## Abstract
Large language models (LLMs) are reshaping the recommender system paradigm by enabling users to express preferences and receive recommendations through conversations. Yet, aligning LLMs to the recommendation task remains challenging: pretrained LLMs often generate out-of-catalog items, violate required output formats, and their ranking quality degrades sharply toward the end of the generated list. To this end, we propose ConvRec-R1, a two-stage framework for end-to-end training of LLM-based conversational recommender systems. In Stage 1, we construct a behavioral-cloning dataset with a Remap-Reflect-Adjust pipeline, which produces high-quality, catalog-grounded demonstrations from powerful blackbox LLMs to warm-start the RL training. In Stage 2, we propose Rank-GRPO, a principled extension of group relative policy optimization (GRPO) tailored to tasks with rank-style outputs. Rank-GRPO treats each rank in the recommendation list as the unit instead of token (too fine-grained) or sequence (too coarse), redefining rewards to remove non-causal credit assignment and introducing a rank-level importance ratio based on the geometric mean of rank-wise token probabilities to stabilize policy updates. Experiments on the on the Reddit-v2 and Redial datasets show that ConvRec-R1 converges faster and achieves higher Recall and NDCG than GRPO-style baselines. Code and datasets are released at https://github.com/yaochenzhu/Rank-GRPO.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces R1 style reinforcement learning and self improvement techniques into conversational recommender systems. It first develops a remap reflect adjust pipeline to improve data quality, and then introduces a ranking GRPO reinforcement learning objective with reward shaping techniques for performance improvement. Experiments demonstrate that it can improve the performance of LLM-based conversational recommender system.

### Strengths
1. The presentation of the paper is clear.
2. The proposed method shows performance improvement.
3. Case studies show that after RL, the ranking output of the model is better, demonstrating that rank-GRPO is functioning properly.

### Weaknesses
1. It will be better if the method can be tested on more datasets. Current only one dataset is used.
2. It will be better if other aspects of conversation quality of the framework such as helpfulness and informativeness, other than the ranking accuracy, can be evaluated.
3. Some claims can be more rigorous. For example, authors claim GRPO is fundamentally misaligned with tasks with rank-style outputs, but the proposed method is still a simple extension of GRPO. If GRPO is fundamentally misaligned, the proposed framework might be something very different from GRPO.

### Questions
Is the method senstive to the predefined prompt templates or is it robust to different templates?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents ConvRec-R1, a two-stage framework for training LLM-based conversational recommender systems. It combines a supervised fine-tuning pipeline (Remap-Reflect-Adjust) with a new reinforcement learning method (Rank-GRPO) that optimizes recommendations at the rank level. The authors present experiments on REDDIT-V2 that demonstrate solid improvements over baselines, achieving near-GPT-4 performance with smaller open-source models.

### Strengths
1) The paper addresses an important and emerging problem i.e aligning LLMs for conversational recommendation, which has practical utility for the industry and is a relevant problem for the community.
2) The authors provide code, data, and detailed implementation notes, making the work easy to reproduce and build upon.
3) The proposed rank-level GRPO is a generally useful method for ranking tasks. It's well motivated, includes gradient analysis, and performs reasonably well in practice.
4) The experimental section is detailed and comprehensive, covering different model scales, baselines, and both SFT and RL stages.

### Weaknesses
1) The core ideas, namely supervised fine-tuning plus RL alignment, primarily extend existing GRPO and RLHF frameworks, without introducing a fundamentally new paradigm.
2) The approach is tightly focused on conversational recommendation and may not generalize well to broader LLM alignment or other ranking tasks, which could limit the paper's impact.
3) The performance improvement over strong prompting baselines (e.g., CRAG) is modest given the added training complexity, and under off-policy settings the method can perform even worse, suggesting limited robustness and stability.
4) All experiments are conducted on the REDDIT-V2 dataset, leaving open the question of how well the method generalises to other domains or item catalogs.
5) The work lacks online or human evaluation, making it unclear whether the improvements translate to better real-world user experience.

### Questions
1) The paper only uses the REDDIT-V2 dataset without sufficient justification. For a research paper, relying on a single dataset is insufficient to demonstrate generality. Why weren't other conversational recommendation datasets considered?

2) ConvRec-R1 performs worse than CRAG under off-policy evaluation, despite its more complex training. What causes this drop? reward misalignment, data shift, or instability in Rank-GRPO? A clearer analysis would strengthen the paper's claims.

3) The paper reports results on models only up to 3B parameters. Have the authors tested or considered larger LLMs to see whether the proposed Rank-GRPO continues to scale effectively with model size?

### Soundness
2

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
The paper proposes ConvRec-R1, a two-stage approach to prevent LLMs from generating out-of-catalog items and to avoid sharp drops in ranking quality. Stage 1 performs supervised fine-tuning (SFT) on high-quality synthetic data produced by GPT-4o; Stage 2 applies Rank-GRPO, an extension of GRPO tailored for ranking optimization. Experimental results demonstrate that ConvRec-R1 achieves strong performance on the evaluated benchmarks.

### Strengths
The paper is well-motivated.

The paper is readable.

### Weaknesses
1. **Performance on larger models (e.g., 7B) is unclear.**
    
    Please provide experimental results or discussions on how the proposed method scales to larger backbones (e.g., 7B parameters). This will help verify whether the observed improvements generalize across model sizes.
    
2. **Baselines in Table 1 are insufficient.**
    
    Table 1 should include more **post-training baselines** specific to LLM-based recommender systems, rather than comparing only SFT or SFT + GRPO. Incorporating recent LLM-agent or ranking-enhanced recommender baselines would make the comparison more convincing.
    
3. **A more detailed ablation study is needed.**
    
    Please include ablation experiments isolating the effects of key components, such as **without remap**, **without reflect**, and **without SFT**. These results would clarify each module’s contribution to the overall performance.

### Questions
see weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This manuscript introduces ConvRec-R1, a two-stage framework for training LLM-based conversational recommender systems. The first stage uses a novel "Remap-Reflect-Adjust" data distillation pipeline to create a high-quality, catalog-aware dataset for SFT. The second stage introduces Rank-GRPO, a new RL algorithm tailored for ranking tasks. Rank-GRPO reframes the RL update unit to the "rank position" level, addressing fundamental flaws in standard GRPO like non-causal credit assignment. Experiments show the framework improves recommendation quality (Recall and NDCG) and allows smaller open-source LLMs to outperform larger, zero-shot models.

### Strengths
1.	Rank-GRPO is a main contribution. The manuscript clearly identifies the core weaknesses of applying standard GRPO to ranking tasks and proposes an new solution by re-framing the problem at the rank level. The technical solutions are well-motivated and supported by theoretical analysis.
2.	The "Remap-Reflect-Adjust" pipeline is a significant engineering contribution that provides a sophisticated solution to the critical data scarcity problem in this domain. It is quite practical for researchers and practitioners aiming to deploy LLM-based CRS.
3.	The experimental design is robust. The results clearly show the value of each stage of the framework and validate the superiority of Rank-GRPO over its baselines. The finding that a well-aligned 3B model can outperform a much larger zero-shot model is certainly relevant.

### Weaknesses
1.	The manuscript utilizes an “LLM as a judge” in its data pipeline but does not adequately discuss or account for known biases of this paradigm, such as position or verbosity bias, which could affect the quality of the SFT dataset.
2.	All experiments are conducted on a single dataset in the movie domain. The manuscript would be stronger with a discussion on the potential challenges of applying the framework to other domains like e-commerce or music.
3.	The manuscript notes that the model's outputs tend to drift out-of-catalog during RL training. This practical limitation deserves more prominent discussion in the main text, including the final out-of-catalog rate and potential mitigation strategies.
4.	The primary results table (Table 1) omits a direct on-policy comparison with GSPO, a relevant and stronger baseline discussed elsewhere in the manuscript. Including this would make the evaluation of Rank-GRPO's advantage more complete.
5.	The multi-step SFT data pipeline involves a lot of components and hyperparameters. The manuscript lacks an analysis of how sensitive the final data quality is to these choices, which would be valuable for reproducibility and practical application.

### Questions
1. Can you elaborate on measures taken to mitigate potential biases (e.g., position bias) from the LLM-as-a-judge used in the "Reflect" step?
2. What is the final out-of-catalog recommendation rate on the test set, and what are potential strategies to better enforce catalog constraints during the RL phase?
3. Why was the GSPO baseline omitted from the main on-policy results in Table 1, given it is a key point of comparison?
4. How sensitive is the model's performance to the new penalty hyperparameters (ϵ_u, ϵ_o) introduced in Rank-GRPO?

### Soundness
1

### Presentation
2

### Contribution
2
