# Probing to Refine: Reinforcement Distillation of LLM Reasoners via Explanatory Inversion

- Avg Score: 5.67
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6, 6, 6

## Abstract
Distilling robust reasoning capabilities from large language models (LLMs) into smaller, computationally efficient student models remains an unresolved challenge. Despite recent advances, distilled models frequently suffer from superficial pattern memorization and subpar generalization. To overcome these limitations, we introduce a novel distillation framework that moves beyond simple mimicry to instill a deeper conceptual understanding. Our framework features two key innovations. \underline{\textit{First}}, to address pattern memorization, Explanatory Inversion (EI) generates targeted ``explanatory probes'' that compel the student to articulate the underlying logic behind an answer, rather than just memorizing it. \underline{\textit{Second}}, to improve generalization, Explanatory GRPO (\texttt{EXGRPO}) uses a reinforcement learning algorithm with a novel Dialogue Structure Utility Bonus, which explicitly rewards the student for maintaining a coherent reasoning process across these probes. Extensive evaluations on 12 datasets demonstrate significant improvements. Using Gemma-7b as the student model, our method yields an average \textbf{20.39\%} increase over zero-shot performance and a \textbf{6.02\%} improvement over the state-of-the-art distillation baselines. Moreover, models distilled with our method show remarkable training efficiency (e.g., surpassing vanilla fine-tuning with \textbf{10-25\%} training data) and strong generalization to out-of-distribution tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a novel framework for distilling reasoning capabilities from large language models (LLMs) into smaller, efficient student models. It addresses the limitations of traditional knowledge distillation, which often leads to pattern memorization and poor generalization. Thus, authors proposed Explanatory Inversion (EI) that asks teacher model to generate explanatory probes that challenge the student model to explain the logic behind answers.

To distill a student model, it has three-stages:
1.	Generate EI probes using a teacher model.
2.	Supervised fine-tuning (SFT) on curated EI-augmented data.
3.	Reinforcement learning via ExGRPO with structured probe dialogues with Dialogue Structure Utility Bonus (DSU).

The student models are Gemma-7B-it and Qwen2.5-7B-Instruct. And the teacher model is Gemini-1.5-Pro. Authors tested on 8 different reasoning tasks (SQA, GSM8k, ANLI). The proposed method improves 6.02%  over SOTA distillation method RevThink. Ablation studiy shows that SFT warm-up and LSFT-aux regularization are crucial for stable RL training.

### Strengths
Authors provide a comprehensive evaluation: Includes in-distribution and out-of-distribution benchmarks, and provides a solid case study that demonstrates improved reasoning in math and commonsense tasks, with structured logic and fewer distractor errors.

Authors introduce novel methods: Explanatory Inversion and ExGRPO, which combine structured explanatory probes with reinforcement learning

### Weaknesses
The paper compares against zero-shot and SFT baselines, but does not include recent reasoning-focused RL distillation methods (e.g., [Divide-or-Conquer](https://aclanthology.org/2024.findings-emnlp.145.pdf), [CoT-Evo](https://arxiv.org/abs/2510.13166v2), and [On-Policy Distillation](https://arxiv.org/abs/2306.13649)). 

The quality of explanatory probes heavily depends on Gemini-1.5-Pro. This raises concerns about scalability and reproducibility for researchers without access to such a strong teacher model. Could authors try open-source alternatives, such as llama70B, for probe generation.

### Questions
The paper highlights success cases but does not deeply analyze where ExGRPO fails.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ExGRPO, a reinforcement-learning–based distillation framework that combines Explanatory Inversion and a Dialogue Structure Utility reward to enhance the reasoning capability and generalization of student LLMs. Empirical results are reported on 12 reasoning datasets, suggesting gains over distillation and data-augmentation baselines.

### Strengths
- The work addresses a genuine problem: retaining reasoning ability during LLM distillation.

- A relatively large evaluation suite with 12 datasets and OOD tests are included.

- The paper is well organized with clear figures and pseudo-formal derivations.

### Weaknesses
- The so-called Explanatory Inversion resembles existing reverse-reasoning or bidirectional augmentation ideas (e.g., “A→Q” vs. “Q→A” reversals) rather than a fundamentally new concept.

- No ablation quantifies whether improvements stem from RL fine-tuning, extra teacher tokens, or the EI data itself. Table 2 mixes multiple knobs (SFT, RL, DSU) but lacks isolating the effect of “explanatory probing.”

- Even the teacher’s “Zero-shot-EI” performance improves, implying that EI augmentation changes the test distribution itself; this raises the possibility of data leakage or prompt-format bias.

- The statistical significance of the improvements is unreported. 

- The filtering pipeline (Eq. 1–2) relies on teacher predictions but gives no statistics on rejection rates or dataset sizes after filtering.

### Questions
Please see Weaknesses

### Soundness
3

### Presentation
2

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
This work proposes a new framework for distilling robust reasoning from large language models into smaller ones. It introduces Explanatory Inversion (EI) to combat pattern memorization by prompting students to explain their reasoning, and Explanatory GRPO (EXGRPO) to enhance generalization via a reward for coherent reasoning. On 12 datasets, the method improves student model performance, training efficiency, and generalization ability.

### Strengths
- The paper is clear in writing and presentation, which is easy to follow.
- The idea is intuitive and explores the reasoning chain constructions of the LLMs, which helps the student model better learns the principles instead of the patterns from the dataset.
- The results are strong and comprehensive, with significant improvement margins and many ablation studies.

### Weaknesses
- The paper used a Dialogue Structure Utility Bonus if the student is engaging in the full k-turn probing dialogue, which leads to better overall outcomes than a partial dialogue. How can the authors prevent reward hacking from using this reward bonus?
-  Why do the authors pick the GRPO-based objective for the RL training? Is there any intuition or reason behind that?
- How does EXGRPO training efficiency compare with other baseline methods?

### Questions
See weaknesses.

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
The paper proposes a model distillation method. It first constructs a high-quality EI training set, ensuring each problem includes a reasonable reasoning expansion, preserves the original logic, and has appropriate difficulty; SFT is then performed on this data. It then introduces ExGRPO, designing rewards based on a Dialogue Structure Utility Bonus to carry out reinforcement-learning-based distillation. Systematic experiments on two student models (Gemma-7B-it and Qwen2.5-7B-Instruct) show improvements over strong baselines on multiple OOD evaluations.

### Strengths
1. The method is well designed and addresses practical issues in model distillation.
2. The paper proposes the Explanatory Inversion strategy and carefully engineers a large set of prompts.
3. The paper introduces the Dialogue Structure Utility Bonus as the reward in reinforcement learning; this design is somewhat innovative.
4. The paper provides thorough comparative experiments and analyses.

### Weaknesses
1. Data generated via the Explanatory Inversion strategy is essentially a form of data augmentation; this part appears largely engineering-oriented, and the core idea is not particularly new, so the academic contribution is limited.
2. Evol-Instruct [1] presents a method for progressively generating complex instructions from simple ones. Although it does not target model distillation, it bears similarities to the paper’s Explanatory Inversion; the paper should add discussion of such related works.

[1] WizardLM: Empowering large pre-trained language models to follow complex instructions

### Questions
1. Mainly those noted under “Weaknesses.”
2. Line 359: “ablation” is misspelled.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies how to distill not only answers but also reasoning procedures into smaller models. It combines an “explanatory probing” data construction with a reinforcement-learning stage that rewards multi-turn, structured explanations before producing the final answer. Across a wide set of tasks and baselines, the method reports consistent gains, with readable presentation and clear motivation.

### Strengths
1. The paper targets a timely and important problem: moving beyond pattern imitation in distillation toward robust reasoning.
2. The narrative flows well, design choices are motivated, and figures/tables are easy to follow.
3. Empirical coverage is broad. Include many competitive baselines and diverse task families; comparisons are thorough and generally fair.
4. The combination of explanation-oriented probes with an RL objective that prefers structurally coherent multi-turn dialogs is a neat, conceptually coherent idea that fits the stated goal.

### Weaknesses
I think the central question remains unclear: does EI teach “understanding,” or is it primarily stronger data augmentation? The evidence presented is largely behavioral (end-task accuracy on in-domain and held-out sets). This does not disentangle genuine conceptual acquisition from targeted exposure to templated probe distributions. In particular: 
1. The DSU/structural reward is still an outcome-level signal (full probe dialog > partial). It does not by itself show that the model internalizes transferable rules, as opposed to learning to perform longer, EI-style rituals. 
2. If EI fosters understanding, predictions should change in directionally correct ways under counterfactual edits (flip a premise, rename variables, swap symbols, introduce irrelevant modifiers). The paper lacks such invariance/causality diagnostics that would separate concept use from surface patterning.
3. The paper reports little on intermediate-step faithfulness/validity (are the stated steps actually correct?), forward↔reverse consistency (e.g., reversal or bidirectional tasks), or error localization (does EI reduce spurious but fluent steps). Such process metrics would directly bear on “understanding” rather than augmentation.

### Questions
See weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes the ExGRPO knowledge distillation framework, aiming to enhance the reasoning ability of large language models (LLMs) by combining Explanatory Inversion (EI) and reinforcement learning (ExGRPO), and effectively distill them into smaller student models.Experimental results show that the student model distilled using this method achieves significant performance improvements on multiple datasets. Compared with existing distillation methods, the ExGRPO method reduces pattern memorization and improves the reasoning ability of the student model. In conclusion, I find the paper's argument relatively clear, and I would give it a score of 6.5 or 7.

### Strengths
1.Diverse interpretive probes generated by EI force student models to understand the logic of questions rather than simply memorizing answers, thereby improving their reasoning ability.

2.ExGRPO, through reinforcement learning and dialogue structure utility rewards, encourages student models to maintain consistency and coherence throughout multi-turn reasoning, which helps them understand and apply complex reasoning structures.

3.Compared to traditional knowledge distillation methods, ExGRPO significantly improves student model performance on cross-distribution tasks, especially demonstrating stronger generalization ability when faced with different domains or unseen data.

4.By using data augmentation methods generated by EI, student models can achieve high reasoning ability with less data and fewer training rounds, significantly improving training efficiency, especially suitable for tasks with limited data.

### Weaknesses
1.Using EI probes significantly increases training costs. Could the authors reduce the number of training samples based on EI to achieve the same training cost and better quantify the contribution of EI?

2.The models all appear to be distilled from Gemini-1.5-Pro. Will exgrpo still perform well even with a relatively weak teacher model?

3.The authors compared many distillation methods and EI probes. EXGRPO's design is ingenious, but it does not seem to compare with existing RL methods to highlight its adaptability and effectiveness.

### Questions
Same as above

### Soundness
3

### Presentation
3

### Contribution
3
