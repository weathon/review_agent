# Proof-Verifier: Enabling Reinforcement Learning from Verifiable Rewards for Mathematical Theorem Proving

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 8, 2

## Abstract
Reinforcement Learning from Verifiable Rewards (RLVR) has revolutionized mathematical reasoning, enabling models like DeepSeek-R1 and OpenAI-o1 to achieve human-level performance on traditional math tasks where answers are single numbers or equations. However, extending RLVR to mathematical theorem proving remains challenging due to the fundamental verification bottleneck: unlike traditional math tasks, theorem proving generates entire reasoning processes that lack reliable automated verification methods for reward signal generation.
In this work, we address this verification bottleneck by introducing Proof-Verifier, the first generative verifier specifically designed to enable RLVR applications in mathematical theorem proving. Proof-Verifier supports both formal and informal language~(e.g., natural language) proofs, providing the detailed verification capabilities essential for effective reinforcement learning. To train Proof-Verifier, we develop a formal-to-informal translation pipeline for high-quality synthetic data generation and employ a novel two-stage coarse-grained to fine-grained reward modeling mechanism.
Experimental validation demonstrates that Proof-Verifier achieves 93\% verification accuracy, enabling reliable reward signals for RLVR applications. We show that Proof-Verifier successfully enables effective test-time scaling (79\% win rate in best-of-N sampling and 32\% improvement in multi-turn proof refinement), and both single-turn and multi-turn RLVR training, consistently improving LLM-based theorem proving performance. Our work establishes the foundation for applying RLVR methodologies to mathematical theorem proving, extending the recent success of reasoning-enhanced models to this challenging domain.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces PROOF-VERIFIER, which enables reliable verification for informal proofs during RLVR. The core contribution is a two-part approach: (1) a data-synthesis pipeline that translates formal proofs to informal ones, and (2) a two-stage reward modeling framework to train the verifier. The authors demonstrate the verifier's high accuracy and its utility in test-time scaling (Best-of-N, refinement) and preliminary RL experiments.

### Strengths
1. The core idea of using formal language verification to train a reward model for informal reasoning is relevant and novel. The formal-to-informal translation pipeline is clever.
2. The experiments in Sections 3.1and 3.2 effectively demonstrate that the resulting verifier achieves high verification accuracy and correlates well with human judgment. The ablation studies also clearly validate the contributions of the proposed data and training components.
3. The inclusion of case studies is helpful for qualitatively understanding the verifier's capabilities in error analysis and feedback generation.

### Weaknesses
1. The most significant weakness is the lack of sufficient experimental validation for the paper's central claim, "enabling RLVR". The paper proposes a verifier for RLVR, but it does not adequately demonstrate its effectiveness in an RLVR training loop. The experiments in Section 4.3 are limited. To be convincing, the authors should have: (1) Conducted a full-scale experiment using their reward model to fine-tune a prover LLM. (2) Evaluated the RL-trained prover on established benchmarks for informal mathematics, such as AIME, and compared this model's performance against strong baselines, such as the base model, a model trained with a different RM, and a model trained with ground-truth rewards.
2. The experimental details in Section 4.2 are critically lacking. The authors state results like "human annotators found that 73% of the feedback..." and "refinement improved pass@k performance from 37% to 51%," but they do not clearly specify the dataset, the exact experimental setting, or the protocol for this evaluation. This makes the results impossible to reproduce.
3. The paper's writing is also weak. The abstract is vague, and the pseudocode occupies a lot of space but does not convey messages clearly.

### Questions
1. The Lean4 compiler provides more than just a binary "True/False" label. It also provides specific error messages. Does this rich error information have the potential to help train the verifier?
2. In Figure 2, the score distributions for both "Human" and "Ours" appear to be strongly multimodal (having multiple peaks). This is somewhat unnatural for a general quality score. Does this reflect the underlying data distribution? Or is it a bias introduced by the scoring metric?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents Proof-Verifier, a novel framework that in-formalizes formal proof results into natural language and trains a proof verifier. Proof-Verifier proposes to use a novel two-stage training strategy, where in the first stage the training aims for overall binary accuracy and in the second stage optimizes for fine-grained scores for problem and proof pairs. The experiment shows the effectiveness of the proposed method, showing good performance in the model's verification accuracy and high correlation to the score of human annotators. The author also provide good theoretical anaylsis in the paper.

### Strengths
- **Novelty and Motivation.** The paper's proposed method is interesting and novel, and is well-motivated. The problem it aims to solve is of great importance in the domain of informal math reasoning. The paper presents a first attempt at trying to use formal feedback to bootstrap a model's informal verification ability, which is of ** significant importance**.

- **Clarity and Experimental Rigor.** The paper is well-written and provides rich experiments. It also provides a theoretical analysis of the proposed method.

- Overall, I think the approach of leveraging the formal proof to help train an informal verifier is well motivated, but also very challenging as the distribution shift exists. But the paper lacks enough experimental results to show that their model effectively addresses this problem.

### Weaknesses
- **Distribution Misalignment Between Formal and Informal Proofs.** The problem of distribution misalignment between formal proof errors and informal solver errors is a major concern. The core idea of using formal proof as a supervision signal to create a verifier has a serious flaw: the errors made by the informal solver might not align with the errors made by the formal proof code. Therefore, it is in doubt that the verifier trained using this data can effectively identify mistakes made by the informal solver.

- **Limited Experimental Comparisons and Evaluation.** The verification accuracy results are shown in Table 1. However, the comparison is incomplete. It would be better to also compare these results with some of the proprietary models like Gemini 2.5 Pro and GPT-5, which are shown in the OPC paper to be the top verification models for informal mathematics.

- **Insufficient Confirmation of Verifier Effectiveness.** The paper's confirmation of the trained verifier's effectiveness is weak. While using the verifier to filter results from the prover is a good step, the paper only provides the result for win rate against the DeepSeed R1 model, which might not be a strong enough competitor to truly confirm the verifier's utility.

### Questions
- Can you provide more experiment results for your own benchmark using GPT-5, Gemini 2.5 Pro, or other frontier models?
- Can you provide the results of your model on a benchmark like OPC, also the best-of-n result on verifiable tasks like AIME? To see the trained verifier have enough ability to generalize to finding errors made by informal solvers.
- (minor) It would be better to use et al. for papers with too many authors in the References.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper trains the first dual-lingual math proof verifier for both formal and informal math. Based on Qwen3-8B, the model is trained by reinforcement learning to output the score, feedback, and the correct/incorrect judgment for the input proof. Two-phase training is adopted where the first phase trains the correct/incorrect judgment with verification reward and consistency reward, and the second phase trains the score and feedback by rewarding the score improvement of provers having received the feedback. The prover achieves 93% verification accuracy on test datasets. The downstream prover achieves further improvement by RLVR with the reward given by the verifier.

### Strengths
Process verification has been a bottleneck for improving LLM's reasoning ability by RL. The paper addresses this important challenge by curating a bilingual dataset with the verifiable proof from lean, translated formal proof, and annotated proof by LLM-as-a-Judge. The dataset contributes to a 9% improvement in verification accuracy when augmented with the baseline OPC dataset. The model post reinforcement learning has achieved a verification accuracy at 93%, which indicates a very well-calibrated reward model. The approach has also results in the first bi-lingual language verifier. The reward model is generative, which provides not only a score but also feedback. The feedback refinement is shown effective both before and after RLVR.

### Weaknesses
1. The informal test dataset has 100 samples annotated by experts while the rest are labeled by Gemini-2.5-pro. The results in the paper can then be interpreted as consistency verification with Gemini-2.5-pro, which is fine for an exploration study.

2. The challenge for training a reward model is generalizability to trajectories sampled from unseen models. It is unclear whether the informal test dataset are sampled from the same series of models that are used to generated proofs for OPC and RFM in the training set.

3. It will be more convincing to benchmark the method against some SOTA generative reward model.

4. What is single-sample baselines through human preference evaluation in section 4.1? A more classic approach to evaluate the reward model is to compare best accuracy of N and accuracy of the best score of N provided by the trained reward model, across N=powers of 2.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper suggests a multi-stage method for training generative reward models for natural language and formal theorem proving. The pipeline consists of a coarse stage where the reward is given by ground-truth correctness labels, and a fine stage where a GRPO algorithm is used to reward textual feedback by the reward score improvements it allows a prover to achieve based on it.

### Strengths
- The paper tackles a highly relevant topic (generative reward modeling to move beyond RLVR) and identifies various techniques and tricks that could be used for training generative reward models.
- The experimental results show that the trained verifier produces valuable feedback.

### Weaknesses
- The method is not compared to other papers and methods (e.g. getting ground truth labels not from informalization of formal proofs but from human-written math corpora with and without perturbations applied).
- The setup "we train a good verifier" is hard to evaluate compared to the more standard question in the literature: "with our verifier, we get better RL numbers than without".
- The win rate comparison lacks a baseline that also uses best-of-N with respect to some reward model, not just comparison to single-shot.
- Verification of formal proofs does not seem a particularly useful target (Table 1 right).
- It is unclear which parts of the pipeline are really useful (there's hardly any difference between baseline and final method in the method ablation).
- Numbers given such as correlations are somewhat incomplete: what is being correlated with what? For binary classification, the full confusion matrix should be given, and not just "correlation" but TPR, TNR, acc, F1 etc. 
- It is surprising that initial multi-turn behavior is conjectured to degrade over the course of a trajectory, I would expect initial multi-turn numbers to increase monotonically with more compute until saturation?
- The main body does not give fundamental details about the experimental setup such as the models used.

Generally, the paper attempts to do many things but doesn't manage to be fully convincing for any of them. As a high-level feedback, I would suggest coupling out the "tech report" ("what we did for this project") from 1-2 scientific papers that go in depth for a crucial design decisions and attempt to convince the readers that THIS should be the way to go in this specific subquestion.

### Questions
-

### Soundness
3

### Presentation
1

### Contribution
2
