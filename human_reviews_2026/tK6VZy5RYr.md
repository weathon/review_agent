# Distill Not Only Data but Also Rewards: Can Smaller Language Models Surpass Larger Ones?

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
**Distillation of large language models (LLMs) has traditionally focused on transferring teacher responses, often assuming access to internal logits. In modern LLM deployment, however, the teacher is typically only accessible as a black-box API or is too large to support online distillation, while simultaneously possessing strong evaluative capabilities that remain underexploited.**
As a result, students learn what to answer, but not which answers are preferable. This gap limits generalization, propagates teacher errors, and prevents students from improving beyond imitation. Therefore, we propose a unified distillation framework that transfers both responses and evaluation ability. Our key idea is to distill reward signals from the teacher, eliminating the need for costly human annotations. However, extracting reliable reward signals from LLMs is challenging because they are optimized for generation rather than evaluation. Therefore, we introduce an adaptive reward distillation strategy that applies majority voting for verifiable tasks and LLM-as-Judge for open-ended tasks.
This yields noisy yet effective self-supervised signals without human annotations. 
To mitigate distribution shift, we systematically collect and label both teacher- and student-generated responses, which are used to train a reward model. The student is first warmed up with supervised fine-tuning on high-quality teacher responses, then refined with reinforcement learning guided by the learned reward model. Experiments on GSM8K, GSM-Plus, MMLU-Pro, and AlpacaEval2 demonstrate consistent gains over supervised fine-tuning, with smaller students in some cases even surpassing their teachers. 
These results highlight our method as a scalable and effective paradigm for training efficient yet competitive LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel KD with RL framework where a student model learns not only from the teacher’s responses but also from the teacher’s evaluation signals, i.e., rewards. The core idea is to train a reward model that aligns with the teacher’s preferences, allowing student outputs to be optimized via PPO against this learned reward signals. The authors also propose separate reward modeling strategies for verifiable and open-ended tasks. Experiments on GSM and MMLU benchmarks show improvements over standard distillation baselines.

### Strengths
- The central idea of jointly distilling both response and reward signals from a teacher LLM is novel and appears effective.
- Separate reward modeling for verifiable tasks and open-ended tasks is a well-motivated design, highlighting that evaluation criteria differ depending on task types.

### Weaknesses
While the motivation and method are promising, the paper lacks deeper analysis to justify whether this RL-based approach is really superior over existing KD methods.

- A core assumption in this paper is that LLMs are unreliable evaluators, which motivates training an explicit reward model. However, training the reward model involves constructing the reward dataset, which itself is built from the teacher LLM judgments of teacher and student responses. If these evaluations are unreliable as claimed, the reward model may inherit their noise and inconsistencies. And if the teacher LLM is good enough to correctly evaluate, there is no need to introduce another reward model. Thus, I’m curious if this reward model has been properly constructed.
- The experimental comparison is limited. One of the primary baselines (**Ours w/o R**) only reflects basic response-level KD using teacher outputs. However, existing KD methods like ImitKD (Lin et al., 2020), GKD (Agarwal et al., 2024), and DistiLLM (Ko et al., 2024) also incorporate student outputs and often use adaptive incorporation with teacher responses. Since the proposed PPO (Eq. 13) uses dataset $\mathcal{D}_{\mathcal{R}}$ which includes responses from both teacher and student, for fair comparison, the paper should include stronger, state-of-the-art KD baselines that similarly exploit student trajectories.

### Questions
- It is unclear how the method performs when the teacher is incapable of providing high-quality responses (i.e., none of the responses reach the target reward $\ell^*$). In such cases, the teacher-student preference dataset $\mathcal{D}_{\mathcal{R}}$ may be skewed toward simpler tasks, limiting the effectiveness of reward model training. Can the method still construct a meaningful reward dataset for those queries?
- Is the reward model robust to noisy or inconsistent teacher scores? How is this addressed during training?

**Minor typos**

- $D_{\text{TeachCan}}$ in Eq. 3
- *Resul* in Sec. 5.3 header

### Soundness
3

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
3

### Summary
This paper proposes a unified distillation framework for LLMs, arguing that traditional SFT (data distillation) is limited because it propagates teacher errors and cannot surpass imitation. The authors introduce reward distillation, which extracts evaluation signals from the teacher without human labels. This is done using majority voting for verifiable tasks and LLM-as-Judge for open-ended ones. A student model is first warmed up with SFT, then refined using reinforcement learning (RL) guided by a learned reward model. This combined approach consistently outperforms SFT, allowing student models to in some cases surpass their teachers.

### Strengths
1. Novel Annotation-Free Framework: The paper proposes a novel reward distillation framework. This method cleverly extracts evaluation signals directly from the teacher (via majority voting or LLM-as-Judge) without needing expensive human annotations.
2. Clear and Easy to Follow: The paper is well-structured and clearly written, making its complex methodology understandable.

### Weaknesses
A primary weakness is the claim that the student can surpass the teacher. This is likely because the student model is trained on the GSM8K and MMLU-Pro data. Even though the teacher generated these answers and majority voting was used to enhance the data quality, the teacher's baseline performance is being compared against a student's fine-tuned performance. A more reasonable comparison would require fine-tuning the teacher model on the exact same high-quality, distilled dataset and then comparing its performance to the student's.

### Questions
N/A

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
- This paper introduces a novel distillation method to transfer the knowledge from the responses, but also the evaluative capabilities from the rewards. 
- The proposed reward distillation method uses a self-supervised process within the teacher-student framework. First, it uses the teacher LLM to generate a variety of responses. Then, it employs an adaptive evaluation strategy to create pseudo labels: majority voting or LLM-as-a-judge. A reward model is then trained on a mix of teacher and student-generated responses to learn these reward signals. Finally, the student model, after an initial warm-up with supervised fine-tuning, is further refined using reinforcement learning guided by this distilled reward model.

### Strengths
- The paper is well-written and easy to follow.
- The core motivation of this paper is to mitigate the distribution shift between the teacher and student, which is a critical challenge and a good research question in knowledge distillation.

### Weaknesses
- For On-Policy RL (PPO), since the rollouts are sampled from the student policy model, there would not exist any distribution shift between the policy and the experience. Therefore, the proposed method does not solve a real question for On-Policy RL training.
- For the verifiable task, this work should compare the proposed Reward Distillation with the standard RLVR method. Why is Reward Distillation better than RLVR? 
- For the non-verifiable task, this work should compare the proposed Reward Distillation with the RLHF with GenRM (LLM-as-a-Judge). Why is Reward Distillation better than RLHF with GenRM? 
- The paper positions itself primarily against traditional distillation but does not sufficiently discuss or compare against the closely related paradigm of self-rewarding or self-improving LLMs[1][2]. These methods also involve a model generating its own training signals or rewards.
- For the reward signal, using the teacher model's signal as a reward has been proposed by On-Policy Distill[3][4], which can provide dense reward signals in the sequence dimension. However, the proposed Reward Distillation only provides an outcome reward signal and should construct data to train an additional reward model.

[1] Self-Rewarding Language Models

[2] Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge

[3] On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes

[4] MiniLLM: Knowledge Distillation of Large Language Models

### Questions
Same as the weaknesses.

### Soundness
2

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
This paper proposes a multi-stage distillation pipeline as an alternative to standard supervised fine-tuning (SFT). The core idea is to distill not only the teacher's data (responses) but also its evaluative capacity (rewards). The method involves (1) an SFT warm-up, (2) training a separate, small reward model (RM) using teacher-generated labels (via majority vote or LLM-as-Judge), and (3) refining the student using Proximal Policy Optimization (PPO) guided by this RM.

### Strengths
The paper's primary strength is its well-motivated strategy of leveraging both the teacher's generative and evaluative capabilities. Empirical results demonstrate that this pipeline outperforms SFT. Warm-up training can further improve results (although same finding has been pointed out by GKD, SKD paper).

GKD: https://arxiv.org/pdf/2306.13649
SKD: https://arxiv.org/abs/2410.11325

### Weaknesses
W1: Limited Novelty and Missing Baselines: This is the most significant weakness. The core idea of using reinforcement learning (via policy gradient) and reward signals for distillation is not new (e.g., GKD, MiniLLM, direct preference knowledge distillation). The paper's novelty is a specific pipeline, yet it fails to compare against these established RL-based distillation baselines. The comparison is limited to SFT, which is a weak baseline making it impossible to assess if this complex SFT+RM+PPO pipeline is any better than simpler, existing RL-KD methods or simply KL loss + reward objective.

W2: The contribution of the reward signal generation is overstated. The components are well-known: "majority voting" is a direct application of Self-Consistency, and "LLM-as-Judge" is a standard technique. The paper's contribution is a synthesis of these existing methods, not a novel mechanism.

W3: The paper's pipeline (SFT -> RM training -> PPO) is significantly complex, but its design choices are poorly motivated. PPO vs. GKD: It is unclear why this complex, multi-stage pipeline is necessary over simpler, end-to-end objectives like GKD, which also incorporate reward signals. The paper provides no justification. 

W4: Separate RM: The paper trains a separate, small RM but never justifies why this is necessary over using the teacher model as the reward source. This is a critical unstated detail (likely for computational cost), which also highlights the next weakness. If RM is expensive, why don't authors consider direct preference KD?

W5: Unalyzed Computational Cost: The paper ignores the massive computational cost of its pipeline. The reward generation requires 10 teacher samples + 30 student samples and evaluations per query. This cost is not analyzed, and it's plausible that a baseline SFT model trained with the same compute budget (i.e., 40x more data) would perform just as well.

GKD: https://arxiv.org/pdf/2306.13649
miniLLM: https://arxiv.org/abs/2306.08543
Direct preference KD: https://arxiv.org/abs/2406.19774

### Questions
Why training a separate RM?

### Soundness
2

### Presentation
2

### Contribution
1
