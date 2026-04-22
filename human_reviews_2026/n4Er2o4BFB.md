# From Correction to Mastery: Reinforced Distillation of Large Language Model Agents

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 2, 6, 8

## Abstract
Large Language Model agents excel at solving complex tasks through iterative reasoning and tool use, but typically depend on ultra-large, costly backbones. Existing distillation approaches train smaller students to imitate full teacher trajectories, yet reasoning and knowledge gaps between the teacher and student can cause compounding errors. We propose SCoRe, a student-centered framework in which the student generates training trajectories and the teacher corrects only the earliest error, producing training data matched to the student's ability and exposing specific weaknesses. The student is first fine-tuned on corrected trajectories. Subsequently, short-horizon reinforcement learning starts from the verified prefix preceding the earliest error, with target rewards assigned at that step. This design encourages autonomous problem-solving beyond imitation and enhances training stability. On 12 challenging benchmarks, a 7B-parameter student distilled with SCoRe matches the agentic performance of a 72B-parameter teacher.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SCoRe (Student-Centered one-step Reinforcement), a novel framework for distilling the capabilities of large, powerful LLM agents into smaller, more efficient models. Unlike traditional Behavior Cloning, where a student passively imitates full teacher trajectories, SCoRe adopts a student-centric approach. It first initializes the student with basic skills and then enters a Mentored Problem-Solving phase. Here, the student attempts tasks independently, and the teacher only intervenes to correct the first error it makes. The student then continues from this corrected point. This method generates capability-matched training data that explicitly targets student weaknesses, theoretically reducing compounding error growth from O(H²) to O(H). A subsequent reinforcement learning phase further refines the model using short-horizon rollouts from verified prefixes and targeted key-step rewards, promoting independent problem-solving over mere imitation. Extensive experiments on mathematical, factual, and deep-search benchmarks show that small 7B-8B models trained with SCoRe can achieve performance comparable to a 72B teacher, significantly outperforming both standard distillation and other RL baselines.

### Strengths
1. The paper introduces a student-centered correction framework (SCoRe) that shifts from traditional teacher-driven imitation to a student-exploration–teacher-correction paradigm. This is a novel and conceptually elegant.
2. Experiments across 12 diverse benchmarks (reasoning, factual QA, deep research tasks) show that a 7B model distilled with SCoRe approaches the performance of a 72B teacher model.
3. The authors provide a clear theoretical analysis showing that SCoRe reduces error propagation from O(H^2) to O(H).

### Weaknesses
1. The framework's efficacy seems to be  contingent on a highly capable teacher model to provide accurate, minimal corrections. This creates a single point of failure; any errors or inconsistencies in the teacher's judgment are directly learned by the student.
2. The core "Mentored Problem-Solving" phase requires extensive, iterative interaction with a large teacher model (e.g., GPT-4) for data generation. This process is computationally expensive and incurs significant API costs, making the training process prohibitively costly despite the final student model being small.
3.  The method's cornerstone—correcting the "first" error—is often subjective in complex reasoning tasks. An initial flawed assumption may only manifest as a visible error steps later, making precise localization challenging. This ambiguity can lead to inconsistent training data and ineffective corrections.

### Questions
1. The entire SCoRe framework relies on a powerful teacher model. How would the performance degrade with a weaker or noisier teacher? Did you perform any sensitivity analysis on teacher quality?
2. Your method hinges on correcting the "first" error. In complex reasoning, an early logical misstep might only manifest as a visible error much later. How do you handle this inherent ambiguity, and did you observe cases where the "first error" was debatable?
3. How does SCoRe fundamentally differ from and improve upon classical interactive imitation learning algorithms like DAgger or HG-DAgger, which also aggregate student-generated data with expert labels?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Training a student model to imitate full teacher trajectories often suffers from compounding errors, since mistakes early in the rollout can accumulate. To address this challenge, this paper proposes SCoRe, a framework in which the student generates its own initial trajectory and the teacher intervenes at the earliest detected mistake, potentially iteratively. This enables more efficient transfer of agentic reasoning and tool-use capabilities from a large teacher LLM to a smaller student model. The authors claim this reduces compounding-error growth from $O(H^2)$ to $O(H)$ and improves RL stability by shortening rollout horizons. Empirical results across 12 benchmarks show that a 7B student model can match the agentic performance of a 72B teacher model.

### Strengths
1. The motivation to make LLM agents more affordable and deployable while preserving strong reasoning and tool-use capabilities is compelling.

2. The results clearly demonstrate consistent improvements over the base model across multiple benchmarks.

3. Figures and tables effectively illustrate the method and its empirical performance.

### Weaknesses
1. The proposed imitation-plus-correction paradigm resembles DAgger-style intervention in traditional RL literature and shares similarities with existing trajectory-repair approaches. Therefore, the degree of conceptual novelty may be limited without a more thorough discussion of prior work.

2. The method assumes a stronger teacher agent capable not only of solving the task but also of accurately identifying and correcting the student’s errors. This constitutes an even higher capability requirement than simply solving the task, and may limit applicability in scenarios where such a teacher is unavailable. 

3. Finally, it is also not unclear how this method scales to frontier-level model improvement. Because the approach fundamentally relies on an even stronger teacher model, the path toward pushing beyond the teacher’s performance remains ambiguous. 

4. The reported student performance appears noticeably weaker than existing smaller-scale models such as DeepScaleR-1.5B-Preview (43.1% on AIME2024, 87.8% on MATH500, 50.0% on OlympiaBench), which surpasses all student models up to 8B parameters in this paper. This raises concerns about the practical significance of the gains demonstrated here.

5. While the paper claims to minimize teacher involvement, the correction process is iterative, and the associated cost (latency, compute, prompts, dollar cost) is not quantified. It remains unclear whether the reduced rollout horizon translates into overall cost efficiency in real deployments.

6. Certain design choices lack justification or clarity. For example, the rationale behind setting the reward to 0.5 when the student’s key step matches the teacher’s correction is not explained, nor is the precise definition of a “correction” formalized.

### Questions
1. What's the Qwen3-8B-Instruct model referenced in Table 2?

2. Are there any failure modes on tasks where teacher guidance provides no improvement or even degrades performance?

3. How sensitive is the framework to formatting? In particular, how does SCoRe handle non-programmatic or free-form tasks, where determining the minimal teacher “leakage” needed to ensure effective learning may be challenging?

4. How does this approach compare with teacher-prefix training or mixed rollout BC+RL strategies that have been widely adopted in recent RL-finetuning work?

5. The theoretical analysis appears to evaluate expectations under the student’s on-policy distribution $d^{\hat{\pi}_i}$. However, since teacher actions are injected during rollout, isn’t the effective execution policy a mixture of student and teacher? Should the proof instead be derived under this mixture distribution?

### Soundness
2

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
This paper introduces SCoRe, a new distillation framework for LLM agents. Instead of the standard "teacher-acts, student-clones" paradigm, SCoRe places the student at the center of trajectory generation. The student explores tasks and the teacher corrects only the earliest error, producing "verified prefixes" and "key steps." In ScoRe, multiple post-training (SFT,RL) paradigms exists. The paper points out: on 12 challenging benchmarks, a 7B-parameter student distilled with SCoRe matches the agentic performance of a 72B-parameter teacher.

### Strengths
- Novel distillation paradigm: Different from traditional "Teacher Acts, Student Clones", this paper propose "Student Explores, Teacher Corrects". (Fig1)
- Comprehensive empirical results: Table 1,2 in paper demonstrate proposed method's effectiveness in mathematical reasoning, factual reasoning and deep search. 
- Strong empirical results: The 7B student nearly matches a 72B teacher, demonstrating substantial efficiency gains.

### Weaknesses
- Lack of comparison with newer distillation methods: The paper mainly compares the proposed method with the traditional “Teacher Acts, Student Clones” paradigm.
- Distillation Paradigm: From this distillation paradigm, the student’s performance ceiling may be constrained by the teacher’s capability. I noticed a brief explanation of this issue in the introduction section. However, since that "the 7B student nearly matches a 72B teacher and even surpasses it on certain tasks," this explanation could be expanded and further deepened.
- Presentation: The full term of SCoRe should be given the first time it appears.

### Questions
1. Could the authors compare their method with the latest distillation approaches? If such a comparison is not necessary, could the authors provide an explanation for that?
2. Could the authors provide a more detailed and in-depth explanation for Weakness 2?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The development of language models into language agents has motivated the exploration of smaller language models as agent backbones, in order to reduce latency and computational cost. This has led to the development of agent distillation, where structured trajectories from larger teacher models are distilled into smaller student models. However, the reasoning and knowledge gap between the student and teacher models often prevents the student from fully learning the teacher’s behavior, and even a single erroneous step can push the student into out-of-distribution states.

This work proposes SCoRe to overcome the limitations of agent distillation. In this framework, the student model generates reasoning trajectories while the teacher intervenes only to correct the earliest detected error, after which the student resumes reasoning. Since the trajectories are primarily generated by the student itself, this approach eliminates the capability and knowledge mismatch introduced in the conventional teacher-acts–student-clones paradigm. The work provides theoretical justification for the observed reduction in error and introduces methods to mitigate the limitations of reinforcement learning (RL) in training large language models.

### Strengths
The strengths of this paper are listed below:

1. The paper introduces a method to address the knowledge and reasoning gap between the student and teacher models in the agent distillation paradigm.

2. The proposed distillation pipeline not only utilizes the final correct trajectories but also leverages key-step correction preference pairs. These preference pairs provide dense rewards during RL training by offering feedback at earlier error points. This approach prevents the student model from being limited to pure imitation and instead encourages independent problem solving.

3. The authors provide a theoretical justification showing that SCoRe reduces the quadratic error growth observed in behavior cloning for long-horizon tasks to linear error growth.

4. They further provide theoretical justification that starting rollouts from verified prefixes reduces gradient variance, leading to a more stable and efficient RL optimization.

5. The authors evaluate their method on several mathematical and factual reasoning benchmarks and demonstrate strong performance gains. They also show results across different model families.

6. They conduct an extensive ablation study to justify the effectiveness of both short-horizon and key-step reward training in RL for mathematical and factual reasoning datasets.

### Weaknesses
The weaknesses of the paper are listed below:

1. Rewarding the model merely for avoiding its original mistake can be problematic for several reasons. First, it provides no guarantee of correctness; the new action may still be wrong but simply differ from the previous error. This can mislead the learning process, as the model is reinforced for being different rather than for being correct. Second, it introduces ambiguity in credit assignment, as the model might accumulate positive rewards for logically invalid steps. Overall, this design risks reinforcing superficial behavioral changes instead of genuine reasoning improvements, potentially leading to unstable or suboptimal learning dynamics.

2. It would be nice to check the performance of SCoRe on benchmarks where the model has not been trained on.

### Questions
1. Please clarify the use of R_avoid reward.
2. Please outline the limitations of your pipeline observed during the experiments.

### Soundness
4

### Presentation
4

### Contribution
4
