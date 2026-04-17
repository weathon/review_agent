# Socratic-Zero : Bootstrapping Reasoning via Data-Free Agent Co-evolution

- Decision: Reject
- Scores: 4, 6, 6, 4, 4

## Abstract
Recent breakthroughs in large language models (LLMs) on reasoning tasks rely heavily on massive, high-quality datasets—typically human-annotated and thus difficult to scale. While data synthesis or distillation offers a promising alternative, existing methods struggle with inconsistent data quality and an inability to dynamically adapt to the evolving capabilities of the model, leading to suboptimal training signals. To address these limitations, we introduce Socratic-Zero, a fully autonomous framework that generates high-quality training data from minimal seed examples through the co-evolution of three agents: the Teacher, the Solver, and the Generator. The Solver continuously refines its reasoning by learning from preference feedback on both successful and failed trajectories; the Teacher adaptively crafts increasingly challenging questions based on the Solver's weaknesses; and the Generator distills the Teacher's question-design strategy to enable scalable, high-fidelity curriculum generation. This closed-loop system produces a self-improving curriculum—requiring no pre-existing tasks or labels. Remarkably, starting from only 100 seed questions, our Socratic-Solver-8B achieves an average gain of +20.2 percentage points over prior data synthesis methods across seven mathematical reasoning benchmarks (AMC23, AIME24-25, Olympiad, MATH-500, Minerva, and GSM8K), with consistent gains on both Qwen3 and GLM4 series models. Even more surprisingly, synthetic data from Socratic-Generator-32B enables student LLMs to achieve superior performance compared to other state-of-the-art (SOTA) commercial LLMs on these benchmarks, including Qwen3-235B-A22B, DeepSeek-V3.1-671B, GPT-5, and Gemini-2.5-Pro.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Socratic-Zero, a novel framework that generates high-quality training data from minimal seed examples through the co-evolution of three agents: Teacher, Solver and Generator. The Solver learns via DPO, the Teacher evaluates and guides problem generation, and the Generator imitates the Teacher to create new tasks. Through iterative co-evolution, the system improves reasoning performance on math reasoning benchmarks.

### Strengths
1. The paper is well-written and clearly presented, making it easy to follow.

2. It introduces a creative closed-loop framework where the Solver, Teacher, and Generator co-evolve without relying on large external datasets.

3. The proposed method achieves notable performance improvements on mathematical reasoning tasks under the given experimental setup.

### Weaknesses
1. One concern lies in the experimental design, which lacks clarity in several aspects. Although the proposed method achieves the highest scores on math benchmarks compared to its baselines, the experimental setup is not well-aligned with common practices. Many strong baselines are missing, such as direct distillation from the same teacher model. Moreover, the reported math scores are not directly comparable to other works that improve reasoning on Qwen3-8B/14B, and there appear to be inconsistencies. For example, the Qwen3 technical report lists a math score of 60.8 for Qwen3-8B-base, while Table 1 in this paper reports 48.8. These inconsistencies make it difficult to assess whether the proposed approach truly outperforms simpler alternatives like distillation.

2. The paper does not clearly specify how many co-evolution iterations were conducted between the Solver, Teacher, and Generator, nor does it report the reasoning performance after each iteration, which would be important for understanding the effectiveness and dynamics of the co-evolution process.

3. It remains unclear whether the proposed method generalizes beyond math reasoning, such as to other domains like code reasoning.

### Questions
See the weaknesses section

### Soundness
2

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
The paper addresses a key bottleneck in advancing the reasoning capabilities of Large Language Models (LLMs): their heavy reliance on massive, human-annotated datasets. To overcome this, it introduces Socratic-Zero, a fully autonomous framework that bootstraps a model's reasoning abilities from a minimal set of 100 seed questions, requiring no further external data.   

The framework is built on a co-evolutionary system of three interacting agents:

The Solver: The LLM being trained. It continuously improves by attempting to solve problems and learning from preference feedback on its own successful and failed attempts using Direct Preference Optimization (DPO).   

The Teacher: A powerful, frozen LLM that acts as an oracle. It evaluates the correctness of the Solver's solutions and, crucially, generates new, more challenging problems that are specifically designed to target the Solver's identified weaknesses.   

The Generator: A model trained to distill and scale the Teacher's problem-generation strategy. By learning what makes a good, challenging question, the Generator can produce a high-quality, adaptive curriculum at scale without constant reliance on the much larger Teacher model.   

This closed-loop system creates a self-improving curriculum that dynamically adjusts to the Solver's evolving skill level. The empirical results are significant: the Socratic-Solver-8B model achieved an average performance gain of +20.2 percentage points over previous methods. Furthermore, synthetic data produced by the Socratic-Generator-32B was used to train a student model that ultimately outperformed much larger, state-of-the-art commercial LLMs on several mathematical reasoning benchmarks.

### Strengths
The paper presents a methodology for leveraging large models to synthesize high-quality data from a small set of seed examples, effectively enhancing the performance of smaller models. A particularly distinctive contribution is the subsequent use of the resulting data pairs to train a separate, medium-sized "Generator Model." This dedicated Generator is a unique feature, offering a specialized and potentially more efficient tool for synthesizing supervised fine-tuning (SFT) data compared to relying solely on the original, larger Teacher Model.

### Weaknesses
Socratic-Zero framework combines several techniques in a novel and effective way, its core components are built upon established paradigms in the field, which could be seen as a limitation on its fundamental novelty.

Heavy Reliance on a "Teacher" as a Form of Knowledge Distillation: The entire system's success is predicated on the existence of a powerful, fixed "Teacher" model that serves as a "reasoning oracle". This framing positions Socratic-Zero less as a system that creates knowledge from scratch and more as a highly sophisticated and efficient knowledge distillation framework. The Solver's ultimate reasoning capability is fundamentally capped by the knowledge and reasoning ceiling of the Teacher model it learns from. The innovation lies in distilling the curriculum generation process rather than just answers, but it remains a form of knowledge transfer from a larger, more capable model to smaller ones.

Similarity to Iterative Self-Training and Self-Play Frameworks: The core loop—where a model generates data based on its performance, which is then used for further training—is conceptually similar to iterative data augmentation, self-training, and self-play methodologies. For instance, the paper's baselines, like LLM2LLM, already use an iterative process of generating questions from failures. Socratic-Zero's main differentiators are the introduction of the Generator for distillation and the use of DPO for learning, but the underlying iterative, self-improving cycle is a known concept in the literature.

### Questions
q1: The Role and Reliability of the Teacher Model

The methodology's reliance on the Teacher Model's output raises a fundamental question concerning the data integrity. Specifically: Is there a mechanism to guarantee that the generated $\left(q_{\tau}, a_{\tau}\right)$ pairs are universally correct (i.e., constitute ground truth)? Alternatively, is the primary objective of this approach to align the Student Model with the Teacher Model's generation capability? If the latter is true, the final performance ceiling of the proposed method is inherently bounded by the proficiency of the Teacher Model, suggesting that a stronger Teacher Model would directly translate to superior final results. Clarification on this dependency is needed.

q2: Clarity on Data Generation Parameters (Section 4.3.1)

Clarification is required regarding two inconsistencies observed in the data generation process detailed in Section 4.3.1:

Question Count Discrepancy: Step 1 mentions that each model generates five questions per seed. However, the resulting total of 3,000 generated questions from 1,000 seed questions implies a generation factor of three per seed, not five. Please clarify this numerical inconsistency.

Timeout Parameter: The rationale behind the 600-second timeout setting in Step 2 is unclear. Does this imply that the model must generate the full context length (4096 tokens) within this duration? This parameter seems unusually generous and requires a more detailed technical explanation regarding its necessity and practical impact.

q3: Definitions of Training Stages

The results tables (e.g., Table X, Y, Z) frequently reference "Stage 1," "Stage 2," and "Stage 3." These stages are not explicitly defined or correlated with the corresponding steps in the training methodology within the main body of the manuscript. Please clearly articulate what each of these stages represents and how they map to the overall training progression.

q4: The Functional Utility of the Generator Model

The precise functional contribution of the Generator Model to the enhancement of the Solver Model's capabilities requires clarification. Is the training of the Generator Model merely a necessary byproduct of the Solver training process? The manuscript notes its use only for generating the synthesis SFT data for evaluation, but it does not detail any mechanism for utilizing the trained Generator to actively boost the Solver's performance (e.g., by replacing or assisting the initial Teacher Model in a subsequent iteration). Clarification on the Generator's role beyond evaluation is requested.

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
This paper presents a teacher-solver-generator co-evolution approach to data generation for training large language models. Using a feedback loop, the solver continuously optimizes the reasoning via preference learning, from positive and negative examples. The multi-agent design addresses an existing challenge of scaling of human-annotated datasets, and shows superior performance compared to state-of-the-art methods across various reasoning benchmarks.

### Strengths
- the paper addresses a crucial problem of high-quality data synthesis, for optimization of LLMs.
- the multi-agent framework and co-evolution preference learning mechanism is novel, and the learning framework is scalable
- the presentation is very clear and easy to follow

### Weaknesses
- It is unclear how much capability the teacher model requires to have in order to generate high-quality question-answer pairs. For example, for problem sets where the teacher may also experience difficulty solving or evaluating the solution, the framework may fail to adapt
- The generator model provides better scalability, however, it is unclear if the teacher model alone can provide decent performance without the generator model.

### Questions
Can the author describe how does the model perform if only using the teacher model, without the generator model?

### Soundness
3

### Presentation
3

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
The paper introduces Socratic-Zero, a novel, fully autonomous framework designed to bootstrap and refine the reasoning capabilities of LLMs with minimal reliance on external, human-annotated data. The core innovation lies in a closed-loop, data-free co-evolution of three distinct agents: the Solver (which learns and refines reasoning via preference feedback on both successful and failed trajectories), the Teacher (which adaptively creates an increasingly difficult curriculum aligned with the Solver's current weaknesses), and the Generator (which distills the Teacher’s question-design strategy to facilitate scalable curriculum generation). This system aims to address the inconsistent data quality and static adaptation inherent in existing data synthesis and distillation methods by continuously generating high-quality, targeted training signals from minimal seed examples.

### Strengths
1. The core architecture fundamentally circumvents the reliance on massive, costly, human-annotated datasets, which is the current scaling bottleneck for high-quality reasoning tasks. This "data-free" approach is highly valuable for domains where annotation is prohibitively expensive or complex.

2. The systematic protocol for seed selection, requiring a 30-70% success rate for initial problems, demonstrates a careful attempt to ensure capability-aligned initialization. This ensures the co-evolutionary loop starts from a robust and productive equilibrium.

### Weaknesses
1. While the Generator is intended for scalability, the paper itself concedes that "computational efficiency optimizations" are a necessity for future work. This suggests the current multi-agent, co-evolutionary loop is likely highly resource-intensive (training three models and maintaining constant interaction), which fundamentally challenges its scalability and practicality compared to simpler, static distillation pipelines.

2. Expanding the framework to new domains is cited as a major area for future work, indicating that the current co-evolutionary success is tightly coupled to the initial domain alignment. This limits the "data-free" claim; the methodology still requires significant external effort (or pre-existing capability) before the closed-loop self-improvement can function in a new area.

### Questions
The paper highlights that the Solver learns from preference feedback over successful and failed trajectories. Given that the Teacher explicitly crafts the curriculum to target the Solver's weakness zone (30–70% success rate), how does the preference modeling distinguish between a "valuable failure" (i.e., a well-reasoned attempt that simply misses the final answer) and a "chaotic failure" (i.e., a nonsensical trajectory that offers little instructional value)? Furthermore, have the authors explored an ablation study comparing this preference learning approach against a simpler Policy Gradient or PPO-style method that optimizes the Solver directly using the Teacher's difficulty signal as a scaled reward, and if so, what were the trade-offs in sample efficiency and final performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a framework for improving reasoning capabilities in LLMs through a three-agent co-evolution process involving a fixed **Teacher** (Qwen3-235B-A22B), a smaller **Solver** (largest is Qwen3-14B-base) and **Generator** (Qwen3-32B). The Teacher adaptively designs problems for the Solver based on its failures, while the Solver learns via preference optimization. The Generator distills the Teacher’s question-design strategy through weighted fine-tuning.

**Solver evaluation** (Table 1): On math and reasoning benchmarks (GSM8K, MATH, AIME, etc.), the approach shows large gains over static and LLM2LLM data-augmentation baselines (56.1 vs. 40.7 and 40.9, respectively, for Qwen3-8B-base).

**Generator evaluation** (Table 5): In a separate ablation, fine-tuning a smaller student model (DeepSeek-R1-Distill-Llama-8B) on data from the co-evolved Qwen3-32B Generator yields higher performance than training it on data from SOTA models (37.22 vs. 36.62 against GPT-5).

### Strengths
- The paper **addresses the important problem of enabling models to learn effectively from a limited set of examples**, proposing a curriculum-learning strategy in which a small seed set (just 100 examples, Table 8) is progressively enhanced in line with the Solver’s evolving capabilities.

- The **main results of the paper are substantial and well-supported by experiments**, with the proposed method providing ~15% improvement over the baseline (Table 1).

- **Evaluation is comprehensive and appropriate**, encompassing  seven math and reasoning benchmarks, multiple model architectures, and all relevant parts of the system (solver and generator separately evaluated, in sections 4.2 and 4.3, respectively).

- **Design of individual components is sound** — the curriculum update mechanism (Section 3.1), DPO-based Solver training (Section 3.2), and weighted distillation for the Generator (Section 3.3) are all well-motivated.

- **Experimental configurations are exhaustively documented and the work is fully reproducible**, with code provided in the supplementary material, and detailed training configurations, hyperparameters, and prompt templates included in the appendix.

- The **breadth of supplementary details and discussions in the appendix is impressive**, aiding in the comprehension of the work. For example, including examples of synthetically generated datapoints (appendix F), and extensive comments on convergence analyses (appendix K).

### Weaknesses
- **The final statement of the abstract is ambiguously worded, leading to potential misinterpretation of the results**. It reads: “Synthetic data from Socratic-Generator-32B enables student LLMs to achieve superior performance compared to other SOTA commercial LLMs…” As written, this implies that the resulting student LLMs themselves outperform SOTA commercial LLMs. However, as Table 5 clarifies, the intended meaning is that student models trained on data generated by Socratic-Generator-32B outperform those trained on data generated by SOTA LLMs, not that the students surpass the SOTA LLMs in absolute performance.

- It is **unclear how the Generator fits into the Solver’s training process, and the paper’s claims about the cohesiveness of the three-agent system are therefore potentially misleading**. The paper repeatedly presents the Generator as a central component: for instance, the abstract states that “the Generator distills the Teacher’s question-design strategy to enable scalable, high-fidelity curriculum generation,” and this is immediately followed by the Solver’s results, creating the impression that the Generator is directly involved in those outcomes. However, the training procedure in Figure 3 and the curriculum update in Section 3.1 suggest that the Generator is orthogonal to the Solver–Teacher loop. It does not appear to contribute to the curriculum on which the Solver is trained, nor to the main results in Table 1 (including the reported +20.2 % aggregate improvement). The authors should clarify exactly which components are used to produce the reported results and, ideally, include experiments where the co-evolved Generator directly participates in the Solver’s training process.

- The **exact details of the overall training procedure is hard to follow** - the paper could be greatly enhanced by providing a pseudocode of the exact training procedure. This is in fact promised at the very end of section 3.1: “The full training procedure is summarized in Algorithm 1”. However, no figure corresponding to Algorithm 1 exists in the paper that’d outline the full training procedure (there is Algorithm 1 in the appendix, but it pertains to the theoretical challenge framework).

- **Impact is limited by its reliance on the Teacher model being more powerful than the Solver**, with experiments using Qwen3-235B-A22B-Instruct-2507 as the Teacher and Qwen3-14B-base as the biggest solver. The paper would be improved by a scaling law study, showing how the methodology performs as a function of the model size gap between the Teacher and Solver.

- The **Teacher model acts both as a problem enhancer and solution verifier of the same problems, risking a bias**. While the authors discuss several mitigation strategies in appendix I (dual-verification with the inclusion of MathRule answer extraction, human review), the work could be improved by ablating a configuration where the judge differs from the enhancer.

- **Experiments section could have more detailed dataset descriptions**:
How many total samples across how many prompts were generated for a) for the MetaMath baseline, b) for the LLM2LLM baseline at each stage, c) for the Socratic-Zero approach at each stage.
Which dataset was used as the seed for the Static Augmentation and LLM2LLM approaches?

- Some **typos**:
  - The first sentence on top of page 5 reads: “The Solver and the Generator are co-evolving guided the Teacher” - misses the word “by”.
  - In section 4.1, the paragraph heading for “Baselines” is repeated, reading: “Baselines. Baselines.”
  - In section 4.3.1, a sentence reads: “We prompted each generator with 1,000 seed problems from SAND-Math and tasked with producing five augmented variants per seed, resulting in 3,000 total generated problems per model”. The numbers don’t add up - I believe it should read “three augmented variants per seed”, as mentioned in section 4.1.
In paragraph 3 of appendix M: “the system generates thousands of ly valuable problems…” - misses the beginning of “highly”.

The paper is strong overall, but I cannot recommend acceptance in its current form due to the unclear and at times misleading presentation of the system’s cohesiveness and the results (see the first three “drawback” points). I encourage the authors to clarify these aspects and to demonstrate the integration of the Generator with the Teacher–Solver system, or alternatively provide an extended discussion explaining why this integration is not pursued.

### Questions
See "Drawbacks" section

### Soundness
2

### Presentation
1

### Contribution
3
