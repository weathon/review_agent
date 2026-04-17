# DéjàQ: Open-Ended Evolution of Diverse, Learnable and Verifiable Problems

- Decision: Reject
- Scores: 8, 4, 2, 4

## Abstract
Recent advances in reasoning models have yielded impressive results in mathematics and coding. However, most approaches rely on static datasets, which have been suggested to encourage memorisation and limit generalisation. We introduce \dejaq, a framework that departs from this paradigm by jointly evolving a diverse set of synthetic mathematical problems alongside model training. This evolutionary process adapts to the model's ability throughout training, optimising problems for learnability. We propose two LLM-driven mutation strategies in which the model itself mutates the training data, either by altering contextual details or by directly modifying problem structure. We find that the model can generate novel and meaningful problems, and that these LLM-driven mutations improve RL training. We analyse key aspects of \dejaq, including the validity of generated problems and computational overhead. Our results underscore the potential of dynamically evolving training data to enhance mathematical reasoning and indicate broader applicability, which we will support by open-sourcing our code.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces DéjàQ, a novel framework for improving the mathematical reasoning abilities of Large Language Models (LLMs) through dynamic dataset evolution. Instead of relying on static datasets, DéjàQ co-evolves a diverse set of synthetic problems alongside model training.

The core of the contribution lies in three LLM-guided mutation operators that generate new problems: a "setting mutator" that alters the problem's narrative context, a "distractor mutator" that adds irrelevant information, and a "symbolic mutator" that modifies the underlying mathematical structure.

Crucially, the same model being trained is used to perform these mutations, creating an efficient, self-bootstrapping system. The paper also provides a thoughtful analysis of the validity of generated problems and the computational overhead of the system.

### Strengths
- **Novel Framework**: The framework is a novel and well-designed synthesis of evolutionary algorithms, curriculum learning, and self-improvement for LLM post-training.

- **Strong Empirical Performance**: The method demonstrates significant and consistent improvements over well-chosen baselines on a variety of mathematical reasoning benchmarks, including both in-distribution and out-of-distribution tasks.

- **Efficient Bootstrapping**: The use of the same model for both training and data generation is a key strength, eliminating the common reliance on more powerful external "teacher" models and making the system more self-contained and efficient.

- **Thorough Analysis**: The paper includes a high-quality analysis of the method's potential failure modes, including the verifiability of generated problems over time (Section 5.2) and the practical resource implications (Section 5.3).

- **Clarity of Presentation**: The paper is written with clarity, and the figures, particularly the system overview in Figure 1, are highly effective at conveying the core ideas.

### Weaknesses
- **Unresolved Teacher-Student Lag**: The paper identifies a key limitation where the "teacher" does not improve with the "student", leading to a higher proportion of invalid problems among high-learnability candidates after training. While identifying this is a strength of the analysis, the paper frames it as future work and does not propose or test a mechanism to resolve it. This might limit the truly "open-ended" nature of the evolution in the long run without further modification.

- **Scalability Questions**: The experiments are conducted on a 7B model. While the results are excellent, it remains an open question how the dynamics of this tightly coupled system would scale to much larger models (e.g., 100B+ parameters). For instance, the quality of mutations might improve, but the rate of student improvement might also accelerate, potentially exacerbating the teacher-student lag.

- **Dependence on Initial Seed Data**: The evolutionary process begins from a seed set of problem templates (GSM-Symbolic). While the mutations, especially the symbolic one, introduce significant novelty, the framework may still be fundamentally constrained by the mathematical concepts present in the initial seed data. It is unclear if the system could evolve problems requiring entirely new types of reasoning not represented in the seed set.

### Questions
- The symbolic mutator is arguably the most powerful operator. Could you provide a more detailed breakdown of its typical failure modes? For instance, beyond producing an incorrect final answer, how often does it generate problems that are logically inconsistent, ambiguous, or unsolvable?

- The analysis in Appendix A.3 mentions the model spontaneously generating problems in Spanish, which highlights the open-ended nature of the system. Did you observe any other surprising emergent behaviors? Specifically, did the symbolic mutator ever introduce mathematical operations or concepts that were not present in the original GSM-Symbolic templates, thereby increasing the conceptual complexity of the archive?

- In your related work, you cite Rainbow Teaming for its use of MAP-Elites and an archive to generate diverse problems. Have you considered citing OMNI-EPIC? It seems highly relevant and arguably closer to your work compared to Rainbow Teaming, as it also describes an open-ended evolutionary process that maintains an archive of generated tasks to create a curriculum of increasing difficulty.

### Soundness
4

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
This paper proposes DÉJÀQ, a self-bootstrapping framework that simultaneously evolves a curriculum of synthetic math problems and trains a 7B LLM via RL with verifiable rewards. Using MAP-Elites, it maintains a diverse archive of problem-answer pairs indexed by human-defined settings; three in-model mutators (setting, distractor, symbolic) continually rewrite problems, while an estimated learnability score filters offspring for neither-too-easy-nor-impossible instances. The same 7B model serves as both rollout model and mutation "teacher", yielding gains on GSM-Symbolic and MATH-500 without external labels or larger teachers.

### Strengths
* The paper introduces a fully self-contained framework that co-evolves a curriculum of synthetic math problems and improves a solver without any external oracle or larger teacher. By uniting MAP-Elites, learnability-based filtering, and three LLM-driven mutators inside a single RLVR loop, it removes the usual dependency on hand-written templates or proprietary generators, which is promising for community adoption.

* Experiments show consistent gains over benchmarks: +6–7 % accuracy on GSM-Symbolic subsets and +1.6 % on MATH-500, together with better tail robustness (CVaR).

* Resource measurements prove the extra inference calls fit within the idle time of the existing rollout server, validating practical deployability.

### Weaknesses
A substantive assessment of the weaknesses of the paper. Focus on constructive and actionable insights on how the work could improve towards its stated goals. Be specific, avoid generic remarks. For example, if you believe the contribution lacks novelty, provide references and an explanation as evidence; if you believe experiments are insufficient, explain why and exactly what is missing, etc.

* After post-training, high-learnability pairs being increasingly likely to be invalid (Fig. 4), indicating the teacher can no longer invent genuinely new, correct problems as the student becomes stronger.

* About "Verifiable": symbolic mutator lets the same model rewrite the question and supply the new chain-of-thought + ground-truth answer; there seems no adequate method to guarantee the validity or correctness.

### Questions
1. In Section 5.2, the paper mentioned: "Since our RLVR process optimises only the student's performance and leaves the teacher static, this mismatch likely exacerbates the problem." What exactly are the student and teacher in RLVR? Is it not the same model that trains and generates data? Why is the teacher static?

2. What reward is used in RLVR? During data generation, is format checking required and should a format reward be used to guide the process?

3. Section 4.4 lists many tricks without detailed explanations-could additional operational descriptions be added? After learnability scores decay, are high-learnability problems replenished during training? Otherwise, how is it ensured that problems are not selected repetitively while still keeping high-learnability problems for training?

4. The paper mentioned: "We do not evaluate the distractor or symbolic mutators in isolation, as they cannot produce cross-category mutations." In fact, only the symbolic mutator changes the problem answer and requires altering the computation steps in the output-why is the comparison focused only on whether the category changes?

5. Why do the experimental results show improvement on symbolic tasks, yet on GPT-Eval-ID the performance of DÉJÀQ-A worsens while only DÉJÀQ-S improves, even though the invalidity of new problems generated by DÉJÀQ-S increases with training?

6. The GSM dataset contains relatively simple problems. Could scalability and generality be issues for this method? The paper only conducts experiments on a 7B model; would smaller models struggle to generate high-quality questions and answers, while larger models can already solve most problems in the GSM dataset? Moreover, among the three mutation methods, none is designed to alter the difficulty level of the problems.

7. The experimental baselines are too few; there is no comparison with performances of other same-scale open-source models.

8. The classification prompt is NOT found in Appendix E.

### Soundness
3

### Presentation
4

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
The paper introduces DejaQ, a framework for open-ended evolution of synthetic training data in reasoning domains, particularly mathematical problem solving. Instead of relying on static datasets, DejaQ co-evolves problem–answer pairs alongside model training using LLM-driven mutations (setting, distractor, symbolic). These mutations aim to increase dataset diversity and adapt difficulty to the model’s current capability. The authors integrate this approach with RLVR and MAP-Elites to manage diversity and learnability. Experiments with QWEN2.5-7B-INSTRUCT show performance gains over standard RL baselines and domain randomisation, especially in robustness and out-of-distribution generalisation.

### Strengths
- Interesting integration of evolutionary search and RL-based fine-tuning for dataset generation.
- Demonstrates a novel use of LLM-guided mutations that preserve verifiability while diversifying data.
- Evidence that DejaQ improves robustness and OOD performance.

### Weaknesses
- Should also show results on a different family of LLMs (e.g., llama) instead of just Qwen. Different families of LLMs might have different behaviours.
- Another ablation of mutating the samples but not having the learning progress sampling would be useful to see which components contribute to the algorithm's overall performance.
- The authors assume that GPT-5-mini is a "reasonable" oracle. A better scientific practice would be to show on a dataset or human annotations on how good GPT-5-mini is as a judge.
- For the result of post-training and invalidity base rate, it would be interesting to see the same plot (fig 4) for each mutation separately. Since "post-training raises the invalidity base rate for the setting mutator but lowers it for the all mutator", seeing the changes for each type of mutation could give more insight into the differences between DejaQ-A and DejaQ-S.

- Since defining learning progress is a key part of the paper, it is missing a lot literature on learning progress. 

The concept of learning progress in prediction or curiosity-driven networks originates from Schmidhuber’s early work on artificial curiosity in 1991 (see historical overview in https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5508364). It was later formalized by Oudeyer and Kaplan in 2007 as a computational mechanism for intrinsic motivation (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4141061, https://pmc.ncbi.nlm.nih.gov/articles/PMC2533589/). Around 2013, Oudeyer’s group introduced the notion of competence progress, measuring improvement in goal achievement or task completion to drive exploration and skill acquisition (https://www.sciencedirect.com/science/article/pii/S0921889012000644 and related works from 2013–2014). Since 2018, this principle has been integrated into intrinsically motivated deep reinforcement learning frameworks (https://arxiv.org/abs/1810.06284, https://arxiv.org/abs/1906.08190). More recently, similar approaches have been applied in complex environments such as Minecraft (https://arxiv.org/pdf/2106.14876) and in LLM-guided data generation settings https://arxiv.org/abs/2306.01711).

### Questions
- The behaviour descriptors are handcrafted (i.e., manually inspected by the authors to come up with the templates). Could this part be potentially automated? e.g., approaches like QDAIF (https://arxiv.org/abs/2310.13032) or ACES (https://arxiv.org/abs/2310.10692)
- As with Goodhart's law, when a measure becomes a target, it ceases to be a good target. Did they authors see any pathologies happening when optimizing for the proposed learning progress metric? Discussion on how this issue could be solved would be useful.
- It is said that the "initial learnability become stale as the model improves", and so the "learnability scores are decayed over time". It would be useful to see an ablation whereby the learnability scores are recalculated, to see how much of this is a problem.
- The authors show an ablation of keeping the same evolutionary process but resampling from the initial dataset. Is the resampling based on the learning progress metric in this baseline?
- How do the authors know quantitatively/ qualitatively if a task set is out-of-distribution?
- What is the "risk parameter alpha"?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces DéjàQ, a system for RLVR post-training where the dataset is evolved by a curriculum as well as LLM mutation. Evaluation results with Qwen2.5-7B-Instruct show the merits of this approach.

### Strengths
1. The paper presents a novel application of LLM mutators (to RLVR post-training)
2. The fact that the dataset evolution process can reuse the same inference infrastructure makes the approach more practical.
3. There are detailed analyses on robustness, maintaining verifiability, and resource requirements and hardware utilization.
4. The paper is well-written.

### Weaknesses
1. The paper would benefit from a clearer or more detailed description of the training process of DéjàQ and baselines. See Questions 2 and 3 below. (For example, the paper currently reads as though the training set and test set could’ve been identical (or at least the set of GSM-Symbolic templates used is identical across the training set and test set), and clarification from the authors would be appreciated.)

2. The fact that the RLVR baselines result in *worse* performance than the base model suggests that they were not implemented/engineered/tuned properly, since properly implemented RLVR should not result in worse performance. If that is indeed the case, then comparison with these baselines is not meaningful.

### Questions
1. Line 300 says “We do not evaluate the distractor or symbolic mutators in isolation, as they cannot produce cross-category mutations.” Why does the inability to produce cross-category mutations justify not evaluating distractor/symbolic mutators in isolation?

2. Could you explain more about how the seed training set is generated? Lines 192-193 say that the seed training set is GSM-Symbolic, but I’m not aware of an explicit training split for GSM-Symbolic. (https://huggingface.co/datasets/apple/GSM-Symbolic only contains a test split.)

3. Can you describe the “resampling” baseline in more detail? The current explanation (line 297) is not very clear to me.

4. In Algorithm 1, how often is the dataset evolved? In other words, how many iterations of (2) occur for every iteration of (1)?

5. Some of the benchmarks (GPT-Eval-ID, GPT-Eval-OOD) were generated by an LLM. How was it ensured that these synthetically generated benchmarks are of high quality (e.g., are error-free)?

### Soundness
2

### Presentation
3

### Contribution
3
