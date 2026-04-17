---
job_id: 86e5e5d5-f7cc-4b7d-af26-737c50554cee
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 0JLUFJMo5p.pdf
paper: Dynamic Task-Embedded Reward Machines for Adaptive Code Generation and Manipulation in Reinforcement Learning
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a reinforcement learning framework (DTERM) for code generation/manipulation with dynamic, task-conditioned rewards, which is clearly within ICLR’s scope (RL, representation learning, neurosymbolic aspects for code).

## Minimum Quality
Pass ✅.  
The submission includes Abstract, Introduction, Related Work, Method (Sections 3–4), Experiments/Results (Section 5, incl. Tables 1–2 and Figures 2–4), and a Conclusion section (although the conclusion text is partially corrupted). Despite substantial weaknesses in rigor and clarity, the work is a bona fide ML research paper rather than a non-scientific document.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
No hidden instructions, prompt injection attempts, or manipulative text targeting automated reviewers are apparent in the provided content.

---

# Expected Review Outcome:

## Summary

The paper proposes Dynamic Task-Embedded Reward Machines (DTERM), a framework for reinforcement learning on code generation and code manipulation tasks that dynamically weights multiple reward components (e.g., compilation success, test case pass rate, style, efficiency) based on task embeddings. A hypernetwork takes transformer-based task embeddings (e.g., CodeBERT) as input and outputs task-specific weights for sub-reward components, with additional mechanisms such as FiLM-style conditioning, prototype-based hierarchical adaptation, and optional multimodal extensions. Experiments are reported on several code benchmarks (CodeXGLUE, APPS, DeepFix, HumanEval), comparing DTERM to static reward baselines and showing improved performance in Table 1 and more robust performance on “unseen tasks” in Figure 2, along with an ablation study in Table 2.

## Strengths

1. **Timely problem and relevant setting**  
   The paper targets RL for code generation and manipulation, where reward design is indeed multi-objective (syntax, semantics, efficiency, style, etc.) and typically hand-tuned. Framing dynamic, task-aware reward composition as the central problem is well-motivated in Section 1 and Section 3.2, and this is an area of active interest in both RL and code LLM communities.

2. **Clear core idea: task-conditioned reward weighting via a hypernetwork**  
   The central mechanism, equations (5)–(6), is straightforward and conceptually clean: use a learned mapping from task embedding $\mathbf{e}_t$ to softmax-normalized weights $\alpha_i$ over pre-defined reward components $R_i(s,a)$, then combine them linearly. This is an understandable and implementable approach that many practitioners could adopt, even if the underlying theory is not deep.

3. **Reasonable modular architecture and integration story**  
   Figure 1 (the DTERM architecture diagram) usefully highlights the modularity: a task embedding generator, a hypernetwork weight generator, and a modular reward decomposer feeding into an RL policy optimizer with compiler integration. This compositional design is practical; in principle it can wrap around existing code-RL pipelines and RLHF setups with minimal disruption.

4. **Empirical results indicate benefits over static rewards**  
   Table 1 shows consistent improvements of DTERM over three baselines (Uniform, Expert-Tuned, GradNorm) across five task types (Summarization, Translation, Completion, Repair, Problems). The gains are non-trivial in some cases, e.g., +12.7 BLEU-4 on Translation and +10.5% relative improvement in Fix Rate. Figure 2 further supports the claim that DTERM degrades less on “unseen tasks” than static schemes, suggesting that task conditioning is at least empirically useful.

5. **Ablation study and reward analysis are directionally informative**  
   Table 2 isolates the contributions of key components on HumanEval: removing the hypernetwork or task embedding substantially hurts Pass@1, and removing FiLM or compiler feedback also degrades performance. This ablation supports the claim that each component matters. Figure 3, which plots the proportions of different sub-rewards in the final reward for each task type, is helpful to qualitatively validate that the system behaves sensibly (e.g., higher compilation/test emphasis for repair vs. more style/efficiency on other tasks).

6. **Interesting qualitative case and training dynamics**  
   The qualitative example in Section 5.6, where DTERM prioritizes fixing a null pointer exception over stylistic tweaks, makes the method intuitively appealing and fits the intended behavior of a task-aware reward system. Figure 4, showing a smoothly decreasing meta-training loss, suggests that the joint optimization of reward weights and policy does not explode or oscillate wildly.

## Weaknesses

1. **“Reward machine” terminology and theoretical framing are misleading / underdeveloped**  
   The title and Section 2.2–3.5 strongly evoke formal reward machines (à la Icarte et al., 2022), i.e., finite-state automata with explicit stateful reward logic. However, the actual method is essentially a *stateless linear combination* of scalar reward components with task-dependent weights, as per equations (5)–(6). No explicit automaton, state transitions, or temporal reward logic is defined. Section 3.5 only mentions reward machines conceptually; Section 4 never formalizes how DTERM instantiates a reward machine. This gap is not only conceptual but also affects clarity: readers expecting a formal reward-machine model will be confused why DTERM is just a hypernetwork-driven weighting of $R_i$. If the intent is metaphorical, the paper should say so and avoid overstating the connection; if not, then a rigorous FSM-based definition and use in code tasks is missing.

2. **Methodological under-specification and internal inconsistencies in the architecture**

   - **Equations (5) and (9) are not clearly integrated.** Equation (5) defines $\alpha_i$ directly as a softmax over $\mathbf{w}_i^\top \mathbf{e}_t + b_i$. Equation (9) then defines $\alpha_i$ as a convex combination of prototype-specific weights $\alpha_i^{(k)}$ with attention scores $a_k$ from equation (8). The paper does not explain how these two parameterizations co-exist. Is (5) only used in the non-prototype variant and (9) in the prototype variant? Are they combined? If (9) is the “final” choice, (5) becomes redundant; this conflict needs to be resolved.
   - **Objective and training of the hypernetwork and prototypes are unclear.** The text states that prototypes $\{\mathbf{p}_k\}$ and $\alpha_i^{(k)}$ are “learned during meta-training on many different types of tasks” (Section 4.3), but the paper never specifies the meta-objective or learning procedure. Is there a separate meta-optimization over tasks, or is everything trained jointly with PPO on the union of tasks? What is the exact loss used to learn prototypes and attention parameters $\mathbf{W}_a$ in equation (8)?
   - **RLHF integration (equation (12)) is ambiguous.** Equation (12) introduces $R_{\text{RLHF}} = \alpha_{\text{pref}} R_{\text{pref}} + \sum_{i=1}^{n-1} \alpha_i R_i$, but there is no formal description of how $\alpha_{\text{pref}}$ is produced (via hypernetwork or fixed?), how human preference data is collected for code tasks, or whether RLHF experiments are even run. This looks more like a sketch of a possible extension than an implemented, evaluated component, yet the abstract and Section 4.6 present it as integrated.
   - **Compiler feedback mechanism (equation (11)) is simplistic and underspecified.** $R_{\text{compile}}=\exp(-\lambda k)$ uses the number of compiler errors $k$, but the paper does not describe how $k$ is obtained in realistic compile logs (some compilers batch errors, some stop early), whether different error severities are distinguished, or how this interacts with incremental code edits. Without those details, it is hard to judge if this is a meaningful signal beyond a simple “compiles / does not compile” indicator.

   These omissions collectively make it difficult to reproduce the method or fully assess its soundness.

3. **Experimental setup lacks crucial detail, and evaluation is shallow given the ambitious scope**

   - **Task splits and “unseen tasks” are not defined.** Figure 2 plots “Cross-task generalization performance measured by normalized reward values” over “Unseen Tasks 1–10”, but the paper never explains how “unseen tasks” are constructed. Are they new problem IDs within the same benchmark, entirely held-out datasets, or new task types (e.g., from translation to repair)? Without a concrete description of the training vs test task distribution, the zero-shot/generalization claim is not verifiable.
   - **CodeXGLUE is referenced but unspecified.** Section 5.1 mentions “The CodeXGLUE dataset (?)” with a missing citation and does not indicate which specific sub-datasets are used, how splits are defined, or what train/validation/test sizes are. This is a major reproducibility gap.
   - **RL details are opaque.** Section 5.1 only states “We train using PPO with learning rate 3e-5 and batch size 32” and using 4 V100s. There is no information on sequence lengths, horizon T in equation (1), reward discount $\gamma$, number of environment steps, or how trajectories are generated for code tasks that usually involve long sequences and expensive compilation/execution.
   - **No statistical uncertainty or variance reporting.** Although the authors mention “3 random seeds”, Tables 1 and 2 have no standard deviations, confidence intervals, or significance tests. Some gains may be within noise, especially on smaller benchmarks such as HumanEval. This is particularly problematic for Table 2, where differences of a few points (e.g., 22.7 vs 21.1) could be non-significant.
   - **Baselines are incomplete for code RL.** The baselines (Uniform, Expert-Tuned, GradNorm) focus on static reweighting strategies. There are no comparisons to modern RL-for-code setups that integrate compiler/test-based rewards directly (e.g., more recent CodeRL-style methods), nor to alternative adaptive reward approaches. This weakens the claim that DTERM is meaningfully advancing the state of the art rather than modestly improving a custom baseline configuration.

4. **Inconsistencies, noise, and apparent copy-paste errors raise concerns about care and polish**

   - The **Conclusion** (Section 6) abruptly switches to “The Dual Selfular-Acting Machine (DSAM.Mouth Rachel) A new method for analyzing the dual selfular acting machine (DSAM), a generative text model architecture akin to one employed by ChatGPT.” This text is completely unrelated to DTERM and appears to have been copy-pasted from another document. This is not a minor typo; it undermines confidence that the manuscript was carefully prepared and reviewed by the authors.
   - Throughout the paper there are numerous grammatical errors and unfinished phrases, e.g., “The Word xog e is a resulting embedding e” (Page 4), “Bat var ‘Learning from choice of model (RLHF)” (Section 4.6), “complete subsets of experiments” (Section 5), “Case studies show late improving the generation” (Section 5.6). Such issues are frequent enough that they hamper readability and leave ambiguity about the intended meaning in some places.
   - Some references are malformed (“The CodeXGLUE dataset (?)”, “application of hypernetworks for reward function generation *(7)*”, “constrained optimization *(7)*”), implying missing or placeholder citations.

   While these issues are not strictly technical flaws, at ICLR standards they significantly affect clarity and perceived reliability.

5. **Mathematical formulation is mostly superficial and omits important details**

   - The reward composition is always linear (equation (6)), so the method does not address potential interactions or non-linear trade-offs among sub-rewards. There is no analysis of how the softmax weighting interacts with the magnitude and scale of $R_i(s,a)$; in practice, if different reward components have different ranges or variances, $\alpha_i$ alone may not be sufficient to balance them. The paper does not discuss normalizing or rescaling each $R_i$, nor does it justify the choice of using a softmax vs unconstrained weights.
   - Equation (7) introduces FiLM modulation $\mathbf{h}' = \gamma_i(\mathbf{e}_t)\odot\mathbf{h}+\beta_i(\mathbf{e}_t)$, but the paper never specifies what $\mathbf{h}$ actually is: is it an intermediate feature from a learned reward-estimation network, from the policy, or from code embeddings? Without this, one cannot reproduce or reason about the effect of FiLM. Moreover, there is no loss function written that uses $\mathbf{h}'$ explicitly, so the role of FiLM in the optimization pipeline is purely narrative.
   - The hierarchical prototype mechanism (equations (8)–(9)) also lacks a concrete training objective. In particular, if $\alpha_i^{(k)}$ are prototype-specific weights, there should be some regularization or diversity-promoting term encouraging different prototypes to represent different reward profiles; otherwise, trivial solutions (all prototypes identical) are possible. This is neither discussed nor addressed mathematically.

   These issues are not fatal, but they mean the paper lacks the level of mathematical precision and insight that would be expected for a method with “reward machines” and “hierarchical adaptation” in the title.

6. **Empirical analysis is limited relative to the claims**

   - **Table 1** provides only a single scalar metric per task, with no per-dataset breakdown, no data size, and no discussion of variance. For example, the “Problems | Pass@1” row aggregates over APPS but does not distinguish levels of difficulty or show where DTERM helps more. There is no analysis of how DTERM behaves at different training budgets or reward noise levels.
   - **Figure 3** is interesting but shallow: it shows proportions of five sub-rewards per task type, but there is no quantitative link between these proportions and task performance. For example, would enforcing a hand-designed weighting similar to the learned bars in Figure 3 reproduce most of the gains, or are the dynamic per-instance variations essential? This would be a natural additional experiment that is missing.
   - **Table 2** only reports ablations on HumanEval, which is relatively small. There is no cross-task ablation (e.g., on CodeXGLUE or DeepFix) to show that the same components matter more broadly.

7. **Missing and weakly contextualized related work**

   The Related Work section focuses mostly on older or high-level references; it misses several directly related and recent works on hypernetworks for task adaptation and adaptive reward design in RL and sequence generation. This contributes to an incomplete understanding of DTERM’s novelty and positioning; see the “Potentially Missing Related Work” section below for concrete examples.

8. **Questionable level of *true* novelty beyond standard mechanisms**

   At its core, DTERM combines a standard task embedding encoder (e.g., CodeBERT), a simple softmax over linear projections (equation (5)), and an off-the-shelf PPO training loop. Hypernetwork-based parameter generation and FiLM-style conditioning are well-known techniques. The hierarchical prototype mechanism is conceptually standard attention over learned prototypes. There is no new theoretical result, no novel RL algorithm, and the “reward machine” dimension is not formalized. While the application to code tasks with multi-objective rewards is interesting, the conceptual step from “static weighted sum of sub-rewards” to “task-embedding-conditioned softmax weights” is not very large. Given the weaknesses in experimental rigor and positioning, this makes it hard to argue for a high-contribution score at ICLR.

## Potentially Missing Related Work

1. **Rezaei-Shoshtari et al., “Hypernetworks for Zero-shot Transfer in Reinforcement Learning”, 2022**  
   This work studies hypernetworks for zero-shot policy transfer across tasks, directly relevant to DTERM’s goal of zero-shot adaptation to unseen coding tasks via task embeddings and hypernetworks (Sections 4.1 and 4.3, Figure 2). It should be cited in the Related Work on hypernetworks in RL (Section 2.3) and compared conceptually when discussing cross-task generalization.

2. **Ma et al., “Eureka: Human-Level Reward Design via Coding Large Language Models”, 2023**  
   Eureka uses large language models to design reward functions, with strong connections to dynamic reward modeling for tasks involving code and formal feedback. This is directly related to DTERM’s narrative of automating reward engineering (Sections 1 and 4.6). It should be discussed in Section 2.2 (Dynamic Reward Modeling) and Section 4.6, especially regarding RLHF and code-oriented reward design.

3. **Huang et al., “Continual Model-Based Reinforcement Learning with Hypernetworks”, 2020**  
   This paper uses hypernetworks for task-adaptive dynamics modeling in continual RL. It is relevant to DTERM’s hypernetwork-based adaptation (Sections 4.1–4.3), and should be mentioned in Section 2.3 when summarizing prior uses of hypernetworks for task-conditioned behavior and generalization.

4. **Chan et al., “Neural Keyphrase Generation via Reinforcement Learning with Adaptive Rewards”, 2019**  
   Chan et al. explore adaptive reward functions in RL for sequence generation, balancing syntactic and semantic metrics. This is conceptually close to DTERM’s multi-component reward balancing for code generation (Section 3.2). It should be cited in Section 2.2 as a prior example of adaptive reward composition in sequence tasks and compared in terms of mechanism (e.g., hand-designed vs learned task-conditioned weighting).

5. **Ren et al., “HyPoGen: Optimization-Biased Hypernetworks for Generalizable Policy Generation”, 2025**  
   HyPoGen introduces optimization-biased hypernetworks to generate generalizable policies across tasks. Given DTERM’s use of hypernetworks and prototypes for cross-task generalization (Section 4.3, Figure 2), this work should be discussed in Section 2.3 and in the experimental discussion of generalization performance.

6. **Jin et al., “Deep Reinforcement Learning with Task-Adaptive Retrieval via Hypernetwork”, 2023**  
   This paper proposes task-adaptive retrieval using hypernetworks in RL, closely related to DTERM’s task embedding-driven adaptation. It should be added to Section 2.3 and compared to DTERM’s approach of generating reward weights instead of retrieval parameters.

7. **Yuan et al., “MARBLE: Music Audio Representation Benchmark for Universal Evaluation”, 2023**  
   While focused on audio, MARBLE leverages hypernetwork-based task adaptation for representation learning, which parallels DTERM’s idea of using hypernetworks for task-aware configuration. It could be briefly mentioned in Section 2.4 or 2.3 as an example of hypernetwork-driven task adaptation in another domain.

In each case, these works are not just loosely related; they address hypernetwork-based adaptation or adaptive rewards and should be integrated into the motivation, positioning, and comparison to better clarify what is truly new in DTERM.

## Questions

1. **Clarification of the learning protocol for prototypes and task generalization**  
   How exactly are the prototypes $\{\mathbf{p}_k\}$ and $\alpha_i^{(k)}$ in equations (8)–(9) trained? Please describe the meta-training or joint training procedure, including the loss function, task sampling strategy, and any regularization or diversity-promoting terms. Also, how are “unseen tasks” in Figure 2 defined relative to the training tasks?

2. **Precise role and implementation of FiLM modulation (equation (7))**  
   What is the source of the feature vector $\mathbf{h}$ being modulated by $\gamma_i(\mathbf{e}_t)$ and $\beta_i(\mathbf{e}_t)$? Is it a hidden representation from a learned reward estimator, the policy network, or a code encoder? Please detail the architectural connections and show where FiLM enters the objective. An ablation figure or table indicating the effect of FiLM on different datasets beyond HumanEval (Table 2) would be helpful.

3. **Normalization and scaling of sub-reward components $R_i(s,a)$**  
   Since equation (6) uses a linear combination of raw sub-rewards, how are their scales handled? For instance, compilation success is binary, test case pass rate is a fraction, BLEU can range widely, and efficiency might be a continuous runtime score. Are these components normalized to comparable ranges before weighting, or does the hypernetwork implicitly learn to handle unnormalized magnitudes? Providing explicit formulas or implementation details for each $R_i$ would increase confidence in the empirical results.

4. **Missing details on datasets and splits, especially CodeXGLUE**  
   Please specify exactly which CodeXGLUE sub-datasets are used for Summarization, Translation, and Completion, and how train/validation/test splits are defined. Are the Expert-Tuned weights drawn from a particular prior paper for each task, or hand-designed for this work? Adding a table with dataset statistics and split sizes would help.

5. **Statistical significance and variability**  
   Can you provide results in Tables 1 and 2 with mean ± standard deviation over the 3 seeds, and possibly conduct simple statistical tests (e.g., paired t-tests) to verify that the improvements of DTERM over baselines are robust? This is particularly important for small benchmarks like HumanEval.

6. **Clarification of RLHF experiments and scope of claims**  
   Is equation (12) implemented and evaluated in any experiment in the paper, or is it purely conceptual? If implemented, please describe the human preference data, comparison baselines, and concrete metrics. If not, I recommend reframing Section 4.6 as future work or a potential extension, and clarifying that the empirical evaluation does not cover RLHF.

7. **Fixing the Conclusion and improving overall clarity**  
   The current Conclusion refers to “Dual Selfular-Acting Machine (DSAM)”, which appears to be unrelated. Please replace it with a concise summary of DTERM, key findings (e.g., from Table 1, Figure 2, Table 2), and limitations. Along with a careful proofreading pass to fix typographical errors (e.g., “The Word xog e…”, “Bat var ‘Learning from choice of model”), this would substantially improve the paper’s presentation quality.

Author responses that provide clear and detailed answers to these questions, along with corrected text and possibly expanded experiments, could improve my assessment.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The core idea (task-conditioned weighting of sub-rewards via a hypernetwork) is reasonable and consistent with the empirical trends shown in Table 1, Figure 2, and Table 2, but the method is under-specified in several key places (prototypes, FiLM, RLHF integration), and the experimental protocol lacks important details and statistical rigor.

## Presentation Rating

2: fair.  
The overall structure is standard and most equations (1)–(12) are readable, and Figures 1–4 and Tables 1–2 help illustrate the approach and results. However, persistent grammatical errors, missing citations, and a completely off-topic Conclusion section significantly degrade clarity and polish.

## Contribution Rating

2: fair.  
The paper addresses a relevant problem and proposes a plausible solution that appears empirically helpful, but the conceptual step over known hypernetwork/FiLM-based adaptation is modest, the “reward machine” angle is not realized formally, and the evaluation and positioning do not convincingly establish strong novelty or impact.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper has an interesting and practically relevant idea with encouraging empirical evidence (notably Table 1 and Figure 2) and some nice analysis (Figure 3, Table 2). However, the combination of methodological under-specification, missing experimental details, weak statistical treatment, misaligned “reward machine” framing, and serious presentation issues (especially the erroneous Conclusion) makes it fall short of ICLR standards in its current form. With significant revision and clarification, it could become a solid contribution.

## Reviewer Confidence

4: confident.  
I am comfortable with RL, hypernetworks, and code generation/LLM-based methods, and I carefully checked the equations and experimental descriptions provided. Some experimental details are missing by design, which limits full verification, but my assessment of the paper’s strengths and weaknesses is unlikely to change dramatically.