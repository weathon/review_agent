# Answer-Consistent Chain-of-thought Reinforcement Learning For Multi-modal Large Langauge Models

- Decision: Reject
- Scores: 6, 4, 2, 4, 4, 6

## Abstract
Recent advances in large language models (LLMs) have demonstrated that reinforcement learning with verifiable rewards (RLVR) can significantly enhance reasoning abilities by directly optimizing correctness, rather than relying solely on supervised imitation. This paradigm has been extended to multimodal LLMs for complex video and image understanding tasks. However, while outcome-driven RL improves answer accuracy, it can inadvertently decouple the reasoning chain from the final answer, leading to situations where models produce inconsistency between the reasoning trace and final answer. In our experiments on multiple-choice visual question-answering tasks, the standard GRPO method yields only 79.7% consistency on MMVU between the reasoning steps and the chosen answers, indicating frequent mismatches between the answers and the reasoning. To this end, we propose Answer-Consistent Reinforcement Learning (ACRE) that modifies the GRPO algorithm with an auxiliary consistency check. After the model generates a chain of thought and an initial answer for a given question, we shuffle the answer options and prompt the model again with the same reasoning trace to predict a second answer. We design a consistency-verification reward that grants a high reward only if both the original and the post-shuffle answers agree and are correct; otherwise, a lower reward is assigned accordingly. This mechanism penalizes reasoning-answer misalignment and discourages the model from relying on spurious patterns, such as option ordering biases. We evaluate ACRE on challenging Video Reasoning benchmarks and image reasoning benchmarks, achieving an average 2.2 improvement over the GRPO baseline.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper identifies and addresses the "reasoning-answer inconsistency" problem in Multi-modal Large Language Models (MLLMs) fine-tuned with reinforcement learning, where models may produce flawed reasoning that coincidentally leads to a correct answer, or vice-versa. To mitigate this, the authors propose a novel reward shaping method called Answer-Consistent Reinforcement Learning (ACRE). ACRE enhances the training signal by introducing a consistency check: a high reward is granted only when the model's generated reasoning trace is robust enough to yield the correct final answer both with original and shuffled multiple-choice options. Through comprehensive experiments and two newly proposed metrics (CACR and OSCR), the paper demonstrates that ACRE not only improves task accuracy on video and math reasoning benchmarks but also significantly enhances the faithfulness and robustness of the model's reasoning process.

### Strengths
1. This paper clearly identifies the “reasoning-answer inconsistency” problem, which warrants attention. The two metrics proposed by the authors, CACR and OSCR, also serve as reasonable quantitative tools.
2. The proposed ACRE method is an elegant solution. The option-shuffling mechanism for reward shaping is a simple but highly effective way to enforce reasoning robustness.
3. The claims are supported by a solid experimental setup. The evaluation is thorough, including not only accuracy metrics but also generalization tests, ablations, and qualitative analyses.
4. The paper is well-written and easy to follow.

### Weaknesses
1. The core mechanism of ACRE, option shuffling, is inherently tied to the multiple-choice question format. It is unclear how this method could be extended to more open-ended generation tasks where a discrete set of answer choices is not available. The paper would be strengthened by a discussion of this limitation.

2. The introduction powerfully motivates the work by quantifying the initial CR-WA and WR-CA rates (18.4% and 2.5%) on MMVU, providing a very concrete problem statement. However, a significant weakness is that the experimental evaluation never revisits these specific metrics to show how ACRE affects them. The paper instead shifts to evaluating improvements on the proposed proxy metrics, CACR and OSCR. While these proxies are insightful, the failure to demonstrate a direct reduction in the originally-cited error rates leaves the core argument feeling incomplete.

### Questions
1. The benchmarks used to evaluate CACR (Table 1) and OSCR (Table 2) are partially overlapping but not identical. For instance, MathVista is used for CACR, while TempCompass is used for OSCR. Could the authors elaborate on the rationale for this choice?

2. The verification step in ACRE assumes that a sound reasoning trace t should be self-sufficient and lead to a final answer independent of the option format (e.g., the letter 'A', 'B', 'C'). However, one could argue that a valid reasoning style might conclude with a statement like, "...and therefore, the correct choice is B." In this scenario, even if the underlying logic is perfect, the fixed trace t would fail the consistency check after option shuffling, thus unfairly penalizing the trajectory. Did the authors consider this potential bias? Does ACRE implicitly favor a specific format of reasoning traces that avoids explicit references to option letters?

3. ACRE's reward mechanism is designed to penalize inconsistency. Is it possible, however, for a model to learn a reasoning process that is flawed yet highly robust, leading to the same incorrect answer consistently, both before and after option shuffling? While this would be penalized by the ground-truth reward, could the strong inductive bias towards 'consistency' introduced by ACRE sometimes trap the model in a local minimum of being 'confidently wrong,' rather than encouraging it to explore pathways to correct reasoning?

4. The related work section provides a good overview of outcome-based RL methods (like GRPO) and general preference optimization techniques (like DPO). However, the discussion could be significantly strengthened by including a complementary line of research that focuses on enhancing reasoning via process supervision. For example, recent works like video-SALMONN-o1 (Sun et al., 2025) introduce methods such as pDPO, which use step-level rewards within a preference optimization framework to directly supervise the correctness of the reasoning process itself.

### Soundness
4

### Presentation
4

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
The authors propose a new RL framework for multimodal LLMs based on answer consistency. They first observe that a model’s reasoning process and its final answer are often logically inconsistent. To address this issue, the authors introduce an auxiliary consistency check during GRPO: by shuffling the answer options and re-prompting the model with the same reasoning trace, they evaluate whether the model produces a consistent answer. The proposed method shows notable improvements on both video and math reasoning benchmarks.

### Strengths
- The paper is easy to follow and well-written.
- The initial observation that GRPO can produce logical inconsistencies between the model's reasoning and its final answer is insightful and interesting.
- The proposed method seems to be effective on video and math reasoning benchmarks.

### Weaknesses
- Intuition. While the initial observation is interesting, the proposed solution is not entirely intuitive. Simply shuffling the answer options may not be sufficient to ensure alignment between the reasoning process and the final answer.

- Applicability. The proposed method relies on the presence of answer options, making it inapplicable to open-ended settings such as free-form VQA tasks.

- Generalizability. The authors demonstrate the effectiveness of the method only on a Qwen-based model, raising questions about its generalizability to other V&L architectures.

### Questions
1. Scalability. If the model size is scaled up (e.g., using Qwen2.5-VL-32B), will the proposed method remain effective and provide consistent improvements?

2. Can the proposed approach generalize to broader multimodal reasoning benchmarks such as MMMU?

### Soundness
3

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
3

### Summary
This paper studies the reasoning–answer inconsistency that emerges when outcomeonly reinforcement learning is applied to multimodal, multiple-choice reasoning. This paper mainly contributes to the following two points:

1). Proposing two complementary tests—the CoT and Answer Consistency Rate (CACR) and the Option Shuffling Consistency Rate (OSCR), and showed that vanilla GRPO improves answer accuracy yet erodes consistency between the generated chain-of-thought (CoT) and the final answer. 

2). Proposing Answer-Consistent REinforcement Learning (ACRE), a GRPO-compatible reward shaping scheme that enforces shuffle-invariant agreement conditioned on correctness.

Although this paper find the wrong matching between the CoT and final answer, and the disorganized option and final answer, it did not conduct a comprehensive and complete experiment and analysis on this situation. The proposed improvement scheme is also lack of complete description and ablation experiments (this part will be described in detail in "Weaknesses"). Finally, this paper does not explain the reasons for its application in multimodal field alone, rather than more widely used in GRPO, so it is recommended to reject this paper.

### Strengths
This paper finds that the application of GRPO in multimodal fields (specifically in video reasoning and mathematical tasks) will lead to the inconsistency between the CoT and the final answer, and the shuffle of options in QA task will lead to the vulnerability of the final answer. Some experiments and visual displays are carried out. Therefore, complementary tests, CACR and OSCR, are proposed to assess the impact of this phenomenon.

### Weaknesses
The weaknesses of this paper focus on the method, experiment and writing.

In method：

1). This paper does not explain why GRPO is studied in multimodal rather than single text or vision. 

2). This paper does not explain why the first reasoning trace is used to generate the second final answer in the proposed scheme ACRE. And there is no ablation experiment designed to compare the results of generating the final answer from scratch instead of using the first reasoning trace.

3). In Chapter 4, there is no specific description and analysis of the three consistency-shaping coefficients in Formula 3, which is lack of logic. It's confusing to read.

In experiment：

1). In Section5.1, the last row of Table 3, “ACRE” does not explain the experimental setup and the model on which the method ACRE was applied. Is it based on Qwen2.5-VL-7B-CoT-SFT in the paper?

2). In Section5.3, in the “Hyperparameters” part of ablation experiment, the consistency-shaping coefficient setting was obtained without ablation experiment on α_1 (specifically in line 463).

In writing：

1). In Section 5.1, this paper directly uses SFT model from Video-R1 and identifies it as Qwen2.5-VL-7B-CoT-SFT. Then in the "Train-data" column of Table 3, the training data of Qwen2.5-VL-7B-CoT-SFT should be the data of Video-R1-SFT, that is, Video-R1-CoT-165k expressed in line 323.

2). Figure 4 is not used in this paper.

3). Some table descriptions are out of line with the conclusion analysis. For example, the analysis in Chapter 3 explains the settings in Chapter 5.

4). The expression of figure in the paper is inconsistent. Figure 2 is expressed by “Figure 2” (line 250), while figure 1 and figure 3 are expressed by “Fig. 1” (line 85), “Fig. 3” (line 377).

5). This paper is weak in expression, and some formulas and experimental settings are confusing to read.

### Questions
In the description of ACRE in Chapter 4, it is possible that the order of the shuffled options is the same as that of the original options. Why not remove this situation?

### Soundness
2

### Presentation
1

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
This paper proposes **ACRE (Answer-Consistent Reinforcement Learning)**, an RL that optimizes LLM based on the consistency between the COT reasoning and its final answers. ACRE improves GRPO by introducing an auxiliary **consistency-verification reward**: the LLM is prompted with the reasoning trace and shuffled answer options. Different rewards are assigned depending on whether both answers match and both answers are correct.
Experiments on multimodal reasoning benchmarks, including video and math tasks, demonstrate that ACRE improves both video and math benchmarks.

### Strengths
1. The proposed ACRE method is conceptually simple yet effective, integrating a consistency-verification reward into GRPO without introducing heavy architectural changes.
2. Experimental results are convincing, showing consistent improvements across video and math benchmarks.

### Weaknesses
1. The approach is limited to multiple-choice QA, which limits its scalability. It is unclear whether ACRE generalizes to open-ended or generative reasoning tasks.
2. There are several writing-related errors or typos, such as:
  * L118-119: "...respectively. Even out-performing..." -> "...respectively, even out-performing..."
  * L154: it seems a period is omitted.
  * Equation (2): notations $x, q, o$ are not defined
  * L346-347: "Then $S(q) = $" is the sentence complete? For L431: "it can be further optimized since.", it is the same.

### Questions
1. The method design is not tailored for multimodal models only. How is ACRE performed on text-only LLMs?
2. How is ACRE performed on larger-scale models, such as 32B models?
3. Although ACRE improves the performance on reasoning tasks, the OSCR is still much lower than SFT models after RL. How is OSCR correlated with the understanding or reasoning performance?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem of "reasoning-answer inconsistency" in Multimodal Large Language Models (MLLMs) that are fine-tuned with reinforcement learning (RL). The authors observe that standard outcome-driven RL methods like Group Relative Policy Optimization (GRPO), while improving final answer accuracy, can lead to models where the generated Chain-of-Thought (CoT) does not logically support the final answer. They identify two failure modes: Correct Reasoning but Wrong Answer (CR-WA) and Wrong Reasoning but Correct Answer (WR-CA).
To mitigate this, the authors propose Answer-Consistent REinforcement Learning (ACRE), a reward-shaping method built on top of GRPO. ACRE introduces a consistency check: after a model generates a CoT and an initial answer, the multiple-choice options are shuffled. The model is then re-prompted with the same CoT and the shuffled options to produce a second answer. A high reward is given only if the model is both correct and consistent across both rounds (i.e., the answers match). This mechanism is designed to penalize reliance on spurious shortcuts like option ordering and enforce a stronger link between reasoning and the final decision. Experiments on video and math reasoning benchmarks show that ACRE improves consistency and achieves modest accuracy gains over the GRPO baseline.

### Strengths
1. Excellent Problem Diagnosis and Analysis: The paper does a fantastic job of not just identifying but also quantifying the "reasoning-answer inconsistency" problem. Citing specific figures, such as finding 18.4% of samples in MMVU are CR-WA, provides strong motivation and highlights a real, non-trivial issue in current RL fine-tuning practices for MLLMs.

2. Intuitive and Well-Designed Method: ACRE is an elegant and targeted solution to the identified problem. Using option shuffling as a perturbation to test the robustness of a reasoning trace is a very direct and clever way to enforce consistency. It is a simple idea that is executed effectively.

3. Insightful Attention Analysis: The attention visualization in Figure 3 is a major strength. It provides a concrete, detailed explanation for the observed behavior. The finding that GRPO's attention collapses onto the system prompt and option indices, while ACRE's attention is redistributed to the content of the options and the output tokens, offers a compelling, low-level justification for why ACRE improves robustness.

### Weaknesses
1. Modest Empirical Gains: The reported performance improvements, while consistent, are quite small. An average gain of +2.2% on Video Reasoning and +1.5% on Math Reasoning (Table 3) over the GRPO baseline is an incremental advance. Given that the method introduces additional computational cost, the significance of this gain is debatable.

2. Significant Computational Overhead for a Small Gain (?) : The ACRE method requires a second forward pass through the model to generate a second answer for the consistency check. The authors are transparent about this, reporting in Table 4 a 24% increase in training GPU hours (4.5h vs 5.6h). This is a non-trivial increase in training cost for the modest accuracy improvements achieved.

3. Limited Scope of the Solution: The proposed consistency check is inherently designed for multiple-choice question-answering tasks. It is unclear how the core idea of ACRE could be generalized to more open-ended tasks where answer options cannot be easily shuffled, limiting the broader applicability of the method. The option-shuffling mechanism is specific to multiple-choice formats. How do you envision the principle of "answer-consistent RL" extending to other important multimodal tasks that lack a discrete set of answer choices, such as open-ended VQA, visual grounding, or instruction-following?

### Questions
I rate this marginally below the average. If the authors could refer to the weakness above and address most of them, I would be willing to raise the scores.

### Soundness
3

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
3

### Summary
This paper addresses the "reasoning-answer inconsistency"  that arises when training Multi-modal Large Language Models (MLLMs) with outcome-based reinforcement learning (RL) methods like GRPO. The authors observe that while GRPO improves final answer accuracy, it can decouple the reasoning chain (Chain-of-Thought, CoT) from the answer, leading to situations where the model produces correct reasoning but a wrong answer (CR-WA) or, conversely, flawed reasoning that coincidentally results in a correct answer (WR-CA). To quantify this, the paper introduces two metrics: the CoT Answer Consistency Rate (CACR) and the Option Shuffling Consistency Rate (OSCR). To solve the inconsistency problem, the authors propose Answer-Consistent Reinforcement Learning, a reward-shaping scheme that modifies GRPO. ACRE works by performing an auxiliary consistency check. After the MLLM generates a CoT and an initial answer for a multiple-choice question, the answer options are shuffled. The model is then prompted again with the original reasoning trace and the shuffled options to produce a second answer. A high reward is given only if both the original and post-shuffle answers are correct and identical. This mechanism penalizes reasoning-answer misalignment and discourages reliance on spurious patterns like option ordering.

### Strengths
1. The paper's primary strength is its clear diagnosis of the "reasoning-answer inconsistency"  and the cleverness of the proposed solution. The idea of using an auxiliary consistency check based on option shuffling to validate a reasoning trace is highly original. It directly targets the model's reliance on spurious correlations, such as option ordering, which is a known but difficult-to-solve problem.
2. The authors provide convincing empirical evidence for their claims. They first show that standard GRPO training erodes consistency (Tables 1 and 2), validating the premise of their work. They then demonstrate that ACRE not only recovers this consistency but also surpasses the CoT-SFT baseline, all while achieving superior accuracy on the final task.
3. The paper goes beyond just reporting scores. The attention visualization in Figure 3 provides a valuable qualitative insight, suggesting that ACRE learns to attend more to the content of the options and its own reasoning, while GRPO disproportionately attends to the system prompt and option indices.
4. The paper shows that ACRE achieves strong results while being trained on a relatively small dataset (ACRE-9.2k). It even outperforms a model (Video-R1-7B) trained on a much larger dataset (260k samples), highlighting the efficiency of the proposed reward-shaping scheme.

### Weaknesses
1. The most significant weakness is that the ACRE framework, as presented, is fundamentally dependent on a multiple-choice question (MCQ) format. The core consistency check relies on the existence of a discrete set of answer options that can be shuffled. This limits the method's direct applicability to many other important tasks, such as open-ended question answering, code generation, or mathematical proof generation, which lack this structure.
2. While the conclusion briefly mentions "alternative rephrasing strategies" like using an LLM to paraphrase the query, this is purely speculative. It's unclear how this would work in practice, how the consistency reward would be calculated (especially without a single ground-truth answer), or what the computational and implementation costs of such a strategy would be.
3. The method inherently requires a second forward pass through the model to generate the post-shuffle answer $\tilde{a}$. The authors report this as a +24% increase in training GPU hours, which, while noted as "acceptable", is a non-trivial overhead.

### Questions
1. How do the authors envision adapting ACRE to open-ended generative tasks (e.g., free-form visual question answering or text generation) where there are no explicit "options" to shuffle?
2. Following up on the idea in the conclusion: If an LLM were used to produce "semantically equivalent paraphrases" of a question, how would the consistency-verification reward $r_c$ be formulated? In an open-ended setting, the original answer $a$ and the new answer $\tilde{a}$ would both be free-text. Would this require another LLM-as-Judge to determine if $a$ and $\tilde{a}$ are "consistent" and "correct"? This seems to add significant complexity and potential sources of noise.

### Soundness
3

### Presentation
3

### Contribution
3
