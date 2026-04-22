# Test-Time Alignment for Large Language Models via Textual Model Predictive Control

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
Aligning Large Language Models (LLMs) with human preferences through finetuning is resource-intensive, motivating lightweight alternatives at test time. We address test-time alignment through the lens of sequential decision making, a perspective that reveals two fundamental challenges. When actions are defined at the token level, as in guided decoding, alignment suffers from the curse of horizon. Conversely, when actions are at the response level, as in traditional iterative refinement, the curse of dimensionality emerges. To resolve this trade-off, we draw inspiration from Model Predictive Control (MPC) in control theory to propose Textual Model Predictive Control (TMPC), a novel predictive planning framework adapted for aligning LLMs at inference time. A key limitation of standard MPC is its reliance on predefined, hard segment boundaries, which are often absent in text generation. TMPC overcomes this by introducing two principles inspired by hierarchical reinforcement learning: (1) Hindsight Subgoal Identification, where TMPC analyzes generation subgoals to retrospectively identify high-reward intermediate outputs as subgoals. This allows the framework to discover meaningful, task-specific planning steps (e.g., a sentence in machine translation or a bug fix in code generation.). (2) Subgoal-Conditioned Re-Generation, where these identified subgoals are used to guide subsequent planning iterations. By conditioning on these proven, high-quality subgoals, TMPC ensures stable improvement by building upon previously validated successes. TMPC is evaluated on three tasks with distinct segmentation properties: discourse-level translation, long-form response generation, and program synthesis. The results demonstrate that TMPC consistently improves performance, highlighting the generality. Project page: https://rl-bandits-lab.github.io/TMPC/.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents Textual Model Predictive Control (TMPC), a novel framework for test-time LLM alignment. It insightfully frames the problem as a trade-off between the 'curse of horizon' in token-level optimization and the 'curse of dimensionality' in response-level optimization. TMPC introduces a middle ground by first identifying meaningful "subgoals" from generated text and then optimizing generation conditioned on these discovered segments. The framework uses two core principles: Hindsight Subgoal Identification to find high-reward text segments, and Subgoal-Conditioned Re-Generation to build upon them, achieving strong results without updating model parameters.

This is a very good paper with a highly promising and original core idea. The conceptual framing is excellent, and the initial results are strong. However, its claims of novelty rest on an unsubstantiated comparison, and the paper writing to clarity of its core mechanisms and analysis needs improvement. With the addition of a crucial heuristic baseline comparison and a clearer analysis of its mechanics, this paper has the potential to be a top-tier.

### Strengths
* The paper's primary strength is its excellent framing of the alignment problem as a trade-off between token-level and response-level, providing a clear motivation for a new class of solutions. The central concept of planning via subgoals is elegant. Figure 1 is a great illustration.
* The method is designed to be task-agnostic, discovering subgoals automatically rather than relying on pre-defined structures. This is a significant advantage over methods that require task-specific engineering.
* Bringing ideas from control theory is interesting.

### Weaknesses
* The paper's central claim is that its hindsight identification of subgoals is a key innovation. However, given this is for test-time alignment, users can easily use a cheap, task-specific heuristic (e.g., sentence boundaries for translation) for their own task. So, I think task-specific handmade subgoal is a very strong and obvious baseline. Without showing that TMPC's discovered subgoals outperform these simple heuristics, the true value of the complex discovery mechanism is questionable.
* The paper's idea is good but a better writing may improve the quality and make it easier to follow. The central concepts of "subgoal" and "re-generation" remain abstract due to a lack of concrete examples, and the results are not deeply explored maybe due to limited space. We strongly suggest  maybe a better use of appendix and a concise revision on the methods background and literature to using the space to provide the essential examples and analysis needed to fully validate the framework's contribution.

### Questions
* To justify the core novelty of hindsight identification, is TMPC demonstrably better than using simple, human-curated subgoals? For instance, in the paragraph-level MT task, how does TMPC's performance compare against a version that uses pre-defined sentence boundaries as the subgoals? This single experiment is critical to proving the value of your proposed mechanism over cheaper, task-specific alternatives.
* Figure 2 suggests subgoals from different rollouts are composed as conditioning for the next generation step. How does the model ensure narrative and logical coherence when concatenating these segments, which may originate from conflicting generation paths? Can you provide analysis showing this composition does not harm the fluency and continuity of the output?
* The concept of a "subgoal" remains abstract. Could you provide concrete examples of the subgoals TMPC actually identifies across the three tasks? For program synthesis, what does an "abstract" subgoal that resolves a single test case look like in the generated text? This is essential for understanding what the model is learning to value as a planning step.
* The subgoal buffer B is a finite resource. In very long generation tasks, what prevents the buffer from being filled with early, locally-optimal subgoals, thus hindering exploration in later stages of generation? Is there a mechanism to manage the buffer's contents beyond a simple capacity limit, such as diversity promotion or aging out older subgoals?

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
4

### Summary
This paper formulates LLM-generated subgoals as a model predictive control problem (TMPC). They identify subgoals via hindsight replay and use the identified subgoals to regenerate the final answer. The have extensive results on against many baseline methods on various datasets.

### Strengths
-Figure 1 clearly shows the high-level idea of the approach compared to traditional methods. The caption is also descriptive and clear to reinforce the ideas in the Figure and introduction

-The paper is easy to follow and mythology is clear

-The proposed approach outperforms baselines in various tasks. I specifically appreciate the failure cases in baselines such as in L406.

### Weaknesses
The related works discussion lacks a discussion on subgoal generation for LLM/VLM-based tasks, where prior work already exists [1, 2]. I do think there is novelty in formulating this as a test-time model predictive control problem.

[1] Logeswaran, Lajanugen, et al. "Few-shot Subgoal Planning with Language Models." Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2022.

[2] Wang, Jiawei, et al. "Discovering intrinsic subgoals for vision-and-language navigation via hierarchical reinforcement learning." IEEE Transactions on Neural Networks and Learning Systems 36.4 (2024): 6516-6528.

### Questions
Minor Suggestion:
-There is an extra comma on L28
-Maybe move the legend of Figure 1 to the top or bottom and spanning across all three methods to make it more clear it applies to all three

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Textual Model Predictive Control (TMPC), a novel framework for aligning large language models (LLMs) with human preferences at test time by casting the generation process as a sequential decision-making problem. TMPC draws inspiration from model predictive control (MPC), incorporating Hindsight Subgoal Identification to automatically detect useful intermediate subgoals and Subgoal-Conditioned Re-Generation to iteratively refine generations. The framework is evaluated on long-form machine translation, long-form response generation, and program synthesis, demonstrating improvements over several training-time and test-time baselines.

### Strengths
1. TMPC elegantly adapts concepts from control theory (specifically model predictive control) to the LLM test-time alignment setting, innovating with subgoal identification and iterative planning, as detailed in the mathematical framework of Section 4 and illustrated in Figure 2.

2. The methodology is general: TMPC is applied to tasks with both natural and abstract segment boundaries (e.g., machine translation sentences, code unit tests, long-form response chunks), supporting claims about its versatility.

3. Empirical results cover multiple domains using both automatic and large-model-based human-proxy metrics. On WMT'24 translation (Table 1), TMPC outperforms both test-time and competitive training-time alignment baselines; for example, it achieves state-of-the-art SEGALE scores and minimal NA Ratios across several language directions, substantially exceeding methods like ARGS and GenARM.

### Weaknesses
1. Reward-model dependence / shared-judge bias. Long-form tasks use similar reward models for both alignment and evaluation, inviting bias and potential reward gaming; despite noise-robustness tests, evidence for agreement with human preferences and cross-evaluator consistency is limited.

2. Underspecified subgoal & buffer aggregation. The threshold α, buffer 𝓑 update/size policy, and aggregation function 𝒢 (composition of non-contiguous subgoals, overlap limits, length control) are not concretely specified; scalability as buffer/search grow is untested.

3. Unquantified compute/latency cost. Multi-round rollouts and rewrites likely incur higher cost than token-level guidance (e.g., ARGS, GenARM), but per-task wall-clock time, GPU-hours, and throughput/latency are not reported.

4. **Missing comparisons to sequence-level rewriting baselines.** No direct comparison against sequence-level rewriters (e.g., **aligner**, **sentence aligner**), leaving TMPC’s advantage over these methods unclear.

### Questions
1. Can the authors provide more formal analysis or empirical ablation regarding the impact and tuning sensitivity of the buffer threshold $\alpha$ and the aggregation function $\mathcal{G}$? For example, does increasing $\alpha$ risk filtering out necessary diversity or causing premature convergence?

2. What explicit strategies does TMPC employ to avoid redundancy or incoherence when assembling non-contiguous subgoals in the generation process? Are there failure cases where the compositionality breaks down?

3. Can you include more thorough ablation on the contributions of Hindsight Subgoal Identification versus Subgoal-Conditioned Re-Generation? A demonstration on at least one benchmark of removing or varying each component in isolation would help clarify this.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper considers the problem of test-time alignment of a fixed large language model (LLM) with a given reward function that encodes externally specified preferences. The paper proposes an iterative improvement method, called Textual Model Predictive Control (TMPC), that combines (i) identification of promising, high-reward subgoals (tokens, sections of text, etc.) within a response with (ii) aggregation of promising subgoals into subsequent prompts. The proposed method is loosely based on the standard model predictive control (MPC) formulation, with the MPC-based formulation of Section 4.1 serving as the point of departure; Principles 1 and 2 of Section 4.2 are the core components of the approach, however, and these are not clearly MPC-based. Experimental evaluation of TPMC against relevant baselines on machine translation, long-form response generation, and program synthesis problems is provided. The experimental evaluation indicates that TMPC outperforms baselines on two-thirds of the machine learning tasks, outperforms all baselines on the long-form response generation task, and achieves higher pass rates than all baselines on the program synthesis tasks.

### Strengths
The test-time alignment problem has seen immense interest from the community in recent years, so the topic of the paper is timely. The proposed TMPC approach is an intuitively appealing and natural approach to this problem, so the paper is likely of interest to the community. Drawing inspiration from MPC through the formulation given in Section 4.1 provides motivation and technical clarity for the (non-MPC) specifics of the proposed approach described in Section 4.2. The experiments consider three well-known benchmark problems and compare the proposed approach with a good variety of appropriately chosen, well-known baseline methods. The experimental results support the effectiveness of the proposed TMPC approach.

### Weaknesses
1. The connection to MPC feels somewhat overstated. Section 4.1 provides enough context to justify describing the approach as loosely MPC-inspired. However, since the main elements of the method are the non-MPC components outlined in Principles 1 and 2 (Section 4.2), framing as an MPC-based approach may be a bit strong.
2. The experimental results, though promising, contain some drawbacks that are not adequately discussed. First, though it is claimed on lines 401-402 that "TMPC consistently outperforms all test-time alignment baselines" on the machine translation tasks, Table 1 shows that TMPC is in fact outperformed by TPO on the Chinese-to-Russian translation task. In addition, as mentioned in footnote 3 on page 8, on the long-form response task TPO was stopped after two iterations instead of the four iterations used elsewhere due to implementation and/or hardware issues, not problems inherent to TPO. For this reason, the comparison results presented in Figure 3 are incomplete with respect to TPO; it is especially important to have an accurate picture of TPO's performance given that TPO was a top competitor on the machine translation tasks.

### Questions
1. In what way is the formulation of the test-time alignment problem as a sequential decision-making problem (lines 97-98) novel?
2. By what mechanism does the does the design of TMPC "achieve a better balance between accurate credit assignment and the size of the search space" (lines 240-241)?
3. Is it true that Principles 1 and 2 on page 5 are not clearly related to MPC?
4. Can you elaborate on the fact that GPT-4o outperforms or is highly competitive with TMPC in the experiments presented in Table 1?
5. Can you comment on the fact that on lines 401-402 it is stated that "TMPC consistently outperforms all test-time alignment baselines" on the machine translation tasks, yet Table 1 shows that TMPC is outperformed by TPO on one of the three tasks?
6. As mentioned in footnote 3 on page 8, on the long-form response task TPO was stopped after two iterations instead of four iterations; does this negatively impact the fairness of the comparison between TPO and TMPC in Figure 3?

### Soundness
3

### Presentation
3

### Contribution
3
