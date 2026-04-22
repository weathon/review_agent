# Adaptive Social Learning via Mode Policy Optimization for Language Agents

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
Effective social intelligence simulation requires language agents to dynamically adjust reasoning depth, a capability notably absent in current studies. Existing methods either lack explicit reasoning or employ lengthy Chain-of-Thought reasoning uniformly across all scenarios, resulting in excessive token usage and inflexible social behaviors in tasks such as negotiation or collaboration. 
To address this, we propose an $\textbf{A}$daptive $\textbf{S}$ocial $\textbf{L}$earning ($\textbf{ASL}$) framework in this paper, aiming to improve the adaptive reasoning ability of language agents in dynamic social interactions. To this end, we first identify the hierarchical reasoning modes under such context, ranging from intuitive response to deep deliberation based on the cognitive control theory. We then develop the $\textbf{A}$daptive $\textbf{M}$ode $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{AMPO}$) algorithm to learn the context-aware mode adaptation and reasoning. Our framework advances existing research in three key aspects: (1) Multi-granular reasoning mode design, (2) Context-aware mode switching in rich social interaction, and (3) Token-efficient reasoning with depth adaptation. Extensive experiments on the benchmark social intelligence environment verify that ASL achieves 15.6\% higher task performance than GPT-4o. Notably, our AMPO outperforms GRPO by 7.0\% with 32.8\% shorter thinking chains, demonstrating the advantages of our AMPO and the learned adaptive reasoning ability over GRPO's solution.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes ASL, a framework for enabling language agents to dynamically adjust reasoning depth in social interactions. The key idea is to design hierarchical reasoning modes inspired by cognitive control theory, ranging from intuitive responses to deep deliberation, and to train agents via Behavioral Cloning and a novel AMPO algorithm. AMPO leverages both mode-level and sample-level advantages to adapt reasoning depth to context, improving performance while reducing token usage. Experiments on the SOTOPIA and SOTOPIA-Hard benchmarks show that ASL consistently outperforms strong baselines such as GPT-4o and GRPO, achieving higher goal completion and social reasoning efficiency.

### Strengths
1. Novelty: this paper introduces adaptive reasoning depth for social intelligence tasks, addressing a capability that has been largely overlooked in prior work.

2. Methodological rigor: the work clearly integrates cognitive theory, specifically Hierarchical Cognitive Control, into the design of reasoning modes, and couples it with a principled RL-based optimization approach.

3. Strong empirical results: experimental results demonstrate substantial improvements over strong baselines in both performance metrics, such as goal completion and strategic reasoning, and efficiency metrics, such as token usage reduction.

4. Comprehensive evaluation: the authors conduct ablation studies, mode analysis, human evaluation, and case studies, all of which support the claims regarding effectiveness and adaptive behavior.

5. Practical Relevance: by adapting reasoning depth, the approach reduces computational costs and addresses a key limitation of Long Chain-of-Thought methods in social reasoning tasks.

### Weaknesses
1. Novelty: while the idea of switching between long and short chains of thought is not entirely new, the authors present it in a well-structured and coherent manner, supported by thorough experimental validation. Therefore, I do not view novelty as a major concern in this work.

2. Generalization: the paper primarily evaluates ASL on the SOTOPIA benchmarks. Extending the experiments to other social reasoning datasets or multi-agent interaction tasks would further demonstrate the generality and robustness of the proposed approach.

### Questions
See those in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose an Adaptive Social Learning (ASL) framework to improve the adaptive reasoning ability of language agents in dynamic social interactions. They start by designing specialized reasoning modes that structure the social agent’s cognitive processes; then develop the Adaptive Mode Policy Optimization (AMPO) algorithm to learn the context-aware mode adaptation and corresponding reasoning patterns. The authors perform experiments on benchmarks of social intelligence environment and verify the novelty of ASL over existing models like GPT-4o and existing training algorithms like GRPO.

### Strengths
1. The work is of good technical quality. The 3-step ASL framework (mode design, BC, AMPO) is logical and well-executed. The reasoning modes are grounded in cognitive science (HCCT). The design of AMPO is intuitive and well-justified. The experimental validation is exceptionally thorough.
2. The paper is well-written. The problem statement is clear and the proposed solution is easy to follow.
3. The novelty of the paper's method is significant. It demonstrates significant improvement on model social intelligence and also addresses the critical bottleneck of reasoning cost (token consumption) in LLM agents.

### Weaknesses
1. Strong dependence on hand-crafted reasoning modes.
2. Scalability of AMPO with more modes: AMPO is demonstrated with $N=4$ modes. However, the calculation of the mode-level advantage $A^{\mathcal{M}}$ requires gathering sufficient samples for **each mode in each output group** to compute stable statistics (average reward and length). As the number of modes $N$ increases, this approach could suffer from sample inefficiency, making it difficult to scale.
3. Minor advice on notation: the mode notation of reward is a bit confusion. It's a superscript on singular rewards ($r_i^{m(i)}$), but a subscript on average rewards ($\bar r_{m_k}$).  It is more user-friendly to unify the notation.

### Questions
1. Could the authors elaborate on the selection process for the specific cognitive "actions" (e.g., Assess, Deduction)?
2. The framework includes two different length penalties: an "answer length reward" ($r_i^l$) in the base reward $r_i$, and an "average token length" ($\overline{l}_{\mathcal{M}_{k}}$) in the mode-level advantage $A^{\mathcal{M}}$. Why are both necessary?
3. Given the "single-turn training paradigm", how does the agent learn the _long-term_ consequences of a mode choice?

### Soundness
3

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
4

### Summary
This paper introduces ASL with four reasoning modes for LLM agents and an agent training algorithm AMPO with both mode-level advantage and sample-level advantage. Experiments showcase the performance improvement and token savings.

### Strengths
The paper’s main novelty is explicit, discrete reasoning modes for social interaction, plus dual-level advantages so the policy learns when to use shallow vs. deep reasoning. The proposed AMPO is an insightful combination of psychological knowledge and computational agent training. The algorithm and experiments are clearly presented in the paper. Abundant experiments showcase the improvement of the efficiency of the agent's reasoning process. This work is practically meaningful.

### Weaknesses
1. The human evaluators only provide the winner of the two outputs without participating in the reward design process, which still may make the results suffer from design/bias variance and proprietary drift. 
2. The large format penalty and length penalty may bias the policy toward short/over-structured outputs, potentially suppressing socially nuanced behaviors.
3. The efficiency-motivated single-turn RL reduction could miss multi-turn dependencies typical in social dialogue.

### Questions
1. How sensitive are gains to the length-reward coefficient and target length?
2. Beyond GPT-4o/Qwen-72B, how do results change with a different LLM judge (e.g., Claude, Llama-3.1-70B-Instruct)?
3. Can the policy discover additional modes or interpolate between modes?

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
3

### Summary
This paper introduces Adaptive Social Learning (ASL), a framework that equips language agents with the ability to adaptively choose a reasoning mode (from a small set of structured modes) during open-ended social interaction. The paper provides a clear technical path and supplies significant empirical evidence that the proposed approach improves goal achievement while reducing unnecessary long chains of thought. The paper reports improved GOAL metrics and token-efficiency over baselines, and includes ablations, OOD tests, and human evaluation.

### Strengths
1. Novel conceptual framing. Mapping hierarchical cognitive control into a small set of explicit, controllable reasoning modes and operationalizing them via control tokens is an original and compelling idea for resource-aware, controllable LLM behavior in social settings.
2. Practical algorithmic contribution (AMPO). The combined mode-level and sample-level advantage estimation in AMPO is a thoughtful improvement over mode-agnostic RL approaches; the technique is intuitive and appears effective in practice.
3. Comprehensive empirical work. The experiments are broad: multiple backbones, several baselines, ablations (reward components, BC vs RL, etc.), OOD evaluations, and human judgments. Reported results consistently indicate improvements in both objective goal metrics and token efficiency.

### Weaknesses
(A) Reproducibility gaps (major). Key elements required for exact reproduction are not fully specified in the paper: 1. The exact BC prompt templates and a set of representative prompt-->response examples for each mode are not included. 2. The full evaluator/judge prompt used to compute reward (and any LLM scoring parameters such as temperature, max_tokens, whether CoT is used) is not fully published. 3. Precise hyperparameters for BC and AMPO (learning rates, batch sizes, number of per-state rollouts G, clip/epsilon values are not presented in a runnable config form. 4. Implementation details for the format compliance detector (how control tokens are parsed, what counts as a format violation, and how penalties are applied) are omitted.
(B) Theory / analysis depth. The method is primarily empirical; there is limited theoretical analysis of AMPO's properties (e.g., variance/bias behavior of the combined advantage estimator, or conditions under which mode-level info benefits policy improvement). A brief formal or intuitive analysis would strengthen confidence in the approach.
(C) Single-turn decomposition vs long-horizon consistency. AMPO optimizes in a single-turn decomposition (i.e., decomposing multi-turn interaction into per-turn RL updates). The paper does not sufficiently analyze whether or how this decomposition affects long-horizon strategic consistency. This is a plausible limitation that requires either analysis or empirical comparison.

### Questions
1. Can more details be provided? Like seeds for runs, confidence intervals or error bars, the exact statistical tests used, and sensitivity analyses for critical hyperparameters.
2. Most experiments are on SOTOPIA (and its Hard variant) plus demo OOD tests. How is the proposed method performing on additional domains (or at least provide more clearly discussion about task generality)?

### Soundness
3

### Presentation
3

### Contribution
3
