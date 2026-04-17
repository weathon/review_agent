# Competitive Multi-Agent Delegation for  LLM Reasoning

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Large Language Models (LLMs) have shown impressive capabilities in natural language generation, yet they remain limited in complex and multi-step reasoning. We propose COMMAND: COMpetitive Multi-AgeNt Delegation, a framework where a principal LLM assigns tasks to multiple agent LLMs. Agents compete in an environment where utilities depend on both their internal confidence and the principal’s evaluation, incentivizing answers that are higher-quality and better aligned with the principal. We establish theoretical guarantees demonstrating that, under fair comparison, multi-agent systems such as COMMAND provably outperform their single-agent counterparts. Moreover, each agent, via online learning, achieves sublinear regret and its average policy will converge to a Nash equilibrium. Empirical evaluations on multiple benchmarks demonstrate that COMMAND yields significant improvements in factual accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes COMMAND, a game-theoretic framework that enhances LLM reasoning through competition among multiple agents evaluated by a principal model. Each agent independently generates candidate answers and receives rewards based on both its internal confidence and the principal’s ranking feedback. Theoretically, the authors prove that multi-agent delegation yields higher expected utility than single-agent setups under fair comparison. Empirically, COMMAND improves accuracy across math benchmarks.

### Strengths
- The paper addresses an interesting and relevant problem: how to enhance the reasoning performance of LLMs through competitive multi-agent delegation under a principal–agent framework. The idea of coordinating multiple policies via a principal for self-improvement without explicit fine-tuning is conceptually appealing.
- The framework is supported by theoretical analysis, providing regret bounds and convergence guarantees under online mirror descent, which adds mathematical grounding to the proposed approach.
- The work connects multi-agent learning and game theory with LLM reasoning, an angle that is potentially useful for understanding cooperative-competitive dynamics in large models.

### Weaknesses
- Many implementation details are missing or unclear, making it difficult to fully understand or reproduce the method. For example:
  - It is not clearly explained how the principal aggregates responses or computes the global utility mentioned in Line 111.
  - The MCTS process is under-specified: what constitutes a node, how rollouts are defined, and how branching decisions are made remain ambiguous.
  - The paper does not specify the number of iterations in the competitive loop or the stopping criterion for convergence.
- In Table 1, it is unclear which base models are used for the reported baselines. Are they the same as the principal’s model, or different ones? Moreover, since the policies in COMMAND may include stronger models than the principal, the paper should include a baseline reflecting the best single policy’s performance for fair comparison.
- Table 2 and Figure 2 seem to indicate that COMMAND does not outperform the strongest policy in the ensemble and may even slightly underperform it, suggesting that the competitive interaction does not yield consistent gains. I'm not sure whether I misinterpret the results. Correct me if I'm wrong.
- The experimental scope is limited, focusing solely on mathematical reasoning tasks. Broader evaluation on other domains would strengthen the generality of the claims.
- The practical contribution is somewhat limited. Although the theory is sound, the improvement margins are small and the framework lacks clear insights into why competition helps or when it may hurt. Also, the relationship between the consumed computation and the performance of COMMAND and other baselines is not reported. It seems that COMMAND may require significantly more compute than other baselines, while yielding only marginal improvement.
- No ablation or sensitivity analysis is provided to isolate the effects of key components (e.g., number of agents, principal choice). Without this, the contribution feels more conceptual than empirically validated.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes COMMAND, a training-free framework that uses game-theoretic principles to improve LLM reasoning through competitive multi-agent delegation. In this setup, a principal LLM ranks answers submitted by multiple agent LLMs. Each agent's utility function combines two components: its internal confidence (measured by self-consistency across its own samples) and the principal's ranking feedback. Agent policies are then updated using mirror descent with the Hedge algorithm.
The authors make three main theoretical claims: (i) multi-agent delegation provably outperforms single-agent approaches under fair candidate budgeting, (ii) each agent achieves sublinear regret, and (iii) the time-averaged policies converge to an approximate Nash equilibrium. Empirically, COMMAND shows accuracy improvements on MATH, GSM8K, and GSM-Hard compared to few-shot CoT, a simplified r* (rStar) variant, and a "Principal-alone" baseline (Sections 1–3, Section 4.2 Table 1, Figure 2).
However, I notice this work appears quite similar to https://arxiv.org/abs/2506.08292—I'd appreciate the authors clarifying the relationship.

### Strengths
Motivated game formulation. The "delegation game" design is elegant: agents optimize a utility that combines their own self-consistency signal with the principal's ranking feedback. This explicitly aligns each agent's search process with the principal's evaluation criteria. The ranking-based reward structure (top=+1, bottom=−1) and the mirror-descent policy updates using exponential weights provide a concrete, implementable mechanism that requires no fine-tuning (Sections 2.2–2.3, Algorithm 1).

Theoretical foundations built. The analysis builds on well-established assumptions (Pareto-optimal play, agent symmetry, non-negative alignment) and standard online learning theory. Theorem 1 formalizes why delegation outperforms single-agent approaches under equal candidate budgets. Theorem 2 proves O(√T) regret with learning rate η=1/√T. Theorem 3 establishes that time-averaged policies converge to a ξT(δ)-approximate Nash equilibrium. Together, these results provide COMMAND with a principled theoretical backbone (Sections 3.1–3.3).

Reasonable experimental setup. The evaluation uses heterogeneous 7–8B parameter agents (Mistral-8B-Instruct, Zephyr-7B-Beta, Phi-3-Mini-Instruct, Falcon-7B-Instruct) with LLaMA-2-7B as the principal. Candidate generation employs MCTS with 16 rollouts at depth 5. The benchmarks span MATH, GSM8K, and GSM-Hard with clearly reported sample counts of 300/300/320 respectively (Section 4.1).

### Weaknesses
Significant overlap with concurrent work—this is my primary concern. The closest contemporary work is ECON (From Debate to Equilibrium, arXiv:2506.08292), which is not cited in the paper. ECON also formulates multi-LLM coordination as a game and seeks a (Bayesian) Nash equilibrium with regret guarantees. While ECON uses a hierarchical RL procedure rather than training-free mirror descent, it reports 11.2% mean gains across six reasoning and planning benchmarks (ICML 2025). Given this substantial overlap in problem formulation and approach, I have concerns about the novelty of the contribution, which has influenced my score.

Theory-practice gap in symmetry assumptions. Assumption 1-ii requires symmetric agents sampling from "the same distribution D" (Section 3.1, page 4). However, the experimental agents come from different model families (Mistral, Zephyr, Phi-3, Falcon) with inherently different sampling distributions. While the paper argues these models have "comparable capacity" and use "identical sampling procedures," this doesn't satisfy the formal symmetry requirement. This gap weakens the applicability of Theorem 1's theoretical comparison to the experimental results (Section 4.1, page 6).

Missing critical baseline for Theorem 1. Theorem 1's central claim compares single-agent versus multi-agent performance under equal total candidate budgets. However, the empirical "Principal" baseline doesn't appear to use the same total number of candidates as the multi-agent system (where each agent runs 16 MCTS rollouts). The paper doesn't report a "single-agent with the same total candidate pool" ablation, so the core theoretical prediction isn't directly validated experimentally (Sections 3.1, 4.1–4.2).

Limited evaluation scope. All tasks are math-centric; there's no evaluation on code generation, planning, or open-ended QA where verification is more challenging. The evaluation uses relatively small subsets (300/300/320 examples) without reporting confidence intervals or significance tests. Additionally, rStar is implemented as a simplified verifier-only variant, which may not represent a strong baseline (Sections 4.1–4.2).

### Questions
Direct test of Theorem 1. Could you add a single-agent baseline that receives the same total candidate budget as the multi-agent system? For example, one agent could select from the union of all candidates produced by the multi-agent pool. This would directly test Theorem 1's theoretical setup (Sections 3.1, 4.2).

Connection to Bayesian equilibrium. Is there an interpretation of your mirror-descent updates as seeking a (Bayesian) equilibrium under uncertainty? This might help clarify the relationship to ECON.

Robustness to violated symmetry. What happens when Assumption 1-ii (symmetry) is violated, as it is with your heterogeneous agents? Do you have any theoretical extensions or empirical ablations studying scenarios where agent utilities come from different distributions? (Sections 3.1, 4.1)

Principal model sensitivity. How sensitive are the results to the choice of principal? If you swap to a different model family (e.g., Mistral or Llama-3) or use a verifier-based reward, do both the absolute accuracy and relative gains change significantly? (Section 4.1)
Generalization beyond math. Can COMMAND handle tasks without easily verifiable solutions, such as planning, code synthesis, or open-ended QA? Do you have any preliminary results beyond mathematical reasoning? (Sections 4.1–4.2)

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces COMMAND, a game-theoretic framework for improving LLM reasoning through competitive multi-agent delegation. In this framework, a principal LLM assigns reasoning tasks to multiple agent LLMs that generate candidate answers and compete for rewards. Each agent's utility combines its internal confidence with the principal's ranking-based evaluation, incentivizing both high-quality outputs and alignment with the principal's preferences. Empirical evaluations on GSM8K, MATH, and GSM-Hard demonstrate modest accuracy improvements over baselines.

### Strengths
1. The framework is training-free, requiring no fine-tuning or parameter updates, and uses only inference-time computation. 

2. The paper provides three theorems with complete proofs establishing that multi-agent systems can outperform single-agent counterparts.

3. Experiments show  gains in mathematical reasoning compared to single-agent baselines.

### Weaknesses
1. The paper claims "under fair comparison, multi-agent systems outperform their single-agent counterparts". However, recent work "Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?" has shown that Majority Voting accounts for most performance gains in multi-agent systems, and proved theoretically that debate alone does not improve expected correctness. Therefore, I have two concerns: 
- How do the authors clarify the contradiction between their theory and recent theoretical results? 
- Is the competitive delegation mechanism adding value beyond simple aggregation? Without comparisons to majority voting baselines, it is unclear whether the gains stem from the game-theoretic mechanism or simply from having more independent samples to aggregate. 

2. The experiments use relatively old and weak models, such as LLaMA-2-7B. These models have limited reasoning capabilities, making the experiments less convincing. The paper should validate the approach on stronger, more recent models such as Qwen3.

3. All experiments focus exclusively on mathematical reasoning tasks. To substantiate claims about general multi-agent LLM reasoning, the paper should evaluate on diverse domains and standard benchmarks such as MMLU, HumanEval, and HellaSwag. The current narrow evaluation severely limits the generalizability of the findings.

4. The paper uses Monte Carlo Tree Search, but its focus is on the advantages of a multi-agent system over a single agent. Therefore, comparing MCTS + multi-agent with a single agent is unfair and requires ablation experiments. However, I did not see any relevant ablation experiments in the paper, making it difficult to believe that the performance improvement comes from multi-agent.

5. The paper's presentation makes it easy for readers to get lost. And the paper missed some critical details, such as baseline implementations (Which LLM does each baseline use?). These details are essential for reproducibility and fair comparison.

### Questions
The paper does not clearly specify which LLM is used for baselines e.g., Few-shot CoT. Are they using LLaMA-2-7B-Instruct? This information is critical for fair comparison.

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
This paper proposes COMMAND, a training-free method for multi-agent LLM reasoning based on competitive delegation. Multiple agent LLMs generate candidate answers and compete for rewards determined by a principal LLM's ranking, combined with their internal confidence. The method employs game theory with agents updating policies via online mirror descent to reach the Nash equilibrium. Experiments on GSM8K, MATH, and GSM-Hard show performance improvements over baselines, though gains are modest (2-9%). The paper provides theoretical guarantees for convergence and regret bounds.

### Strengths
This paper provides theoretical guarantees showing that the multi-agent framework of COMMAND improves over its single-agent counterpart.

Unlike RL-based approaches or fine-tuning methods, COMMAND works purely at inference time, making it practical for immediate deployment without requiring additional training resources.

Tables 3-4 validate key aspects of Assumption 1, with ~90% Pareto-optimal play compliance and positive correlation between principal and agent utilities (0.15-0.50).

### Weaknesses
The paper's evaluation setting is quite odd. The evaluation only used 300 questions from the GSM8K and MATH datasets and 320 questions from the GSM-Hard dataset. In fact, the complete MATH500 test set only has 500 questions, which wouldn't introduce significant computational overhead. Furthermore, the reasoning chains in GSM8K are not very long, and the reviewer considered the computational overhead to be completely acceptable. Conducting experiments on the complete test set will be more convincing.

The paper compares against only three baselines (Few-shot CoT, rStar, Principal), missing several essential comparisons: multi-agent debate, self-consistency, etc.

The paper should include an ablation where all agents use the same LLM (e.g., all agents are LLaMA-2-7B-Instruct). Without this, it is impossible to determine whether performance gains come from the multi-agent mechanism or simply from having one stronger model (e.g., Mistral) in the agent pool.

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
3
