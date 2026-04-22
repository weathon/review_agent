# Unlocking the Power of Multi-Agent LLM for Reasoning: From Lazy Agents to Deliberation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Large Language Models (LLMs) trained with reinforcement learning and verifiable rewards have achieved strong results on complex reasoning tasks. Recent work extends this paradigm to a multi-agent setting, where a meta-thinking agent proposes plans and monitors progress while a reasoning agent executes subtasks through sequential conversational turns. Despite promising performance, we identify a critical limitation: lazy agent behavior, in which one agent dominates while the other contributes little, undermining collaboration and collapsing the setup to an ineffective single agent.  In this paper, we first provide a theoretical analysis showing why lazy behavior naturally arises in multi-agent reasoning. We then introduce a stable and efficient method for measuring causal influence, helping mitigate this issue. Finally, as collaboration intensifies, the reasoning agent risks getting lost in multi-turn interactions and trapped by previous noisy responses. To counter this, we propose a verifiable reward mechanism that encourages deliberation by allowing the reasoning agent to discard noisy outputs, consolidate instructions, and restart its reasoning process when necessary. Extensive experiments demonstrate that our framework alleviates lazy agent behavior and unlocks the full potential of multi-agent framework for complex reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes DR MAMR, a series of enhancements for reinforcement learning algorithms focused on multi-turn multi-agent systems like those introduced in ReMA (Wan 2025). Here, the focus is on multi-agent systems that alternate between a meta-agent, whose purpose is to provide higher-level planning and form intermediate tasks, and a reasoner agent, whose purpose is to reason about and solve tasks proposed by the meta-agent. DR MAMR identifies a “lazy agent” shortcoming in ReMA, where a single agent does all of the planning and reasoning, while the other agent produces short (or no) outputs. The authors then propose fixes to this involving modifications to the loss function and methods for provider denser credit attribution via Shapely values. The authors train models at 3B, 7B, and 14B scale and evaluate on math tasks.

### Strengths
- Shapely values are an interesting, and to my knowledge, novel way to provide dense credit attribution in multi-turn settings with relatively small computational overhead (a 0.5B embedding model is required to compute semantic similarity). The ablation result shows that such credit attribution has a large impact on performance.
- The proposed changes and additions to the loss and reward formulation are principle and experimentally backed with controlled ablations.
- The paper is generally well-written and the problem of the lazy agent is well-motivated

### Weaknesses
- Relatively weak baseline choices: The second performing baseline across model scales is GRPO training. There exist many algorithmic improvements on top of GRPO that enable better performance, such as DAPO, DR GRPO. The advantage of these methods is that they do not require such complicated rollout and reward approaches needed by ReMA and DR MAMR. Even a comparison across one model size, like 7B, would be helpful.
- The authors focus only on math domains, eschewing other reasoning domains/benchmarks. As a starting point, I would love to see if DR MAMR models generalize to other domains, e.g., evaluating on GPQA-Diamond, MMLU-Pro, etc.
- Theoretical presentation: The proposed theorem is a trivial re-arrangement of the defined $g, Z$ terms. The full proof of this result reveals the simplicity, as it spends the majority of the space deriving the expression for $Z$ (by simply performing some algebraic manipulations involving gradient computations), then concluding with some additional algebraic manipulations. Overall, I find this result overstated: It does not introduce any novel result, nor does it use novel techniques in the proof. Presenting such a result as a standalone theorem needlessly complicates presentation.

### Questions
- Do authors compare Shapely-valued dense rewards with alternative methods, like using a process reward model? 
- Nit: GRPO stands for Group Relative Policy Optimization, not Preference Optimization (L46)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies why lazy-agent behavior emerges in multi‑agent LLM reasoning (one agent effectively does all the work while the other contributes little) and proposes a theoretical framework to address this issue by tracing the problem to a bias in multi‑turn GRPO: the turn‑length normalization favors shorter trajectories, implicitly discouraging collaborative back‑and‑forth. Then introduce Dr. MAMR with three components: (i) remove the turn‑length normalization identified by the theory; (ii) add a Shapley‑inspired step‑level causal‑influence signal computed by grouping semantically similar steps across rollouts(which is common is MARL setting); and (iii) add a verifiable reward for restart via a <restart> control that lets the reasoning agent discard noisy history and begin afresh. On seven math benchmarks with Qwen2.5 7B and 14B‑Instruct, Dr. MAMR outperforms single‑agent GRPO, VRP(CoT), and ReMA. And improves stability (ReMA collapses; Dr. MAMR does not).

### Strengths
-- Motivation: The paper motivates that lazy‑agent behavior collapses multi‑agent systems into single‑agent reasoning, squandering collaboration benefits, which is an issue observed in MARL and newly shown here in LLM multi‑agent reasoning. The motivation is explicit in the introduction and contribution summary.

Well structured: 

-- Crisp theoretical diagnosis of the cause. Theorem 1 in (Sec. 5.1, p. 5) formalizes how the 1/T normalization in multi‑turn GRPO biases the gradient toward shorter (fewer‑turn) continuations with equal final reward, explaining the emergence of lazy agents; the authors also argue the bias is pronounced because turns are far fewer than tokens. (Sec. 5.1, p. 5) 

-- Methodologically well‑scoped fix. The Shapley‑inspired causal‑influence estimator averages masked‑history log‑likelihood deltas over groups of semantically similar steps, reducing phrasing bias and variance. A “who helped whom” question. (Sec. 5.2, p. 6; Appx. B.3, p. 18)

### Weaknesses
-- Domain generality is narrow. All experiments are on math; claims about “complex reasoning tasks” would be stronger with code, planning, or QA tasks where verifiability is harder. Which can lead to another question:
-- Restart relies on verifiable end‑state. The restart reward needs a checkable final answer; outside math (or without ground‑truth/validators), the mechanism may not be directly applicable. Moreover, ablations show restart helps but less than CI/normalization fixes, suggesting narrower impact. (Sec. 5.3 p. 7; Table 2 p. 9)

-- Theory relies on an equal final reward assumption. Theorem 1 contrasts continuations with equal final reward but different lengths; this is a clean but restrictive setting. It is less clear how strongly the conclusion holds under more realistic reward noise, token‑level normalizations/clipping, or different credit schemes. (Sec. 5.1, p. 5) 

-- Sensitivity of the CI estimator is underexplored. Grouping uses Qwen2.5‑0.5B embeddings and a cosine‑similarity threshold of 0.9; \alpha = \beta = 0.1 are fixed across runs for stability. The paper lacks ablations on embedding choice/threshold and \alpha /\beta, so robustness of CI to these design choices is uncertain. (Appx. B.2–B.3, p. 18)

-- Connection to MARL: Comparative context to prior “lazy‑agent” literature could be tighter. The paper cites MARL work on lazy agents but does not empirically compare to mechanisms from that line (e.g., influence‑based credit or external‑state influence). This limits continuity with prior method. (Sec. 2, p. 3; cf. ICML’23 “Lazy Agents” & Lazy‑MDPs)

### Questions
Besides the weakness, I have something to ask:

-- Theory stress tests: Beyond removing 1/T, did you try alternatives (e.g., adaptive normalization by effective turns that exclude no‑ops, or per‑role normalization)? Do the theoretical predictions hold when rewards differ slightly (near‑equal) or when token‑level normalizations are active? (Sec. 5.1, p. 5)

-- The phenomenon of “lazy agents” has been studied in MARL (e.g., LAIES at ICML’23; Lazy‑MDPs). This paper’s contribution is to diagnose and mitigate it in multi‑agent LLM reasoning with a targeted loss fix, a tractable step‑level influence signal, and a restart‑credit mechanism. What's your next plan for further work to combine this work?

### Soundness
3

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
The paper studies lazy agent behavior in multi-agent LLM reasoning, where one agent does most of the work and the other contributes little. It attributes the collapse to a turn-averaging term in multi-turn GRPO that implicitly favors shorter dialogues. Their method Dr. MAMR removes the turn normalization, estimates step influence with a Shapley-style average over semantically similar steps across rollouts, and equips the reasoner with a restart token rewarded only when restarting improves the verifier-aligned likelihood of the final step. On math benchmarks with Qwen2.5 models at 3B, 7B, and 14B, the approach outperforms GRPO, VRP prompting, and ReMA, and shows better training stability and pass-at-K scaling. Ablations indicate that normalization removal and the influence estimator are key contributors.

### Strengths
- Clear diagnosis of the lazy agent failure mode with a simple insight into how turn normalization biases learning toward short dialogues.
- Method that is principled and modular with debiased objective, a practical influence estimator that reduces single-trajectory and phrasing bias, and a verifier-aligned restart mechanism.
- Consistent gains over strong baselines across multiple model sizes and math benchmarks, with improved stability and better pass-at-K behavior.
- Useful ablations and training dynamics analyses that support the design choices.

### Weaknesses
- Empirical scope restricted to math reasoning; claims about multia-gent benefits would be stronger with code and other domains requiring reasoning.
- Since the paper emphasizes single vs multi-agent comparison, it would be good to include compute vs performance for single agents and DR MAMR.

### Questions
A threshold of 0.9 was used for grouping, have you tried to vary this value and see how it affect the learning outcome of DR MAMR?

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
3

### Summary
This paper examines a multi-agent LLM reasoning framework, ReMA, and finds that it induces lazy agent behavior (where one agent contributes little to the final result). The paper examines why this lazy agent behavior occurs, and introduces a verfiable reward mechanism that allows the agent to discard previous outputs and restart reasoning. This setup improves multi-agent reasoning behavior over the original ReMA on standard math benchmarks, and shows improved performance compared to single-agent frameworks.

### Strengths
* The problem is clearly identified and the theoretical analysis gives some insight into why this occurs (though it is an interesting choice to call this Dr. MAMR if the paper goes out of its way to claim that this is distinct from Dr. GRPO).

* The introduced causaul influence mechanism is a nice way to measure the impact of the restart action in a computationally-efficient manner (though the effectiveness of the proposed semantic similarity is not really analyzed).

* The results show pretty clear improvements over the original ReMA.

### Weaknesses
* My biggest concern is that the lazy agent behavior (and corresponding proposed solution) is only applicable to one recent multi-agent framework, ReMA. Given that this is a recent (and to this point not very widely used) framework with relatively poor performance (it under-performs single-agent GRPO), it's unclear whether this problem/solution will have much impact. For example, there's nothing to indicate that other multi-agent LLM frameworks will induce similar behavior. The fact that no other multi-agent LLM frameworks are included as baselines do not give me much confidence here.

* Experimental results are only over one model family (Qwen 2.5) and one type of benchmark (math).

* There's no sensitivity analysis on the new additional hyperparameters, e.g. alpha/beta. There's no analysis on whether there's a risk of potentially entering modes where restarts are over-triggered, for example.

### Questions
1. Fig. 3 is also not clear. What is the "performance gap" that is being measured here? Is this ReMA+ val. acc - ReMA val. acc? Why does the gap increase between Pass@1 and Pass@16 for AIME25/AIME24?

2. I don't understand Fig. 2. Particularly the difference from (b) to (c) to (d). What is the "causal-effect gap" here? A higher causal influence is better, yes? But it doesn't seem like there is any significant difference in the reasoning agent's causal influence from (b) to (d), for example.

### Soundness
2

### Presentation
2

### Contribution
2
