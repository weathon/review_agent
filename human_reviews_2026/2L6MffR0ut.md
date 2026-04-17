# Multi-agent Coordination via Flow Matching

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
This work presents MAC-Flow, a simple yet expressive framework for multi-agent coordination. We argue that requirements of effective coordination are twofold: *(i)* a rich representation of the diverse joint behaviors present in offline data and *(ii)* the ability to act efficiently in real time. However, prior approaches often sacrifice one for the other, *i.e.*, denoising diffusion-based solutions capture complex coordination but are computationally slow, while Gaussian policy-based solutions are fast but brittle in handling multi-agent interaction. MAC-Flow addresses this trade-off by first learning a flow-based representation of joint behaviors, and then distilling it into decentralized one-step policies that preserve coordination while enabling fast execution. Across four different benchmarks, including $12$ environments and $34$ datasets, MAC-Flow alleviates the trade-off between performance and computational cost, specifically achieving about $\boldsymbol{\times14.5}$ faster inference compared to diffusion-based MARL methods, while maintaining good performance. At the same time, its inference speed is similar to that of prior Gaussian policy-based offline MARL methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes MAC-Flow, an offline MARL framework that (1) learns an expressive flow-matching–based joint policy via behavioral cloning, then (2) factorizes / distills it into decentralized one-step policies optimized with a behavioral-regularized actor–critic objective and the Individual-Global-Max (IGM) principle. The pitch is to retain diffusion-like expressiveness for multi-modal joint action distributions while achieving Gaussian-policy–like inference speed. Experiments span different dataset, reporting faster inference vs diffusion MARL at comparable performance, and similar speed to Gaussian baselines

### Strengths
1. The paper clearly identifies—and structures the solution around—the core trade-off in offline MARL: diffusion policies capture multi-modal joint behaviors but are slow, while Gaussian one-step policies are fast but brittle for coordination.

2. The paper connects to flow-matching literature and positions MAC-Flow as a MARL counterpart to single-agent flow-distillation and Flow Q-Learning, showing conceptual continuity with recent advances.

3. The two-stage design is guided by the Individual-Global-Max principle, aligning per-agent policies with the global optimum and providing a standard, scalable route to decentralized execution

3. On four benchmarks, MAC-Flow demonstrates that you can retain coordination performance while achieving faster inference than diffusion-based MARL—matching Gaussian-policy speed without its expressiveness limits.

### Weaknesses
1. The method references “mathematical guarantees” around joint-to-factorized policy learning with IGM, but the paper (as given) does not present a formal theorem / conditions under which the distilled one-step policies provably preserve the global optimum of the learned joint flow. 

2. While the authors claim flow matching combines diffusion’s expressiveness and Gaussian’s speed, the paper lacks a deeper justification for why flow matching provides better inductive bias or coordination representation in MARL. There is no theoretical or empirical comparison against diffusion-based MARL such as Graph Diffusion for Robust Multi-Agent Coordination (2025).

3. The evaluation omits stress tests under partial observability, agent scaling, and dynamic teammate shifts—scenarios where flow-based policies might degrade.

### Questions
1. Under what assumptions does the factorized one-step policy set preserve the argmax consistency with the flow-learned joint policy? Any error decomposition (e.g., from flow fitting + distillation + value learning) that bounds sub-optimality? 

2. Can you report MI(agent i; agent j | o) or diversity / mode-coverage metrics comparing joint flow vs distilled policies, and ablate IGM on/off?

3. How does the policy behave under partial-observability stress (masking % of features) or opponent distribution shift?

4. Provide cases where MAC-Flow under-performs Gaussian or diffusion policies—what patterns are not captured by the distilled factorization?

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
2

### Summary
This paper tackles a long standing tension in offline multi agent RL between expressive centralized training and fast decentralized execution. The authors first learn a centralized joint policy over all agents’ actions via flow matching, then distill it into one step per agent policies suitable for test time. Empirically, MAC-Flow performs competitively on standard cooperative benchmarks while delivering large gains in inference speed relative to diffusion style generators, and it stays comparable to simpler Gaussian MARL baselines in wall clock.

### Strengths
1. Learning a rich joint policy with flow matching, then distilling to per agent policies, directly addresses the coordination vs speed trade off that many of us have run into in offline MARL. I think the training to deployment narrative is easy to follow and feels usable.
2. The comparisons include both diffusion based policies and conventional MARL baselines, and the results highlight a strong reduction in inference latency while retaining returns. Given how often test time latency matters in multi agent control, this is an impactful result.
3. The writing is clear, the algorithmic pipeline is spelled out, and there are helpful ablations that point to which pieces do the heavy lifting.

### Weaknesses
1. The reliance on an individual global max style factorization is an obvious pressure point. The paper would be stronger with stress tests on heavily coupled tasks where separability breaks down. Bounds are good, but concrete counterexamples or failure modes would build trust.
2. The link from small $W_2$ to small value loss hinges on a Lipschitz $Q_{tot}$ and on distributional proximity that may not hold uniformly. It would help to see empirical measurements of these quantities during training to show that the assumptions are not only theoretically convenient but also reasonably satisfied.
3. Most of the complexity discussion focuses on test time. Training a flow over joint actions can be expensive, and the main text gives less visibility into training wall clock, memory, and scaling with agent count and action dimension. I want to know the total cost to get the speedups.
4. It is not clear how the approach behaves under partial observability, larger discrete alphabets, or mixed cooperative competitive games. Even a brief study in one of these regimes would help position the method more broadly.

### Questions
1. Could you include tasks where interaction strength is dialed up and report both $W_2\big(\pi^{\text{joint}},\prod_i \pi_i\big)$ and the corresponding return gaps?
2. Can you provide training cost curves and memory footprints as functions of agent count and action dimension, along with sensitivity to the number of flow integration steps during training. A head to head with an autoregressive joint policy would be especially informative.
3. How does MAC-Flow adapt when observation is local rather than global? If policies condition on $o_i$ at test time while the joint flow used $s$ and a central critic at training time, do your guarantees change, and what additional assumptions would be needed for the value gap bounds to carry over?

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
This paper introduces MAC-Flow, a framework for multi-agent coordination that learns a flow-based joint policy from offline data and distills it into decentralized one-step policies to balance performance and inference speed. It achieves ~14.5× faster inference than diffusion-based MARL methods across 4 benchmarks while maintaining good performance, with ablation studies confirming the value of its two-stage strategy.

### Strengths
1. MAC-Flow effectively balances the trade-off between multi-agent coordination performance and inference speed.
2. The resulting individual policy supports seamless offline-to-online fine-tuning.
3. The ablation studies show the components of MAC-Flow are effective.

### Weaknesses
1. There's still a small performance gap compared to diffusion policies (DoF), especially on SMACv2, showing room for improvement in handling highly stochastic multi-agent environments.
2. It requires offline datasets with diverse joint behaviors for effective training, and its performance may degrade when using low-quality or limited offline data.
3. The baselines for continuous control are weaker due to the absence of diffusion and flow-based policies.

### Questions
1. Can you provide some explanations on why MAC-Flow generally underperforms compared to DoF on SMACv2? Does it mean that individual policies are unreliable when facing complex and stochastic environments?
2. How is the training cost of MAC-Flow compared to other baselines?

### Soundness
3

### Presentation
3

### Contribution
3
