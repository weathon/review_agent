# Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards

- Decision: Accept (Poster)
- Scores: 4, 6, 8

## Abstract
Compound AI systems integrating multiple components, such as Large Language Models, specialized tools, and traditional machine learning models, are increasingly deployed to solve complex real-world tasks. However, optimizing compound systems remains challenging due to their non-differentiable structures and diverse configuration types across components, including prompts, hyperparameters, and model parameters. To address this challenge, we propose Optimas, a unified framework for effective optimization of compound systems. The core idea of Optimas is to maintain one Local Reward Function (LRF) per component, each satisfying a local–global alignment property, i.e., each component’s local reward correlates with the global system performance. In each iteration, Optimas efficiently adapts the LRFs to maintain this property while simultaneously maximizing each component’s local reward. This approach enables independent updates of heterogeneous configurations using the designated optimization method, while ensuring that local improvements consistently lead to performance gains. We present extensive evaluations across five real-world compound systems to demonstrate that Optimas outperforms strong baselines by an average improvement of 11.92%, offering a general and effective approach for improving compound systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a method to optimize an agentic system comprising of multiple LLMs and other components. To do so, it first elicits preferences data for each local component and artifically get the preference label that "aligns" with the global system output, in the sense that if response A gets better system reward, then response A is preferred over B locally.

This preference elicitation process is done many times, and each LLM in the system is progressively adjusted/fine-tuned so the system performance improves.

### Strengths
1. The presentation of the paper is good. I could quickly understand what their method is about.
2. Their method makes sense from an intuitive point of view - since we cannot do end-to-end learning of the system (due to perhaps, non-differentiable components), we can slowly adjust each component so that it improves the overall system (by looking at how each local step affects the system output. Despite so, there are some flaws in this approach (see weaknesses).
3. Experimental results are comprehensive (I do have some questions regarding some of the set up and validity).

### Weaknesses
1. The method is intuitive. However, it has several flaws. The biggest flaw is that it seems to be performing some form of local parameter updates with respect to a non-differentiable reward signal. As far as I know, this has the same flavour as popular gradient-free training such as REINFORCE. Why didn't the authors mention such approaches or use it as the baselines (I'm quite sure we can use it as a baselines).
2. In addition, what makes this method slightly worse than prior approaches such as REINFORCE is that it optimizes each component in a round-robin/random fashion. This means the approach cannot be globally optimal because we are adjusting one component/subset of parameters at a time right? It is misleading to claim that their approach guarantees optimality in this case (we need some extra convex assumption of the optimization landscape).
3. In addition, I feel some of the experiment set up is contrived. For instance, how would a single strong model perform in answering the task directly, instead of creating a complicated system to answer it? This seems like a simple baseline that should be discussed. In addition, it seems like the local rewards have a positive correlation with the global rewards - how does simply improving the local models directly (locally) or using stronger local models fare as a baselines?
4. Lastly, could the authors clarify what they meant that we are reducing the number of system calls? Everytime you try to elicit a preference of the local component, we need to do a forward pass of the system to check the global rewards, is this correct? So we definitely need to accumulate many system calls eventually right?

### Questions
See my weakness section above.

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
The paper proposes OPTIMAS, a framework to optimize compound AI systems by learning component-wise Local Reward Functions (LRFs) that are trained to be locally–globally aligned with the task’s global reward.

### Strengths
1. The paper clearly motivates why compound AI systems are hard (non-differentiable, heterogeneous knobs) and frames a practical local–global alignment objective.

2. The paper unifies optimization across prompts, hyperparameters, model selection, and model parameters within one iterative loop.

3. The paper offers interpretability hooks—probing LRF preferences (e.g., brevity bias matching F1) to explain why updates help.

### Weaknesses
While OPTIMAS reduces the number of full system runs, it introduces non-trivial computational overhead from (i) training and adapting the shared 8B-backbone Local Reward Functions (via LoRA) and (ii) generating preference labels that require downstream sampling to estimate expected global rewards. In settings that enable parameter fine-tuning (e.g., PPO for some modules), total token and wall-clock costs may exceed prompt-only baselines. The paper would be strengthened by a cost breakdown (tokens, time, and dollars) that separates LRF training/adaptation, downstream sampling per preference pair, and any PPO steps, and by a cost-normalized comparison to DSPy/TextGrad.

### Questions
1. How often does local improvement (per LRF) fail to translate to global improvement in practice, and what diagnostics or safeguards detect misalignment early?

2. How sensitive is performance to the LRF backbone size and adaptation frequency? Could you show accuracy vs. backbone (1B/3B/8B) and vs. adaptation interval?

3. Are there failure cases where local–global alignment deteriorates over long runs? If so, what mitigation (e.g., trust regions, re-labeling frequency) is most effective?

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
3

### Summary
The paper proposes an effective and efficient framework, **OPTIMAS**, for optimizing compound AI systems with heterogeneous configurations (e.g., prompts, model parameters, and hyperparameters) as a whole. The key idea is to maintain a **Local Reward Function (LRF)** for each component. Before optimizing a component, the corresponding LRF is updated using preference data so that answers preferred by the global reward are also preferred locally (achieving **local–global alignment**). The component is then optimized to maximize its LRF reward.

Theoretically, the authors show that under certain regularity conditions, training an LRF to prefer outputs associated with higher global reward aligns it with the global reward, and optimizing a component with the LRF is equivalent to optimizing it using the global reward. They further show that, under additional assumptions, the optimization converges to a component-wise maximum.

Empirically, the method achieves consistent and substantial improvements across five real-world compound AI systems over five baselines (4 other methods and 1 unoptimized baseline). The authors also validate the mechanism itself: local optimization improves global reward, the trained LRFs exhibit stronger alignment with global rewards than a simple LLM judge, and higher alignments lead to higher global rewards. Additionally, they analyze interpretability and efficiency of the method.

### Strengths
1. The presentation is clear and well-organized.
2. The addressed problem—optimizing a compound AI system that could have heterogeneous configurations as a whole rather than its parts independently—is timely and practically important.
3. Theoretical analysis under some assumptions provides partial guarantees for the effectiveness of the method.
4. Empirical evaluation is comprehensive and convincingly supports the method’s effectiveness and mechanism.

### Weaknesses
There appear to be some problems in Theorem 4.1 and Lemma B.1. Specifically, Equation 4 defines a loss with a leading minus sign, so Theorem 4.1 should refer to the **minimizer** of Equation. 4, not the **maximizer**. Similarly, in Lemma B.1 the optimization should be formulated as an **argmax** of the expected log-likelihood. The proof shall remain conceptually sound after correcting these signs.

Minor typo issues:
- Line 94: “optimized” → “optimize”
- Line 119: add comma before “While”
- Line 249: “Eq. equation 3”
- Line 256: “r_j” → “r_i”

Otherwise, the paper is technically solid.

### Questions
In Theorem 4.1, should it be the 'minimizer' of equation 4  instead of the 'maximizer'. And in Lemma B.1 should it be formed as an argmax problem instead of argmin?

### Soundness
4

### Presentation
4

### Contribution
4
