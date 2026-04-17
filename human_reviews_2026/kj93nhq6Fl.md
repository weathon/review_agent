# PICACO: Pluralistic In-Context Value Alignment via Total Correlation Optimization

- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
In-Context Learning has shown great potential for aligning Large Language Models (LLMs) with human values, helping reduce harmful outputs and accommodate diverse preferences without costly post-training, known as In-Context Alignment (ICA). However, LLMs' comprehension of input prompts remains agnostic, limiting ICA's ability to address value tensions—human values are inherently pluralistic, often imposing conflicting demands, e.g., stimulation vs. tradition. Current ICA methods therefore face the Instruction Bottleneck challenge, where LLMs struggle to reconcile multiple intended values within a single prompt, leading to incomplete or biased alignment.
To address this, we propose PICACO, a novel pluralistic ICA method. Without fine-tuning, PICACO optimizes a meta-instruction that incorporates multiple values to better elicit LLMs' understanding of them and improve alignment. This is achieved by maximizing the total correlation between specified values and LLM responses, which theoretically reinforces value conformity and reduces distractive noise, resulting in more effective instructions. Extensive experiments on five value sets show that PICACO works well with both black-box and open-source LLMs, outperforms several recent strong baselines, and achieves a better balance across up to 8 distinct values.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a novel in-context learning method named PICACO, which aims to address the challenge faced by LLMs in adhering to multiple, even conflicting, human value systems. PICACO optimizes a meta-instruction to maximize the total correlation between the model’s responses and a target set of human values, thereby achieving more balanced and effective multi-value alignment without fine-tuning the underlying model parameters. Experimental results demonstrate that PICACO consistently outperforms existing ICA baselines across diverse value compositions and different LLM architectures, achieving superior steerability, robustness, and value conformity.

### Strengths
1. The definition of the multi-value alignment problem is novel and of significant importance.
2. The introduction of information theory concepts into the in-context alignment task is not only intuitively reasonable but also provides a solid mathematical foundation for the optimization objective, surpassing simple heuristic methods.
3. The proposed method is effective for both black-box models and open-source models.

### Weaknesses
1. Although the proposed method demonstrates some empirical ability to mitigate value conflicts, it lacks a theoretical framework for modeling the intrinsic nature of such conflicts. This limitation constrains its reliability and interpretability in extreme or complex ethical scenarios.

2. PICACO depends heavily on an external model (e.g., *GPT-4o-mini*) as the value evaluator $q_{\omega}$, which may introduce evaluation bias.

### Questions
1.What are the computational and sampling costs of PICACO?

2.How can the PICACO framework be extended to accommodate non-static or dynamic value systems? For example, in interactive dialogue scenarios, users may introduce new values or modify existing ones during the conversation.

3.What do you perceive as the main challenges in achieving such online or incremental pluralistic value alignment? When confronted with irreconcilable deep value conflicts—for instance, those rooted in cultural or religious sensitivities—maximizing total correlation may yield compromise-like responses that lack substantive content. How do you envision PICACO, or future work building upon it, incorporating an explicit and interpretable value trade-off framework to handle such cases?

4.To what extent can the optimized meta-instruction generalize to unseen tasks or domains that were not included in the training compositions? Does the meta-instruction retain its alignment capability across out-of-distribution prompts or novel value compositions?

### Soundness
2

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
4

### Summary
The paper proposes a training-free framework (PICACO) for pluralistic alignment: instead of fine-tuning an LLM, it iteratively (i) generates and filters responses to a task using value scorers, and (ii) refines a meta-instruction so that future generations better satisfy multiple, potentially concurrent values (e.g., safety, helpfulness, style). The method is cast as maximizing a TC-like objective with two components: a value-conformity scorer $q_\omega(\cdot)$ and a redundancy-reduction scorer $q_\phi(\cdot)$, so that responses are both value-aligned and not merely surface copies. It runs as an EM-like loop over instructions and responses.

### Strengths
**S1: Problem framing**: the paper targets a realistic setting—multi-value / pluralistic alignment without re-training the base LLM—and gives a clear formulation of “generate → score → refine instruction” as an iterative procedure.

**S2: Simple, practical mechanism**: the two-step loop (response enhancement + instruction refinement) is lightweight, LLM-only, and can be plugged on top of existing models without extra finetuning or RL, which is attractive for third-party / resource-limited users.

**S3: Empirical coverage**: the paper evaluates on several tasks and value sets, showing consistent improvements over single-instruction baselines and demonstrating that the redundancy penalty helps avoid superficial, template-like responses.

### Weaknesses
**W1: Objective not fully grounded.** There is a gap between the theoretical formulation and the actual implementation of $q_\omega(\cdot)$ and $q_\phi(\cdot)$ in this paper.
 - **W1.1**. In the experiments, treating GPT-4o-mini as $q_\omega(\cdot)$ is somewhat misaligned with the theory. In the derivation, $q_\omega(\cdot)$ is presented as a variational distribution, i.e., a controllable/optimizable distribution that approximates the true posterior. Under that definition, it should be a small, explicitly parameterized model that we train ourselves, with parameters $\omega$, and that outputs a probability for each value. In practice, however, the authors use GPT-4o-mini as $q_\omega(\cdot)$, which is not “variational inference” but rather “using an off-the-shelf LLM as a value scorer.” In other words, they do not actually learn a variational distribution; they simply let GPT-4o-mini act as an oracle judge to rate how well a response satisfies the (k)-th value. This theory–implementation gap weakens the claimed probabilistic grounding: the empirical gains may come from using a strong external judge, not from the proposed variational objective.
 - **W1.2**. Redundancy detector $q_\phi(\cdot)$ is a weak spot. The whole “punish surface-level alignment” idea hinges on $q_\phi(\cdot)$ ​ being able to tell copying from genuine alignment, but the paper does not say (i) how $q_\phi(\cdot)$ ​ is trained, (ii) on what data, (iii) how robust it is across domains. If $q_\phi(\cdot)$ ​ is even slightly mis-specified, the algorithm may penalize fluent but correct responses or, conversely, fail to penalize template copying.

**W2: Ablation is incomplete.** You may need ablations such as: remove redundancy term $q_\phi(\cdot)$; fix the instruction pool (no refinement); use random instead of TC-guided instruction sampling. Without these, we can’t tell which part actually gives the lift.

### Questions
**Clarity of notation and definitions (Lines 194–195).**
The writing in this part is quite opaque. You state that “*s is the content where v is derived …*”, but it is unclear what **s** concretely denotes. Is **s** an in-context example or task instruction? What does it mean, operationally, that “the value (v) is derived from (s)” — does (s) *contain* value-specific textual evidence, or does it *index* the ICL example from which the value was learned? Please provide a precise definition of **s**, describe how it is obtained in practice, and, ideally, add a small running example to demonstrate how ((x, s, v)) are related.

**Contradictory objective (Lines 230–231).**
   You write that you “maximize $log q_\phi(s|x_i, y_{i,j})$ and minimize $log q_\phi(s|x_i, y_{i,j})$”. Mathematically, this is seriously self-contradictory, since it optimizes the same quantity in opposite directions. Is this a typo, or are there actually *two* different scorers involved (e.g., a positive/value-consistent scorer and a negative/redundancy scorer) that were written with the same symbol $q_\phi(\cdot)$ ? Please clarify the intended objective and explain how the same term could be both *maximized* and *minimized* in the algorithm.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes PICACO, a black-box in-context alignment method that optimizes a meta-instruction by maximizing conditional total correlation so an LLM can express multiple, potentially conflicting human values without fine-tuning.

### Strengths
The method is practical. It works with black-box LLMs, requires no parameter updates.

This paper demonstrates rigorous theoretical derivation.

### Weaknesses
Readability: Although the paper shows the method from a theoretical view, it should also include some more intuitive, semantic explanations to improve readability.

Ablation: The authors need to test the contribution of qφ and M2 to the final results.

Cost: How much resource (GPU hours or API cost) does it take on average to align one instruction?

### Questions
see Weaknesses

### Soundness
3

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
The paper introduces PICACO, a training-free framework for pluralistic in-context alignment (ICA) that enables large language models to follow multiple, potentially conflicting human values simultaneously. It formulates meta-instruction optimization as conditional Total Correlation maximization between target values and responses, using an EM-like loop for response enhancement and instruction refinement. Tested across five value sets (Helpfulness, Harmlessness, HH-Balance, Confucianism, and Modern Liberalism) and multiple LLMs, PICACO consistently outperforms ICA baselines such as OPRO, URIAL, and Modular Pluralism, achieving better balance across up to eight values and improved safety robustness without finetuning.

### Strengths
Reframes pluralistic in-context alignment as conditional Total Correlation optimisation, offering a theoretically grounded and training-free approach that overcomes the Instruction Bottleneck in existing ICA methods.

Demonstrates rigorous empirical evaluation across five value sets and multiple LLM families (open and black-box), showing consistent improvements in alignment balance, robustness to optimisers, and enhanced safety under jailbreak tests

The paper clearly motivates the problem, explains the method with mathematical precision, and provides intuitive examples (e.g., reconciling Tradition vs. Hedonism) to illustrate nuanced value alignment

Advances the frontier of alignment research by showing that meta-instruction optimisation without finetuning can yield balanced, culturally adaptive, and safety-consistent alignment across diverse moral frameworks

### Weaknesses
Much of the evidence relies on LLM-as-judge (e.g., GPT-4o/DeepSeek judging outputs), which can favour certain response styles and blur whether gains reflect true value adherence versus judge preference.

The method relies on conditional Total Correlation (TC) maximisation. But there isn’t a clean ablation isolating TC from other design choices (e.g., pool construction, filtering) or comparing against alternative dependencies (e.g., MI, HSIC).

All reported comparisons are among in-context alignment methods rather than training-based ones. Even if the performance might be lower, understanding the gap is important.

### Questions
Can the authors add a small human evaluation?

The study only contrasts PICACO with other ICA methods. Could the authors include a lightweight SFT or DPO baseline to quantify the remaining performance gap and support the claim that PICACO offers competitive training-free alignment?

### Soundness
3

### Presentation
3

### Contribution
2
