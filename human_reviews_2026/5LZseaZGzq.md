# Untargeted Jailbreak Attack

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 6, 4, 0

## Abstract
Existing gradient-based jailbreak attacks on Large Language Models (LLMs), such as Greedy Coordinate Gradient (GCG) and COLD-Attack, typically optimize adversarial suffixes to align the LLM output with a predefined target response. However, by restricting the optimization objective as inducing a predefined target, these methods inherently constrain the adversarial search space, which limit their overall attack efficacy. Furthermore, existing methods typically require a large number of optimization iterations to fulfill the large gap between the fixed target and the original model response, resulting in low attack efficiency.

To overcome the limitations of targeted jailbreak attacks, we propose the first gradient-based untargeted jailbreak attack (UJA), aiming to elicit an unsafe response without enforcing any predefined patterns. 
Specifically, we formulate an untargeted attack objective to maximize the unsafety probability of the LLM response, which can be quantified using a judge model. Since the objective is non-differentiable, we further decompose it into two differentiable sub-objectives for optimizing an optimal harmful response and the corresponding adversarial prompt, with a theoretical analysis to validate the decomposition. In contrast to targeted jailbreak attacks, UJA's unrestricted objective significantly expands the search space, enabling a more flexible and efficient exploration of LLM vulnerabilities.
Extensive evaluations demonstrate that \textsc{UJA} can 
achieve over 80\% attack success rates  against recent safety-aligned LLMs with only 100 optimization iterations, outperforming the state-of-the-art gradient-based attacks such as I-GCG and COLD-Attack by over 20\%.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces the Untargeted Jailbreak Attack (UJA), a gradient-based method for jailbreaking Large Language Models (LLMs). Existing attacks, such as GCG and COLD-Attack, are described as "targeted", meaning they optimize an adversarial prompt to align the LLM's output with a predefined target response. The paper states this approach constrains the optimization search space.

UJA, in contrast, is an untargeted attack. Its objective is to maximize the unsafety probability of the LLM response without enforcing a predefined pattern. This unsafety probability is quantified using a judge model.

Because this objective is non-differentiable, the paper proposes a two-stage optimization process:
1.  Stage 1 (Adversarial Response Optimization): An optimal harmful response is identified by optimizing over the judge model's representations.
2.  Stage 2 (Adversarial Prompt Optimization): An adversarial prompt is optimized to make the target LLM generate the response found in Stage 1. This stage uses a "gradient projection" technique to transfer information between the different token spaces of the judge model and the target LLM.

The paper's contributions include the formulation of this untargeted objective and the two-stage optimization method. Evaluations in the paper show UJA achieves over 80% Attack Success Rate (ASR) with 100 optimization iterations, which is reported as outperforming baselines by over 20%.

### Strengths
Originality: The work's originality lies in its critique of existing "targeted" jailbreak attacks and its novel formulation of an "untargeted" objective. This new problem definition, which maximizes unsafety probability, presents a distinct conceptual approach to gradient-based attacks.

Clarity: The paper is well-structured. It clearly articulates the limitations of prior work and logically presents its proposed two-stage methodology. The authors identify the non-differentiable nature of their objective and detail their proposed solution.

Quality: The methodological quality is high. The paper provides a theoretical justification for its decomposition of the non-differentiable objective. This rigor extends to the empirical evaluation, which includes comprehensive ablation studies that proactively address key questions about the framework's components, such as the necessity of the response optimization stage and the impact of different judge models.

Significance: The paper's significance is demonstrated by its strong empirical results. The proposed method achieves a substantially higher attack success rate on benchmarks compared to existing state-of-the-art methods, establishing a new and more effective baseline for jailbreak attacks.

### Weaknesses
The paper's primary weakness lies in its "untargeted" claim. The methodology is more accurately described as a dynamic target-finding attack. The process is not target-free; Stage 1 is explicitly designed to find a single, optimal target response, and Stage 2 then optimizes the prompt to match that specific target. This is still a form of targeted optimization, which re-introduces the kind of constraint the paper claims to overcome.

A second, and more fundamental, limitation is the framework's critical dependency on the judge model. As suggested by the ablation study on different judges, the attack's effectiveness is fundamentally capped by the classification performance and biases of the chosen judge model. The method is optimized to generate a response that fools that specific judge, which may not be the same as generating a response that is universally or practically harmful. The attack's success is relative to its own component, meaning it may simply be overfitting to the judge's specific vulnerabilities.

Finally, the two-stage, two-model architecture introduces significant practical complexity. It requires white-box gradient access to both the target LLM and the judge model, plus a "gradient projection" step to bridge their vocabularies. This is a far more complex setup than single-model baselines. The paper's focus on efficiency in iterations may obscure the true computational cost and setup overhead of this more intricate optimization.

### Questions
The "untargeted" claim is confusing. The method finds a specific target response in Stage 1 and then optimizes the prompt to match that target in Stage 2. Could the authors clarify how this is fundamentally different from a targeted attack, other than the target being generated dynamically?

The ablation study on judge models is a key part of the paper. Were any other judge models used in the experiments besides GPTFuzzer and Llama-Guard-3?

The paper highlights efficiency in terms of iterations. However, the two-stage, two-model approach seems computationally complex. Could the authors provide a comparison of the total wall-clock time and memory usage required for UJA versus the baselines (e.g., GCG, COLD-Attack) to achieve their reported results?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Untargeted Jailbreak Attack (UJA), a gradient-based jailbreak method that does not require predefined target responses. Unlike methods like GCG and COLD-Attack that optimize for predefined prefixes, UJA maximizes the unsafety probability of LLM responses via a judge model. It decomposes the non-differentiable objective into two differentiable sub-objectives: optimizing an optimal harmful response and the corresponding adversarial prompt. Extensive experiments on 8 LLMs (e.g., Llama-3, Qwen-2.5) and 2 datasets (AdvBench, HarmBench) show UJA achieves over 80% attack success rate (ASR). It also exhibits strong transferability and robustness against defenses like Perplexity and SmoothLLM.

### Strengths
1. The paper tackles an important problem inherent in existing gradient-based jailbreak methods: their reliance on predefined affirmative response prefixes (e.g., "Sure, here is...") as optimization targets. This dependency requires prior knowledge of model-specific response patterns and can lead to optimization futility when the chosen prefix poorly aligns with the target LLM's natural output distribution.

2.  The consistent superiority over seven baseline methods provides strong empirical support.

3. UJA overcomes a fundamental challenge in transferring optimization signals across the target model and judge model with different tokenization schemes.

### Weaknesses
1. In the experimental section, the paper only uses GPTFuzzer as the judge model to provide feedback, which raises concerns about the reliability of the proposed method.

2. The method exhibits high dependence on the judge model, often requiring a specially fine-tuned judge model. This actually imposes constraints on the method’s practicality. In new malicious scenarios (e.g., biosecurity scenarios), these fine-tuned judge models lack relevant training and may need to be re-fine-tuned. The authors should objectively examine the potential limitations of the method.

3. Lacks discussion on some existing works. For instance, there are black-box jailbreaking suffix optimization works that also do not require targets and share the same idea of using a judge model to provide feedback signals (e.g., [1] An Optimizable Suffix Is Worth A Thousand Templates: Efficient Black-box Jailbreaking without Affirmative Phrases via LLM as Optimizer; [2] Jailbreaking Black Box Large Language Models in Twenty Queries). The authors should discuss them in the paper.

### Questions
1. Have the authors considered testing the method’s performance when using other judge models for feedback?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes UJA (Untargeted Jailbreak Attack) — a novel gradient-based jailbreak framework that, unlike prior targeted methods (e.g., GCG, COLD), removes the dependence on predefined target prefixes.
UJA formulates jailbreak as maximizing the unsafe probability of LLM outputs judged by an external classifier

### Strengths
1. Experiments cover several models and datasets, showing reasonably consistent improvements in ASR over certain baselines.

2. The proposed method shows strong transferability across architectures, with improved efficiency (fewer iterations, lower GPU cost).

### Weaknesses
- Dependence on the judge model. 
The method heavily relies on a differentiable judge model $J(\cdot)$ to compute gradients of “unsafe probability.” In practice, such judges encode alignment criteria that are usually proprietary or confidential. Using them for optimization exposes sensitive alignment information and might not reflect a realistic attacker setting.  

- Insufficient methodological clarity.
The paper does not specify critical implementation details.  For example, it remains unclear:
  - What exactly is z? Is it a pooled embedding, token-level hidden representation, or logits vector?
  - How to obtain the final prompt $p$ based on $z$

- Incomplete baseline comparison.  
The experimental section omits several relevant and recent jailbreak approaches such as $\texttt{AdvPrompter}$ and R1-style reinforcement-based methods.

### Questions
Q1. In Table 3, models such as Mistral and Vicuna appear to have weak defense capability—they tend to produce unsafe outputs even when conditioned with compliance instructions. However, the baseline $\texttt{llm-adaptive}$ achieves unusually low ASR despite being a context-adaptive method.  Can the authors explain why $\texttt{llm-adaptive}$ performs so poorly under these conditions?

Q2. The optimization process of UJA essentially maximizes a scalar unsafe score from a judge model, which resembles a reinforcement learning setup (reward maximization). Why a purely gradient-based approach was preferred, and what advantages or limitations it brings compared to an RL formulation.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The paper attempts to address limitations in existing 'targeted' gradient-based jailbreak attacks, which aim to force a model to output a predefined phrase, constraining effectiveness and efficiency. It proposes the Untargeted Jailbreak Attack (UJA), which instead seeks to elicit any unsafe response, thereby broadening the potential adversarial search space. It is implemented by maximizing the unsafe score of the target LLM's response, obtained from a judge model.

### Strengths
The paper identifies a genuine problem clearly. Nevertheless, it is flawed in other aspects (see "Weaknesses").

### Weaknesses
● How does the proposed algorithm ensure that a response is genuinely unsafe, rather than merely deceiving the employed judge model?

● The writing strongly requires enhancement, including but not limited to streamlining verbose descriptions and ensuring the correct use of mathematical notation.

● Line 301: How is it guaranteed that a valid candidate jailbreak prompt is obtained?

● Regarding the experiments: Why does I-GCG perform worse than GCG? Furthermore, why does LLM-Adaptive perform poorly when attacking Vicuna and Mistral? In its original paper, the success rate against these two models was close to 100%.

### Questions
See Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1
