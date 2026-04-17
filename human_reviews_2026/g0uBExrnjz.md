# Excessive Reasoning Attack on Reasoning LLMs

- Decision: Reject
- Scores: 4, 6, 6

## Abstract
Recent reasoning large language models (LLMs), such as OpenAI o1 and DeepSeek-R1, exhibit strong performance on complex tasks through test-time inference scaling.
However, prior studies have shown that these models often incur significant computational costs due to excessive reasoning, such as frequent switching between reasoning trajectories (e.g., underthinking) or redundant reasoning on simple questions (e.g., overthinking).
In this work, we expose a novel threat: crafting adversarial inputs to exploit excessive reasoning behaviors.
However, directly optimizing for excessive reasoning is non-trivial because reasoning length is non-differentiable. 
To overcome this, we introduce a proxy framework that approximates the long reasoning objective and shapes token-level behavior:
(1) Priority Cross-Entropy Loss, a modification of the standard cross-entropy objective that emphasizes key tokens by leveraging the autoregressive nature of LMs;
(2) Excessive Reasoning Loss, which encourages the model to initiate additional reasoning paths during inference; and
(3) Delayed Termination Loss, which is designed to extend the reasoning process and defer the generation of final outputs.
We optimize and evaluate our attack for the GSM8K and ORCA datasets on DeepSeek-R1-Distill-LLaMA and DeepSeek-R1-Distill-Qwen. 
Empirical results demonstrate a 3x to 6.5x increase in reasoning length with comparable utility performance.
Furthermore, our crafted adversarial inputs exhibit transferability, inducing computational overhead in o3-mini, GPT-OSS, DeepSeek-R1, and QWQ models.
Our findings highlight an emerging efficiency-oriented vulnerability in modern reasoning LLMs, posing new challenges for their reliable deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates a new type of adversarial vulnerability in reasoning large language models (LLMs), referred to as Excessive Reasoning Attack. The authors argue that reasoning-oriented models, such as DeepSeek-R1 and Qwen variants, tend to perform unnecessary or redundant reasoning that can be exploited to increase inference-time computation. To formalize this, they propose a proxy optimization framework consisting of three components: (1) Priority Cross-Entropy Loss to emphasize informative tokens, (2) Excessive Reasoning Loss to encourage repeated or branched reasoning, and (3) Delayed Termination Loss to postpone output completion. Using the GCG optimization method, they generate short adversarial suffixes that substantially extend reasoning trajectories on GSM8K and ORCA benchmarks. Experiments show increased reasoning length, latency, and energy usage, with minimal change in accuracy, and demonstrate partial transferability across several commercial LLMs (o3-mini, GPT-OSS, DeepSeek-R1, QWQ).

### Strengths
1. This paper introduces Excessive Reasoning Attack, a novel adversarial attack that differs from prior works focusing solely on content manipulation or refusal-based safety issues. It specifically targets the reasoning process of LLMs, exposing a new dimension of vulnerability related to inference efficiency. 
2. The paper proposes three complementary differentiable proxy losses—Priority Cross-Entropy (PCE), Excessive Reasoning (ER), and Delayed Termination (DT)—which effectively address the non-differentiability of reasoning length.
3. The experiments yield several meaningful insights, such as the observation that optimized adversarial suffixes remain effective across models sharing the same tokenizer.

### Weaknesses
1. White-box Assumption and Limited Practicality

This paper introduces Excessive Reasoning Attack, a novel adversarial attack targeting reasoning LLMs. I acknowledge that such an attack poses a more substantial threat to online LLM services (e.g., OpenAI, Google, and Alibaba Cloud) than to open-source models. However, the proposed method relies on a white-box assumption, requiring full access to model weights and gradients. This dependency makes it inapplicable to black-box commercial models. Although the results indicate certain transferability, the optimization mechanism still depends heavily on white-box access, which considerably limits the real-world practicality of this attack.

2. Demand on Computational Resources

The experiments are conducted on 7–8B models (LLaMA and Qwen), which can be handled with a single A100 80GB GPU. However, since the method requires gradient access—a memory-intensive operation—it entails significant computational overhead, which would
become more severe for larger LLMs. It is recommended that the computational cost and GPU memory consumption of the proposed framework be quantitatively analyzed to clarify its scalability and practicality.

3. Limitations of Experimental Design

- Stronger Baselines are Needed

The baselines in Table 1 are not well-aligned with the stated objective of this paper, inadvertently amplifying the effectiveness of the proposed method. Most compared methods are not designed to extend the reasoning length of LLMs. For instance, Engorgio Prompt primarily targets general DoS attacks, not reasoning-specific ones, while CatAttack introduces distractive sentences that mainly add semantic noise rather than lengthening reasoning. As shown in Table 1, these baselines barely increase reasoning length compared to the original setting. Therefore, stronger baselines are needed to empirically evaluate the performance of the proposed method.

- Limitation of Transferability Experiment

It would be beneficial to test the transferability of the proposed adversarial attack on more advanced reasoning LLMs, such as GPT-5, Claude, and Gemini, rather than only on open-source or relatively weak models listed in Table 4. Such experiments would better demonstrate how different reasoning capabilities affect robustness against this attack.

### Questions
See the weaknesses above.

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
This paper investigates a novel denial-of-service (DoS) style threat against reasoning-focused large language models (LLMs), which are prone to inefficient behaviors like redundant or overly long reasoning chains. The authors propose the "Excessive Reasoning Attack," which crafts adversarial suffixes to compel the target model to generate excessively long reasoning trajectories, thus significantly increasing inference latency, energy consumption, and overall computational cost without degrading task accuracy. To achieve this, they introduce a composite loss function comprising three components: Priority Cross-Entropy (PCE) Loss, Excessive Reasoning (ER) Loss, and Delayed Termination (DT) Loss, optimized using a gradient-based search (GCG). The results show a substantial increase in reasoning length (3x to 6.5x) and resource usage, demonstrating transferability across various commercial and open-source models.

### Strengths
1.  **Novel and Practical Attack Objective:** The paper identifies and exploits a highly relevant vulnerability: the efficiency and resource consumption of reasoning LLMs. Unlike attacks targeting answer correctness, this focuses on *economic* and *operational* damage (computational overhead), which is a critical, underexplored threat in commercial LLM deployment (akin to a DoS attack).
2.  **Strong Empirical Validation and Transferability:** The attack demonstrates high efficacy, successfully increasing reasoning length by several multiples across different model architectures (LLaMA and Qwen variants). Furthermore, the strong transferability of the adversarial suffixes to black-box commercial models like 03-mini and others highlights the generalized nature of this efficiency vulnerability in the reasoning mechanisms themselves.

### Weaknesses
1.  **Ambiguity in Causality of Performance Gain:** The paper observes that the attack, while lengthening reasoning, sometimes increases task accuracy. The analysis attributes this to increased capacity allocation, but a more in-depth exploration of *why* the attack's specific, lexically biased reasoning leads to *better* answers is needed to fully understand the mechanism.
2.  **Tokenizer Dependency in Transferability:** The transferability analysis, particularly the difference between the LLaMA and Qwen optimized suffixes on the target models, points to a strong dependence on tokenizer alignment. This suggests the attack's universality may be bounded by tokenization schemes, a limitation that should be discussed more explicitly in the conclusion.

### Questions
Please see the weakness.

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
The paper proposes a white-box adversarial suffix attack that increases the reasoning token length (and thus latency/energy) of reasoning-oriented LLMs without substantially degrading task accuracy. Because reasoning length is non-differentiable, the authors optimize three differentiable proxy losses: (i) Priority Cross-Entropy (PCE) that reweights supervision toward prompt-dependent tokens, (ii) Excessive Reasoning (ER) that boosts the probability of sentence-initial deliberation tokens, and (iii) Delayed Termination (DT) that suppresses end and beginning of sentence tokens. They search adversarial suffixes with GCG. On GSM8K/ORCA, the attack increases reasoning tokens by 3 to 6.5X, with large latency and energy overhead; adversarial suffixes also transfer to several commercial/closed models.

### Strengths
1) The decomposition into PCE, ER, DT is well-motivated by the goal (longer reasoning). Ablations isolate each piece (Table 5; Table 11), showing monotone increases when objectives are combined.

2) On LLaMA-8B-R1-distill (GSM8K greedy), reasoning tokens rise from 668 to 1914 and latency 24.3s to 54.9s with only 10-token suffixes (Table 1). Similar or larger effects on Qwen-7B-R1-distill. The designed attack works experimentally.

3)Cross-model tests (Table 4) show the attack is not brittle. This connects to prior literature on universal triggers and GCG-style suffixes shaping model behavior.

### Weaknesses
1) Only 50 examples * 2 datasets * 3 runs = (approx.) 300 evaluations. No statistical test on accuracy differences. For GSM8K (app. 8.5 k train / 1 k test), using 50 samples risks > +- 3 points variance; hence “no degradation” claims are statistically inconclusive.


2) The threat model assumes full gradient access (white-box), while commercial systems (o1/o3-mini) are black-box API only. Transferability results (Tab. 4) are modest (<600 tokens gain) and could arise from stochastic sampling noise.

3) The paper assumes that maximizing the PCE/ER/DT composite correlates with reasoning-token count, yet this link is never formally validated. Equation (3) weights by prompt-sensitivity (Δ log p), but the proof that this expectation increases generation length is missing.
A simple counterexample exists: tokens with high prompt-sensitivity can be function words (e.g., “Therefore”) that appear early without elongating reasoning.
A more principled surrogate, e.g., REINFORCE or length-aware policy gradient as in Wu et al., 2025 (When More is Less), would better justify optimization correctness.


4) (not a major weakness) I noticed that related work has been provided in Appendix, which is very non-standard. This is not grounds for rejection, but I think including some of the related work would be helpful.

**References:**
[1] Wu Y, Wang Y, Ye Z, Du T, Jegelka S, Wang Y. When more is less: Understanding chain-of-thought length in llms. arXiv preprint arXiv:2502.07266. 2025 Feb 11.

### Questions
1) GSM8K and ORCA are math-only; do you expect similar behavior on symbolic-reasoning datasets (e.g., MATH Bench)?

2) How do you verify that longer reasoning corresponds to distinct reasoning trajectories, not repetition? ANy qualitative study?

3) What happens if $\beta$ is reduced from 50 to 1? Does effect vanish?

4) Please confirm whether the importance score (Eq2.) is correct. I might be wrong but current formula appears mathematically reversed.

### Soundness
3

### Presentation
3

### Contribution
2
