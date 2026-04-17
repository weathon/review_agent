# LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Human cognition naturally engages with abstract and fluid concepts, whereas existing reasoning models often rely on generating discrete tokens, potentially constraining their expressive capabilities. Recent advancements aim to address this limitation by enabling large language models (LLMs) to generate soft, abstract tokens, thus facilitating reasoning within a continuous concept space. In this paper, we investigate the $\textit{Soft Thinking}$ capabilities of various LLMs through a systematic analysis of their internal behavior using a suite of probing techniques. Contrary to the prevailing belief that Soft Thinking supports parallel exploration of diverse reasoning paths, our findings reveal that $\textbf{LLMs behave as single-threaded reasoners}$—they predominantly rely on the token with the highest probability in the soft input to predict the next step. This behavior induces a greedy feedback loop that suppresses alternative reasoning paths and undermines the benefits of transmitting richer information via Soft Tokens. To address this $\textit{Greedy Pitfall}$, we propose $\textbf{Stochastic Soft Thinking}$, which introduces stochasticity to break free from the greedy tendency. Our experiments demonstrate that incorporating $\textit{randomness}$—particularly with the $\textbf{Gumbel-Softmax trick}$—can alleviate the limitations of vanilla approaches and unleash the potential of Soft Thinking, resulting in superior performance across eight reasoning benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies Soft Thinking, a decoding method where the next input embedding is a weighted average of token embeddings according to the model’s predicted probability distribution. The community has suggested that Soft Thinking enables parallel reasoning, where multiple reasoning paths are explored simultaneously.
The authors perform a careful empirical investigation across multiple large reasoning models (e.g., DeepSeek-R1-Distill-Qwen-32B, QwQ-32B, Skywork-OR1-32B) and several math, QA, and coding benchmarks. Using logit and layerwise analyses, they find that LLMs behave as single-threaded reasoners: even when given continuous inputs, they collapse almost deterministically to the top-1 trajectory. This “Greedy Pitfall” shows that Soft Thinking, as commonly implemented, is functionally equivalent to greedy decoding.
To mitigate this, the authors propose Stochastic Soft Thinking: injecting randomness through Dirichlet or Gumbel-Softmax sampling to restore diversity. The Gumbel-Softmax variant improves reasoning accuracy and stability, with theoretical justification via Luce’s choice axiom.

### Strengths
- Well-motivated study: The paper addresses a timely and widely misunderstood topic. Many groups have speculated about “continuous reasoning,” and this work decisively clarifies what actually happens.
- Empirically clean and thorough: Multiple strong models and diverse benchmarks are used. The behavioral evidence for “single-threaded reasoning” is compelling and well visualized.
- Clear diagnosis and fix: The “Greedy Pitfall” is an elegant conceptual summary of why Soft Thinking fails. The Gumbel-Softmax correction is simple but principled.
- Theoretical framing: The link to Luce’s choice axiom provides a clean justification for their stochastic sampling choice, grounding the fix in existing probabilistic decision theory.

### Weaknesses
The main limitation of this paper is that its contributions concern only decoding behavior and not any internal mechanisms of reasoning. Despite its title, the paper does not truly “demystify” the model’s reasoning process. The finding that Soft Thinking collapses into top-1 decoding is empirically strong but does not explain why the model’s internal computation prefers that collapse or how it represents/sorts out alternate hypotheses. The work therefore improves our understanding of sampling dynamics, not of reasoning per se.
A second issue concerns the claim that Soft Thinking remains “on distribution.", this assumption is doubtful, as it was never trained on such mixed representations. The paper does not analyze whether this produces distortions in internal representations, altered activation norms, or other side effects.

### Questions
1. Soft token off-distribution effects: You claim that mixing token embeddings does not take the model off-distribution, yet the model was never trained on combinations of embeddings. Did you observe any qualitative or quantitative side effects from this?
2. Depth of single-threadedness: You show that Soft Thinking quickly collapses to the top-1 trajectory, but is this collapse uniform across layers or token positions? Do early layers ever represent meaningful multi-path structure, or is the single-threadedness present from the start?

### Soundness
3

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
3

### Summary
Prior work (soft thinking (Zhang et al., 2025)) processes the chain-of-thought context as a convex combination of token embeddings according to each position's model output distribution. This paper improves soft thinking by incorporating random sampling in the CoT process, where sampling is modelled by Dirichlet distribution and Gumbel-softmax distribution.

### Strengths
- The detailed description in Appendix D is very helpful in understanding soft token thinking. I highly recommend the authors to incorporate the idea of Appendix D into the main text if space allows.
- The benefits of soft token thinking incorporated in RL training is promising, especially on larger models, according to the evidence in Appendix E.

### Weaknesses
- The Dirichlet scaling parameter is described by $\gamma$ in Line 335 but $\alpha$ is used in Line 370.
- The term "Greedy" has been mentioned multiple times, but I find it difficult to understand without its precise definition on what it refers to. For example, "Greedy Token Thinking" is mentioned Section 4.4 along with "discrete Token Thinking", but the difference of the two methods is not discussed.

### Questions
- It has been mentioned that vanilla Soft Thinking fails to explore different reasoning paths. How does Stochastic Soft Thinking, e.g., Dirichlet sampling and Gumbel-softmax sampling, allows the model to explore different reasoning paths? Is there any evidence to support this?
- How many samples are drawn (from Dirichlet distribution and Gumbel-softmax distribution) to form each soft thinking token? An ablation study on the effect of number of samples to the performance would give a better picture on whether soft token thinking benefits from the mixture of embeddings.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the underlying mechanisms of Soft Thinking and challenge the prevailing belief that it would allow LLM to explore multiple reasoning paths in parallel. The primary experiments show that the standard Soft Thinking approach achieves similar results to greedy decoding and their analysis reveals that the subsequent generation is dominated by single token with highest probability. Numerical metrics such as JS Divergence and ROUGE-L similarity are applied to verify the findings. The paper then proposes Stochastic Soft Thinking to address above issues. They evaluate Dirichlet Sampling and Gumbel-Softmax Trick on Deepseek, QwQ, Skywork models in mathematical reasoning benchmark and Gumbel-Softmax method shows consistent improvement.

In short, the paper discover the Greedy Pitfall of Soft Thinking and address the problem by injecting stochastacy.

### Strengths
1. Clear Motivation and Problem Framing: The central claim that LLMs are "single-threaded reasoners" and that Soft Thinking defaults to a greedy process is an important observation on the drawback of previous method.
2. Sufficient Analysis to support the Greedy Pitfalls: The evidence from output probability (JS Divergence) , hidden state representations (Logit Lens) , and sequence-level output (ROUGE-L) effectively support the hypothesis.
3. Practical and Effective Solution: The paper address the issue by Stochastic Soft Thinking, particularly using Gumbel-Softmax which is practical and shows consistent improvement against vanilla Soft Thinking. The method is evaluated on various math, coding and QA benchmarks. This paper also conduct ablation study to demonstrate the necessity of randomness and softness.

### Weaknesses
1. The paper's modest performance gains are not shown to be statistically significant. The proposed "Stochastic Soft Thinking" method involves randomness. However, the reported average improvements are small (ranging from +0.42 to +1.05 points). For five of the eight benchmarks, the authors report Pass@1 scores, which may be sensitive to run-to-run variance. The reproducibility statement confirms these experiments were run with a single "fixed random seed". This is insufficient to demonstrate the stability and hyperparameter sensitivity of proposed method.

2. Causal evidence for “mitigating the Greedy Pitfall” is incomplete. The paper shows accuracy gains and an input-level softness–randomness analysis (Fig. 5), but it does not re-run the core behavioral probes used to diagnose the pitfall—JS Divergence, logit-lens intersection trajectories, or ROUGE-L similarity to the greedy trace—after introducing randomness. Without those, it’s hard to conclude that improvements come from reducing top-1 dominance rather than other effects of stochasticization. Please repeat the diagnostic analyses under the stochastic methods to substantiate the claim.  

3. The "Greedy Pitfall" is only demonstrated at a fixed temperature, lacking a crucial ablation study. The paper's entire premise that vanilla Soft Thinking fails is based on its performance at a fixed temperature of 0.6. The "Greedy Pitfall" is defined as the model's reliance on the top-1 token. The numerical dominance of the top-1 token is directly controlled by the temperature setting. It is plausible that simply increasing the temperature (e.g., to 1.0 or even higher) would flatten the probability distribution, make the soft tokens inherently less greedy, and potentially solve the pitfall without requiring the proposed stochastic methods. (Logit Lens analysis probes a single manually-balanced token with probability 0.6/0.4, this is not a substitute for a full sequential generation where a high temperature would be applied at every step of the generation.)

4. Formatting Issue: The authors should consider adjusting the font size in Figure 5 and legend in Figure 3.

### Questions
1. Could the authors please provide the mean and variance (or confidence intervals) over multiple runs with different seeds to validate that these improvements are statistically significant and not an artifact of randomness? Could the authors provide analysis on the stability of the method at different randomness level?
2. Could the authors repeat the behavior probes under Dirichlet/Gumbel soft tokens and report whether top-1 dominance is addressed?
3. Could the authors provide an ablation study showing the end-to-end performance of both vanilla Soft Thinking baseline at higher temperatures to confirm this is an inherent flaw and not just a result of the low-temperature setup?
4. Could the authors please revise the figures mentioned in weakness 4?

I will raise my score once above issues are properly addressed.

### Soundness
2

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
2

### Summary
This paper tackles the problem of learning from human feedback when that feedback contains systematic noise and biases. The authors propose OPEN (Objective Preference Elicitation from Noisy feedback), which uses an auxiliary objective inference model to recover true preferences from noisy observations.

### Strengths
The paper addresses a genuinely important problem since human feedback in practice is often inconsistent and biased, especially when dealing with complex tasks or subjective preferences. The theoretical framework connecting noisy observations to latent objectives through a probabilistic model is well motivated, and I appreciate that the authors provide both convergence guarantees and empirical validation. The experiments on LLM summarization and dialogue tasks show meaningful improvements over baseline RLHF methods, with particularly strong results when feedback quality degrades.

### Weaknesses
you can improve your writing, authors.
The auxiliary model for objective inference adds considerable complexity to the training pipeline, and I'm concerned about the computational overhead this introduces compared to standard RLHF. While the synthetic experiments are convincing, the real world experiments could benefit from more diverse evaluation settings beyond just text generation tasks. The paper also doesn't fully address how to set the hyperparameters for balancing between the inferred objectives and raw feedback, which seems crucial for practical deployment.

### Questions
How sensitive is the method to the choice of prior over objective functions?

### Soundness
2

### Presentation
1

### Contribution
2
