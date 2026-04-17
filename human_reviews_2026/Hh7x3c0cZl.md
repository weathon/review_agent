# Theoretical Modeling of Large Language Model Self-Improvement Training Dynamics Through Solver-Verifier Gap

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
Self-improvement is a significant techniques within the realm of large language model (LLM), aiming to enhance the LLM performance without relying on external data. Despite its significance, generally how LLM performances evolve during the self-improvement process remains underexplored. In this paper, we theoretically model the training dynamics of self-improvement via the concept of solver-verifier gap. This is inspired by the conjecture that the performance enhancement of self-improvement stems from the gap between LLM's solver capability and verifier capability. Based on the theoretical framework, we further show how to model the entire training trajectory. This framework allows quantifying the capability limit of self-improvement by fitting the theoretical model to the experiment results. We validate the effectiveness of the theoretical framework on various LLMs and datasets. Beyond self-improvement, we extend our analysis to investigate how external data influences these dynamics within the framework. Notably, we find that under limited external data regimes, such external data can be utilized at any stage without significantly affecting final performances, which accords with the empirical observations.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper develops a physics-inspired theoretical framework to explain how large language models (LLMs) improve themselves without external data. It introduces the concept of a solver-verifier gap—the difference between an LLM’s ability to generate solutions (solver) and its ability to evaluate them (verifier)—as the main driver of self-improvement. The authors model training dynamics using coupled differential equations, showing that solver and verifier capabilities evolve exponentially toward limits determined by their initial gap. Experiments on Phi and Llama models across multiple datasets validate these exponential laws and confirm that verifiers consistently outperform solvers. The paper further extends the framework to cross-improvement, where limited external data enhance verifier capability; theoretical and empirical results show that the timing of using such data has minimal impact on final performance. Overall, the study provides a unified quantitative model linking LLM self-improvement behavior, convergence limits, and external data effects

### Strengths
This paper is highly original in formalizing LLM self-improvement through a physics-inspired solver-verifier gap framework, transforming an empirical phenomenon into a quantitative model. Its quality is strong, with clear theoretical derivations, well-controlled experiments, and convincing exponential fits that validate the framework across multiple models and datasets. The writing is clear and logically structured, effectively integrating intuition, math, and visualization. In significance, the work provides a foundational step toward understanding autonomous model improvement and offers practical insights for optimizing self- and cross-improvement strategies, making it both conceptually innovative and broadly impactful for future LLM research.

### Weaknesses
Some claims appear to be a bit strong - "Self-improvement is among the most prominent techniques within the realm of ...".

I also advocate for using the term "generation-verification" gap to respect the originality.

The time-invariant assumption sounds a bit strong: this assumes that the key parameters governing training dynamics — such as the cross-improvement effect γ, and the solver/verifier rate coefficients α and β remain constant across all epochs. Some more ablations would be helpful to understand their effects.

Some limiting behaviors are worthy of examination - such as how γ α and β would affect the convergence.

In EvoLM, model developers have found that base models that have undergone mid-training can be more easily adaptable using RL. The work should take into account such factors and see how this affects the self-improvement behaviors. Ideally, a controlled comparison using base, mid, post-trained checkpoints would be an insightful add. Because the field of post-training can heavily depend on upstream training, I am leaning towards advocating for strong accept if this confounder is taken into account.

### Questions
How does the base model FLOPs (and families) affect the analysis? 

how do models from different families have different exponential decay functions, which would lead to the modeling of solver and verifier uncertainties (or their gaps) toward asymptotic limits?

### Soundness
3

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
3

### Summary
The paper investigates the training dynamics underlying the self-improvement process of large language models, aiming to uncover the mechanisms driving performance evolution without relying on external data. The authors propose a theoretical framework based on the solver–verifier gap, which explains that the essential source of self-improvement lies in the disparity between a model’s generation (solver) and self-evaluation (verifier) capabilities. Inspired by the potential energy framework in physics, the authors formulate a set of coupled differential equations to characterize how the solver and verifier capabilities evolve over time (i.e., training iterations), and derive that their capability growth follows an exponential convergence pattern.

### Strengths
1. The paper provides an interesting and effective theoretical exposition of LLM self-improvement, accompanied by a relatively rigorous theoretical proof.

2. It conducts diverse empirical studies on mathematical tasks across multiple models, which enhances the credibility of the results.

### Weaknesses
1. The relationship between model capability and output uncertainty is not clearly articulated. The connection between a model’s accuracy on different tasks, its output uncertainty, and its underlying “capability” remains ambiguous. Why can output uncertainty serve as a valid indicator of model capability? How is this metric related to output accuracy? Although the authors provide some explanation in the appendix, the paper lacks further theoretical and empirical justification for the appropriateness of this metric.

2. The theoretical analysis relies on numerous and rather simplified assumptions. In particular, the energy-based assumptions regarding the training process may fail to capture the complex dynamics involved in model optimisation.

3. The experiments are limited to mathematical reasoning tasks (GSM8K, Math) and lack broader validation on code generation or other reasoning benchmarks.

### Questions
Please refer to the relevant points in the Weaknesses section. If the authors can provide clarification and improvements, I would be very happy to raise my score.

### Soundness
2

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
3

### Summary
The paper studies the training dynamics during LLM self-improvement training. The paper uses the solver-verifier gap as the key concept for measuring the self-improvement dynamics.  The solver-verifier gap is defined by the log ratio of policy's probability over the best of N generation and the individual generations. The paper proposes to model the generator's capability change and solver's capability change as a differential equation, and thus the change of gap can be measured as the energy function. Experiment results are provided to support the theoretical models. The paper further studies the dynamics of cross-improvement.

### Strengths
1. The paper studies the important problem of modeling the LLM self-improvement process, which models a very complex system with simple differential equations. 

2. The paper provides experiments to verify the validity of the proposed models.

### Weaknesses
1. I had a difficult time even trying to understand the basic definitons such as solver uncertainty and verifier uncertainty. Are terms like $U_s(t)$ and $U_v(t)$ random variables, since $y_i$ are random variables? If so, how do we even understand these terms and so the gap term? How are they measured in practice?

2. I don't see the necessity of formulating everything as differential equations as the LLM self-improvement update is discrete. 

3. The exeperiment setup seems unclear. For example, it is unclear to me what TF and QE methods are. 

4. The result of the gap narrows during self-improvement does not seem new to me; it appeared in many previous works already. 

5. The paper needs to highlight what are the significance of the current results.

### Questions
1. Although some experiments seem to suggest that $E(t)$ can be linear wrt $G(t)$, can the authors show if this is even possible in any toy settings, where we can even have the policy follows some bernoulli distribution parametrized by $p$, and we have some updating dynamics $d p(t)/dt$, can we say something that $E(t)$ is linear wrt $G(t)$?

2. Can the authors explain why does eq 12. a reasoning model?

3. In the final sentence of section 5.2, how did we reach the conclusion that the small difference in accuracy validates prop. 5.1?

### Soundness
2

### Presentation
2

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
This paper develops a theoretical framework to explain how LLMs improve themselves without external data. It introduces the solver–verifier gap, where the solver represents generation ability and the verifier represents evaluation ability. Using coupled differential equations inspired by potential energy, the authors model the training dynamics and show that solver and verifier capabilities converge exponentially.

### Strengths
1. First clear formalization of LLM self-improvement dynamics through solver–verifier interactions.
2. The differential-equation approach captures exponential convergence behavior that aligns well with empirical trends.
3. Experiments across datasets and LLMs demonstrate strong fits (R² > 0.9) to the theoretical model, reinforcing its predictive power.
4. The extension to limited external data regimes adds practical insight into data allocation strategies.

### Weaknesses
1. The model is phenomenological rather than derived from first principles; it lacks a deep mechanistic explanation of why solver-verifier dynamics follow this form.
2. Assumes linear potential energy and time-invariant coefficients (α, β), which may not generalize to all LLM training settings.
3. The connection between theoretical variables (E(t), G(t)) and practical LLM learning signals remains abstract.
4. The paper omits discussion of computational cost, convergence rate sensitivity, and implications for large-scale training.

### Questions
See weakness above.

### Soundness
3

### Presentation
3

### Contribution
3
