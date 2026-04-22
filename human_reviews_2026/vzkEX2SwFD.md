# Activation Steering with a Feedback Controller

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
Controlling the behaviors of large language models (LLMs) is fundamental to their safety alignment and reliable deployment. However, existing steering methods are primarily driven by empirical insights and lack theoretical performance guarantees. In this work, we develop a control-theoretic foundation for activation steering by showing that popular steering methods correspond to the proportional (P) controllers, with the steering vector serving as the feedback signal. Building on this finding, we propose Proportional-Integral-Derivative (PID) Steering, a principled framework that leverages the full PID controller for activation steering in LLMs. The proportional (P) term aligns activations with target semantic directions, the integral (I) term accumulates errors to enforce persistent corrections across layers, and the derivative (D) term mitigates overshoot by counteracting rapid activation changes. This closed-loop design yields interpretable error dynamics and connects activation steering to classical stability guarantees in control theory. Moreover, PID Steering is lightweight, modular, and readily integrates with state-of-the-art steering methods. Extensive experiments across multiple LLM families and benchmarks demonstrate that PID Steering consistently outperforms existing approaches, achieving more robust and reliable behavioral control.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper aims to address the lack of theoretical guarantees, presence of steady-state errors, and overshoot issues in existing activation steering methods. It analogizes the inter-layer activation propagation process in LLMs to a dynamical system, framing popular methods as instances of a Proportional (P) controller. Building on this, it introduces the Proportional-Integral-Derivative (PID) controller from control systems theory and proposes the PID Steering framework.

### Strengths
1. This work provides a unified control-theoretic perspective, categorizin existing methods (e.g. ActAdd, DirAblate, Mean-AcT) as specific types of P controllers, thereby logically leading to the integration of the PID controller concept and the proposal of the PID Steering framework.

2. The paper goes beyond conceptual proposal by providing rigorous mathematical derivations, including error dynamic modeling and PID controller discretization.

3. PID Steering is designed as a lightweight, plug-and-play module that can be seamlessly integrated into various existing steering frameworks (e.g., Angular Steering, Mean-AcT) by replacing their original steering vector construction, facilitating widespread adoption.

### Weaknesses
1. Manual tuning of PID parameters: the PID controller parameters $K_p, K_i, K_d $ appear to require manual tuning. Finding optimal settings involves significant trial and error, which is somewhat cumbersome.

2. Lack of analysis on $I, D$ Independent contributions and combinatorial effects. The experimental section fails to demonstrate the independent contributions and combinatorial effects of the $P, I,$ and $D$ modules, since the authors claim their method is a plugin fashion. Lack of the experiment setting to demonstrate the benefits by introducing the $I$ or $D$ to the $P$-class controllers.

3. Incomplete experimental comparison with cited methods: While the introduction and background sections systematically review mainstream activation steering methods, Section 5 does not include comparative tests against all mentioned methods (e.g., ActAdd). This prevents intuitive verification of the PID framework's optimization effects across different underlying $P$ controllers.

### Questions
1. Do the trial-and-error costs associated with manual PID parameter tuning vary across different task scenarios? Could parameters be highly sensitive and difficult/costly to tune in certain task types?

2. If testers used harmful prompts for evaluation, would PID Steering suffer performance loss?

3. Why the activation steering vector is added to every layer?  If added to only part optimal selected layer(s), the method offer advantages over identifying and intervening?

4.  Unclear Visual Representation in Figures: Figure 3, as key evidence supporting the core argument, lacks clarity in visualization and exposition. One theoretical cornerstone is that the integral term eliminates steady-state error. However, the PI curve in the figure ultimately displays saturation and large overshoot, obscuring this advantage and making it difficult for readers to appreciate the effectiveness of this step. Adding a subplot specifically comparing P and PI could highlight this advantage.

### Soundness
2

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
This paper introduces a control-theoretic framework for activation steering in LLMs, named PID Steering. The motivation here is that traditional activation steering methods, such as Activation Addition, Directional Ablation, and Mean Activation Transport,modify internal model activations to control behaviors like toxicity or refusal, but they lack formal guarantees and often rely on empirical heuristics.

Based on this the authors introduce PID for alignment, removing residual bias and mitigating overshoot when aligning.

Results demonstrate how PID control improves convergence and stability, reducing steady-state errors. Empirical experiments across multiple models (Gemma2, LLaMA3, Qwen2.5) and tasks (toxicity mitigation, jailbreak prevention, and image style control) show that PID Steering outperforms prior activation steering methods, achieving more consistent and interpretable control without harming overall model performance.

### Strengths
- The paper proposes an interesting methods of control-theoretic interpretation of activation steering.

- The proposed PID Steering introduces components that correct long-term drift and prevent oscillations making it robust.

- The approach is tested across diverse tasks (toxicity, jailbreaks, style transfer) and modalities (text and images), showing robustness and generality.

### Weaknesses
- Selecting appropriate PID gains (Kp, Ki, Kd) is nontrivial, and may result in suboptimal solutions.

- While results are broadly positive, the paper lacks detailed ablation studies on scenarios where PID might underperform or destabilize.

- Claims that PID Steering is lightweight are mentioned but not benchmarked in terms of latency or inference-time overhead.

- It is unclear to me how controller parameters generalize across model architectures or domains without retraining.

- The much smaller models such as Gemma is also used in this paper. Although, larger models (24B or 70B) are not tested, I was wondering if the authors could already make a size comparison between the models and present the observed differences?

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

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
This paper presents a novel perspective on activation steering for large language models and diffusion models by introducing a control-theoretic framework for managing model behavior. The paper provides theoretical analysis using input-to-state stability ISS and demonstrates empirical gains on toxicity reduction, jailbreak resistance, and style control tasks across several models.

### Strengths
1. It is novel to connect the control theory and activation steering in LLMs.
2. The paper provides a solid mathematical formalization of steering as a dynamic control process. 
3. Experiments are comprehensive, covering both text (toxicity, jailbreak) and vision (style transfer) domains.

### Weaknesses
1. Lack of baselines: The experiments compare primarily against the sequential steering vector ****baseline and, to a lesser extent, static activation addition. This is an insufficiently broad comparison. Many other steering methods, such as CAA and ITI, could serve as competitive baselines. 
2. The paper evaluates toxicity using LLaMA-3-8B both as the generator and as the evaluator, which introduces a self-evaluation bias: the same model family that produces text also judges its toxicity. Such evaluations are not independent and can underreport toxicity. A separate toxicity classifier such as GPT-4 or at least cross-model correlation analysis should be used for reliable measurement.
3. The paper does not report ablations on the $K_P, K_I, K_D$, though the performance of control systems is typically sensitive to them. Similarly, computational overhead, such as inference latency and memory increase is unreported, which is important for large models.
4. The theoretical analysis assumes locally linearized activation dynamics (Eq. 29–32). While this yields tractable ISS proofs, the approximation error may be significant in deep nonlinear transformers, especially under strong steering perturbations.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
