# ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Activation steering, or representation engineering, offers a lightweight approach to align large language models (LLMs) by manipulating their internal activations at inference time. However, current methods suffer from two key limitations: \textit{(i)} the lack of a unified theoretical framework for guiding the design of steering directions, and \textit{(ii)} an over-reliance on \textit{one-step steering} that fail to capture complex patterns of activation distributions. In this work, we propose a unified ordinary differential equations (ODEs)-based \textit{theoretical} framework for activation steering in LLM alignment. We show that conventional activation addition can be interpreted as a first-order approximation to the solution of an ODE. Based on this ODE perspective, identifying a steering direction becomes equivalent to designing a \textit{barrier function} from control theory. Derived from this framework, we introduce ODESteer, a kind of ODE-based steering guided by barrier functions, which shows \textit{empirical} advancement in LLM alignment. ODESteer identifies steering directions by defining the barrier function as the log-density ratio between positive and negative activations, and employs it to construct an ODE for \textit{multi-step and adaptive} steering. Compared to state-of-the-art activation steering methods, ODESteer achieves consistent empirical improvements on diverse LLM alignment benchmarks, a notable $5.7\%$ improvement over TruthfulQA, $2.5\%$ over UltraFeedback, and $2.4\%$ over RealToxicityPrompts. Our work establishes a principled new view of activation steering in LLM alignment by unifying its theoretical foundations via ODEs, and validating it empirically through the proposed ODESteer method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides a unified theoretical framework based on Ordinary Differential Equations (ODEs) for activation steering.

The authors show that conventional one-step activation addition, $\tilde{a} = a + T \cdot v(a)$, can be rigorously interpreted as the first-order Euler discretization of an ODE, $\dot{a}(t)=v(a(t))$. Within this framework, they argue that identifying the optimal steering direction $v(a)$ is equivalent to designing a barrier function, $h(a)$, which separates desirable and undesirable regions in the activation space, guiding the activations toward desirable regions while ensuring it remains there.

Building on this, the authors introduce BODES. BODES defines the barrier function $h(a)$ as the log-density ratio between positive and negative activations, utilizing nonlinear features (specifically, Polynomial Count Sketch) to capture complex patterns. By solving the ODE derived from the normalized gradient of this nonlinear barrier function, BODES performs multi-step, adaptive steering where the direction changes dynamically based on the current activation. Empirically, BODES achieves consistent improvements: 7% on TruthfulQA, 2% on RealToxicityPrompts, and 2% on UltraFeedback.

### Strengths
1. The theoretical framework is novel. Interpreting activation steering (a discrete intervention) as the numerical solution (Euler discretization) of a continuous Ordinary Differential Equation (ODE) is new and overcome the limitation of previous methods in one-step mapping. 

2. Empirical results are quite strong and significant. BODES achieves consistent improvements: 7% on TruthfulQA, 2% on RealToxicityPrompts, and 2% on UltraFeedback. Extensive experiments cover various LLMs and tasks.

3. The paper is clearly written.

### Weaknesses
1. Activation steering is valued for being "lightweight" and inference-time usable. BODES involves: (i) calculating nonlinear features (Polynomial Count Sketch), (ii) computing the Jacobian of this map, and (iii) running a 10-step numerical ODE solver at every generated token. While the components are individually efficient, the cumulative overhead might be substantial compared to a single vector addition (CAA) or matrix multiplication (SEA). Practically, it is beneficial to include a comparison of the average generation latency (e.g., tokens/second or wall-clock time per token) for BODES against the baselines (e.g., CAA or ITI compared in your paper).

CAA - Steering Llama 2 via Contrastive Activation Addition. ACL 2024.
SEA - Spectral editing of activation for LLM alignments. Neurips 2024.
ITI - Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. Neurips 2023

2. The paper attributes performance gains to "improved numerical accuracy" from multi-step ODE solving (Section 5.3) and states the Euler method is used for 10 steps (Appendix C.3). However, in addition to the ablation for single-step BODES vs BODES, it would strengthen the paper if you could ablate on the performacne vs number of steps. And, trying a higher-order numerical methods may further verify the performance gain sourced from the numeircal accuracy.

3. BODES has a strong theoratical solidness but it inevitably introduce addition complexit in inference-time compute and hyperparemeter search, e.g., strength hyperparams listed in Table 5 seems to be sensitive with model choice.

### Questions
1. Table 5 gives the range of $T$ used for different models, given that $T$ controls the overall steering strength and is a critical hyperparameter, could you provide a plot (similar to Figure 2 for layer selection) showing the metric of interest (e.g., Win (%) for UltraFeedback) as a function of $T$ for one model/task pair (e.g., LLaMA3.1-8B on UltraFeedback)? This would illustrate the sensitivity of BODES to the key strength hyperparameter $T$.

2. The optimal intervention layer is selected based on the peak performance of a linear, one-step method (CAA, Fig. 2). Was this layer selection validated for the multi-step, and nonlinear BODES method? Given the difference in steering dynamics, is it guaranteed that the optimal layer for a simple linear steer (CAA) is also optimal for ODE-based steering?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper “Activation Steering for LLM Alignment via a Unified ODE-Based Framework (BODES)” proposes a principled way to steer large language model (LLM) activations toward desirable behaviors without retraining. The authors reinterpret prior *activation addition* techniques (which simply add a fixed direction vector to hidden states) as a special case of an Ordinary Differential Equation (ODE) system.
They define an **activation flow**:

$$\dot{a}(t) = \nabla_a \log \frac{p_+(a)}{p_-(a)}$$

where $(p_+(a)) and (p_-(a))$ are activation densities of *desired* and *undesired* behaviors.
Integrating this ODE (via an adaptive solver) moves activations smoothly toward regions of high $(p_+(a))$ — that is, *truthful, helpful, or harmless* behavior — while guaranteeing stability and forward invariance (once inside the “good” region, activations remain there from the proposition 1).

Experiments on multiple open LLMs show small but consistent gains (≈2–7%) in truthfulness and safety benchmarks,

### Strengths
1. The paper introduces a theoretically grounded and unified ODE-based framework for activation steering.  
2. It enables adaptive, stable, and efficient alignment of LLMs without any retraining.  
3. The method shows consistent improvements across truthfulness and safety benchmarks

### Weaknesses
1. **Minimal performance improvement:** The reported gains (≈2–7%) are small and may not be statistically significant.
2. **Lack of clarity on \(p_+\) and \(p_-\):** The paper doesn’t specify how positive and negative activations are categorized or sampled.
3. **Unverified barrier property:** It is not shown that the learned barrier function \(h(a)\) satisfies Proposition $\(\dot{h}(a)=\nabla h(a)^\top v(a)>0\)$ in practice.
4. **No trajectory-level likelihood analysis:** The paper doesn’t measure or visualize how average likelihood (\(h(a)\)) changes along the ODE trajectory.

### Questions
1. How are $p_+$ and $p_-$ activations defined ?
2. Does activation steering actually increase the *average likelihood* along trajectories compared to unsteered activations?
3. Can the authors verify that the barrier condition \(\dot{h}(a) > 0\) holds for the learned \(h(a)\)?
4. Could this framework generalize to  alignment to reward-based alignment , comparison with self consistency

### Soundness
2

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
4

### Summary
This paper studies activation steering for large language models (LLMs) and reframes it as solving an ordinary differential equation (ODE). It shows that standard one-step activation addition can be viewed as the Euler step of this ODE, providing a unified perspective that connects both input reading and output optimization approaches via a barrier function over activations. The authors propose BODES, which learns a nonlinear log-density-ratio barrier between positive and negative activations using logistic regression on polynomial count-sketch features, and performs multi-step ODE integration along the barrier gradient for adaptive steering. Experiments on Falcon‑7B, Mistral‑7B, and Llama‑3.1‑8B show consistent improvements over one-step baselines, with average gains of about 7% on TruthfulQA and 2% on UltraFeedback and RealToxicityPrompts

### Strengths
1. The barrier-function perspective is well-grounded in control theory, offering a principled view that unifies existing steering methods.

2. The paper is clearly written, and empirical results are promising, showing consistent gains across multiple models and datasets.

### Weaknesses
1. The claimed advantage over output-optimization methods is debatable: both approaches rely on a learned scoring function—the proposed method’s barrier function also requires accurate estimation and introduces additional hyperparameters (e.g., step size, number of ODE steps, solver type, and polynomial sketch settings)


2. It is unclear whether the ODE formulation is necessary. Could similar results be achieved by taking several gradient ascent steps on the barrier function, which might be more efficient?

3. Inference efficiency is not discussed. Solving an ODE can be computationally expensive; a comparison of inference speed versus existing one-step or gradient-based steering methods would strengthen the paper.

4. How sensitive is performance to the ODE solver choice and step size? Does the performance degrade significantly if the ODE is approximated with fewer steps?

### Questions
See Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on activation steering. It innovatively proposes a unified ordinary differential equation (ODE)–based theoretical framework for activation steering, and introduces multi-step activation steering derived from this formulation. The approach outperforms traditional one-step steering methods across multiple models and tasks.

### Strengths
1. The proposed ODE-based multi-step activation steering is well-motivated and demonstrates a meaningful degree of novelty.


2. The experiments cover multiple models and multiple tasks, clearly showing that the proposed method outperforms one-step activation steering baselines.

### Weaknesses
1. The proposed method requires multi-step ODE integration, which introduces additional computational overhead during inference. The paper would benefit from a more detailed analysis and discussion of this extra cost.


2. Some implementation details are insufficiently specified. During training, the method relies on collecting positive and negative activations, but the paper does not clearly describe how these activations are extracted, particularly which token positions are used. Additionally, during inference, steering is applied at every decoding step, it is unclear whether training is consistent with this procedure, i.e., whether activations used for training were also collected across different token positions or only from specific tokens.


3. It is unclear how well the proposed method transfers across datasets or domains. Additional experiments demonstrating whether the proposed method beyond the datasets it was trained on would help clarify the applicability of the approach.


4. Since the paper primarily focuses on alignment tasks, it is important to assess whether the method degrades model utility. Evaluating performance on general-purpose benchmarks such as MMLU would allow for a clearer understanding of whether the proposed steering affects core model capabilities.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
3
