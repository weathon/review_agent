# Lyapunov Guidance: Stabilizing Generative Flows with One-Line Code

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Flow matching has recently emerged as a powerful approach to learning complex data distributions with excellent performance across diverse generative tasks, yet adapting pre-trained flow models to new tasks typically requires costly retraining. To mitigate this issue, post-training guidance methods were proposed as they are lightweight and user-friendly for downstream applications. However, existing guidance methods are unreliable since they usually rely on function approximations and lack structural guarantees of sampling stability. In this paper, we address this challenge by proposing a unified framework, LyaGuide (Lyapunov Guidance for flow matching), which reformulates the guidance in flow matching as a Lyapunov control problem. LyaGuide supports two modes depending on whether the Lyapunov function is a known priori: a model-driven mode for developer-oriented scenarios where the guidance distribution is explicitly specified, and a data-driven mode for user-oriented scenarios where pre-trained models can be adapted with downstream task-specific data. Furthermore, to enforce the stability, we introduce a pseudo projection operator with a closed-form expression that strictly satisfies the Lyapunov condition. Notably, LyaGuide is compatible with any guidance method and can be implemented with a single line of code. Experiments on synthetic datasets and image inverse problems demonstrate that our framework consistently improves sample quality and guidance fidelity while preserving efficiency, and it significantly enhances the performance of existing guidance methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a unified framework, Lyapunov Guidance for flow matching (LyaGuide), which reformulates the guidance in flow matching as a Lyapunov control problem. Based on this framework, the paper designs a pseudo-projection operator with a closed-form expression that strictly satisfies the Lyapunov condition. The experiments demonstrate the effectiveness of the proposed method.

### Strengths
- This paper offers a new framework or perspective for the flow matching guidance problem.
- Based on the stability theory in the controlled system, the paper proposes a projection operation for more accurate guidance.
- The method can be applied to broad scenarios: both explicit and implicit prior knowledge.

### Weaknesses
- l146: What does this sentence 'c is a control term derived from V' mean? Or, what relationship is 'derived from'? This is not clear here, although later there are detailed explanations. It will be better to explain it here.
- The caption of Fig.2 is unclear. Maybe change it to 'Illustration of the pseudo projection $\pi$ and exact projection $\pi^*$? Additionally, the meaning of the points with different colors in this figure is not clear.
- How does the sampling step of flow matching influence the performance of LyaGuide?

### Questions
- l130: Why is designing the stabilizing controller u(x) a major problem in cybernetics field? Or, why is the stable u(x) better? It needs a more intuitive explanation for easier understanding.
- l350: Why should V be locally convex around task-relevant regions? Proposition 3.1 does not include such assumptions. Also, why does the importance sampling weight promote the convexity around high-score samples? And if the locally convexity is needed, what about section 5.1? Can V be locally convex in this setting?
- l366: what is $g_ϕ$? This may need a brief introduction.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces LyaGuide, a Lyapunov function-based method to enhance guidance in flow matching generative models, addressing inefficiencies in adapting pre-trained models to new tasks (e.g., inverse problems like image restoration). Drawing from control theory, it provides a unified, stable, and theoretically guaranteed approach that integrates with existing methods, requires minimal code changes, and improves reliability for applications in drug design, image editing, and beyond.

### Strengths
1. It's an interesting (and the first, to the knowledge of the reviewer) attempt to reformulate the guidance problem in generative models as a control stability problem, and studying it with Lyapunov functions.
2. The proposed theoretical framework includes different guidance techniques, including EBM guidance (since the guidance is the gradient of a time-invariant potential) and posterior sampling in diffusion models and flow matching.

### Weaknesses
1. The presentation can be improved. For example, a more detailed discussion and intuitive understanding of the equivalence between guidance and Lyapunov stability in the main text would be beneficial.
2. The experimental evaluation is relatively limited. E.g., only flow matching on an image inverse problem is evaluated, whereas the authors claim that the framework is generally applicable to EBM and various tasks. Surely, the contribution of this work is largely theoretical, but the soundness would be improved with more empirical evidence.

### Questions
1. In my understanding, Theorem A.2.1 is central to the contributions. However, how the equivalence between Lyapunov stability and guidance is established can be presented more clearly. In my understanding, the proof shows that a locally Lyapunov stable control can be "deformed" to the desired guidance control, but it does not show how it can be deformed. Lastly, it is not entirely clear how the proposed method, after the control is projected to be Lyapunov stable, deforms the stable control to match the desired guidance control.
2. What is causing the Lypapunov guidance results to still be different from ground truth?

I am willing to raise the score if the questions are answered and the concerns are addressed.

Minor:
Line 227 guidnace -> guidance

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces LyaGuide, a framework that unifies various flow-matching guidance methods (e.g., classifier guidance, energy-based guidance, reward guidance) under the theoretical perspective of Lyapunov control. The authors propose interpreting the energy function in generative models as a Lyapunov function and the guidance term as a stabilizing control input. To ensure stability, they introduce a pseudo projection operator that enforces the Lyapunov condition in closed form, claiming compatibility with existing methods and implementation simplicity (“one-line code”). Experiments on synthetic datasets and image inverse problems (inpainting, super-resolution, deblurring) show improvements in stability and performance over baseline guidance methods.

### Strengths
- Unifies diverse guidance techniques in generative modeling under Lyapunov control theory, offering a new theoretical lens.

- The pseudo-projection operator provides a lightweight, closed-form correction that can be easily integrated into existing models.

- The framework can wrap around multiple existing guidance methods, making it versatile.

- Synthetic and image inverse experiments show modest but consistent improvements in stability and quality.

### Weaknesses
- Several stability conditions are unverified or incorrectly generalized from local to global settings.

- Quantitative improvements (e.g., in Fig. 3 and Table 1) are marginal, and visual results show minor differences; the proposed method mainly stabilizes trajectories rather than improving fidelity substantially.

- Fig. 3’s third row labeling appears incorrect, and the “toy example” results do not clearly demonstrate meaningful gains.

- Lyapunov functions ensure local minima convergence, not global optimization, limiting practical utility.

- The phrase “rigorous stability guarantees” is misleading given the local and heuristic nature of the verification.

### Questions
- How can one verify that the proposed $V(x)$ satisfies Lyapunov’s positive definiteness and negative derivative conditions in practice?

- What is the basin of attraction or stable manifold for convergence, and how can users determine if $x_0$ lies within it?

- Does the pseudo-projection operator guarantee convergence when $V(x)$ is non-convex or multimodal?

- How sensitive is the method to scaling parameters $(δ, k)$ in the projection and control terms?

- Line 185 has a typo "exists a a continuously ".

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces LyaGuide, a novel framework that unifies various post-training guidance methods for flow-matching models using Lyapunov control theory. The central idea is to view the guidance process in generative modeling as a stabilization problem: the guidance term acts as a control input ensuring the system’s convergence toward a desired distribution.

### Strengths
1. The theoretical framework is sound and general, unifying several different scenarios of guidance in flow-based model.

### Weaknesses
1, Experiments are neither sufficient. Both reward-guided scenario (scenario 1) and prior-knowledge one (scenario 2) are equipped with highly complete empirical benchmarks, for example, GenEval for scenario 1 and normal image generation for scenario 2. The effectiveness of the guidance method should be evaluated on this well-established benchmarks.

2, Baselines are missing. Since both RLHF and prior-knowledge guidance are included in the theoretical framework, baselines for these two tasks should be then considered, like Flow-GRPO for RLHF and AutoGuidance (it also works on flow matching models empirically).

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
2
