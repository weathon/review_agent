# SP-VLA: A Joint Model Scheduling and Token Pruning Approach for VLA Model Acceleration

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Vision-Language-Action (VLA) models have attracted increasing attention for their strong control capabilities. However, their high computational cost and low execution frequency hinder their suitability for real-time tasks such as robotic manipulation and autonomous navigation. Existing VLA acceleration methods primarily focus on structural optimization, overlooking the fact that these models operate in sequential decision-making environments. As a result, temporal redundancy in sequential action generation and spatial redundancy in visual input remain unaddressed. To this end, we propose SP-VLA, a unified framework that accelerates VLA models by jointly scheduling models and pruning tokens. Specifically, we design an action-aware model scheduling mechanism that reduces temporal redundancy by dynamically switching between VLA model and a lightweight generator.  Inspired by the human motion pattern of focusing on key decision points while relying on intuition for other actions, we categorize VLA actions into deliberative and intuitive, assigning the former to the VLA model and the latter to the lightweight generator, enabling frequency-adaptive execution through collaborative model scheduling. To address spatial redundancy, we further develop a spatio-semantic dual-aware token pruning method.  Tokens are classified into spatial and semantic types and pruned based on their dual-aware importance to accelerate VLA inference. These two mechanisms work jointly to guide the VLA in focusing on critical actions and salient visual information, achieving effective acceleration while maintaining high accuracy. Extensive experiments show that our method achieves 1.5$\times$ lossless acceleration in LIBERO and 2.4$\times$ in SimplerEnv, with up to 6\% average performance gain. Inference frequency and latency improve by 2.2$\times$ in SimplerEnv and 1.4$\times$ in LIBERO.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes SP-VLA, a unified acceleration framework for Vision-Language-Action (VLA) models.
It combines:(1) Model Scheduling – dynamically switching between a full-scale VLA model and a lightweight ridge-regression-based generator depending on whether the action is “deliberative” or “intuitive,” to reduce temporal redundancy; and(2) Spatio-Semantic Token Pruning – pruning vision tokens using accumulated attention scores and Canny-edge information to remove spatial redundancy. Experiments on LIBERO and SimplerEnv demonstrate 1.5–2.4× inference acceleration with nearly lossless task performance, and improved inference frequency and latency.

### Strengths
1. **Novel and conceptually clear idea**

The paper is among the first to jointly address temporal and spatial redundancies in VLA models.
The analogy to human dual-process control (deliberative vs. intuitive actions) is original and intuitively compelling, offering a behavioral perspective on model efficiency.

2. **Comprehensive empirical validation**

Results are reported across multiple VLA backbones (OpenVLA, CogACT) and environments (LIBERO, SimplerEnv) with consistent performance gains.
Ablation studies, sensitivity analyses, and visualizations are well-structured and strengthen empirical credibility.

### Weaknesses
1. **Over-reliance on handcrafted heuristics**

Both the scheduling and pruning modules rely on manually designed heuristics—velocity thresholds, ridge regression fitting, Canny edges, and fixed attention thresholds—rather than adaptive or learnable components. This constrains scalability and contrasts with current trends toward learned compression/scheduling in multimodal LLM research.

2. **Generalization and robustness not sufficiently validated**

The method’s stability under varying architectures, task types, or sensor conditions is not explored. Key hyperparameters (speed thresholds, pruning ratios, deliberative/intuitive ratio) might require re-tuning, reducing transferability. A cross-task or cross-model evaluation would greatly enhance the paper’s impact.

### Questions
1. Could you explain how the main hyperparameters (e.g., speed thresholds, pruning ratios, buffer size) were determined in practice?
Were they tuned by trial-and-error, grid search, or guided by theoretical intuition or prior experiments? Are these parameter settings consistent across tasks and environments, or do they need adjustment for each scenario? 

2. Have you attempted to deploy SP-VLA on a physical robotic platform, or at least measured whether the acceleration results reported in simulation differ from those on real hardware?

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
SP-VLA accelerates Vision-Language-Action by (i) temporal scheduling between a full VLA policy and a lightweight ridge-based action extrapolator triggered by speed/gating signals, and (ii) order-preserving token pruning that keeps both semantic (accumulated attention) and spatial (Canny edges) tokens with a speed-adaptive keep ratio. Across simulated suites and backbones, it reports considerable speedup without accuracy loss.

### Strengths
1. Interesting idea that targets time + space waste; easy to plug into different VLA stacks.

2. Solid gains with small or no accuracy drop; ablations back up the design.

3. Explains why semantics-only pruning breaks VLA (loses spatial order).

### Weaknesses
1. Relies on a few heuristic knobs (speed window, buffer length, gating τ); some sensitivity.

2. The lightweight head assumes near-linear short-horizon motion; can fail under contact/perturbations.

3. Canny edge detection can be fragile in the presence of lighting and material noise. Additionally, there is limited evidence from real-robot experiments and little information on tail latency and energy consumption.

### Questions
Overall, this paper hits a real pain point in VLA—wasted compute across time and space—and the solution is practical enough to drop into existing stacks. I have a few comments and questions as follow.

How do you handle switch latency and thrashing between models? What’s the impact on P95/P99 latency?

How much of the speedup remains if you add tail-latency limits or minimum dwell times between switches?

How often do you fall back to the full VLA under strong nonlinearity, and how quickly does the system recover from a bad extrapolation?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces SP-VLA, a unified framework for accelerating Vision-Language-Action (VLA) models. The authors identify two main sources of redundancy in VLA models: temporal redundancy in action generation and spatial redundancy in visual input. To reduce temporal redundancy, the authors distinguish between deliberative actions, which are handled by the full VLA model, and intuitive actions, approximated by a lightweight generator based on ridge regression. To reduce spatial redundancy, the method prunes visual tokens before feeding them into the language model, by combining semantic attention scores and Canny edge detection. The approach is validated on LIBERO and SimplerEnv, showing latency and control frequency improvements, while maintaining high performance.

### Strengths
* **Perception tokens pruning**: the spatio-semantic dual-aware token pruning strategy is well-motivated and empirically validated. By combining semantic attention with spatial cues, the method preserves critical spatial information achieves faster inference without significant accuracy loss.
* **Comprehensive experiments**: The paper provides extensive experiments on standard benchmarks (LIBERO, SimplerEnv), including ablation studies and comparisons to relevant baselines.

### Weaknesses
* **Loosely-connected contributions**: The paper presents two main ideas (action scheduling and token pruning) that are only loosely connected, apart from the common goal.
* **Limited generality**: the introduced novelties are "simpler" than the VLA model and they introduce a new set of hyperparameters, which hinders generality of the approach. This limitation particularly affects the action part, where different embodiments may have very different action spaces and thus, defining hyperparameters might be more difficult

### Questions
* How would the method work in conjunction with different action prediction heads? E.g. diffusion / flow matching policies
* How can the method deal with multiple cameras?
* How can we train the model to handle multiple embodiments (i.e. robotic setups) at once?

There are several typos in the paper:
* “navigatio and medical robotics” → should be “navigation and medical robotics”.
* “Accleration for Vision-Language-Action Models.” → should be “Acceleration for Vision-Language-Action Models.”
* “LANTENCY AND FREQUENCY” (Appendix) → should be “LATENCY AND FREQUENCY”.
and many more (including "Ride Regression" in Figure 3)

I would recommend the authors to use a grammar checker and thoroughly check their manuscript.

### Soundness
3

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
The paper presents a model scheduling and token pruning approach to make VLA. The main idea is categorizing VLA actions as deliberative and intuitive and then using a smaller model for the intuitive actions. Additionally the paper also incorporates a token pruning approach where tokens are pruned based on semantic and spatial relevance to the task. The paper shows that jointly applying both model scheduling and token pruning helps achieve significant improvements in VLA inference.

### Strengths
1. The paper presents a simple method to speed VLA inference based on a simple observation that most actions predicted by the VLA do not require the full processing power of a VLA (intuitive actions) and can rather be estimated using a super simple ridge regression. 
2. The method shows significant improvement in run-time of VLA on 2 simulated benchmarks.

### Weaknesses
***1. How are hyper-parameters chosen?***

My biggest concern with this paper is that performance of both model scheduling and token pruning are heavily reliant on the hyper-parameters and there is not much detail on how they are chosen. 

Are these simulator/task-dependent? In which case, do you select them on a subset of data and then evaluate on a larger set to see if they transfer?

How can these parameters be selected for real-world tasks where evaluation is not cheap to run and each task might require a different set of hyper-parameters?



***2. Why does performance improve with SP-VLA?***

The paper does not explain much how/why (they mention error correction which is not explained well) performance would improve when using model scheduling and token-pruning where intuitively one would think that performance should drop (at least slightly as we are pruning some information).


***3. No real-world results?***

It would be interesting to see how the method holds in real-world evaluations. It would also be interesting to see if there’s an efficient way to estimate the hyper-parameters required for real-world tasks or if the hyper-parameters discovered in simulation transfer to the real-world setting as well.

### Questions
Please see the weaknesses section for my questions. The paper presents a nice idea for speeding up VLA inference however, I have concerns regarding the hyper-parameters selection and its transfer to real-world setting therefore I’m leaning towards a reject for now. Will be happy to change my mind if the authors can answer questions and other reviewers do not bring major concerns that I might have missed.

### Soundness
3

### Presentation
3

### Contribution
2
