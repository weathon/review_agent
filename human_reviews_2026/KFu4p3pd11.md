# Masked Generative Policy for Robotic Control

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
We present Masked Generative Policy (MGP), a novel framework for visuomotor imitation learning. We represent actions as discrete tokens, and train a conditional masked transformer that generates tokens in parallel and then rapidly refines only low-confidence tokens. We further propose two new sampling paradigms: MGP-Short, which performs parallel masked generation with score-based refinement for Markovian tasks, and MGP-Long, which predicts full trajectories in a single pass and dynamically refines low-confidence action tokens based on new observations. With globally coherent prediction and robust adaptive execution capabilities, MGP-Long enables reliable control on complex and non-Markovian tasks that prior methods struggle with. Extensive evaluations on 150 robotic manipulation tasks spanning the Meta-World and LIBERO benchmarks show that MGP achieves both rapid inference and superior success rates compared to state-of-the-art diffusion and autoregressive policies. Specifically, MGP increases the average success rate by 9\% across 150 tasks while cutting per-sequence inference time by up to 35×. It further improves the average success rate by 60\% in dynamic and missing-observation environments, and solves two non-Markovian scenarios where other state-of-the-art methods fail.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces Masked Generative Policy, which is a new framework for visuomotor imitation learning that models robot actions as discrete tokens and leverages masked generative transformers to efficiently generate and refine action sequences. Unlike autoregressive or diffusion generative policies, MGP tries to generate globally coherent future plans and refine them online. It combines MaskGIT-style generation with robotic action modeling. The experimental results demonstrate state-of-the-art performance on Markovian and non-Markovian control.

### Strengths
- It reframes the policy generation problem as masked generative modeling is new and practical, especially given the latency and horizon challenges in robotics. The tokenization of actions is smart to allow transformer modeling of full sequences.
- The global coherence maintains long-horizon consistency through token memory. The parallel sampling and selective refinement drastically cut latency, leading to high inference efficiency.
- The experimental results are comprehensive and demonstrate the effectiveness of the proposed method across simulations and tasks. While diffusion models model smooth distributions and autoregressive models enforce causality, MGP smartly bridges them using mask-and-refine semantics, achieving both speed and robustness.

### Weaknesses
- The system design and two-stage training are complex. The VQ-VAE and MGT pipeline introduces extra overhead and possible distribution shift between discrete tokens and true continuous actions.
- When predicting all tokens at once, it loses the explicit notion of conditioning the next tokens on the current action. In dynamic control, this can lead to physically inconsistent predictions.
- The model must have enough context to predict consistent future tokens without sequential conditioning. It could work in structured simulation, but may fail with partial observability or noisy real-world sensors where causality exists.

### Questions
- Is the pipeline easy to smoothly transfer to real-world tasks? The robustness to sensor noise, delays, or physical contact uncertainty remains a question, especially when it requires a strong encoder and global context.
- There were few visual rollouts or per-task failure analyses. How token refinement behaves in specific dynamic scenes could be more illustrative.
- A discussion section on the potential domain mismatch and increased complexity of the proposed two-stage training is helpful.

### Soundness
3

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
3

### Summary
This paper introduces Masked Generative Policy (MGP), a new visuomotor imitation learning framework that models robot control as a masked token-generation problem.
MGP first discretizes continuous actions with a VQ-VAE tokenizer, then trains a masked generative transformer (MGT) to reconstruct full action sequences from partially masked tokens conditioned on current observations.

Two inference paradigms are proposed:

- MGP-Short for Markovian, short-horizon tasks: parallel token generation with one or two score-based refinement steps.

- MGP-Long for non-Markovian, long-horizon tasks: predicts the entire trajectory in one pass and adaptively refines uncertain future tokens through posterior-confidence estimation (PCE) as new observations arrive.

Extensive experiments on Meta-World and LIBERO benchmarks show strong gains—up to 35× faster inference and higher success rates (+9% overall, +60% in dynamic or missing-observation settings).
Ablations (MGP-FullSeq, MGP-w/o-SM) validate that PCE-based selective refinement is critical for efficiency and global coherence.

### Strengths
Original idea: creatively transfers masked-generation paradigms (MaskGIT/MUSE) to robotic action synthesis.

Technical soundness: clearly defined VQ-VAE tokenizer, transformer conditioning, and confidence-guided refinement loop.

Empirical rigor: evaluated on 150+ tasks across difficulty levels; includes robustness tests (dynamic, missing-observation, non-Markovian).

Fair comparison: benchmarks against continuous-action (diffusion/flow) and discrete-token baselines under identical encoders and demos.

Ablation insight: MGP-w/o-SM (without score-based masking) confirms that selective refinement improves both efficiency and success rate.

Relevance: unifies the advantages of diffusion (sample quality) and autoregressive (temporal coherence) methods in a parallelizable design.

### Weaknesses
Limited analysis of tokenizer sensitivity: performance may depend on the VQ-VAE codebook design, but this is not explored.

Hyperparameter transparency: the exact confidence-masking threshold and its effect on refinement stability are not analyzed.

Potential complexity: the two-stage training (tokenizer + policy) increases implementation effort; joint end-to-end training would strengthen the approach.

### Questions
How is the confidence-based masking threshold determined? Fixed ratio or adaptive per step?

Does the posterior-confidence estimation ever over-mask or destabilize refinement when confidence calibration drifts?

How sensitive is performance to the tokenizer’s codebook size and discretization granularity?

Would an end-to-end jointly trained transformer + VQ-VAE outperform the current two-stage pipeline?

Discrete tokens normally introduce information loss—what do the authors believe enables MGP’s discrete representation to outperform continuous-action models like Diffusion Policy? Is it the global trajectory modeling, masked refinement dynamics, or some property of the VQ-VAE discretization?

### Soundness
3

### Presentation
3

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
This manuscript proposes a novel imitation learning framework for learning visuomotor policy parameterized by masked generative transformer (MGT), which enables high inference efficiency for closed-loop control while maintaining robustness in long-horizon and non-Markovian tasks. Specifically, two sampling strategies are designed: (1) MGP-Short performs short-horizon sampling and refines action tokens with few iterations for the best performance-efficiency trade-off in Markovian tasks; and (2) MGP-Long samples the full trajectory and adaptively refines tokens with updated observations from the environment to retain global coherence. Experiments demonstrate the strong performance of the proposed methods in Markovian and more challenging tasks.

### Strengths
- Unlike diffusion-based policy, which might require external distillation for fast inference speed, MGP puts less stress on iterative sampling for obtaining clean actions, and has high flexibility of test-time adjustment with proposed sampling strategies.
- MGP-Long iteratively refines the action tokens using the executed actions along with the updated observation to improve trajectory-level coherence, which achieves strong performance in Non-Markovian and dynamic environments, and remains robust to missing observations

### Weaknesses
- Baselines such as diffusion-based policies (e.g. ) as well as VQ-BeT stand out when learning multimodal action distributions, while MGP is also built on top of vector quantization, it is not yet clear how the proposed sampling methods work on tasks with explicit multimodality
- As all tokens are predicted in parallel, the refinement process can be affected if there are low-quality actions predicted initially with high confidence, causing error accumulation throughout the following iterations. Furthermore, it would be helpful to extend the first ablation studies to investigate how many performance gains can be obtained from more refinement steps, especially in more challenging environments.
- Please include standard deviations in the table for thoroughness if multiple seeds are used to aggregate the result.

### Questions
- Typo: “blcoks” -> “blocks” in line 191
- In Figure 3, should the unexecuted token “52” at the bottom left be “53” before Posterior-Confidence Estimation
- In line 269, the authors mentioned four ablation studies were conducted, but in section 4.5, only three of them are elaborated.
- How many actions are encoded into one discrete token? And would that hyperparameter affect performance on different tasks?

### Soundness
3

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
4

### Summary
This paper introduces the Masked Generative Policy (MGP), a novel framework for robot imitation learning that eliminates the inference bottlenecks of diffusion models and the sequential constraints of autoregressive models.
MGP-Short is specifically designed for Markovian tasks, adapting the masked generative transformer for short-horizon sampling. It demonstrates improved success rates on standard benchmarks while significantly reducing inference time.
MGP-Long allows for globally coherent predictions over long horizons, enabling dynamic adaptation, robust execution under partial observability, and efficient, flexible execution. It achieves state-of-the-art results in dynamic, observation-missing, and non-Markovian long-duration environments.
The authors validated the effectiveness of MGP in multiple simulated environments.

### Strengths
The authors conducted a thorough analysis of current action generation methods and proposed MGP to address the latency issues inherent in diffusion-style or autoregressive-style action generation. The paper is clearly articulated and easy to follow. The concept of using MGP to re-predict tokens with low confidence while maintaining those with high confidence is intriguing. Theoretically, this approach could indeed reduce the time consumed in predicting actions.

### Weaknesses
1. I acknowledge that the results in the simulated environment are impressive. However, due to the sim-to-real gap, it is often necessary to demonstrate effectiveness in real-world settings within this field.

2. Regarding the confidence score. Could you analyze the situations that might lead to a lower confidence score? Additionally, how can we ensure the accuracy of the confidence score itself?

3. About the MGP-Long settings. In long sequences, certain objects may cause environmental changes due to previous actions. At this point, the predictions may no longer remain globally coherent, and we would need to generate a new action sequence based on the changed objects.

### Questions
1. The results in the simulated environment are impressive. However, due to the sim-to-real gap, it is often necessary to demonstrate effectiveness in real-world settings within this field. 
2. How can we ensure the accuracy of the confidence score itself?
3. Regarding the MGP-Long settings: In lengthy sequences, some objects may lead to environmental changes as a result of prior actions. When this occurs, the predictions may lose their overall coherence, necessitating the generation of a new action sequence that takes into account the modified objects.

### Soundness
3

### Presentation
3

### Contribution
3
