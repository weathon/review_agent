# WorldGym: World Model as An Environment for Policy Evaluation

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Evaluating robot control policies is difficult: real-world testing is costly, and handcrafted simulators require manual effort to improve in realism and generality. We propose a world-model-based policy evaluation environment (WorldGym), an autoregressive, action-conditioned video generation model which serves as a proxy to real world environments. Policies are evaluated via Monte Carlo rollouts in the world model, with a vision-language model providing rewards. We evaluate a set of VLA-based real-robot policies in the world model using only initial frames from real robots, and show that policy success rates within the world model highly correlate with real-world success rates. Moreoever, we show that WorldGym is able to preserve relative policy rankings across different policy versions, sizes, and training checkpoints. Due to requiring only a single start frame as input, the world model further enables efficient evaluation of robot policies' generalization ability on novel tasks and environments. We find that modern VLA-based robot policies still struggle to distinguish object shapes and can become distracted by adversarial facades of objects. While generating highly realistic object interaction remains challenging, WorldGym faithfully emulates robot motions and offers a practical starting point for safe and reproducible policy evaluation before deployment.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a framework to train a robot control sequence conditioned diffusion model to produce images. The model serves as a virtual world where robot policies can be validated in manipulation tasks. The language model can be used to provide rewards for the evaluation. In the experimental part it is shown that the model provides evaluations that are consistent with real-world evaluations. At its best the model can be used in training better models for real world tasks.

### Strengths
Since the diffusion models can provide realistic looking images and through language (or rather tokens) can be used to condition image generation, it is an interesting path of research for robot learning to use such models as components in robot learning. This idea is not new, but the implementation of this work is rather good and it can provide interesting avenues for the future research. I also believe it could be useful for offline RL.

### Weaknesses
**Major:**

**Moderate:**

 - The qualitative example sequences in Figure 2 are so short that it is barely visible something happens - a longer sequence from task start to task finish would be more interesting to see

 - More details about how much training data is needed to train the model. This could be in the terms of how many hours of video and how many episodes used, how the position of camera (view) affects the results (how much variation was allowed) => for the camera-ready you could add even more discussion about the limitations observed

 - Also some details about the computing setup and training times so that we know if this is doable in a research laboratory

 - Some example images of the out-of-dataset tasks would be interesting to see (i.e. how different they actually are)

**Minor:**

### Questions
I think your work is good and everything is explained in the level the results are replicable. I wish the code and pre-trained models are published since I am sure there are many technical details missing but important for those who want to replicate them.

I am sure the model has limitations, but I also find that it can be used in coarse evaluation of policies trained and perhaps to debug errors during development. I find it valuable and therefore important to be published.

It would be interesting to see how well the diffusion model can generalize given more training data (diversity + amount) and is there some kind of correlation with the policy i.e. could it be used to predict how much training data is needed to be able to learn a policy. And does diffusion model training need more data than learning a well working policy?

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
This paper introduces WorldGym, a framework that uses a video-based world model as an environment for evaluating robot policies. The model predicts future visual outcomes from actions and uses GPT-4o as a semantic reward evaluator, showing strong correlation with real-world policy performance.

### Strengths
The paper introduces a clear and innovative use of video diffusion world models for policy evaluation instead of training, which reframes how offline policy analysis can be done without physical robots. The experiments show consistent correlation between simulated and real results, maintain relative performance rankings across models, and include OOD tests that reveal model weaknesses. Technically, the framework is efficient, combining causal temporal attention and adaptive horizon prediction to support different policy granularities while keeping inference cost reasonable.

### Weaknesses
I believe a more convincing way to show WorldGym’s effectiveness would be to perform reinforcement learning with WorldGym as the environment and then test the resulting policy in simulation or the real world, but the paper doesn’t do that.

### Questions
Have the authors attempted to perform reinforcement learning using WorldGym as the training environment? If not, are there plans to test such sim-to-sim or sim-to-real transfer in future work?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes WorldGym: a learned, action-conditioned video world model used as a unified environment to offline-evaluate robot policies. From a single real initial frame and a policy’s action chunk, the model predicts future frames; a VLM grader turns those into success scores. A key systems choice aligns the prediction horizon with each policy’s action-chunk size to reduce wasted rollout compute. Experiments report strong sim-to-real correlation of success rates, preserved policy rankings across families/sizes/checkpoints, and targeted OOD probes that reveal failure modes. The setup is framed as OPE with learned dynamics and learned reward from videos.

### Strengths
1. Originality: reframes policy evaluation as “rollout in one learned world” rather than per-task simulators; leverages the one-world prior and diverse training data. 
2. Practicality: one real frame + actions, no hand-coded simulators; horizon–chunk alignment is a clean trick that supports mixed policies while saving compute.
3. Clarity: the OPE formulation and rollout protocol are easy to follow; model/policy interfaces are explicit.
4. Significance: high sim-to-real correlation and preserved rankings make this a plausible tool for fast model selection; OOD edits provide actionable diagnostics.

### Weaknesses
1. VLM reward calibration is under-analyzed: the proposed VLM grader is central, but the paper does not show reliability audits (human agreement, prompt/temperature sensitivity, temporal credit). The authors should add thorough calibration and robustness studies.
2. Dynamics fidelity over long horizons is not quantified: the work shows plausibility, but does not report compounding-error metrics (e.g., FVD/LPIPS vs time, controllability under action perturbations). The authors should measure error growth and contact realism.
3. Statistics are light: correlations are promising, but the paper does not provide per-task CIs, bootstrap uncertainty, or stress tests under harder domain shifts (lighting, camera pose, clutter). Add stronger stats and broader shifts.
4. Single-frame initialization sensitivity is unclear: the paper does not analyze bias/leakage from the initial frame. Evaluate with occlusions/crops/background swaps to quantify sensitivity.
5. Efficiency claims are not profiled: horizon–chunk alignment is appealing, but the paper does not show throughput/latency/memory vs baselines across GPUs and chunk lengths. Provide wall-clock profiles and ablations (e.g., with/without diffusion forcing).
6. Related-work positioning lacks shared benchmarks: there is no head-to-head vs closest OPE/world-eval baselines on common tasks. Add small shared batteries and ranking-agreement metrics.
7. OOD analysis is narrow: findings hinge on a few edit types; generality across textures, shapes, adversarial overlays is unknown. Expand OOD suite and report policy-wise degradations with uncertainty.

### Questions
Same in weakness.

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
4

### Summary
This paper introduces WorldGym, a world-model-based environment for evaluating robot control policies without physical deployment. The approach uses an autoregressive, action-conditioned video generation model to simulate robot interactions, with a VLM (GPT-4o) providing task success evaluation. The authors demonstrate strong correlation (r=0.78) between world model success rates and real-world performance across three VLA policies (RT-1-X, Octo, OpenVLA) on Bridge manipulation tasks. WorldGym enables efficient evaluation on OOD tasks and environments through language/image modifications, revealing interesting policy failure modes.

### Strengths
- WorldGym proposes a way to address a genuine need for safe, reproducible, and cost-effective policy testing before real-world deployment.
- WorldGym shows impressive correlation between simulated and real-world success rates. It also does strong empirical validation.
- The single world model generalizes across diverse tasks and environments.
- The paper is well-written, the problem motivation is compelling, and the approach is clearly described.

### Weaknesses
- The paper does provide quantitative metrics on world model prediction quality.
- The world model still shows physics-inplausible predictions, as shown in Figure 14. The paper also does not propose any method to help the video model learn real-world physics.
- The paper does not compare with baselines like building digital twins to do policy evaluation.
- The paper does not show the computational requirements and the inference speed of the world model.
- The paper does not show qualitative results on the simulation and real-world correlation. For example, can authors provide the visualizations of model success and failure cases to see if WorldGym has similar visual renderings compared with the real world? Or providing a fixed set of action replay in both the real world and WorldGym is also meaningful to see the world model gap.

### Questions
See weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3
