# MeanCache: From Instantaneous to Average Velocity for Accelerating Flow Matching Inference

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
We present MeanCache, a training-free caching framework for efficient Flow Matching inference. Existing caching methods reduce redundant computation but typically rely on instantaneous velocity information (e.g., feature caching), which often leads to severe trajectory deviations and error accumulation under high acceleration ratios. MeanCache introduces an average-velocity perspective: by leveraging cached Jacobian--vector products (JVP) to construct interval average velocities from instantaneous velocities, it effectively mitigates local error accumulation. To further improve cache timing and JVP reuse stability, we develop a trajectory-stability scheduling strategy as a practical tool, employing a Peak-Suppressed Shortest Path under budget constraints to determine the schedule. Experiments on FLUX.1, Qwen-Image, and HunyuanVideo demonstrate that MeanCache achieves $4.12\times$, $4.56\times$, and $3.59\times$ acceleration, respectively, while consistently outperforming state-of-the-art caching baselines in generation quality. We believe this simple yet effective approach provides a new perspective for Flow Matching inference and will inspire further exploration of stability-driven acceleration in commercial-scale generative models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces MeanCache, a training-free caching framework designed to accelerate flow model inference in large-scale generative models. Traditional caching methods reuse intermediate representations to save computation but rely on instantaneous velocity fields, which lead to error accumulation. MeanCache addresses this by reframing caching in the average-velocity domain, inspired by the MeanFlow formulation.

### Strengths
Overall, the paper’s motivation and structure are sound. I was able to follow the authors’ intuitions and main ideas, though the mathematical formulations lack sufficient explanation and rigor (see weaknesses). Additionally, I appreciate the comprehensive empirical evaluation on large-scale generative models.

### Weaknesses
The biggest concern I have is the claim that MeanFlow-based caching is more robust than caching in Flow Matching. Although the authors provide an explanatory figure (Fig. 3), this is not convincing, since error accumulation can also be observed in MeanFlow. Can the authors provide high-level intuition and mathematical proofs to justify this claim? Additionally, MeanFlow is an existing idea. The main contributions of the paper come from implementing feature caching within the MeanFlow model. To assess the core contribution, I would like to see how vanilla MeanFlow (without caching) performs. This would help isolate and evaluate the effects of the proposed method more clearly. 


I find some mathematical formulations and claims inaccurate and presented without sufficient explanation.
First, the JVP approximation error can be large when $t - r \gg \delta$. Can the authors show whether this error is negligible (or quantify it)?
Second, the metric used in Eq. (13) depends on timestep discretization, which can vary across applications and the number of sampling steps. This is also connected to my first point, since $u(z_s, t, s)$ is already an approximation of the true vector in Eq. (2). However, I do not find discussions or mathematical proofs that show the approximation of the error.
Lastly, I found some minor details regarding the formulations: 


- L.106 lower case paragraph not in consistency
- Eq. (1) $\hat{x} $ is introduced without explanation
- Eq. (2) is $s > t$? This indicates forward velocity towards prior? Might be better to represent in reverse time consistently in line with L.122 $x_1 - x_0$ sounds more natural.
- Eq. (4) $\mathcal{F}$ and l are not defined elsewhere 
- Eq. (5) JVP is a function of $z, t, s$. Might be better to express explicitly instead of $\hat{JVP}$
- Eq. (7) $\hat{JVP} \approx \frac{d}{dr}$. Not equality by definition. 
- Eq. (15) no explanation of $\mathcal{B}$

### Questions
Why caching in MeanFlow would be better than caching in flow models? JVP term is approximated in MeanFlow too. This part is most confusing and concerning to me.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MeanCache, a training free caching framework for Flow Matching (FM) inference. The method reframes caching from the instantaneous velocity domain to the average velocity domain, using a JVP based estimator to approximate interval average velocities and mitigate local error accumulation. A trajectory stability scheduling strategy casts timestep selection as a peak suppressed shortest path problem on a multigraph; dynamic programming yields the cache placement under a step budget. Experiments on FLUX.1, Qwen Image, and HunyuanVideo report ∼3–4.5× speedups with competitive perceptual quality vs. strong caching baselines.

### Strengths
+ Clear, practical idea: Average velocity caching with a JVP estimator is simple yet impactful; it addresses the well known error accumulation issue at high acceleration ratios.
+ Scheduling formalization: Casting cache placement as a constrained shortest path problem is principled and provides a tunable trade off via $\gamma$ and budget $B$.
+ Strong empirical results: Consistent improvements over strong baselines across image and video; qualitative figures align with quantitative gains.

### Weaknesses
- Heuristic JVP reuse: Approximating $JVP_{t\to s}$ by $JVP_{r\to t}$ is empirically motivated. The paper does not provide theoretical guarantees or error bounds beyond the MeanFlow identities. Performance depends on the span $K$, which is tuned and schedule dependent.
- Stability assumption: The scheduling assumes that “relative changes at fixed timesteps are highly consistent across samples.” This is plausible but not statistically substantiated. Some quantification (e.g., cross prompt correlation of stability costs) would strengthen the claim.
- Cost of graph construction: The paper states the graph is built offline with 50 prompts $\times$ 5 seeds, but the actual overhead (FLOPs/GPU hours) and sensitivity to the prompt set are not reported. It remains unclear how often this needs to be rebuilt across domains or resolutions.

### Questions
- Schedule generalization: If the stability graph is built on one prompt pool/domain, how well does it transfer to another (e.g., different prompt sets or resolutions)? Any results on cross domain reuse?
- Offline cost: Please report the wall clock overhead (and FLOPs) to build the multigraph and the proportion relative to the downstream inference savings.

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
4

### Summary
This paper presents MeanCache, a training-free framework to accelerate Flow Matching inference. It addresses error accumulation in prior caching methods by shifting from an instantaneous to an average velocity perspective. The core idea is to approximate the interval average velocity using a cached Jacobian-Vector Product for better stability. This is combined with a principled scheduling strategy, formulated as a shortest-path problem. Experiments on large-scale models (FLUX, Qwen-Image, HunyuanVideo) demonstrate significant speedups while maintaining or improving generation quality over state-of-the-art caching baselines.

### Strengths
1. MeanCache consistently outperforms state-of-the-art baselines, especially at high acceleration ratios where competing methods collapse.
2. The inclusion of both perceptual metrics and reconstruction metrics provides thorough quality assessment.
3. The paper is very well-written, clearly motivating the problem and lucidly explaining the proposed methodology.

### Weaknesses
1. The paper lacks theoretical analysis (e.g., error bounds) explaining why JVP-based average velocity outperforms TaylorSeer's Taylor expansion, leaving the source of empirical gains unclear.
2. Constructing the multigraph and computing shortest paths incurs preprocessing cost. Table 1-2 report only inference latency, not total time including preprocessing.
3. TaylorSeer encounters OOM on HunyuanVideo, forcing CPU-offload for all methods. This may artificially inflate latency measurements and doesn't reflect realistic deployment scenarios.

### Questions
1. How does the preprocessing time scale with budget $B$?
2. Is the stability map computed on 50 prompts transferable to the entire prompt distribution?
3. What is the memory footprint of storing cached JVPs and the multigraph?

### Soundness
3

### Presentation
4

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
This paper proposes MeanCache, a training-free caching framework for accelerating Flow Matching diffusion models. Unlike prior methods that cache instantaneous model outputs (velocities) at fixed intervals, MeanCache operates in the average-velocity domain. Specifically, it uses cached Jacobian–vector products (JVPs) from earlier timesteps to construct interval-average velocities over longer spans, which are significantly smoother than instantaneous velocities. By reusing these interval-average velocities, MeanCache mitigates the local error accumulation that plagues existing caching schemes. To decide when and how far to cache, the paper introduces a trajectory-stability scheduling strategy: denoising steps are nodes in a graph, edges represent caching intervals weighted by their induced velocity-error, and a peak-suppressed shortest-path search (a budget-constrained graph optimization) selects the caching schedule that avoids large error spikes. In experiments on large-scale models (FLUX.1 image model, Qwen-Image, and HunyuanVideo), MeanCache achieves substantial speedups (e.g. 4.12×, 4.56×, and 3.59×) while maintaining or even improving generation quality compared to state-of-the-art caching baselines. The authors demonstrate that MeanCache preserves both perceptual quality and content consistency under high acceleration. e.g. rare-word prompts remain consistent under 4× speedup for MeanCache, whereas prior methods exhibit severe drift. The key contributions are a new average-velocity perspective for caching and a principled stability-driven scheduling algorithm, together yielding a simple yet effective acceleration scheme.

### Strengths
* MeanCache presents a conceptually simple shift (using average vs instantaneous velocity) yet this new perspective is powerful. The paper explains this insight clearly.

* It achieves large acceleration on realistic large models while retaining high fidelity, outperforming prior caching schemes across image/video tasks.

* The use of the MeanFlow identity and JVP to bridge instantaneous and average velocity is well-founded.

* The trajectory-stability scheduling (peak-suppressed shortest path) is a novel, effective tool for deciding cache placement under budget.

* The method is training-free and was tested on diverse tasks (text-to-image, text-to-video), suggesting broad applicability.

### Weaknesses
* The method depends on approximating a future JVP from past states. As the authors note, the choice of interval K is “critical” and this is a trade-off between error and stability. If the model’s dynamics are highly non-linear, the approximation might degrade. The paper addresses this with scheduling, but it remains an approximation-dependent approach.

* There are several hyperparameters (cache span $K$, budget $\mathcal{B}$, peak-penalty $\gamma$). While the paper provides ablation studies, choosing these may be non-trivial in new settings.

* The scheduling step involves building a multigraph of possible cache edges and solving a shortest-path; the paper does not discuss this overhead.

### Questions
* How sensitive is the chosen cache schedule to the specific prompts or seeds used to build the stability map? In practice, do you compute a single schedule offline per model, or would it need adjustment at runtime for different inputs?

* If the diffusion model is updated (e.g. fine-tuned or improved), must the whole schedule and hyperparameters (like K spans) be recomputed? Is there a way to adapt the existing cache without re-running the full graph optimization?

* How much overhead is incurred when solving the shortest-path?

### Soundness
4

### Presentation
3

### Contribution
3
