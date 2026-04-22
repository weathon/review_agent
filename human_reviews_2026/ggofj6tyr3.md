# Geometry-aware Policy Imitation

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
We propose a Geometry-Aware Policy Imitation (GPI) approach that rethinks imitation learning by treating demonstrations as geometric curves rather than collections of state–action samples. From these curves, GPI derives distance fields that give rise to two complementary control primitives: a progression flow that advances along expert trajectories and an attraction flow that corrects deviations. Their combination defines a controllable, non-parametric vector field that directly guides robot behavior. This formulation decouples metric learning from policy synthesis, enabling modular adaptation across low-dimensional robot states and high-dimensional perceptual inputs. GPI naturally supports multimodality by preserving distinct demonstrations as separate models and allows efficient composition of new demonstrations through simple additions to the distance field. We evaluate GPI in simulation and on real robots across diverse tasks. Experiments show that GPI achieves higher success rates than diffusion-based policies while running 20× faster, requiring less memory, and remaining robust to perturbations. These results establish GPI as an efficient, interpretable, and scalable alternative to generative approaches for robotic imitation learning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Geometry-Aware Policy Imitation (GPI), an imitation learning framework using expert demonstrations as state-space geometric curves. It generates a non-parametric vector field via distance fields, decouples metric learning and policy synthesis, and outperforms diffusion-based policies.

### Strengths
- This paper is overall very clearly written.

- The proposed method is straightforward in implementation yet very effective.

- Real-world experiments are valuable. I appreciate the authors' efforts to conduct real-world validations of their approach

### Weaknesses
**Weakness 1. Analysis of Stability Under Strong Perturbations**

While the paper demonstrates robustness to small Gaussian perturbations and unknown dynamics, it lacks evaluation under strong, real-world perturbations (e.g., sudden object collisions, sensor noise bursts, or large deviations from demonstration trajectories). The Lyapunov stability proof in Appendix A assumes continuous state/action spaces and ideal distance field computation, but no experiments test whether GPI maintains convergence when perturbations violate these assumptions. This gap raises doubts about GPI’s practicality for unstructured environments where severe perturbations are common.


**Weakness 2 Distance Metric Design and Adaptability**

The choice of distance metric (decomposed into $\(d_{rob}\) $and $\(d_{env}\)$) is a critical design lever for GPI, yet the paper relies on hand-tuned weights (e.g., $\(w_{obj}=w_{agt}=w_{\theta}=1.0\$) in PushT) and preselected metric types (Euclidean for positions, geodesic for quaternions) without systematic validation of their optimality. Additionally, while the paper mentions making the metric "learnable and co-optimized with policy synthesis" as future work, it provides no analysis of how suboptimal metric choices (e.g., misweighted \(d_{rob}\) vs. \(d_{env}\)) impact performance—especially in tasks with complex state spaces (e.g., high-dimensional vision + multi-joint robot states).

### Questions
see weakness section

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
This paper proposes Geometry-aware Policy Imitation (GPI), a non-parametric approach to imitation learning that treats demonstrations as geometric curves inducing distance fields in state space. From these fields, GPI derives two control primitives: a progression flow (advancing along trajectories) and an attraction flow (correcting deviations). The method achieves competitive or superior performance compared to diffusion policies while being 20-100× faster and requiring less memory. Experiments span simulation (PushT, RoboMimic, Adroit) and real robots (ALOHA, Franka), demonstrating efficiency, multimodality, and robustness.

### Strengths
- The geometric perspective on imitation learning is intuitive and well-motivated. The idea is quite novel and interesting.
- The policy inference speedup and memory reduction have a significant advantage over Diffusion Policy.
- The 2D example is illustrative.

### Weaknesses
- The policy rollout success rates is almost the same as Diffusion Policy. The authors may use more complex tasks to show the performance differences.
- The real robot experiments are limited and do not report success rates.
- The paper does not compare to VINN (Pari et al. 2022), though cited as the closest prior work.
- The paper does not explore under what conditions the convergence proof holds when dynamics are unknown/misspecified.
- The paper does not show performances when demonstrations are suboptimal or contain mistakes.

### Questions
See weaknesses.

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
The paper proposes Geometry-aware Policy Imitation (GPI), which treats each demonstration as a geometric curve and builds a distance field in state space. Two complementary control primitives—(progression along the curve tangents and attraction via the negative distance gradient in the actuated subspace)—are superposed to form a lightweight, non-parametric vector-field policy. Multiple demonstrations are composed by distance-weighted retrieval (top-K with softmax temperature). This decouples metric learning (state representation / distance) from behavior synthesis (flow composition). Experiments on PushT (state + vision), RoboMimic (Lift/Can/Square) and Adroit (Door/Pen/Hammer/Relocate) plus two real-robot tasks (box flip on ALOHA, fruit handover on Franka) show GPI attains equal or higher success than diffusion policies while being 20–100× faster and more memory-efficient; a Lyapunov-style analysis argues convergence of the flow policy.

### Strengths
1.Clear geometric formulation & interpretability. Turning demonstrations into distance/flow fields yields an intuitive, controllable policy; the roles of attraction vs. progression and the top-K composition are explicit.

2.Efficiency. State-based PushT runs at ~0.6 ms per step vs. ~65 ms for DDIM-10; vision-based uses a light encoder with small memory. No training is needed for state-based settings.

3.Metric learning is flexible: task-specific heads, VAE latents, or pretrained features (ResNet/PCA, CLIP, SAM), and only robot-space distance shapes attraction—vision is used mainly for demo selection.

4.Theoretical sanity check. A concise Lyapunov argument establishes asymptotic convergence in the actuated subspace under the proposed dynamics

### Weaknesses
1.Metric dependence & high-dimensional brittleness. Performance hinges on the chosen distance; SAM underperforms without fine-tuning, and the method relies on learned embeddings for selection while only drob shapes attraction—raising questions under perceptual noise or representation drift.

2.Comparisons and settings. Diffusion baselines are strong but specific (DDPM-100/DDIM-10). It would help to compare against recent flow-matching/streaming-flow policy variants and to ensure action-space/horizon choices align fairly (the method uses H=1 reactive control).

3.Assumption clarity. The convergence proof assumes smooth distance fields and projection to the nearest point on a continuous curve (e.g., splines). Practicalities with discrete demos, non-holonomic constraints, and contact discontinuities are not fully analyzed.

### Questions
1.Retrieval & scaling. What is the exact complexity of one control step (in terms of demos N, trajectory lengths, and latent dimension)? Do you use approximate NN (e.g., FAISS) or down-sampling; how does latency scale to ~10⁶ states? 

2.Adaptive weighting. You set λ₁, λ₂ constant or distance-dependent. Can these be scheduled online (e.g., larger attraction far from demos, or curvature-aware progression) and do you have guidelines for selecting β, K robustly across tasks? 

3.Vision robustness. Since denv does not shape attraction, how do you mitigate mis-selection due to perceptual drift (e.g., lighting/occlusion)? Would a small attraction term in latent space (or a learned cross-term) help?

4.Baselines. Could you add comparisons to flow-matching policies and recent “streaming flow” formulations that also interpret action trajectories as flows? This would clarify whether speed/robustness advantages hold beyond diffusion.

### Soundness
4

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
The paper proposes Geometry-Aware Policy Imitation, a method that treats expert demonstrations as geometric curves in state space instead of discrete state–action samples. Each demonstration defines a distance field, from which two flows are derived, a progression flow using demonstrated tangents, and an attraction flow using the negative gradient of the distance field. Their weighted sum forms a vector field that produces control commands. The paper states that this approach separates metric learning (state representation and distance computation) from behavior synthesis (flow combination). It presents a Lyapunov-based argument that the combined flow is stable in the actuated subspace and converges toward the demonstration. Experiments are reported on Push-T, Robomimic, Adroit and two real-robot tasks.

### Strengths
- Clear and consistent mathematical formulation.
- Demonstrated efficiency on Push-T. 
- Comprehensive documentation of experimental settings, including both state-based and vision-based inputs.
- Modular structure allowing different encoders. 
- The text includes explicit ablations on the number of neighbors, action horizon, and noise levels.

### Weaknesses
- The convergence proof covers only smooth, continuous trajectories in the actuated subspace. The paper does not provide analysis for environments with unactuated dynamics or discontinuities.
- The paper defines multiple possible distance metrics, but no quantitative comparison or sensitivity analysis across metrics is reported.
- Quantitative comparisons are limited to diffusion-based baselines.
- The paper gives descriptive outcomes and system specifications but no numerical success or robustness statistics in the real-robots experiments. 
- The method requires storing features for all demonstrations. The text notes linear memory growth but provides no empirical scaling or retrieval-time measurements.
- Multimodality is illustrated through noise perturbations but not quantified beyond the presented plots.

### Questions
- Can the authors provide quantitative results showing how different distance metrics (Euclidean, geodesic, latent cosine) affect performance?
- Are there measured success rates or task completions for the ALOHA and Franka experiments?
- How does retrieval latency or memory use change with increasing numbers of demonstrations?
- How sensitive is the method to discontinuities or non-smooth demonstrations?
- Can the authors clarify whether diffusion-policy baselines were reimplemented under identical conditions or reused from prior work?

### Soundness
3

### Presentation
4

### Contribution
3
