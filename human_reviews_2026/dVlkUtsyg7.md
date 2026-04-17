# Improving Feasibility via Fast Autoencoder-Based Projections

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Enforcing complex (e.g., nonconvex) operational constraints is a critical challenge in real-world learning and control systems. However, existing methods struggle to efficiently enforce general classes of constraints. To address this, we propose a novel data-driven amortized approach that uses a trained autoencoder as an approximate projector to provide fast corrections to infeasible predictions. Specifically, we train an autoencoder using an adversarial objective to learn a structured, convex latent representation of the feasible set. This enables rapid correction of neural network outputs by projecting their associated latent representations onto a simple convex shape before decoding into the original feasible set. We test our approach on a diverse suite of constrained optimization and reinforcement learning problems with challenging nonconvex constraints. Results show that our method effectively enforces constraints at a low computational cost, offering a practical alternative to expensive feasibility correction techniques based on traditional solvers.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a FAB projection method, which encodes a prediction and context into a latent space, projects onto a simple convex set 𝑆, and decodes back to the original space. A two-phase training is used. Phase 1 reconstructs the feasible set from feasible samples, while Phase 2 structures the latent via a discriminator with a hinge loss, plus a latent loss and a Jacobian regularizer. FAB is then attached as a plug-and-play mapping 𝜙 around a base network.

### Strengths
1. The main contribution is an amortized, non-iterative feasibility mapper that is fast at inference and empirically effective on diverse nonconvex toy sets. 
2. Its plug-and-play design is attractive for deployment, since 𝜙 can be mounted onto arbitrary predictors without rewriting training loops.
3. The Safe RL experiment demonstrates the safety and latency advantages.

### Weaknesses
1. FAB explicitly does not provide hard feasibility guarantees, making it unsuitable for safety-critical regimes, and no robustness evidence is given for distribution shift or dataset coverage failures that the authors acknowledge as limitations. 
2. Comparisons emphasize feasibility, but optimality gaps vary and many classical baselines (e.g., Projected Gradient, FSNet) can reach 100% feasibility as well.
3. The experiments focus on toy nonconvex sets rather than realistic, parameter-dependent instances, so the path to real deployments remains unclear.

### Questions
Two biggest concerns with L2O methods remain (1) the expensive training cost, and (2) the lack of feasibility guarantees. Can the authors discuss more about these two aspects?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes FAB, a fast “approximate projector” built from a conditional autoencoder. Training has two phases: (1) reconstruct feasible points; (2) use a discriminator + simple penalties to make latent samples decode to feasible outputs. Experiments on several synthetic constraint families and one safe-RL task look promising.

### Strengths
1. Easy to attach after an existing predictor to improve feasibility with negligible latency.

2. The two-phase training architecture is clear and easy to follow

3. Tables report near-perfect feasibility and big time gains vs. homeomorphic projection and other baselines

### Weaknesses
It would further strengthen the paper if some form of theoretical guarantee—even a weak or probabilistic feasibility bound—could be established.

### Questions
1. Could you report throughput vs. dimension and memory for different decoder widths and number of decoders, and a cost breakdown?

2. Does a FAB trained on one hazard layout transfer to unseen layouts? Any results under domain shift?

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
5

### Summary
This paper introduces the Feasibility Autoencoder-Based (FAB) method, which uses an offline-trained autoencoder to project infeasible DNN outputs into a structured latent space for fast feasibility enforcement in constrained optimization. Key contributions include a two-phase training algorithm for the autoencoder, empirical speedups, and demonstrations of high feasibility with near-optimal performance on benchmarks like portfolio optimization.

### Strengths
1. The method's originality lies in its pretrained encoder-decoder as a reusable feasibility projector, combining autoencoders with adversarial structuring for non-convex constraints and showing potential for generalization across tasks.

2. Quality is good, with well-designed algorithms, and good presentation.

3. The studied problem is important, as FAB enables real-time constraint handling in practical ML applications, potentially impacting fields like operations research and safe AI.

### Weaknesses
1. The lack of theoretical guarantees, especially on optimality preservation post-projection, weakens reliability, as the method relies on empirical data without bounds on gaps or failure modes. The feasibility and optimality guarantee is not presented and the method itself seems only rely on the performance of the decoder

2. Contributions are vague compared to baselines like homeomorphic projections, which offer low-complexity feasibility guarantees; clearer differentiation (e.g., quantitative edges in non-convex settings) is needed.

3. The rationale of optimality consideration is questionable, please see below.

### Questions
1. How does the autoencoder ensure optimality conservation during projection, and can you provide bounds or discuss failure modes?

2. How does training handle multi-lable T_feas (could have multiple y per x)? Will different y harms the model performance as it could be confused on the targeted y.

3. What are FAB's specific advantages over homeomorphic projections (e.g., in non-convexity or generalization), and how could the pretrained encoder-decoder be extended to new domains?

4. The rationale of optimality consideration is questionable, like the decoder only see feaible/infeasible points, given an infeasible but close to optimal point, how could the model reconstruct a good solution with good optimality performance?

### Soundness
3

### Presentation
3

### Contribution
2
