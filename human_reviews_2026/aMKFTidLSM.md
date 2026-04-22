# Dual Goal Representations

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
In this work, we introduce dual goal representations for goal-conditioned reinforcement learning (GCRL). A dual goal representation characterizes a state by "the set of temporal distances from all other states"; in other words, it encodes a state through its relations to every other state, measured by temporal distance. This representation provides several appealing theoretical properties. First, it depends only on the intrinsic dynamics of the environment and is invariant to the original state representation. Second, it contains provably sufficient information to recover an optimal goal-reaching policy, while being able to filter out exogenous noise. Based on this concept, we develop a practical goal representation learning method that can be combined with any existing GCRL algorithm. Through diverse experiments on the OGBench task suite, we empirically show that dual goal representations consistently improve offline goal-reaching performance across 20 state- and pixel-based tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a new approach to encode a goal state in goal-conditioned reinforcement learning (GCRL). Instead of using the goal state s directly, the work proposes to use a dual representation where the goal is represent as its distance to all other states. It is claimed that this representation has favorable properties which are theoretically proved and verified by the experimental results on simulated environments.

### Strengths
The fundamental idea of the work is simple and interesting. Moreover, the experimental results seem to verify that the simple idea works (at least) in simulations.

### Weaknesses
**Major:**

 - The idea of dual representation in Figure 1 is simple, but also yields to certain questions such as dimension of the dual representation in the case of a large number of states or in the continuous cases where states are infinite - this should be clarified to serve the readers

 - The concept of "temporal distance" should be clarified. It generally suggests "number of steps from s to g", but it is not defined

 - Things get confusing when states and goal are transformed to the embeddings psi(s) and phi(g) that transform the state to N-dimensional vectors. Close to Eqs. (1) and (2) psi and phi are defined to be the output and the goal heads, but heads of what? Then the distance function is approximated by *implicit q learning* without any explanation why this is done and how it is done? Q-learning provides actor and critic networks, but are they used and how?I sought information from the appendix D.2 without finding any more explanation?

 - The theory is nicely explained, but the gap from the theory to actual implementation seems large?

**Moderate:**

 - Even the simple example in Figs. 2 and 3 remains unclear as the meaning of state is not explained (i.e. what means the 64 random states - any random configuration?)

 - If the proposed algorithm needs pre-training with IQL, then how many interaction steps are needed for IQL? If that is substantial, then is this method really improving over other methods with the same number of steps and average over N runs with random initializations (for example, in Figure 4 are those IQL steps added to Dual results)?

**Minor**

### Questions
As the authors see from my comments, I struggled to understand the practical implementation of the proposed method. I hope that the authors clarify me the concepts or explain why my questions/concerns are incorrect? I am RL practitioner and somewhat experienced coder, but I could not implement your method solely based on the manuscript.

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
4

### Summary
The paper proposes to use estimated distances between states as the underlying latent representation in goal-conditioned reinforcement learning (GCRL). This way a goal can be implicitly represented by the distances to all other states, akin to a heuristic estimate. The approach is empirically tested in a set of benchmark domains using deep reinforcement learning on top of the learned goal representation.

### Strengths
The main strength of the work is that the proposed algorithm performs well in practice compared to other algorithms for GCRL. Concepts are clearly presented and experiments seem rather extensive.

### Weaknesses
As mentioned by the authors, several previous approaches estimate distances or metrics for GCRL by embedding states in a latent state space and applying a metric such as the L2-norm in the latent space. The authors propose using the scalar product instead, which seems highly related to the L2-norm since the squared L2-norm equals
\[
\psi(s)^\top\psi(s) + \varphi(g)^\top\varphi(g) - 2 \psi(s)^\top\varphi(g).
\]
Hence the main term for determining the distance (though negated) is precisely the scalar product in the third term. Another difference is that the goal embedding \varphi is potentially different from the embedding \psi used for other states. Though this has potential positive effects such as representing asymmetric distances, applying a metric to vectors from different embedded spaces does not have the same intuitive interpretation. I believe the motivation for these design choices could be explained better in the text.

In general the claim to novelty is a bit weak. Using the embedding for goal representations seems very similar to using the embedding to estimate distance metrics or perform planning. The inner product seems to slightly outperform the L2-norm in Table 4, but this difference is not large enough to account for the improvement in performance compared to other algorithms.

There exists another work in the literature that also learns explicit distance estimates between pairs of states from offline data:

Steccanella and Jonsson, ECML 2022
State Representation Learning for Goal-Conditioned Reinforcement Learning

### Questions
Do you actually use a different embedding for goal states as opposed to other states? What is the motivation for using two different embedding functions?

If planning using the learned distance estimates performs poorly, does this not indicate that the learned distance estimates are not very accurate?

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
The paper proposes a way to encode goals for goal-conditioned reinforcement learning called dual goal representations. The method characterizes a goal by the collection of shortest-time distances from every state to that goal. The authors show that this representation is sufficient to recover an optimal goal-reaching policy and is invariant to observation noise that does not affect latent system dynamics. Practically, they approximate this functional by learning a goal-conditioned value model, taking its ``goal head'' as the goal embedding, and then training any downstream offline GCRL algorithm that consumes this embedding. Experiments on OGBench report average improvements over several prior representation learners across 20 state and pixel tasks, plus ablations comparing inner-product vs metric parameterizations and a study on noise robustness.

### Strengths
* The dual view, defining a goal by its temporal distances to all states, is elegant and connects clearly to the control objective. The sufficiency and noise-invariance theorems make the idea precise and motivate learning a representation that depends only on dynamics rather than raw observations.

* The recipe, ``learn a goal-conditioned value model, use its goal head as the representation, then plug into standard offline GCRL,'' is straightforward and compatible with multiple downstream algorithms. This lowers integration cost for practitioners.

* On OGBench, the proposed representations are competitive or best on a majority of state-based and several pixel tasks, and the paper investigates parameterization choices, the limits of directly extracting a policy from the distance model, and robustness to test-time goal noise. These analyses help interpret when the approach works.

### Weaknesses
* The key invariance result is proved under an Ex-BCMP model with disjoint observation supports per latent state, a strong assumption seldom satisfied by realistic image observations. The paper then validates ``robustness to noise'' by adding Gaussian noise to evaluation goals, which does not convincingly reflect exogenous distractors in observations or dynamics. A more faithful test would use structured nuisance variation tied to the emission model or distractor objects.

* The method compares favorably on several pixel tasks, yet performance on visual puzzles collapses for all representation learners. The paper attributes this to late fusion that is required when goals are embedded, while the ``original'' baseline can use early fusion. This creates a comparison gap that complicates claims about representation quality, since the fusion strategy is entangled with the representation. The suggestion that state-aware representations could resolve this is plausible, but the paper stops short of implementing it.

* The empirical approach largely repurposes a goal-conditioned value model and takes its goal head as a representation, combined with inner-product scoring. While the dual perspective is neat, the implementation feels incremental relative to prior metric and contrastive goal embeddings. The inner-product ``universality'' claim is cited, but the paper does not quantify approximation error or show cases where the metric form fundamentally fails beyond average score tables.

### Questions
* How sensitive is the approach to the accuracy of the learned distance surrogate? Please report correlations between predicted distances and true shortest-path steps on tasks where ground truth can be computed, and study how downstream performance varies as you degrade the value model. This could validate the core mechanism that “better distance, better policy.”

* Can you evaluate noise invariance with structured distractors rather than additive goal noise, for example by adding irrelevant moving objects or textured backgrounds in pixel tasks while keeping latent dynamics fixed? This would better mirror the Ex-BCMP setup used in the theorem.

* To address the early vs late fusion confound, can you try a state-aware goal embedding or a lightweight cross-attention block that allows early interaction while keeping a representation bottleneck, then re-run the visual puzzles and other pixel tasks for a fair comparison? Even a small prototype would strengthen the claim.

* Please expand the ablation on representation dimensionality and the choice of inner-product vs metric forms. Are there tasks where the metric version wins once you scale capacity or adjust negative sampling for metric learning baselines, and how do results change with larger N?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces dual goal representations for goal-conditioned reinforcement learning (GCRL). The core idea is to represent a goal state not by its own features, but by the set of temporal distances from all other states to itself. The paper theoretically proves that this representation is sufficient for learning an optimal policy and, importantly, is invariant to exogenous noise. While the theory establishes these desirable properties to make this concept practical, the proposed algorithm sidesteps the computational challenge by learning a compact, low-dimensional vector that approximates this representation. A comprehensive set of experiments demonstrates the benefit of this approach, showing that it consistently improves the learning performance of several existing GCRL algorithms across a wide range of tasks.

### Strengths
- The paper introduces a novel and well-motivated approach to goal representation based on temporal distances. This "dual" perspective is a strong conceptual contribution with clear potential for applications even beyond the GCRL setting.

- The paper is well-written and clearly structured. The authors effectively build intuition by cleanly separating the ideal theoretical concept from its practical implementation, making the core ideas easy to follow and understand.

- The experimental evaluation is both extensive and convincing. By testing on a diverse suite of environments and integrating their representation with three different downstream GCRL algorithms (GCIVL, CRL, GCFBC), the authors provide strong evidence for the method's general applicability and robustness.

### Weaknesses
- A significant gap exists between the theoretical framework, which analyzes an ideal functional $\varphi^v$, and the practical implementation, which learns a finite-dimensional approximation. Is not obvious that the formal guarantees of sufficiency and noise invariance do transfer to the compressed low dimensional learned representation. The justification for the method's effectiveness therefore rests heavily on its strong empirical validation rather than a direct theoretical proof for the practical algorithm.

- The proposed method employs a two-stage architecture: first learning a representation $\varphi(g)$ by training an approximate value function ($\psi(s)^T \varphi(g)$), and then using $\varphi(g)$ to train a separate downstream policy and, in many cases, a new value function (V_GCRL). This design raises significant questions about efficiency and potential redundancy.
While the paper's ablation study (Table 5) shows that direct policy extraction from the initial value function performs worse, the underlying reason for this is not fully explored. It seems that a deeper analysis is needed to justify why the inner product parameterization is expressive enough to yield a powerful representation, but too restrictive for direct policy extraction. Clarifying this would explain why the apparent redundancy of learning a second value function and discarding the initial state encoder $\psi(s)$ is a necessary design choice rather than a limitation.

- A minor fix is that the paper is missing the citation to the paper "State Representation learning for Goal Conditioned Reinforcement Learning" Steccanella et al. that seems relevant and learn a temporal distance representation used for reward shaping.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
