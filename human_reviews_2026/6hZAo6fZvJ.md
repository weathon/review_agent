# Distributional value gradients for stochastic environments

- Decision: Accept (Poster)
- Scores: 2, 6, 10, 6

## Abstract
Gradient-regularized value learning methods improve sample efficiency by leveraging learned models of transition dynamics and rewards to estimate return gradients. However, existing approaches, such as MAGE, struggle in stochastic or noisy environments,  limiting their applicability. In this work, we address these limitations by extending distributional reinforcement learning on continuous state-action spaces to model not only the distribution over scalar state-action value functions but also over their gradients. We refer to this approach as Distributional Sobolev Training. Inspired by Stochastic Value Gradients (SVG), our method utilizes a one-step world model of reward and transition distributions implemented via a conditional Variational Autoencoder (cVAE). The proposed framework is sample-based and employs Max-sliced Maximum Mean Discrepancy (MSMMD) to instantiate the distributional Bellman operator. We prove that the Sobolev-augmented Bellman operator is a contraction with a unique fixed point, and highlight a fundamental smoothness trade-off underlying contraction in gradient-aware RL. To validate our method, we first showcase its effectiveness on a simple stochastic reinforcement‐learning toy problem, then benchmark its performance on several MuJoCo environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new deterministic policy gradient method in the continuous action space by using Sobolev training, which additionally incorporates the gradient information in the distributional learning. Theoretically, the authors show the contractivity with Max-sliced MMD and Wasserstein distance under some assumptions. The dynamics and reward transitions are also needed to learn to derive a tractable algorithm. Experiments are conducted in a toy example and several Mujoco environments.

### Strengths
1.Incorporating the action-gradient in the distributional learning seems novel.

2.Some theoretical results are provided and also many details have been discussed, which are deferred to Appendix.

### Weaknesses
1. **The motivation is not clear**. Why is introducing action gradients beneficial in distributional learning? Directly combining Sobolev training seems straightforward. A more profound analysis is needed to strengthen the motivation.

2. **Many theoretical results are questionable with impractical or unclear conditions.** I push back against the statement in line 142 he Lipschitz continuity of $\pi$ typically holds when using neural-network function approximation, which is not true. The Lipschitz constant of deep neural networks is often unboundedly large, which depends on the spectral norms of the weight matrix that is often unbounded during the training. The contraction in the theorem only holds when $\gamma \kappa <1$, which is hard to justify for any practical environment. This renders the contraction vacuous and meaningless. What are the mild conditions in Theorem 2, which are very unclear. What is $\kappa$ in a rigorous way?

3. **Limited experiments and insignificant improvements**. In distributional RL literature, published papers have done extensive experiments with significant improvement, making the proposed algorithm deployable. However, the improvement in a toy example is too weak, and the improvement in the Mujoco environments is very limited. Although the robustness setting is helpful to some extent, it is unsatisfactory to find that the proposed algorithm only matches the performance with the baselines, even though it has additionally learned the environment and there is a much higher computational cost.

4. **The writing is not clear.** Many sentences are in bold, but they are hard to follow. For instance, what do the authors want to express in Line 203? It is in bold but may not be strongly related to the other parts of the paper. A lot of details are provided in the appendix, which are very distracting. The main content of the paper is not even self-contained. For example, it is surprised that many key details in Theorem 2 are deferred to the appendix, which largely undermines the readability of this paper. I could not understand why we should not expect the proposed algorithm to be superior over all the environments in general.

### Questions
1.What is the environmental non-differential regarding?

2.The formulation in Eq.10 is questionable. Note that a is a constant value as the initial action, why do you mean the gradient is taken regarding a constant? In addition, the notation of cumulative rewards does not involve the action. Since it is a continuous action space, what does $|A|$ mean? Essentially, it should be $\mathbb{R}^d$.

3.What is the computational cost when we use max-sliced MMD? It seems that it is very high, but we are disappointed to find that the improvement is insignificant even with the higher computational cost.

4.Why do we need to incorporate the TQC trick? Is there any ablation study in experiments compared with other baselines?

5.The introduction of CVAE is less convincing. Why do we not directly train a diffusion model?

6.The theory and the practical algorithm seem to have a gap. The finally proposed algorithm does not enjoy a theoretical contraction guarantee.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper seeks to improve the sample efficiency and performance of actor critic RL algorithms in stochastic environments by estimating the distribution of  action gradients through the critic. To do so it introduces the distributional modeling of returns and their gradients, and also details an approach to use it for actor critic training, which it terms distributional Sobolev training. The practical instantiation uses a form of maximum mean discrepancy to estimate the action gradients via a novel Bellman operator, and uses a variational autoencoder to estimate the one-step transition function.

Evaluations are done on a toy domain to verify that the approach works as intended and then in MuJoCo environments with added stochasticity to evaluate the approach on a standard benchmark.

### Strengths
* The proposed idea is intellectually stimulating and pushes the frontier on how actor critic algorithms work. It presents a novel idea and sound theoretical analysis to justify it.
* The proposed practical algorithm takes reasonable approaches for the required theoretical estimates, such as MMD and MSMMD.
* The use of the toy domain is useful to study if the proposed technique can deal with increased stochasticity as claimed
* The statistical bounds used in the evaluations are heartening and appreciated.
* The theoretical analysis seems to be sound
* The idea would be of interest to researchers who focus on dealing with stochasticity in the environment.

### Weaknesses
* The use of the VAE is a practical requirement to deal with environments without differentiable dynamics. Using a toy environment where the gradients of the environment are known would have been even better to evaluate how well the idea would work if the dynamics did not have to be estimated.
* While the results in the toy domain are impressive, it is unclear why they do not translate as well to the Mujoco environments, where MAGE remains within the margin of error. This difference is not a dealbreaker, since the trends across environments shows that the proposed techniques outperform MAGE across multiple environments.
* Nit: Ordering the methods in Figure 3 so the proposed methods are the rightmost ones and the ones on the left are baselines would make them much easier to read

### Questions
* Are practical considerations of implementing the idea potentially holding back the empirical results? Could the proposed approach be used for other robotics tasks like OGBench with added stochasticity?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This paper considers distributional reinforcement learning, and describes how to incorporate gradient information into distributional temporal-difference learning. Not only is it theoretically well-motivated, it also describes practical means of implementation even in challenging, realistic settings such as non-differentiable environments.

### Strengths
- This paper makes foundational contributions to the important and timely topic of distributional RL.
- It is exceptionally well-written, with both pedagogical clarity and mathematical precision (admittedly, it gets a bit dense at times, but I think that is unavoidable). 
- The positioning against the Related Work is detailed, clear, and well articulated. Furthermore, it gives a balanced and honest discussion of its limitations.
- I appreciate the pedagogical toy problem
- The extensive appendices provide detailed support (mostly proofs) for the claims in the main paper.

### Weaknesses
Honestly, I can't think of any major weaknesses. Of course, it would have been nice to see clear improvements over the baselines on an established benchmark. Still, I think the theoretical and algorithmic contributions by far outweigh the importance of such experimental results. 

Minor things: 
- Some of the figures (and especially the text therein) are tiny and require zooming.
- As a matter of personal preference, I think it's neater to emphasize text with italics rather than bold font, but I'll leave that for your discretion.

### Questions
No further questions.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses a limitation of existing gradient-aware reinforcement learning methods, which improve sample efficiency by incorporating value gradients into critic training. The authors identify that current approaches treat these gradients as deterministic quantities. Thus, they extend distributional RL to model the joint distribution of both the cumulative return and its action-gradient. The proposed algorithm, Distributional Sobolev Deterministic Policy Gradient is built upon three key technical components. First is a novel Sobolev Bellman operator, which forms the theoretical core by bootstrapping the joint distribution of returns and their gradients. Second, to make this operator practical in non-differentiable environments, the method employs a conditional Variational Autoencoder enabling gradient backpropagation through the learned dynamics. Third, the critic is trained to minimize the discrepancy between its predicted distribution and the bootstrapped target distribution.

### Strengths
- Extending the idea of distributional RL to action-gradient is interesting. By proposing to learn the distribution of the gradient, the work offers a principled and more complete model of credit assignment under uncertainty. This moves the field beyond the limitations of prior methods like MAGE, which are confined to modeling the gradient's expectation and are thus vulnerable to noise.
- The formal derivation of the Sobolev Bellman operator is solid.
- Except for theoretical contributions, the paper also provides comprehensive experimental results.

### Weaknesses
- As the authors stated, one weakness of DSDPG is its computational complexity.
- There is a disconnect between the theoretical motivation for the MSMMD metric and its empirical utility. The paper introduces MSMMD primarily to obtain a provable contraction for the Sobolev Bellman operator under a tractable, sample-based metric. Standard MMD does not offer this same general guarantee. However, the empirical results across both the toy problem and the MuJoCo benchmarks show that the MSMMD variant provides at best a marginal performance improvement over the simpler MMD variant, and in some cases performs identically.

### Questions
What are the reasons for choosing the incomplete operator for DSDPG?

### Soundness
4

### Presentation
4

### Contribution
3
