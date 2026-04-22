# Accelerating Model-Based Reinforcement Learning Using Equivariance

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Model-based reinforcement learning (MBRL) is a promising approach for learning effective policies in a data-efficient manner by using learned dynamics models to generate synthetic rollouts for actor-critic trianing, thereby reducing the reliance on costly environment interactions. However, when the learned dynamics model is inaccurate, these synthetic rollouts can introduce bias and deteriorate performance. Fortunately, many domains exhibit symmetries that can serve as powerful inductive biases, enabling the learned models to generalize beyond their training data. In this work, we exploit these inherent symmetries in MBRL and formally define equivariant MBRL for POMDPs. Building on this formulation, we introduce EquiDreamer, a framework that integrates symmetry into both world modeling and policy learning through an equivariant latent dynamics architecture. Experiments on visual continuous control tasks demonstrate that our equivariant MBRL method outperforms both model-based and model-free baselines, achieving strong results with substantially fewer environment interactions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper studies equivariance in model-based reinforcement learning under POMDPs. Its main innovation is an equivariant Recurrent State Space model that reconstructs equivariant feature embeddings, rather than raw images, from a pretrained encoder using frame averaging. The method is evaluated on four tasks from the DeepMind Control Suite and compared against DreamerV3 and DrQ-v2.

### Strengths
* The problem is well-motivated; leveraging environment symmetries can meaningfully improve learning efficiency.
* The approach of reconstructing equivariant feature embeddings instead of raw observations is conceptually interesting.

### Weaknesses
1. The method assumes prior knowledge of the symmetry groups. It would be valuable to explore whether the approach could be extended to learn these symmetries automatically, as in [2,3].

2. Additionally, the method cannot handle approximate symmetries, and this limitation is not discussed. Addressing both these points could significantly strengthen the contribution and impact of the work.

3. I identified potential issues with Proposition 1 and its proof that questions soundness of the paper: Proposition 1 is derived from Equation (3), which in turn is taken from [1], an unpublished paper. Given the central role of this equation, the authors should better justify its validity rather than assuming it holds, especially since [1] has not undergone peer review and appears to be an informal note.

4. The proof relies on two strong and arguably unrealistic assumptions: (1) the dynamics model perfectly approximates the true transition dynamics, and (2) the symmetries are exact and perfectly generalizable. These assumptions are almost impossible to hold in neural network-based models and, importantly, prevent the method from being applicable to environments with approximate symmetries.

5. Results are reported on only four DMC environments out of more than twenty available. The selected tasks are all 2D with small action spaces, despite the method being motivated as generally applicable to 3D symmetries. This raises questions about scalability to more complex environments.

6. The reported baseline results, particularly for DrQ-v2, are inconsistent with the original paper, where the method performs much better. Here, DrQ-v2 fails to learn in nearly all environments, suggesting potential issues with implementation or hyperparameter choices. If different settings were used, the authors should clarify this. In any case, the authors need to provide a proper representation of the baseline, particularly if the baseline is well-established in the community and has a high reproducibility.

7. The paper is missing various baselines from equivariant representation learning algorithms. The method is only compared against Dreamer and DRQ-v2. At least one baseline from each of the following families should be included:
    * Model-free equivariant methods: Deep homomorphic policy gradient [2], or EQR [3]. 
    * Model-based equivariant methods:  EDGI [4], equivariant MuZero [5], SEN [7], or [8].

8. The contributions appear incremental compared to prior work on equivariant model-based RL. The paper does not cite or compare against some key methods in this area, such as [4] and [5].

Given the above methodological, theoretical, and empirical limitations, as well as concerns regarding the validity of the main proposition, I do not believe the paper meets the standard of this venue. Nonetheless, this is a promising research direction, and I encourage the authors to address these issues in future revisions.

### References
[1] Jiang, Nan. "A note on loss functions and error compounding in model-based reinforcement learning." arXiv preprint arXiv:2404.09946 (2024).

[2] Rezaei-Shoshtari, Sahand, et al. "Continuous mdp homomorphisms and homomorphic policy gradient." Advances in Neural Information Processing Systems 35 (2022): 20189-20204.

[3] Mondal, Arnab Kumar, et al. "Eqr: Equivariant representations for data-efficient reinforcement learning." International Conference on Machine Learning. PMLR, 2022.

[4] Brehmer, Johann, et al. "Edgi: Equivariant diffusion for planning with embodied agents." Advances in Neural Information Processing Systems 36 (2023): 63818-63834.

[5] Deac, Andreea, Théophane Weber, and George Papamakarios. "Equivariant MuZero." arXiv preprint arXiv:2302.04798 (2023).

[6] Park, Jung Yeon, et al. "Learning Symmetric Embeddings for Equivariant World Models." International Conference on Machine Learning. 2022.

[7] Zhao, Linfeng, et al. "Equivariant action sampling for reinforcement learning and planning." arXiv preprint arXiv:2412.12237 (2024).

### Questions
1. How are you imposing the value function to be invariant?
2. How does the method work with approximate symmetries in the environment?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes EQUIDREAMER, a model-based framework that incorporates equivariance into the latent dynamics model and policy learning for POMDPs. The method uses symmetries in the environment to improve sample efficiency and generalization, building on the Dreamer architecture by replacing image reconstruction with feature reconstruction and using equivariant neural networks. Experiments on visual control tasks demonstrate improved performance over DreamerV3 and DrQ-v2 baselines.

### Strengths
1. Integrating equivariance principles into a model-based RL framework is an interesting research area, as well as revealing its sample efficiency in many problems.
2. Experimental results show consistent improvements in sample efficiency on several continuous control tasks.

### Weaknesses
1. The overall novelty is limited, as the core idea of applying equivariance to RL has been explored in prior work (e.g, [1][2]), and the extension to model-based RL, while sensible, does not constitute a significant conceptual advance.

2. The claimed benefits of feature reconstruction over image reconstruction are not sufficiently analyzed from the equivariance component, making it difficult to attribute the gains to the proposed novelty.

3. The model structure is confusing, particularly the use of a unified parameterization $p_{\theta}$ for the transition, observation, and reward models. This convolutes the roles of distinct components and lacks a clear justification.

minor typos:
1. In line 149, it seems like the equation form is incorrect.
2. In line 151, it should be "In this paper,..."

[1] Mondal, A. K., Jain, V., Siddiqi, K., & Ravanbakhsh, S. (2022, June). Eqr: Equivariant representations for data-efficient reinforcement learning. In International Conference on Machine Learning (pp. 15908-15926). PMLR.

[2] Grimm, C., Barreto, A., Singh, S., & Silver, D. (2020). The value equivalence principle for model-based reinforcement learning. Advances in neural information processing systems, 33, 5541-5552.

### Questions
1. How does the method scale to more complex symmetry groups or real-world tasks where symmetries merely exist?
2. Were there any environments or symmetry conditions where the equivariant model failed to improve upon the baseline?

### Soundness
2

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
This paper proposes a equivariant MBRL method to capture symmetries in specific POMDP domains. It helps with generalization to unseen equivariant states during training thus improving sample efficiency and meanwhile incorporates physical information. The proposed equivariant model-based RL method, EquiDreamer, shows higher sample efficiency compared to DreamerV3, demonstrating the effectiveness of method proposed.

### Strengths
1. It's an important research subject to discover symmetric patterns during training to reduce abundant exploration and boost sample efficiency.

2. The effectiveness of components of method proposed is clearly supported and discussed with empirical evidence.  The idea of reconstructing in the feature space instead of the original pixel space shows advantage on complex tasks like reacher-hard.

### Weaknesses
1. Visualization of the symmetries discovered could be presented to help understanding the method proposed. It remains somehow vague that whether EquiDreamer actually captured the equivariance between distinct state.

2. There are only empirical results on five tasks in DMC, which weakens the evidence for the effectiveness and generalizability of the method proposed. Results in other continuous control domains like Robodesk, Meta-world which also seem to have inherent symmetries would be preferred.

### Questions
1. Despite DMC, can you list more benchmarks or domains that inherit the equivariant feature?

### Soundness
3

### Presentation
3

### Contribution
2
