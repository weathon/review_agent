# DecompGAIL: Learning Realistic Traffic Behaviors with Decomposed Multi-Agent Generative Adversarial Imitation Learning

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
Realistic traffic simulation is critical for the development of autonomous driving systems and urban mobility planning, yet existing imitation learning approaches often fail to model realistic traffic behaviors. Behavior cloning suffers from covariate shift, while Generative Adversarial Imitation Learning (GAIL) is notoriously unstable in multi-agent settings. We identify a key source of this instability—irrelevant interaction misguidance—where a discriminator penalizes an ego vehicle’s realistic behavior due to unrealistic interactions among its neighbors. To address this, we propose Decomposed Multi-agent GAIL (DecompGAIL), which explicitly decomposes realism into ego–map and ego–neighbor components, filtering out misleading neighbor–neighbor and neighbor–map interactions. We further introduce a social PPO objective that augments ego rewards with distance-weighted neighborhood rewards, encouraging overall realism across agents. Integrated into a lightweight SMART-based backbone, DecompGAIL achieves state-of-the-art performance on the WOMD Sim Agents 2025 benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
* This paper addresses the problem of training instability in multi-agent GAIL for realistic traffic simulation.
* It identifies irrelevant interaction misguidance as a key issue, where the discriminator incorrectly penalizes a realistic ego agent because of unrealistic interactions among its neighbors.
* The proposed solution DecompGAIL introduces a decomposed discriminator architecture that explicitly separates realism into (1) ego-map realism and (2) pairwise ego-neighbor realism.
* This decomposed design structurally filters out the misleading neighbor-neighbor and neighbor-map interaction signals that cause instability.
* The method is enhanced with a social PPO objective that augments the agent’s reward with a distance-weighted sum of its neighbors' rewards, promoting overall population realism.
* The method achieves SOTA on WOSAC.

### Strengths
* DecompGAIL achieves state-of-the-art results on WOMD sim agents and demonstrates significantly improved training stability.
* Irrelevant interaction misguidance is a novel problem that the authors discovered. GAIL has been notoriously difficult to train.
* The authors show strong empirical validation that the proposed approach significantly improves training stability and in addition achieves strong results on WOSAC.
* The ablation study validates that each proposed component contributes to the strong results.

### Weaknesses
* While the solution of decomposing the discriminator is sound it is engineered and strongly dependent on the input features that are used by the sim agent model. For example, it’s unclear how this approach could be applied to a simulator model that leverages images and sensor sim.

### Questions
* Are there failure cases or poor quality examples that could be included?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work proposes DecompGAIL, which aim to address the instability of the Generative Adversarial imitation Learning. The authors proposes to decomposes the realism to ego-map and ego-neighbor to avoid weakly relevant neighbor–neighbor interactions. Alghough it shows better training stability, the model is built upon a very strong pretrained backbone, and the results are saturated (only minimal gain) on the overall realism. The authors should conduct a more comprehensive study of why do we need Generative Imitlation Learning framework, and highlight how this GAIL-finetuning provided a different aspects, dimensions of traffic simulation

### Strengths
- The paper is well-written and the proposed Decomposed GAIL method is more stable than prior GAIL baselines.
- DecompGAIL achieves competitive results on the Sim Agent Challenge 2025, though the gain is very small.
- The main potential of this work is the discriminator, which can provide a useful realism signal compared to prior metrics (see weakness section)

### Weaknesses
- Qualitative results are not interesting and are very similar to previous works
- Overall, DecompGAIL’s advantage compared to prior works is unclear, given that the performance improvements are very small (± 0.01 realism score).
- The pretrained backbone already attains a high discriminator score (~0.5) from Figure 3, which suggests the added GAIL module provides minimal gain.
- I suggest the authors to start with a underperformed backbone w/ GAIL finetuning, and the main potential of this work is to  showcase the discriminator:whether it provides a uesful realism signal, compared to prior metrics such as Waymo Sim Agents Challenge
- In Sec 5.4, the authors claimed that BC suffers from a covariate shift problem by mentioning higher collision likelihood and very minimal gain on realism. This may not hold true as the Sim Agent Challenge measures distributional realsim instead of rule satisfaction; see [1].


[1] Wang M., Wang J., Ye T., Chen J., Yu K. (2025). Do LLM Modules Generalize? A Study on Motion Generation for Autonomous Driving. CoRL.

### Questions
- How does this work compared to supervised fine-tuning, what are the advantages of using GAIL, given that the gain is very minimal  compared to the original SMART 0.05 
- The definition and handling of “ego” vs “neighbor” agents is unclear: during training, are all agents’ policies updated simultaneously, or is only the ego agent’s policy learned while neighbors remain fixed? How does this affect memory usage and training dynamics (especially when using a decomposed discriminator)?
- Please address the weakness sections

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses a key limitation in multi-agent Generative Adversarial Imitation Learning (GAIL) for traffic simulation, where existing discriminators penalize realistic agents simply because other nearby agents behave unrealistically. The authors call this phenomenon irrelevant interaction misguidance.
To solve this, they propose DecompGAIL, which decomposes the discriminator into two terms: ego–map realism (how the ego vehicle interacts with the static environment) and ego–neighbor realism (how it interacts with relevant nearby agents). Irrelevant neighbor–neighbor or distant-agent interactions are filtered out through distance-weighted aggregation. In addition, a social PPO reward encourages coherent scene-level realism by adding distance-weighted neighbor rewards to each agent’s objective. Experiments on the WOMD Sim Agents 2025 benchmark show that DecompGAIL improves realism metrics and training stability compared with prior multi-agent GAIL baselines.

### Strengths
- Clearly identifies and mitigates the irrelevant interaction misguidance problem.

- The decomposition is simple, computationally efficient, and compatible with existing frameworks.

- Demonstrates improved realism and stability on public benchmarks with comprehensive ablations.

- Offers a general recipe that can transfer to other multi-agent imitation settings.

### Weaknesses
- Comparisons omit Wasserstein or gradient-penalized discriminators, making it unclear whether decomposition alone drives the gains.

- The social reward could unintentionally amplify correlated neighbor behaviors.

- The related work section omits several closely related papers:

A. Kuefler, J. Morton, T. Wheeler, and M. J. Kochenderfer, “Imitating Driver Behavior with Generative Adversarial Networks,” IEEE Intelligent Vehicles Symposium (IV), 2017, pp. 204–211.

R. P. Bhattacharyya, B. Wulfe, D. J. Phillips, A. Kuefler, J. Morton, R. Senanayake, and M. J. Kochenderfer, “Modeling Human Driving Behavior through Generative Adversarial Imitation Learning,” CoRR, 2020.

H. Chen, T. Ji, S. Liu, and K. Driggs-Campbell, “Combining Model-Based Controllers and Generative Adversarial Imitation Learning for Traffic Simulation,” IEEE ITSC 2022, pp. 1698–1704.

K. Brown, K. Driggs-Campbell, and M. J. Kochenderfer, “Modeling and Prediction of Human Driver Behavior: A Survey,” arXiv:2006.08832, 2020.

### Questions
How sensitive is performance to the decay parameters $\alpha$ and $\beta$ for interaction weighting and social rewards?

When freezing or fine-tuning the map encoder during discriminator training, how does stability or realism change?

Could the social reward cause feedback loops where agents learn to exploit mutual rewards without improving realism?

### Soundness
3

### Presentation
3

### Contribution
3
