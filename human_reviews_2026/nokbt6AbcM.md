# HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Robotic Assembly

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Long-horizon manipulation tasks represent a significant challenge in robotics, demanding both strategic, high-level reasoning and fast, precise, low-level control. While recent advances in generative models have shown promise in generating behavior plans for long-horizon tasks, they often lack a principled framework for hierarchical decomposition and struggle with the computational demands of real-time execution, due to their iterative denoising process. In this work, we introduce $\textbf{Hierarchical Diffusion-Flow}$ ($\texttt{\textbf{HDFlow}}$), a novel hierarchical planning framework that optimally leverages the strengths of $\textit{diffusion}$ and $\textit{rectified flow}$ models. $\texttt{\textbf{HDFlow}}$ employs a high-level diffusion planner to generate sequences of strategic subgoals in a learned latent space, capitalizing on diffusion's powerful exploratory capabilities. These subgoals then guide a low-level rectified flow planner that generates smooth and dense trajectories, exploiting the speed and efficiency of ordinary differential equation (ODE)-based trajectory generation. This hybrid approach synergistically combines the strengths of both models to overcome the limitations of single-paradigm generative planners, enabling robust and efficient long-horizon planning. We evaluate $\texttt{\textbf{HDFlow}}$ on four challenging furniture assembly tasks, where it significantly outperforms state-of-the-art methods. Project website: https://hdflow-page.github.io/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes HDFlow, a hierarchical planning framework for long-horizon robotic assembly tasks that combines diffusion models for high-level subgoal planning with Rectified Flow for low-level trajectory generation. The method incorporates a contrastive-trained world model and manifold-aware EBM guidance to ensure generated subgoals lie on the manifold of feasible plans. Evaluated on FurnitureBench with both simulated and real robot experiments, HDFlow achieves superior performance over baselines.

### Strengths
###### **Solid system design and engineering execution.**

The paper proposes a well-motivated hierarchical architecture that effectively leverages different generative models for different planning levels. The key insight that high-level planning requires multi-modal exploration while low-level planning prioritizes speed and smoothness demonstrates good understanding of the problem structure. The hybrid design using diffusion models for high-level subgoal planning and Rectified Flow for low-level trajectory generation is a principled engineering choice that avoids the computational bottleneck of iterative denoising at all hierarchical levels.

###### **Principled theoretical analysis**

The authors provide formal analysis of the guidance gap problem in Proposition 4.1, quantifying the error bound of EBM guidance. Proposition 4.2 further establishes convergence guarantees for the alternating projection approach, ensuring the method is mathematically sound.

###### **Comprehensive experimental validation.**

The ablation study systematically quantifies the contribution of each component. This transparency allows readers to understand the role of each module. Beyond simulation results on FurnitureBench, the paper includes real robot experiments, which validates the practical applicability of the method.

### Weaknesses
###### Limited methodological novelty—primarily an engineering integration of existing techniques

While the system design is competent, HDFlow essentially combines well-known components without fundamental innovation. The use of Rectified Flow for low-level planning is a natural application of its known speed advantages over diffusion models, not a conceptual breakthrough. The core contribution is demonstrating that this particular combination works on FurnitureBench, but this represents incremental engineering rather than algorithmic insight. 

###### Narrow task focus contradicts the paper's broad title and claims of generality.

The paper is titled "FOR LONG-HORIZON ROBOTIC ASSEMBLY" and positions HDFlow as a general framework for long-horizon manipulation. However, the method contains no assembly-specific design choices, no reasoning about part connections, geometric constraints, or contact-rich manipulation strategies that characterize assembly tasks. If HDFlow is genuinely a general imitation learning method for long-horizon tasks, the authors should validate it across diverse benchmarks beyond furniture assembly. The exclusive evaluation on FurnitureBench raises concerns about overfitting the approach to this single benchmark. The lack of cross-domain validation significantly weakens the paper's contribution claims and makes it difficult to assess whether the design choices generalize beyond this specific task distribution.

###### Missing analysis of multi-modal exploration capabilities claimed in the abstract. 

The paper emphasizes that the high-level diffusion planner provides "exploration and multi-modal diversity to discover viable sequences of subgoals", positioning this as a key advantage over single-paradigm approaches. However, the experimental section provides no evidence of this multi-modality. Are there furniture assembly tasks where multiple valid assembly orders exist (e.g., assembling legs in different sequences)? Can HDFlow discover and execute these different strategies? Without visualizations of diverse generated subgoal sequences or quantitative measures of trajectory diversity, this claimed advantage remains unsubstantiated. Adding experiments that demonstrate the planner generating multiple qualitatively different solutions to the same task would strengthen the paper's narrative about the benefits of the hierarchical diffusion approach.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
HDFlow is a latent hierarchical planner. Based on the latent space learned using a RSSM formulation, it uses a diffusion based high-level sub-goal latent generation given the start and final goal latents, followed by a flow-based low-level planner to connect two subsequent sub-goals. The training process requires positive and negative trajectories to train the RSSM world model with an auxilliary constrastive loss. The positive-negative rollouts are further used to learn an energy function to guide the high-level planner as well. The high-level planner further imposes a projection step where a noisy sequence of subgoal latents is projected into the closest sequence from a successful trajectory. The low-level planner is flow-based and useful for fast execution.

### Strengths
1. The use of diffusion for learning a diverse sequence of subgoals and flow matching for fast trajectory generation is powerful. This is backed by ablations with hierarchical diffusion only and hierarchical flow only baselines.

2. The use of manifold-aware Energy-Based Model guidance and projection based on successful sub-goal sequences is interesting. It ensures that the generated sequence of subgoals are feasible, connects the given start and goal latents, and leads to a successful sequence.

### Weaknesses
1. It feels like the high-level planner is just looking up successful trajectories from the set used to construct the projection manifold. If the projection happens at a trajectory level, this literally becomes selecting one of the successful goal reaching trajectories. How many trajectories are used to construct this manifold? Performing the projection step on a 2048 size latent for 300 denoising steps seems like very compute expensive?

2. The fact that an interpolation of 10 latent trajectories lead to a feasible latent trajectory in the latent space points to the fact that the demonstrations are very similar in nature or there are concentrated clusters in the latent space. 

3. A core problem of hierarchical approaches is that: at inference, high-level planner might output a sequence of subgoal that is out of distribution of the learned low-level policy. It seems, with the hard projection step into a feasible "seen" sequence of subgoals, the authors are mitigating this.

Minor note: the webiste did not work for me.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Hierarchical Diffusion-Flow (HDFlow), a hierarchical generative planning framework that combines high-level diffusion models with low-level rectified flow models. Building on Hierarchical Diffusion approaches, HDFlow aims to exploit the strength of diffusion-based planning (high-level) with rectified flow models (low-level). The method is evaluated across both simulation and real-world robotic settings, showing consistent gains in success rate and inference efficiency over prior baselines.

### Strengths
1. The paper demonstrates HDFlow’s effectiveness across simulation and real-world domains, providing empirical evidence that the hierarchical integration improves both planning quality and execution reliability.

2. Results indicate that HDFlow achieves higher success rates while reducing inference time, addressing a known limitation of diffusion-based planners.

### Weaknesses
1. The framework relies on an external latent representation (RSSM) trained via contrastive objectives. While this improves performance, it introduces an additional pretraining stage that complicates reproducibility and may limit generalization to unseen domains. Notably, Table 3 shows a significant degradation without the contrastive latent, indicating limited robustness of the hierarchical design alone.

2. Conceptually, the contribution appears as an incremental combination of existing paradigms of Hierarchical Diffusion and Rectified Flow, without introducing a fundamentally new mechanism.

3. The core research question (“Is a single generative modeling paradigm optimal for all hierarchy levels?”) has already been explored in SHD and HDMI, which demonstrated similar multi-level generative decompositions. The paper could strengthen its claim by clarifying what specific failure modes of single-paradigm approaches HDFlow overcomes.

### Questions
None

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces HDFlow, a novel hierarchical planning framework designed to solve long-horizon, contact-rich robotic assembly tasks. The core idea is a novel hybrid hierarchical planning framework consisting of a diffusion and a rectified flow models. By leveraging the strength of each type of generative models, the proposed hybrid method, HDFlow, mitigates the limitations of single-paradigm generative planners, and enables efficient long-horizon planning.

Specifically, HDFlow uses a diffusion model as a high-level planner for its better exploration and multi-modal capability; and it uses a rectified flow model as a low-level planner, leveraging its nature as an Ordinary Differential Equation (ODE) to rapidly and deterministically generate smooth, dense trajectories between two given subgoals from the high-level planner.

This hybrid "Diffusion-Flow" approach aims to get the best of both worlds: the robust, multi-modal planning of diffusion and the inferential speed of flow models.

The authors evaluate HDFlow on four challenging tasks from the FurnitureBench benchmark in both simulation and the real world. The results show that HDFlow significantly outperforms all baselines, including flat diffusion planners (Diffuser, DD) and pure-diffusion hierarchical planners (SHD, HDMI).

### Strengths
1. The central idea of a hybrid Diffusion-Flow hierarchy is well-motivated. While hierarchical diffusion planners exist, they typically use diffusion at both levels, thereby inheriting the inference latency bottleneck. HDFlow's insight on that high-level strategic planning and low-level trajectory generation have fundamentally different requirements (exploration vs. speed) is well-articulated. The addition of a manifold-aware EBM to guide the diffusion process is also a contribution that addresses a known issue (manifold deviation) in generative planning.

2. Experiments demonstrates that HDFlows significantly outperforms multiple baselines on the challenging furnitureBench benchmark, which features long-horizon, contact-rich robotic manipulation. A set of ablation studies, sampling speed comparison are provided, along with real-world robot evaluation.


3. The paper is well-written and easy to follow. The problem is clearly motivated, and the proposed method is developed logically. The figures are also effective at communicating the method's design and benefits.

### Weaknesses
1. The papers provide a set of ablation studies in Table 3. 
It is a bit surprising that removing the Contrastive WM alone will significantly reduce the performance. The authors claim that the proposed contrastive loss objective can enable a notion of progress towards a goal. While the contrastive objective does drag $z_k$ and $z_G$ closer and pushing 
$z_k$ and $z_j$ away, can the authors provide further explanation on how this effectively "enable a notion of progress towards a goal". Will the objective compromise the planner if $z_k$ and $z_G$ is too close/similar? How is the training stability of this loss?


2. For the world model, I would like to see ablation studies of loss $\mathcal{L}_{IDM}$. It would also be interesting to see how will HDFlow perform if directly using DINOv2 features as latent $z$, without training an additional RSSM world models.

3. How is the scalability of RSSM as the world models (especially for long-horizon, high dimension state space)? Qualitative reconstruction results of the world models and relevant analysis are needed. In addition, are there any other architecture choices? 

4. In Section 4.1, I cannot find where $q_{\phi}$ is defined. While it is defined in the appendix, I recommend defining all important notations in the same section.

5. The authors should provide more details regarding the loss in Eq. 18, for example how $\mathcal{P}_{{M}}$ is obtained at training-time. From Step 2 in Line 306, it seems like $\mathcal{P}$ is obtained based on the $k$ nearest neighbors of the Tweedie's estimate. Is it also the case for training time? Pseudocode for training and inference is needed.

6. This paper only evaluates the proposed method on several robot assembly tasks, leaving the method's performance on other tasks unknown. Evaluating on a diverse set of tasks can help better understand capability of HDFlow. For example, some commonly used planning benchmark, such as Maze or MujoCo locomotion, can be incorporated.

### Questions
See the weakness above.

### Soundness
3

### Presentation
3

### Contribution
3
