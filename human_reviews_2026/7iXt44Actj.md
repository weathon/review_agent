# Value Matching: Scalable and Gradient-Free Reward-Guided Flow Adaptation

- Decision: Accept (Poster)
- Scores: 4, 2, 4, 6

## Abstract
Adapting large-scale flow and diffusion models to downstream tasks through reward optimization is essential for their adoption in real-world applications, including scientific discovery and image generation. While recent fine-tuning methods based on reinforcement learning and stochastic optimal control achieve compelling performance, they face severe scalability challenges due to high memory demands that scale with model complexity. In contrast, methods that disentangle reward adaptation from base model complexity, such as Classifier Guidance (CG), offer flexible control over computational resource requirements. However, CG suffers from limited reward expressivity and a train-test distribution mismatch due to its offline nature. To overcome the limitations of fine-tuning methods and CG, we propose Value Matching (VM), an online algorithm for learning the value function within an optimal control setting. VM provides tunable memory and compute demands through flexible value network complexity, supports optimization of non-differentiable rewards, and operates on-policy, which enables going beyond the data distribution to discover high-reward regions. Experimentally, we evaluate VM across image generation and molecular design tasks. We demonstrate improved stability and sample efficiency over CG and achieve comparable performance to fine-tuning approaches while requiring less than 5% of their memory usage.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes an online algorithm called Value Matching (VM) to learn a value function within an optimal control framework, addressing the scalability challenges faced when adapting large-scale flow and diffusion models to downstream tasks via reward optimization. VM enables flexible control over the complexity of the value network, offering adjustable memory and computational requirements. It supports optimization with non-differentiable rewards and operates in an on-policy manner, allowing it to go beyond the training data distribution and explore high-reward regions. Experiments on image generation and molecular design tasks demonstrate that VM achieves better stability and sample efficiency compared to Classifier Guidance (CG).

### Strengths
1.It learns the scalar value function directly rather than estimating guidance gradients, leading to more stable training.
2.It offers significant theoretical and practical value by enabling low-cost reinforcement learning (RL) fine-tuning.

### Weaknesses
1.The evaluation is insufficient, with comparisons limited to only CT-PPO.
2.The core contribution is primarily encapsulated in Equation 11, but the paper devotes excessive space to background, making it hard to follow.

### Questions
1.Please clarify how your work differs from the following approaches, and discuss the performance gap:
Inference-Time Alignment Control for Diffusion Models with Reinforcement Learning Guidance
Efficient Controllable Diffusion via Optimal Classifier Guidance
2.Although your method reduces fine-tuning costs, can it outperform the following full fine-tuning approaches?
Large-scale Reinforcement Learning for Diffusion Models
Training Diffusion Models with Reinforcement Learning
DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes value matching (VM), an online approach that relies on a small learned value network for guiding the model towards higher rewards. The proposed method is derived from optimal control theory and effectively learns a vector field network and a value network that simultaneously approximate the running cost. On downstream generation tasks, the proposed approach achieved better performance across multiple objective rewards.

### Strengths
- The paper's proposed method is built upon adjoint matching (AM), which is intuitive and easy to follow given the AM framework.
- The paper, built upon AM, also shares some desirable mathematical properties or guarantees.
- The proposed method has better running time compared to gradient-based AM models, but the performance comparison remains unclear (see weaknesses).

### Weaknesses
- Although the theoretical part in the paper seems solid to me, the experimental evaluations have **very limited baselines and focus only on small-scale datasets or reward functions** to convincingly demonstrate the effectiveness of the proposed approach.
  - **Regarding the number of baselines**. For each task, only the CFG and a single model were used as the baseline, even though many existing works were mentioned in the paper. For example, for image generation, there are gradient-based methods such as Adjoint Matching, SOC, FlowGrad [1], and DFlow [2], as well as other RL approaches such as ReFL [3] and [4] (gradient-free) and DRaFT [5] (gradient-based). For molecule generation, DFlow and OCFlow [6] achieved decent generations with almost perfect stability and validity. Current results do not fully demonstrate the superiority of the proposed approach over existing work.
  - **Regarding the evaluation metric and reward function**. The reward function used in the experiments in this paper is either a toy example from existing work or a non-standard metric not established in previous work on the same task, further weakening the paper's claims, as most metrics are not necessarily comparable. 
    - For example, the standard task in previous flow matching guidance papers on image generation almost all focus on guiding text-to-image models like SD2/3 and use robust metrics like CLIP score, PickScore, or HPSv2 to prevent easily hacking the rewards (e.g., [6] demonstrated that the compression metric can be easily hacked to almost perfect but meaningless generations). For molecule generation, it also remains unclear why the authors did not follow the standard molecule generative modeling evaluation setup in [7] (uses CFG), DFlow, and OCFlow to evaluate the generation and compare the results with these existing works easily. 
    - For molecule generation, it is widely known that quantum chemical calculations are either computationally expensive (ab initio methods) or highly inaccurate (empirical or semiempirical methods). It remains unclear what class the calculation method used in the experiments belongs to, how accurate and generalizable it is (GEOM-Drug is known to have some unreasonable configurations), and what the computational time is, which is crucial for reproducibility. In addition, crucial molecular properties like stability and validity were never mentioned or compared in the paper. Therefore, I am highly skeptical about why the authors did not follow the standard and easy approach in existing work but opted for a seemingly far more complex setup. I would highly discourage such an approach when comparing with baselines for fairness.

To summarize, for a paper emphasizing the scalability of the proposed method, I believe the existing experiments are, in contrast, limited in scope and poorly credible in supporting its fundamental claims. 

- **The theoretical contributions in this paper are limited to me**. The core idea is almost identical to [VGG-Flow](https://openreview.net/forum?id=6MmOy2Ji8V). The scale of the experiments and the number of baselines in this paper fall significantly short of those in VGG-Flow, even if the latter is to be considered concurrent. In addition, there are existing works that have explored the role of the value function or its equivalent, the Q-function, for generative modeling, such as [8] and [9]. Despite different application domains, the underlying core ideas are pretty similar to me. Given that the theoretical results primarily come from the adjoint-matching paper, the theoretical contributions are limited.
 
- The method's scalability hinges on the value network being "significantly smaller than the base model." But what happens when the reward function is extremely complex? A small network may fail to accurately model the true value function, creating a new performance bottleneck. Is there a trade-off between VM's memory savings and its ability to represent a complex reward landscape? This weakness also echoes in the paper's limited, small-scale evaluation, as thoroughly mentioned in the first part.

- The algorithm learns by regressing the value network's predictions $V_{\theta}(x_t, t)$ onto a Monte Carlo estimate of the cost functional $\hat{J}$ (Eq. 12). This target $\hat{J}$ is based on a single sample trajectory and also depends on the current value function itself. This can be a very high-variance target, which is known to make value-based RL difficult to stabilize. The paper uses a weighting scheme, but the inherent stability of this learning process, especially for very long trajectories, is a potential concern.

- To sample from VM, one must run both the large base model and the (smaller) value network at every step to compute the guiding gradient $\nabla V_{\theta}$. While this is still much faster than other gradient-based inference-time optimization schemes, it's not "free" and adds computational overhead compared to using a single fine-tuned model. Other gradient-free approaches, such as ReFT and [6], should be benchmarked to support a more credible claim. Additionally, this issue may be coupled with the expressive power of the value net for more complex rewards mentioned above, and it may not be easy to find a balance. 

[1] Liu, Xingchao, et al. "Flowgrad: Controlling the output of generative odes with gradients." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[2] Ben-Hamu, Heli, et al. "D-flow: Differentiating through flows for controlled generation." arXiv preprint arXiv:2402.14017 (2024).

[3] Luong, Trung Quoc, et al. "Reft: Reasoning with reinforced fine-tuning." arXiv preprint arXiv:2401.08967 (2024).

[4] Fan, Jiajun, et al. "Online reward-weighted fine-tuning of flow matching with wasserstein regularization." The Thirteenth International Conference on Learning Representations. 2025.

[5] Clark, Kevin, et al. "Directly fine-tuning diffusion models on differentiable rewards." arXiv preprint arXiv:2309.17400 (2023).

[6] Wang, Luran, et al. "Training free guided flow matching with optimal control." arXiv preprint arXiv:2410.18070 (2024).

[7] Hoogeboom, Emiel, et al. "Equivariant diffusion for molecule generation in 3d." International conference on machine learning. PMLR, 2022.

[8] Zhang, Shiyuan, Weitong Zhang, and Quanquan Gu. "Energy-weighted flow matching for offline reinforcement learning." arXiv preprint arXiv:2503.04975 (2025).

[9] Alles, Marvin, et al. "FlowQ: Energy-Guided Flow Policies for Offline Reinforcement Learning." arXiv preprint arXiv:2505.14139 (2025).

### Questions
Please refer to the list of weaknesses above. In addition:
- VM is an *on-policy* algorithm (Algorithm 1), which means it discards past trajectories after each update. On-policy methods are generally known to be sample-inefficient. While the paper shows VM is more efficient than CG, how does its absolute sample efficiency compare to (hypothetical) offline value-based methods?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Value Matching (VM), a novel, scalable, and gradient-free algorithm for reward-guided adaptation of large-scale flow and diffusion models. VM uses less memory than finetuning methods and offers improved stability compared to classifier guidance.

### Strengths
1. By decoupling the value learning from the base model, VM requires a separate, small value network. This reduces memory usage by over 95% compared to fine-tuning.
2. VM operates on-policy, which gives it enhanced reward expressivity and stability compared to CG.
3. The presentation of this work is clear and easy to follow, with abundant theoretical justifications.

### Weaknesses
1. The main quantitative results in Figures 8 and 9 focus only on Reward, KL, and FID. Why not directly compare the generated samples' diversities?
2. The paper contrasts VM's simplicity with the "extensive hyperparameter search" required by CT-PPO. However, this work does not include essential ablations studies on the architecture and size of the separate value network. It is important to know the sensitivity of VM's final performance to variations in the value network. Does VM also require a careful choice of a value network to support acceptable performance?
3. A concern lies in the selection of inference-time techniques. While CG is a classic baseline, its performance and controllability are not comparable to other non-fine-tuning guidance methods. Why not compare with those in experiments?
4. Figure 6 does not compare with any baselines. I think any training or non-training methods can perform well for the compressibility task.
5. In Figure 9, why is VM only compared with CT-PPO? For example, for aesthetic scores, many direct propagation algorithms can easily achieve >7 aesthetic reward in 10 epochs, but they are not mentioned. The current results are also confusing. It seems unclear what this means "VM demonstrates performance comparable to CT-PPO but with more predictable and stable behavior."

### Questions
see above

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
3

### Summary
The authors propose an online algorithm for learning the value function of the flow matching models by formulating the reward-guided generation as an optimal control problem. By drawing the analogy that classifier guidance is seen as offline value function learning, the method enables reward-guided generation without reward fine-tuning the base flow matching model. Specifically, it trains a separate value model as the value function. This admits flexible reward model design (in terms of both architectures and model sizes) and significantly reduces memory usage compared to directly fine-tuning the base model. The method achieves comparable performance with PPO.

### Strengths
+ The connection between CG and VM is intuitive (VM viewed as an online generalization of CG)
+ The results of VM achieving comparable performance with PPO on image and molecule generation tasks look promising.

### Weaknesses
+ One of the major benefits of using reward guidance over directly doing reward fine-tuning (if the reward is non-differentiable, one can use an approach similar to [1]) is the reduced memory footprint. However, one can adopt LoRA to effectively reduce the memory requirement of the latter. I suggest the author compare different LoRA setups to show the trade-off of memory usage vs. fine-tuned performance of the baseline method to better illustrate how the proposed method does in preservation performance while using much less memory.
+ Since the value model is decoupled from the base model, it would be good to perform a bit of a scaling study on the scale of the value model.
+ Optimization with more reward functions will make the results more convincing.
+ It would be good to add [1] to related work as it also incorporates value function learning for reward fine-tuning flow-matching models.

[1] Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward, CVPR 2025

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
