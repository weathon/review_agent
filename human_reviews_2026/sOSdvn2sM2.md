# Diffusion Policy through Conditional Proximal Policy Optimization

- Avg Score: 5.50
- Decision: Reject
- Scores: 2, 6, 8, 6

## Abstract
Reinforcement learning (RL) has been extensively employed in a wide range of decision-making problems, such as games and robotics. Recently, diffusion policies have shown strong potential in modeling multi-modal behaviors, enabling more diverse and flexible action generation compared to the conventional Gaussian policy. Despite various attempts to combine RL with diffusion, a key challenge is the difficulty of computing action log-likelihood under the diffusion model. This greatly hinders the direct application of diffusion policies in on-policy reinforcement learning. Most existing methods calculate or approximate the log-likelihood through the entire denoising process in the diffusion model, which can be memory- and computationally inefficient. To overcome this challenge, we propose a novel and efficient method to train a diffusion policy in an on-policy setting that requires only evaluating a simple Gaussian probability. This is achieved by aligning the policy iteration with the diffusion process, which is a distinct paradigm compared to previous work. Moreover, our formulation can naturally handle entropy regularization, which is often difficult to incorporate into diffusion policies. Experiments demonstrate that the proposed method produces multimodal policy behaviors and achieves superior performance on a variety of benchmark tasks in both IsaacLab and MuJoCo Playground.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Diffusion Policy through Conditional Proximal Policy Optimization (DP-CPPO), a novel on-policy reinforcement learning (RL) framework that integrates diffusion generative models with Proximal Policy Optimization (PPO). The key motivation stems from the difficulty of computing action log-likelihoods within diffusion models, which traditionally limits their use in on-policy settings.

### Strengths
The method avoids the costly recursive log-likelihood computation typical in diffusion-based RL (GenPo or DACER), replacing it with a simple Gaussian-based PPO step.

Unlike prior diffusion methods (FPO), which struggle with entropy terms, DP-CPPO supports entropy regularization analytically through a Gaussian lower bound, enabling controlled exploration in policy learning.

### Weaknesses
- The explanation at the end of 3.2 is far-fetched. EMA is a parameter smoothing but not a KL constraint.
* Missing comparisons with recent **on-policy diffusion methods** such as GenPo and DPPO.
* Absent results on **off-policy diffusion-RL methods** (**DIME**, **DACER**, **MaxEntDP**) evaluated on OpenAI Gym MuJoCo benchmarks.
* Multimodal behaviors are only demonstrated qualitatively in a **Multi-Goal** environment; **no quantitative metrics** (e.g., KL-divergence, mode count, action entropy) are reported.
* No **wall-clock performance comparison** with **FPO** under equivalent computational resources, making the claim of “high computational efficiency” unsubstantiated.
* No **variance statistics across random seeds** (only mean and visualization curves are reported), making it difficult to assess robustness under high-variance RL training.

Ablations

**Regularization Terms:** Independently and jointly ablate the entropy and score regularizations.

**EMA Mechanism:** Evaluate performance degradation when EMA is removed.

**Flow Steps:** Conduct a systematic study on how the number of flow steps affects the reward.

### Questions
How tight is the bound of this entropy related to mutual information? How will this error affect the experimental parameter setting? Can you provide an experimental analysis?

Given that the monotonic improvement of the strategy cannot be strictly guaranteed at the end of Section 3.2, is it reasonable to regard the diffusion process as a policy iteration process?

### Soundness
2

### Presentation
1

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
This paper presents a novel and efficient framework to train diffusion policies in on-policy reinforcement learning. Instead of directly optimizing diffusion policies, this paper aligns the policy iteration process with the diffusion process. Specifically, at each policy improvement step,  the old diffusion policy is forzen and a Gaussian residual policy is trained via conditional PPO to optimize the combination of these two models. Then, the combination policy is distilled into a new diffusion policy using the flow matching loss. The old diffusion policy is updated utilizing the EMA technique, and the proposed policy iteration can be approximately regarded as monotonically improving. Additionally, entropy regularization and score-based regularization are incorporated to enhance exploration and training stability, respectively. Experiments on IssacLab and MuJoCo Playgroud demonstrate that the proposed method effectively learns multimodal policies and achieve superior performance across various benchmark tasks.

### Strengths
1. The paper addresses a key and challenging problem—applying diffusion policies in on-policy reinforcement learning.
2. The proposed training framework is both efficient and elegant, as it only requires optimizing the Gaussian residual policy and training diffusion models using the simple flow matching loss, thereby avoiding the intractable computation of diffusion model log-likelihoods.

### Weaknesses
1. The paper would be strengthened by including a comparison with DPPO[1], which also employs an on-policy RL algorithm (PPO) to train diffusion policies.
2. Moving the learning curves from the appendix to the main text would provide a clearer comparison of performance against baseline methods.

### Questions
1. In line 268, the paper states that the score-based regularization term, which tends to let a Langevin dynamics update toward the standard Gaussian, can accelerate and stabilize training. Why such a term can accelerate and stabilize training?
2. Is the proposed training paradigm—aligning policy iteration with the diffusion process—also applicable to off-policy reinforcement learning? Have the authors experimented with replacing PPO with off-policy algorithms when training the Gaussian residual policy? 

[1] Allen Z Ren, Justin Lidard, Lars L Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha Majumdar, Benjamin Burchfiel, Hongkai Dai, and Max Simchowitz. Diffusion Policy Policy Optimization. International Conference on Learning Representation. 2025.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a novel policy parameterization scheme and training algorithm. By parameterizing the new policy as a convolution of a reference policy and a conditional Gaussian kernel , the policy optimization process and the entropy regularization step are made simple and efficient. The authors demonstrate the algorithm's stability and effectiveness through extensive experiments.

Main Contributions:

A Novel DP-CPPO Framework: The authors propose a new reinforcement learning framework (DP-CPPO) that effectively supports the use of diffusion models in on-policy algorithms and, notably, is compatible with entropy regularization.

Innovative Policy Parameterization: By parameterizing the new policy as a convolution of a reference policy and a conditional Gaussian kernel, the policy optimization process and entropy regularization are made simple and efficient.

Tractable Entropy Regularization: The framework naturally resolves the difficulty of computing the diffusion policy's entropy $\mathcal{H}(\pi_{\theta})$. The authors achieve efficient exploration by maximizing a tractable lower bound of the entropy, namely the entropy of the Gaussian.

### Strengths
The main strength of this paper lies in its novel and significant methodology, which cleverly bypasses the intractable $log \pi(a|s)$ computation in on-policy diffusion training by treating each policy iteration as a denoising step. The method is computationally efficient, with GPU memory occupation comparable to PPO while maintaining reasonable training times. Furthermore, it elegantly solves the difficult entropy regularization problem by optimizing a tractable entropy lower bound, a key feature lacking in methods like FPO. Strong empirical results, including demonstrated multi-modal capabilities, excellent benchmark performance, and ablation studies proving the necessity of all components, confirm the method's effectiveness.

### Weaknesses
The method has several weaknesses. First, it introduces a policy fitting step (Flow Matching) after the optimization step (CPPO), which creates an approximation error whose cumulative impact on convergence is unassessed. Second, the algorithm relies heavily on an EMA approximation to ensure monotonic improvement, which is not a theoretical guarantee and may fail if policy updates are too large.

### Questions
1. Regarding stability, how does the Flow Matching fitting error behave during training? What impact does this error have on the stability of the CPPO optimization step? Is there a risk of the fitting process lagging behind large policy updates?

2. What is the effective batch size (or number of samples) used to update the flow model (Eq. 12) in each policy iteration?

3. Instead of training the flow model for a fixed number of epochs in each iteration (Algorithm 1, Line 4-5), have you considered an adaptive update scheme? For instance, training the flow model until its loss (Eq. 12) converges below a specific threshold. What effect might this have on the overall algorithm's stability and computational efficiency?

### Soundness
3

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
4

### Summary
This paper proposes a diffusion-based on-policy RL method. Concretely, given an existing policy \(\pi^{k}\), a single-step Gaussian kernel is trained using the PPO objective (Equation (11)) such that \(\pi^{k}\), when passed through the kernel, produces a target distribution \(\pi^{\mathrm{target}}\). Subsequently, Flow Matching is used to fit this target distribution and thereby complete training. In addition, the authors investigate score-based regularization of the Gaussian kernel to align it with a single diffusion step; ablation experiments reveal its efficacy. The work is evaluated on benchmarks such as MuJoCo Playground and IsaacLab, and demonstrates top performance compared to other baselines.

### Strengths
1. The method is simple yet novel: it separates the Flow Matching stage from reinforcement-learning policy improvement, by first training a Teacher policy under a familiar PPO training paradigm and then using Flow Matching to learn that Teacher policy.
2. The ablation studies are thorough: through experiments the paper convincingly shows the importance of both the entropy regularization (which allows the method to outperform FPO) and the score-based regularization (which prevents the Teacher policy from diverging and thus makes Flow Matching feasible).

### Weaknesses
1. Some experimental descriptions are insufficiently clear. The meaning of “Flow” vs. “Flow + Residual” is ambiguous. The original text only uses the phrase *“diffusion-only (denotes ‘Flow’) policy and the combined policy”* (lines 390–391) without further clarifying exactly what “Residual” constitutes.
2. Details of the training process are missing. Since the method relies on Flow Matching to fit \(\pi^{k}\) after \(p_{\boldsymbol{\theta}}\), the paper should report: evidence that Flow Matching converges to the Teacher policy. Although Table 5 reports `training epochs = 15` and `mini batches = 4`, I am concerned that this is insufficient to guarantee that the Flow-Matching stage has converged to the Teacher policy.
3. Baselines are too few. The paper uses only FPO and PPO as baselines, which limits the strength of the empirical claims. Even though the authors mention (line 382) the implementation difficulty of FPO in a Torch-based IsaacLab environment, the omission of other relevant baselines is still a drawback. For example, the following open-source diffusion-RL baselines are available:

   * DACER ([https://github.com/happy-yan/DACER-Diffusion-with-Online-RL](https://github.com/happy-yan/DACER-Diffusion-with-Online-RL)) – JAX implementation
   * QVPO ([https://github.com/wadx2019/qvpo](https://github.com/wadx2019/qvpo)) – PyTorch implementation
   * DPPO ([https://github.com/irom-princeton/dppo](https://github.com/irom-princeton/dppo)) – PyTorch implementation
   * DIPO ([https://github.com/BellmanTimeHut/DIPO](https://github.com/BellmanTimeHut/DIPO)) – PyTorch implementation
     Given the availability of these implementations, it seems feasible to include them as baselines.

### Questions
In Section 3.3 you present score-based regularization and show its empirical benefit. My question is: why is it necessary to align the one-step Gaussian kernel with a one-step diffusion update? More specifically, would a simpler regularizer such as \(\mathrm{KL}(p_\theta||p_{\theta_{old}})\) (to prevent the kernel from making large jumps) suffice? In other words, the score-based regularizer seems somewhat analogous to enforcing a TRPO-style trust region; is the more sophisticated “score alignment” strictly required?

### Soundness
3

### Presentation
3

### Contribution
3
