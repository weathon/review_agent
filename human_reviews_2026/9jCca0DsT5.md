# Unrolled Networks are Conditional Probability Flows in MRI Reconstruction

- Decision: Reject
- Scores: 6, 8, 2, 2

## Abstract
Magnetic Resonance Imaging (MRI) offers excellent soft-tissue contrast without ionizing radiation, but its long acquisition time limits clinical utility. Recent methods accelerate MRI by under-sampling $k$-space and reconstructing the resulting images using deep learning. Unrolled networks have been widely used for the reconstruction task due to their efficiency, but suffer from unstable evolving caused by freely-learnable parameters in intermediate steps. In contrast, diffusion models based on stochastic differential equations offer theoretical stability in both medical and natural image tasks but are computationally expensive. In this work, we introduce flow ODEs to MRI reconstruction by theoretically proving that unrolled networks are discrete implementations of conditional probability flow ODEs. This connection provides explicit formulations for parameters and clarifies how intermediate states should evolve. Building on this insight, we propose Flow-Aligned Training (FLAT), which derives unrolled parameters from the ODE discretization and aligns intermediate reconstructions with the ideal ODE trajectory to improve stability and convergence. Experiments on three MRI datasets show that FLAT achieves high-quality reconstructions with up to $3\times$ fewer iterations than diffusion-based generative models and significantly greater stability than unrolled networks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper addresses accelerated MRI reconstruction from under-sampled k-space by establishing a theoretical equivalence between unrolled networks and conditional probability flow ODEs. It demonstrates that each cascade in an unrolled model corresponds to a forward-Euler step of a conditional probability flow, thereby imposing explicit constraints on the timestep schedule and model hyperparameters. This formulation provides a clear and useful theoretical perspective—viewing unrolled MRI reconstruction as a discretized conditional probability flow—which naturally leads to an ODE-grounded training scheme that is both simple to implement and empirically effective. Building upon this insight, the authors introduce Flow-Aligned Training (FLAT), which (i) enforces ODE-consistent timestep scheduling, (ii) fixes step sizes and weights through the ODE mapping, and (iii) introduces intermediate “velocity alignment” supervision via a composite loss. Experiments show that FLAT achieves higher or comparable PSNR and SSIM using only 12 steps, outperforming diffusion-based baselines that typically require 50–1000 iterations.

### Strengths
- **Unified view of unrolled networks and probabilistic flow models.**  
  The paper presents a significant conceptual advance by unifying *unrolled reconstruction networks* and *conditional probabilistic flow ODEs*. It formally shows that the cascaded updates in unrolled networks can be interpreted as **forward Euler discretizations** of conditional probability flows, thus providing a clear theoretical framework for understanding unrolled MRI reconstruction.  
  This perspective addresses several long-standing issues in unrolled models, including:  
  - Unstable or redundant reconstruction trajectories with unclear intermediate meanings.  
  - Poor interpretability due to backpropagation-based hyperparameter tuning.  
  
By redesigning the unrolled structure from a **probabilistic flow** perspective, the paper achieves better interpretability and stability while maintaining strong reconstruction performance. This work offers conceptual value not only to the unrolled MRI community but also to the broader field of deep iterative reconstruction, potentially inspiring more interpretable and theoretically grounded iterative architectures.

### Weaknesses
- **Inadequate diffusion-based baseline comparison.**  
  The comparison against diffusion-based methods is insufficient. The paper claims that FLAT outperforms traditional unrolled networks and is faster and better than diffusion-based approaches. However, the chosen baseline—**MC-DDPM**—is not representative or competitive. I strongly recommend including at least one **SOTA diffusion-based method**, such as **DDS [1]**, to substantiate the claimed advantages of FLAT over diffusion models.  


> [1] Chung, Hyungjin, Suhyeon Lee, and Jong Chul Ye. **"Decomposed Diffusion Sampler for Accelerating Large-Scale Inverse Problems."** *ICLR*, 2024.

---
While the proposed idea is novel and promising, the experimental section requires further strengthening. More robust comparisons and additional analyses (especially addressing the questions below) are necessary to convincingly validate the method. With these improvements, I would be very willing to reconsider my evaluation positively.

### Questions
- **Line 226 clarification:** The statement “conditional flow ODE evolving from the under-sampled initialization $x_1 = y$ towards the fully-sampled $x_0$” is not precise. Since $y$ denotes under-sampled k-space measurements, the initialization should more appropriately be $x_1 = A^\top y$.  

- **Performance on fastMRI Knee dataset:**  
  Table 1 shows that FLAT does not significantly outperform traditional unrolled networks on the fastMRI Knee dataset and even underperforms in some metrics. Could the authors clarify the reason for this performance gap?  

- **Complex loss function design:**  
  The proposed training involves a highly complex loss composition. Could the authors analyze the contribution of each component and whether all terms are necessary?  

- **Ablation interpretation (Table 3):**  
  From Table 3, removing the paper’s three main contributions theoretically reduces FLAT to an E2E-VarNet with the proposed complex loss. The results show that this “degraded” model still performs notably better than a standard E2E-VarNet, suggesting that the improvement may largely stem from the loss function itself. If so, would applying the same loss function to a vanilla E2E-VarNet yield even higher performance—perhaps surpassing FLAT—on the fastMRI Knee dataset?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a new method (FLAT) for viewing unrolled reconstruction approaches as flow models. This insight allowed led to three critical changes in training unrolled networks for reconstruction (1) unrolls viewed as cascaded time steps must satisfy constrains as a trajectory (2) normally free hyper parameters in unrolled methods are fixed to satisfy ODE (3) at intermediate time steps (unrolls) the images are aligned to the desired trajectory not just the final image. The authors show through several experiments that enforcing these constraints on their unrolled network led to SOTA performance compared to other existing methods in MRI reconstruction.

### Strengths
Overall, this paper provides a novel and useful insight into existing unrolled techniques which I think would be of interest to the broader MRI recon community since unrolled techniques are very popular. The connection between flows and unrolls is interesting and clearly improves performance which is great. The authors did a good job comparing to other SOTA methods.

### Weaknesses
I do believe that the paper would benefit from testing their method on various acceleration levels of MRI data. They show results for R=8 but I would also like to see what their performance gains are at higher (and) lower acceleration levels like R=4 and 12. Additionally I would like to know what the wall clock time is for running inference of their method vs. the other methods presented.  They present the number of iterations compared to other techniques, but it would be nice to see the actual timing.

### Questions
1.	I am not sure about the statement in lines 238-239 where it is stated that $p(y|x_t)=N(y|Ax_t,\sigma^2)$. Isn’t this only true for t=0?
2.	Is the initial reconstruction $x^k = y$ the pseudo-inverse reconstruction?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper aims to develop a flow ODE characterization for unrolled networks. The idea is interesting, but unfortunately there are fundamental flaws.

### Strengths
- I think the overall idea of using a flow ODE characterization to describe unrolled networks is great. Hence I really wanted to like this paper. Unfortunately both the theory and execution has substantial flaws.

### Weaknesses
1) The proof of the main result is fundamentally flawed:
- The argument hinges on writing out p(y|x_t). Unfortunately Eq. 1 does not apply to intermediate points on the trajectory, which is well-known in the literature. p(y|x_t) would need to be calculated as (in the authors' notation): \int p(y|x_0) p(x_0| x_t) dx_0, since we only know the relationship between y and x_0 (i.e. Eq. 1). This breaks down the whole proof. There are many works on approximating this integral in the diffusion inverse problems literature.
- The authors can see this fails by considering their own definition of x1 = y (incidentally I'm surprised they are trying to come up with a velocity field from k-space to image domain)
- The second part of the proof that is questionable is the statement "this velocity aligns with the gradient of the conditional log-density" Why is this true? This is not shown. 
- Also fundamentally one would expect (5) to have dependence on y in this setup, instead (5) is describing an unconditional flow on the image set, with no knowledge of measurements.

2) The authors seem to be unaware of MRI reconstruction literature. Much of the motivational claims are untrue or incomplete:
- "hyperparameters such as step size and weighting coefficients are typically set through heuristics or empirical tuning" 
Almost all algorithm unrolling frameworks learn step sizes and weighting coefficients jointly with the proximal operator/regularization neural network. This statement is therefore incorrect. Furthermore, these are considered parameters of the unrolled network, not hyperparameters. This joint learning of regularization and data fidelity parameters is the main advantage of algorithm unrolling over plug-and-play type methods.
- "they are typically trained with supervision only at the final cascade"
This is partially true. First off, it seems the authors are not distinguishing between unrolled networks with shared parameters (i.e. each cascade uses the same CNN and step sizes/weighting parameters as in MoDL) vs. those with unshared parameters (as in E2E-VarNet). In the former case, it is easy to train with supervision at the output. Even in that case, one can do weak intermediate supervision by training a single cascade first, replicating it for T cascades, and fine-tuning the T-cascade version (as in MoDL). In the latter case, the typical approach is to first train the shared parameter version, then fine-tune the unshared version on that. There are also works that propose intermediate supervision (e.g. doi: 10.3390/e27090929), but the benefit of this is marginal especially in the first shared setup.
- Naturally, the erratic behavior observed is related to the unshared version. This is usually not seen in the shared parameter setup, which has much "flatter" behavior across cascades.

3) Experiments are performed on either DICOM images or single-coil datasets, which have limited utility in MRI.

4) The proposed algorithm only extends to gradient descent/proximal gradient descent type algorithms, and do not explain the more successful variants based on variable splitting (e.g. ADMM)

5) The authors set \sigma = 1, but this has a physical meaning in the derivation as the observation noise, so one cannot arbitrarily set it to any value they want.

6) The authors not only use the flow-based loss term, but multiple other loss terms. It is unclear if the comparison networks used the same additional loss terms. An ablation study on the effect of each term in (13) is clearly missing.

7) 12 steps is not a speed-up over existing unrolled networks. MoDL has 10 steps, E2E-Varnet readily has 12 as well.

Minor points:
- "Φk(·) is the learned regularizer (often implemented by a CNN)" 
Often the proximal operator corresponding the regularizer is implemented with a CNN.
- Flow matching literature typically uses the opposite indexing, with x0 being the noise and x1 being the data distribution.
- \sum \delta_k = -1 is an interesting insight, but this is counter-acted by arbitrarily setting \sigma = 1. 
- There is something fundamentally wrong with the fully-sampled k-space shown in Fig. 1. Perhaps due to some DICOM processing.

### Questions
These are already covered in the weaknesses:
- How does the method work on multi-coil MRI data?
- Why is \sigma = 1? If it is used as the noise level in the dataset, how does it affect the results?
- What is the effect of each term in the loss function?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper re-interprets unrolling networks as flow matching. It views the iterations derived from optimization algorithms in unrolling networks as time steps in flow matching and aligns intermediate estimation, x_k in unrolling network,with the conditional path x_t in flow matching. Authors formulated it mathematically and claimed the newly derived losses trains unrolling network better. The alignment of optimization iterations in unrolling network to the conditional path in flow matching seems new to me. However the motivation is weak and not well grounded. Please find my concerns below.

### Strengths
1. This paper is well-written and has a comprehensive literature review.
2. This paper includes experiments that compares the proposed method to methods that are from different categories.

### Weaknesses
1. The authors claim that the core innovation of FLAT defines the velocity alignment between unrolling iterations and flow matching. It is formulated by first defining two different velocities at the k-th timestep:

	a.  **Ideal Discretized Velocity ($v_{t_k}$):** This is the target velocity, defined as the discrete temporal derivative using the ground truth.
	    $v_{t_{k}} = (x_{t_{k}}^{\*} - x^{(k+1)}) / (t_{k} - t_{k+1})$, $x_{t_{k}}^{*}$ is the linearly interpolated ground truth at time $t_k$. $x^{(k+1)}$ is the network's prediction from the *previous* iteration.  
	b.  **Network's Predicted Velocity ($v^{(k)}$):** This is the velocity predicted by the network at the current step.
	    $v^{(k)} = (x^{(k)} - x^{(k+1)}) / (t_{k} - t_{k+1})$, $x^{(k)}$ is the network's output at the current step $k$.   

        Why is this ideal ode path is better than original path derived from optimization iterations? If this is true, how much influence can this loss terms make?

2. As a following-up questions, how you select the hyper-parameters? Could you explain why there are so many other loss terms? How you balance the weights for each term? Why the weights for the velocity alignment term is so small? Can we only keep the velocity alignment term? What if we have the other loss terms for other unrolling methods.

3. Not sure if it is right to replace $\nabla_{x_t} \log p(x_t)$ with the velocity field $v_\theta(x_t, t)$, as the unrolling network actually does not learn this and has no generative modelling for it. The unrolling network to me just learns a discriminative model. Would you further comment on this?

4. If this velocity alignment help us in training unrolling network, how would the number of discrete steps affect the performance? The more the better?

5. Which specific diffusion model based method you used for comparison? DDPM and DDIM samplers seems to be very broad. Or would you specify how you implement the method using diffusion models?

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
