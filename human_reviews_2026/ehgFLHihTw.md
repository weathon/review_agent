# Score-based Membership Inference on Diffusion Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Membership inference attacks (MIAs) against diffusion models have emerged as a pressing privacy concern, as these models may inadvertently reveal whether a given sample was part of their training set. We present a theoretical and empirical study of score-based MIAs, focusing on the predicted noise vectors that diffusion models learn to approximate. We show that the expected denoiser output points toward a kernel-weighted local mean of nearby training samples, such that its norm encodes proximity to the training set and thereby reveals membership. Building on this observation, we propose \textbf{SimA}, a single-query attack that provides a principled, efficient alternative to existing multi-query methods. SimA achieves consistently strong performance across variants of DDPM, Latent Diffusion Model (LDM). Notably, we find that Latent Diffusion Models are surprisingly less vulnerable than pixel-space models, due to the strong information bottleneck imposed by their latent auto-encoder. We further investigate this by differing the regularization hyperparameters ($\beta$ in $\beta$-VAE) in latent channel and suggest a strategy to make LDM training more robust to MIA. Our results solidify the theory of score-based MIAs, while highlighting that Latent Diffusion class of methods requires better understanding of inversion for VAE, and not simply inversion of the Diffusion process

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose an effective membership inference attack against diffusion models. The key idea is to set $\epsilon$ to zero, which enables an effective attack with a single query. They provide theoretical support for the method and show experimentally that it improves TPR. They also find that existing attacks, including the proposed method, perform poorly on latent diffusion models (LDMs).

### Strengths
* Proposed a simple yet efficient and effective method.
* Theoretically demonstrated that membership can be distinguished.
* Experimentally showed an improvement in TPR.
* Found that attacks, including the proposed method, perform poorly on latent diffusion models (LDMs).

### Weaknesses
* Although the method appears effective, it does not seem novel in terms of either the approach itself or the motivation compared to existing methods.
* The paper only discusses the advantages of setting ε to 0 versus non-zero values at a high level. It would be stronger if the authors provided a theoretical or experimental analysis of specific cases where such differences might arise.
* Most modern diffusion models are based on LDMs, and the fact that the proposed method is not effective compared to other approaches in this setting makes its overall impact appear limited.
* Except for TPR, the improvements in ASR and AUC appear marginal.
* The paper is not well organized. The key idea, setting ε = 0, appears only later in the method section, even though it is central to the approach. This concept should have been introduced earlier, ideally in the introduction and at the beginning of the method section, along with a clear discussion of how it differs from prior methods in the related work. The authors also observed that all attacks, including their proposed one, perform poorly on LDMs, and they analyzed this issue. They further showed that increasing the β value can mitigate such attacks. However, this part feels somewhat disconnected from the main storyline and could potentially be developed into a separate paper.

### Questions
* The method is particularly effective on guided diffusion. What is the reason for this?
* ASR and AUC are similar, but the method shows notably better performance only in TPR. What is the reason for this?

### Soundness
3

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
3

### Summary
This paper presents an innovative membership inference attack method called SimA for diffusion models. The core idea is that the norm of the predicted noise at a data point serves as a strong indicator of its membership, based on the theoretical insight that the denoiser output points towards a kernel-weighted local mean of nearby training samples. SimA operates as a ​​single-query attack​​, improving efficiency over multi-query baselines while achieving competitive performance across DDPM, Guided Diffusion, and Latent Diffusion Models. The study reveals that ​​LDMs are inherently more robust​​ to MIAs due to the information bottleneck imposed by their latent autoencoder.

### Strengths
1. The paper provides a rigorous derivation linking the predicted noise norm to kernel-weighted local means of training data, offering a unified explanation for score-based MIAs.
2. SimA requires only a single query per sample, reducing computational cost compared to multi-query methods.
3. The study identifies the unexpected robustness of LDMs to MIAs, linking it to VAE regularization hyperparameters and proposing strategies to enhance privacy without sacrificing generative quality.

### Weaknesses
1. The paper does not fully explore why VAE-based bottlenecks mitigate MIA beyond empirical correlations with β. A deeper analysis of how exactly the latent space bottleneck disrupts the membership signal would have strengthened this finding.
2. The method is sensitive to the selection of the timestep t and requires manual tuning for different datasets. Is it possible to develop a principled method for selecting the optimal t automatically, thus eliminating the need for this empirical sweeping?

### Questions
See the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on MIAs against DMs. It proposes SimA, a novel single-query MIA framework that leverages the norm of predicted noise vectors, rooted in the insight that the expected denoiser output points to he kernel-weighted local mean of nearby training samples. The paper validate SimA across pixel-space DMs and LDMs on multiple datasets. SimA outperforms multiquery baselines in ASR and AUC. The paper also shows that LDMs exhibits suprisely less vulneralbe to MIAs due to the strong information bottelneck and the need to focus on VAE inversion for MIA on LDMs.

### Strengths
1) The paper introduces SimA，a single-query MIA that unifies existig multiquery mothods into a score based framework to avioid relying on Monto Carlo sampling or multi-step queries.
2) The paper gives extensive and rigoruous experimental validation across 11 datasets and 3 major DM classes.
3) The paper identifies a critical problem, that is, LDMs are less vulnerable to MIAs than pixel-space models.
4）The paper addresses a new direction of MIA research from diffusion inversion to VAE inverison.

### Weaknesses
1) The paper points out the problem that MIAs exhibit poor performance on LDMs, but it does not pratically resolve this problem.

2) The paper's MIAs are designed under ideal academic assumptions (unrestricted gray-box information access, minimal defenses) that do not align with commercial realities.

### Questions
SecMI requires multiple queries to assemble a statistical profile,  but SimA demands richer single-query information (noise vector norms). While SimA appears more efficient than SecMI in query count, both methods fail in real-world commercial settings due to strict black-box constraints.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes SimA, a single-query diffusion MIA that leverages the norm of the predicted noise vector (score) from diffusion models. Theoretical analysis shows that the denoiser output points toward a kernel-weighted local mean of training samples, with its norm encoding proximity to the training data. Empirically, SimA achieves strong performance across various diffusion models.

### Strengths
1. The paper presents a novel approach to Membership Inference Attack (MIA) by analyzing the norm of the estimated noise.

2. The experiments, particularly the investigation involving the β-VAE, provides interesting results.

### Weaknesses
+ Clarity of Core Assumption in SimA​. The paper posits that for member samples, the local mean converges to the training sample itself, implying that at $t=0$, the predicted noise $\epsilon_{\theta}(x_0,t=0)$ (where $x_0$ is the clean training sample) should converge to zero. However, in practice, the output at $t=0$ often resembles Gaussian noise with a norm significantly larger than 0. This discrepancy raises questions about the validity of the local-mean assumption in real-trained models, especially regarding how well the theoretical peak-to-sample alignment is achieved under finite training and network parameter update.

+ Peak Formation for Non-Members​. Figure 1 suggests that members correspond to peaks in the smoothed probability distribution, but it does not sufficiently address whether non-member samples can also form local peaks. If the generative process navigates probability gradients, non-members that are plausible under the data distribution may also lie at or near critical points (local maxima), leading to small noise norms similar to members. What is the structure of the probability distribution near non-members? Why non-members can not form a peak? If non-members also form a peak, its local mean also converges to zero. That means nonmembers and nonmembers can not be distinguished by **the gradient**.

+ ​Distinction Between Data Manifold and Learned Manifold​. I find that the author seems to try to conflate the true data distribution with the distribution learned by the neural network. The argument assumes that the diffusion model’s score function accurately reflects the kernel-smoothed empirical distribution of the training set. In reality, the network only minimizes a training loss and may not perfectly capture the data manifold. The geometric properties of the learned score field could differ significantly. This gap may undermine the core assumption that the local mean around a query point is well-approximated by the model. 

+ (Minor) Relationship Between Naive Loss and SimA​ (Line 311-323). The “Naive” baseline (e.g., loss-based MIA) can be viewed as computing $||\epsilon_{\theta}+(-\epsilon)||$, which resembles SimA’s ||\epsilon_{\theta}|| but with an added noise term $-\epsilon$. If the noise term averages out over multiple queries, one might expect the Naive method to converge to SimA’s performance with increased sampling. However, the performance of Naive Loss does not scale with the number of query. How to explain this?

+ (Minor) It is quite strange that PIA achieves the best performance in CIFAR-10 (Table 1), which contrasts with [1] that shows PIA is generally unstable. 

[1] Membership inference on text-to-image diffusion models via conditional likelihood discrepancy. Neurips 2024.

### Questions
See the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
