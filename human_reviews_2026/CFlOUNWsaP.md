# Scaling Image and Video Generation via Test-Time Evolutionary Search

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
As the marginal cost of scaling computation (data and parameters) during model pre-training continues to increase substantially, test-time scaling (TTS) has emerged as a promising direction for improving generative model performance by allocating additional computation at inference time. While TTS has demonstrated significant success across multiple language tasks, there remains a notable gap in understanding the test-time scaling behaviors of image and video generative models (diffusion-based or flow-based models). Although recent works have initiated exploration into inference-time strategies for vision tasks, these approaches face critical limitations: being constrained to task-specific domains, exhibiting poor scalability, or falling into reward over-optimization that sacrifices sample diversity. In this paper, we propose \textbf{Evo}lutionary \textbf{Search} (EvoSearch), a novel, generalist, and efficient TTS method that effectively enhances the scalability of both image and video generation across diffusion and flow models, without requiring additional training or model expansion. EvoSearch reformulates test-time scaling for diffusion and flow models as an evolutionary search problem, leveraging principles from biological evolution to efficiently explore and refine the denoising trajectory. By incorporating carefully designed selection and mutation mechanisms tailored to the stochastic differential equation denoising process, EvoSearch iteratively generates higher-quality offspring while preserving population diversity. Through extensive evaluation across both diffusion and flow architectures for image and video generation tasks, we demonstrate that our method consistently outperforms existing approaches, achieves higher diversity, and shows strong generalizability to unseen evaluation metrics.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes EvoSearch, a method for improving image and video generation models through test-time evolutionary optimization. Rather than training or fine-tuning model weights, EvoSearch adapts the sampling process of diffusion models dynamically at inference. It formulates generation as a search problem where multiple candidate are evolved to maximize the reward. Empirical results on benchmarks show consistent gains in perceptual quality metrics.

### Strengths
The paper presents a training-free method to improve generative models at inference. EvoSearch applies to many diffusion architectures for both image and video generation and is among the first to study test-time scaling for video models, which broadens its impact across modalities. The experiments show consistent gains on perceptual quality metrics, indicating practical value.

### Weaknesses
While the proposed method brings additional performance gains across tasks, the method relies heavily on heuristic choices for the evolution schedule, mutation strategy, and initialization of noise parameters. How these parameters are chosen? Can authors give justifications for the parameters? Also, this may limit scalability for new model, task, and reward. 

Although the method is new, test-time scaling has been explored extensively. The paper needs a more convincing case for why EvoSearch should work and why it should outperform existing methods such as best of N and particle sampling. The approach is framed as evolutionary optimization, but the paper does not explain in a principled way how the search improves the denoising trajectory. As written, the gains appear purely empirical. Providing more intuitions and formal derivations would strengthen the claims. 

Lastly, running evolutionary search at test time introduces computational overhead, especially for video generation where reward and generative model are large. The paper lacks a thorough analysis of runtime and trade-offs compared to standard sampling methods.

Minor 
- L.234, 304 Eq. equation typo

### Questions
Please see weaknesses.

### Soundness
2

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
The paper proposes a new type of test-time scaling (TTS) method for diffusion models, inspired by evolutionary algorithms. Unlike previous approaches that require $N$-times sampling (Best-of-$N$) or are restricted to initial ODE stages (particle sampling), the proposed method enables multiple sampling trajectories starting from intermediate states $x_t$, with the ability to modify the initial noise based on intermediate reward evaluations. For both image and video generation with flow-based models (noting that denoising diffusion models are a special case of flow-based models), the proposed method achieves superior quality and diversity.

### Strengths
- Performance is promising.
- The paper provides extensive analysis including toy example and ablation study with hyper-parameters.
- The proposed idea seems to be novel.

### Weaknesses
- The writing could be improved. Although the paper focuses on reinterpreting TTS from the perspective of evolutionary algorithms, the terminology used is unconventional. Consequently, the roles of hyperparameters such as the population scheduler $K$ and evolution schedule $T$ are unclear and difficult to interpret.

- Line 265: Chung et al., 2023 should be corrected to [1].

- Computing the reward on the fully denoised $x_0$ (lines 264–268) is not new. [2], which is missing from the related work, also performs evaluation on the fully denoised $x_0$.

- Lines 192–194: the paper claims that $N$-times sampling for Best-of-$N$ is computationally inefficient. However, the proposed method also performs repeated sampling for evaluation (lines 259–268). While this may be slightly more efficient since sampling starts from an intermediate step $t$ instead of the initial state $x_T$, the main source of computational overhead remains comparable to previous methods.

- The paper suggests that using an SDE is necessary for exploration. However, this has already been explored in [3], which demonstrated that SDE-based dynamics are required for fine-tuning diffusion models.

Reference

[1] Noise2Score: Tweedie's Approach to Self-Supervised Image Denoising without Clean Images, NeurIPS 2021

[2] Training Diffusion Models Towards Diverse Image Generation with Reinforcement Learning, CVPR 2024

[3] Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control, ICLR 2025

### Questions
- Could the authors provide guidance on how to select appropriate hyperparameters when applying EvoSearch to a new generative model?

### Soundness
3

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
2

### Summary
This paper explores test-time scaling (TTS) for vision generative models and proposes EvoSearch, an evolutionary search–based approach for enhancing the denoising trajectory in both diffusion and flow-based generative models. Instead of modifying model architecture or training objective, EvoSearch leverages evolutionary mechanisms during inference to identify better intermediate latent states as parents for subsequent denoising steps. The paper presents extensive experiments across image and video generation tasks, showing improvements in visual quality and sample diversity.

### Strengths
- Novelty: Using evolutionary algorithms to optimize denoising trajectories during inference is original and conceptually appealing.
- Strong empirical evaluation: Extensive experiments, ablations, and analysis support the method's effectiveness.
- General applicability: Works across diffusion and flow-based models, and both images and videos.
- No training overhead: Fits well within the TTS paradigm—improving sampling quality without retraining.

### Weaknesses
- Clarity of algorithm description: The current presentation of EvoSearch's workflow, particularly in the overview figure and pseudocode, lacks clarity and makes it difficult to precisely understand the evolutionary operators.
- Ambiguous visualization (Figure 3): The diagram includes unexplained symbols (e.g., check marks, arrows, population filtering meaning, mutation signs), making the process non-transparent. It would strongly benefit from a redesign with clearer semantics and annotations explaining the denoising vs. generation stages.
- Reward computation description unclear: There is confusion regarding whether the reward r(x_0) is computed per-parent or aggregated, and how it aligns with the expected reward definition under p(x_0|x_t). This should be explicitly clarified step-by-step.

### Questions
- What does the check mark on the middle image mean?
- Why is the reward value in selection shown as 0.01, and why are 3 elites highlighted?
- In the mutation block, what do the different operators (→, =, + etc.) represent?

Reward Computation in Algorithm 2:
At t = 0, if we have 10 parent states, do we compute 10 rewards or 1 aggregated reward?
- If only 1 reward is computed, what does it correspond to?
- If 10 rewards are computed, how does this align with Eq.2's expectation over p(x_0, x_t)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes Evolutionary Search (EvoSearch), a novel test-time scaling framework for generative models in images and videos. Instead of training new models or fine-tuning, EvoSearch allocates extra computation at inference to improve output quality by iteratively refining generations. It treats the diffusion/flow model’s denoising process as an evolutionary algorithm: a population of sample trajectories is evolved through selection and mutation at each step. This approach aims to preserve diversity while gradually steering samples toward higher reward. EvoSearch is a general method applicable to both diffusion models (e.g. SD 2.1) and flow-based generative models (e.g. Flux) for image and video generation. Empirically, the authors show that as inference compute (“number of function evaluations”, NFEs) increases, EvoSearch produces higher-quality outputs that outperform baseline strategies like best-of-$N$ and particle filtering on several benchmarks. Notably, using EvoSearch, a smaller text-to-video model (Wan 1.3B) was able to match or surpass the performance of a 10× larger model (Wan 14B) given the same inference-time budget. The paper positions EvoSearch as a general, training-free approach to test-time alignment and improved generation quality for vision models, drawing inspiration from biological evolution to navigate the high-dimensional search space of generative model outputs.

### Strengths
EvoSearch introduces a fresh perspective by framing inference-time optimization as an evolutionary process. It leverages selection and mutation operators tailored to the diffusion denoising trajectory, which is an original way to maintain diversity and avoid the collapse seen in earlier methods.

When enough computation is allocated, EvoSearch does improve generation quality. The experiments show notable gains in output metrics as the number of inference steps increases. The experimental results indicate that given a high test-time compute budget, EvoSearch can find higher-reward samples than prior methods, validating its effectiveness in principle.

The evolutionary strategy explicitly tries to avoid the diversity collapse problem. By always selecting a population of top samples and mutating them with randomness, EvoSearch preserves multiple promising candidates instead of focusing on a single trajectory. The authors note that previous particle methods were limited by a fixed initial pool and could converge to similar results. In contrast, EvoSearch’s design (inspired by natural evolution) inherently maintains population diversity, which is important for generative tasks to not over-optimize the reward at the cost of mode collapse.

### Weaknesses
While EvoSearch is novel in framing the problem as an evolutionary algorithm, the practical advantages over existing particle sampling methods are not very convincing. In the Stable Diffusion 2.1 image experiments, EvoSearch’s improvement over a standard particle filtering baseline is quite small. One could argue EvoSearch is a variant of particle sampling with an evolutionary selection twist, yielding comparable results to known methods on images. This weakens the claim of a significant contribution. Also, in FLUX experiments, EvoSearch’s gains come at the cost of very high computation, raising concerns about practicality. The authors had to go up to around 5000 NFEs before EvoSearch clearly overtook the particle sampling baseline, which is an enormous number. In terms of wall-clock time, this is on the order of 20+ minutes per single image generation (thousands of seconds on a single GPU) for one prompt. This dramatically limits real-world usability.

Also, it formulates the guidance as a reward (from models like ImageReward or CLIP score) and uses a gradient-free evolutionary search to optimize it. However, many of these reward models are differentiable, which raises the question: why not use gradient-based optimization? Recent works have shown that directly optimizing the initial noise or latent via gradient ascent on a differentiable objective can be extremely effective. For instance, ReNO (Eyring et al., 2024) optimizes the initial noise in a one-step diffusion model by backpropagating the reward gradient, achieving significant gains in prompt fidelity within just ~50 iterations (around 20–50 seconds of compute). Likewise, D-Flow (Ben-Hamu et al., 2024) differentiates through the entire generative process of a diffusion/flow model to adjust the noise, effectively guiding generation by continuous optimization rather than random mutation. These approaches show that gradient-based test-time alignment is feasible and often faster or more direct than heuristic search. The fact that EvoSearch does not compare against any such differentiable methods is a notable omission. It’s unclear if the authors avoided gradients due to concerns like non-convexity or local optima; regardless, no discussion is provided. This is a weakness because the paper doesn’t justify its design choice of an evolutionary (gradient-free) strategy when, in principle, the reward signal is differentiable. Without comparing to methods like ReNO or D-Flow, we don’t know if EvoSearch’s gradient-free approach is actually more effective, or if it might be outperformed by simply taking gradient steps on the noise (which could potentially reach a high-reward solution much faster).

The authors did not include a comparison with the latest particle sampling advancement, namely the Rollover Budget Forcing (RBF) method introduced by Kim et al. (2025). RBF is specifically designed to improve inference-time sampling by adaptively reallocating the diffusion steps budget, and it has been shown to outperform prior particle filtering approaches. In fact, the paper’s related work section cites this method (Kim et al., 2025b) and notes that it “demonstrates superior results” over naive baselines. By failing to compare against RBF in the experiments, the evaluation is incomplete, and we don’t see how EvoSearch stacks up against the state-of-the-art in test-time sampling. Given that RBF was published in March 2025 and targets a very similar problem (scaling diffusion/flow model inference), it should have been included as a baseline. This omission is critical because RBF could potentially already solve some challenges (e.g., better compute allocation, maintaining diversity) that EvoSearch addresses.

The paper aspires to present a general-purpose test-time scaling approach, but in doing so it may be spreading its contributions thin. The most compelling results are in the video generation domain, where test-time optimization isnovel. However, for image generation, the method doesn’t substantially advance beyond what was already known (as discussed above). This disparity suggests that the paper’s contribution might have been better framed with a narrower scope: for example, focusing on test-time evolutionary search for video generation as the primary innovation.

### Questions
Have you considered leveraging gradient-based optimization on the reward function instead of (or in addition to) evolutionary search?

The results indicate EvoSearch can require thousands of steps (and very long wall-clock time) to fully exploit the reward. How practical is this method, and are there ways to reduce the compute required?

### Soundness
2

### Presentation
3

### Contribution
2
