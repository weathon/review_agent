# Adaptive Destruction Processes for Diffusion Samplers

- Decision: Reject
- Scores: 8, 4, 6, 2

## Abstract
This paper explores the challenges and benefits of a trainable destruction process in diffusion samplers -- diffusion-based generative models trained to sample an unnormalised density without access to data samples. Contrary to the majority of work that views diffusion samplers as approximations to an underlying continuous-time model, we view diffusion models as discrete-time policies trained to produce samples in very few generation steps. We propose to trade some of the elegance of the underlying theory for flexibility in the definition of the generative and destruction policies. In particular, we decouple the generation and destruction variances, enabling both transition kernels to be learnt as unconstrained Gaussian densities. We show that, when the number of steps is limited, training both generation and destruction processes results in faster convergence and improved sampling quality on various benchmarks. Through a robust ablation study, we investigate the design choices necessary to facilitate stable training. Finally, we show the scalability of our approach through experiments on GAN latent space sampling for conditional image generation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper jointly learns generation and destruction in a discrete-time sampler, enabling state-dependent, learnable variance in both directions. This sidesteps continuous-time path-KL issues when variances differ and is supported by bounded parameterizations plus stabilization tactics (shared backbone, separate optimizers, target nets, replay). Empirically, it helps most in few-step regimes and narrow-mode targets; second-moment objectives (TB/VarGrad) are competitive with or better than PIS while being more memory-friendly. A style-transfer/latent-space demo shows the approach scales. Main trade-offs: weaker ties to continuous-time bridges, hyperparameter sensitivity (especially destruction LR), occasional instability (e.g., TLM at large steps), limited breadth beyond faces, and extra compute for some conditional setups.

### Strengths
1. Clear motivation; concrete contribution: learnable, state-dependent variance for both directions in discrete time.

2. Thoughtful engineering for stability with solid ablations.

3. Good theoretical positioning vs. IPF/SB/CMCD/GFlowNet.

### Weaknesses
1. Continuous-time connection & guarantees. Because the method allows different variances for generation vs. destruction, the link to continuous-time bridge formalisms seems less direct. Could you clarify what guarantees still hold in discrete time (e.g., well-posedness, stability, convergence/consistency), and how you ensure or demonstrate them in practice? Any safeguards to prevent pathological transitions when the two variances diverge?

2. Hyperparameter sensitivity & tuning. The approach appears sensitive to learning-rate ratios—especially for the destruction model—which may affect stability and reproducibility across datasets/seeds.  How to guarantee the stability of training across settings?

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposed a trainable destruction process technique during discrete diffusion model's sampling process. In detail, the method proposed a novel view for diffusion sampling. The experimental results showed that the deconstruction process yielded in fastering convergence, and higher ELBO. And the ablation studies show the effectiveness of the decomposition process.

### Strengths
1. Framework novelty. The method firstly extend the traditional diffusion process into learnable variances in an unified theoretical framework. 

2. Integration of stability mechanism. The paper involved reinforcement-learning stabilization tools inspired by reinforcement learning's view. And Table 2 systematically evaluate the performance of each tool. 

3. Scalability to high-dimensional tasks. Section 4.4 demonstrated the capability of the method to higher dimension image generation tasks, which leads to boarder applications.

### Weaknesses
1. Insufficient theoretical analysis. Although there is unified framework and well-defined processes, no analysis of the convergence or gradient bias of KL divergence is provided. 

2. Lack of continuous-time analysis. There is no proof for the equivalence between the generation and th destruction processes as T goes to infinity. 

3. Limited evaluation to TLM. The paper proposed TB and TLM, but the main experiments were conducted by TB.

### Questions
1. In equation 14, does the choice of proposal distribution tied to the off-policy exploration in Section 3.3?

2. Is there analysis showing the connection between the learnable variances between two processes?

3. How will the method behaive if the energy landscape violates smoothness assumptions (non-Lipschitz E(x))? Does the KL objective in Eq. (13) remain well-defined under this situation?

4. Were target networks and PER ablated individually or only in combination (Table 2)? How sensitive is performance to the PER replay ratio?

5. The paper alternates between KL-based and second-moment objectives. Is there a principled reason not to use hybrid losses (weighted KL + TB)?

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
2

### Summary
This paper introduces a novel framework for training diffusion samplers by jointly optimizing both the generative and destruction processes in discrete time. Unlike prior approaches that fix the destruction process or assume continuous-time SDE dynamics, the authors propose learning both forward and reverse transitions, with decoupled, state-dependent variances parameterized via neural networks. This flexibility enables better adaptation to limited-step sampling regimes and complex energy landscapes.

### Strengths
1. Novel joint training of generation and destruction processes in diffusion samplers, enabling improved convergence and sampling quality, especially in few-step regimes.

2. Flexible design with state-dependent, decoupled variances for both processes—only possible in discrete-time formulation—leading to enhanced adaptability to complex energy landscapes.

### Weaknesses
1. Limited visual results: The paper presents few qualitative or visual examples (only human faces in Fig. 4), making it difficult to fully assess sampling quality, especially in image-related tasks.

2. No discussion of limitations: The paper lacks a section acknowledging potential limitations (e.g., scalability to more complex distributions, sensitivity to architecture choices), which raises concerns about generalizability.

### Questions
See the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a method for training diffusion samplers, which are models designed to sample from an unnormalized density (defined by an energy function) without access to data samples. Specifically, it introduces the joint training of both the generation and destruction processes by viewing them as discrete-time policies. The core novelty is decoupling their variances, which enables the means and variances of both processes to be learned as state-dependent neural networks, rather than being fixed or constrained by continuous-time theory. The experiments demonstrate that this joint training approach results in faster convergence and improved sampling quality across various small scale benchmarks, especially when the number of generation steps is limited or the energy landscape has narrow modes. This scalability is validated on a high-dimensional GAN latent space sampling task, showing quantitative and qualitative benefits for conditional image generation.

### Strengths
1. The problem studies is novel and interesting
2. The proposed training objective, Second-Moment Divergence, deviates from the standard KL formulation, which is an interesting direction to explore. 
3. The design space is meticulously swept over, with the key ingredients for stable training reported in this paper. 
4. The advantage is most pronounced when the number of sampling steps is small. The paper shows that on some tasks, its method with as few as 5 steps can outperform 20-step samplers that use fixed variances. 
5. Their experiments on synthetic dataset, though limited by their scale, are very explanatory.

### Weaknesses
1. Despite the non-trivial efforts to stabilize the training, the joint training process is inherently unstable. The Trajectory Likelihood Maximization (TLM) objective, one of the main candidates for generically training the destruction process, is "unstable and often leads to divergent training" when the number of steps is large. 
2. The method also seems to be highly sensitive to hyperparameters. L265-266, "tuning relative learning rates is critical for stable training". 
3. The results on GAN, which is regarded as scalability test of the method, has mixed results.

### Questions
Theoretically speaking, would the trained sampler guaranteed to approximate the target distribution as the number of sampling steps goes to infinity?

### Soundness
2

### Presentation
3

### Contribution
2
