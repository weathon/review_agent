# Forward-only Diffusion Probabilistic Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
This work presents a forward-only diffusion (FoD) approach for generative modelling. In contrast to traditional diffusion models that rely on a coupled forward-backward diffusion scheme, FoD directly learns data generation through a single forward diffusion process, yielding a simple yet efficient generative framework. The core of FoD is a state-dependent stochastic differential equation that involves a mean-reverting term in both the drift and diffusion functions. This mean-reversion property guarantees the convergence to clean data, naturally simulating a stochastic interpolation between source and target distributions. More importantly, FoD is analytically tractable and is trained using a simple stochastic flow matching objective, enabling a few-step non-Markov chain sampling during inference. The proposed FoD model—despite its simplicity—achieves state-of-the-art performance on various image restoration tasks. Its general applicability on image-conditioned generation is also demonstrated on diverse image-to-image translation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a probabilistic forward-only diffusion model (FoD) that incorporates Geometric Brownian Motion (GBM) to transform an image into another that is in two distributions separately. The authors claimed that by introducing a mean-reversion term $\mu - x_t$​ into both the drift and diffusion coefficients of the SDE, they defined an analytically solvable forward-only process, eliminating the need to approximate or learn a reverse SDE. However, I have several concerned. First, I believe this claim of FoD is actually a backward diffusion (as I explained in Weakness section).  Second, in conventional diffusion models, the process involves diffusing an image  to noise (forward), and then reconstructing from noise to an image that closely remember the original image (reverse) although one can always  transform between two deterministic images. The setup in this paper actually resembles the GOUB., more recently; UNIDB. Although the authors proposed to  add the state dependency into the diffusion coefficient, there is no proof of the effect of this diffusion. Indeed the author almost has not discussion on the coefficients. Instead, the authors used extremely small diffusion coefficients, this made it actually more close to deterministic method, reducing the diffusion effects, which I believe making this paper relatively weak contribution to the field.

### Strengths
The paper is well written. The authors effort to consider diffusion coefficients as state dependent although the implementation is weak.

### Weaknesses
1. It seems the authors misinterpretation of forward diffusion as backward. This can be observed from they actually started from $x_T$ ∼ $p_{data}$ to  $x_0$ ∼$ p_{prior}$. The training step stated in the paper actually resembles the backward diffusion diffusion step in the conventional diffusion. There is not much different between this method in terms forward/backward as the diffusion itself is not a reason to stay on forward only as what we can see from GOUB.

2. P4. Line 178, as the author mentioned, "The subtractive form of the logarithm reflects that
the flow field decays multiplicatively from its initial value with a stochastic exponential scaling", this translate to the original image difference, the variance of the noise can explore, I wonder how the authors address this issue. The only solution in the paper is that they set the $ e^{-\int_0^t (\theta_s+\frac{1}{2} \sigma_s^2) ds}= 0.001$. The authors should have discussed the impact of selecting such small diffusion coefficients. This small value seems to make it almost a deterministic process. 

3. As I mentioned in the summary and item 2, the authors used extremely small diffusion coefficients, this made it actually more close to deterministic method, reducing the diffusion effects. This can be observed in several sequences of images. For example, the second row images (over various time steps) in Figure 1 seemed so deterministic.

4. The following paper should be referred:
1). K Zhu et. al. UniDB: A Unified Diffusion Bridge Framework via Stochastic Optimal Control
2). G Kim et. al. Diffusion-based generative model for financial time series via geometric Brownian motion

5.  PROPOSITION 3.1 seems like an existing result and no need to prove again.
6. Although the paper compared their method to GOUB in terms of metrics such as FID, they did not provide the GOUB results in the figures of images that compares the results. It will be good to seem their comparisons in images as they are similar diffusions between two deterministic images.

### Questions
1. The authors need to discuss more how selection of the coefficients impact the behavior of the diffusion. 
2. Choice of $\theta$ and $\sigma$ affects the performance? 
3. In P4, line 146, the authors stated that As $t \rightarrow \infty$, the SDE converges to a stationary state $x_T ∼ N(x_t | \mu, \lambda^2)$. This formula does not make sense to me. Please check if t is T.

### Soundness
2

### Presentation
3

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
This manuscript proposes a new forward diffusion process for generative modeling. Specifically, the authors leverage a mean-reverting style SDE and apply the mean-reversion terms to both the drift and diffusion coefficients. The designed SDE has a closed-form solution, and the only unknown term is the simulation target. Following the spirit of denoising score matching, the author proposes a simple regression objective to learn it and thus enable simulating the designed SDE. For the sampling option, the author explored both the Euler method and the first-order discretization. Experimental results demonstrate that the proposed technique is applicable on image-to-image tasks.

### Strengths
* The writing is easy to follow.
* Using the proposed mean-reverting-style SDE for generative modeling looks novel to me.

### Weaknesses
* The claim about "simpler, single" diffusion process is somewhat unconvincing to me. According to the training algorithm and sampling procedure, the effort is almost the same as that of the diffusion models, and the training objective itself needs approximation.
* Another main claim of the paper is that the proposed method can be viewed as a stochastic counterpart to flow matching. However, to me, it would be necessary to compare the established stochastic counterpart, known as diffusion bridges or bridge matching [1, 2], of the flow matching model, both conceptually and empirically (on image-to-image benchmarks).
* For empirical results, it would be better to add Gassuain-to-image generation tasks to demonstrate the effectiveness of the proposed framework.



## References
[1] Peluchetti, Stefano. ‘Non-Denoising Forward-Time Diffusions’. (2023)

[2] Shi, Yuyang, et al. ‘Diffusion Schrödinger Bridge Matching’. (NeurIPS 2023)

### Questions
N/A

### Soundness
2

### Presentation
3

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
This paper introduces forward-only diffusion, which replaces the standard diffusion SDE with a mean-reversion term in both the drift and diffusion terms. They derive a tractable solution to the SDE that determines the conditional distribution $p(x_{t+1}|x_t, \mu)$ and parameterize their neural network to learn the flow $\hat{\mu}_{\phi} - x_t$. This enables a tractable loss to minimize the KL between the ground truth conditional and the model's estimate. Experimental results on image restoration tasks are provided, demonstrating improvements over baselines.

### Strengths
**(S1)**: Elegant formulation. A single, forward-only SDE for diffusion with the mean-reversion term is a neat formulation for image-restoration and conditional generation tasks. The mean-reversion term enables a state-dependent denoising process that dynamically adjusts to different corruption levels within an image. To me, this makes a lot of sense for conditional generation.

**(S2)**: Tractability and flexibility. The SDE with mean-reversion in the diffusion term still yields a unique, tractable solution and enables a simple loss function that is stable to train. This is a strong point in favor of the method. Providing both Markovian and non-Markovian sampling strategies for this forward-only SDE further improves the flexibility of this approach.

**(S3)**: Good ablations. Nullifying the diffusion term and reverting the SDE back to the flow-matching ODE clearly demonstrates worse performance on image restoration tasks, in terms of structural similarity metrics. This demonstrates the need for stochasticity in the mean-reverting SDE. Another ablation on fast-sampling is helpful. 

Overall, I think the idea and application of this paper is novel and interesting, and so I would recommend it for acceptance. If some of my concerns outlined below are addressed, I would be happy to raise my score.

### Weaknesses
**(W1)**: Poor unconditional generation. The FID scores on CIFAR-10 (7.89 for FoD-SDE, 5.01 for FoD-ODE) are not competitive with standard forward-backward diffusion models (e.g., Score SDE @ 2.38) or even other forward-only ODE models like Rectified Flow (2.58). While noted as a limitation, this positions FoD as more a specialized method for conditional image generation tasks than as a general generative model. 

**(W2)**: Limited exploration of conditional image generation tasks. The paper mainly focuses on image restoration, which is a low-entropy task (i.e. the source is already close to the target image). Some qualitative examples for image-to-image translation are provided, but a more extensive evaluation on translation tasks would be useful to support the generality of this approach. Additional experiments on text-to-image or latent-diffusion architectures would be further welcome for completeness.

**(W3)**: Missing comparisons. Some recent work on diffusion bridges (denoising diffusion bridge models) tackle similar problems as this paper. Comparison and a more detailed discussion around bridge models is missing. While FoD is an instantiation of stochastic interpolant methods, the paper could be strengthened by a more detailed comparison to other recent SI-based methods that have also been applied to image restoration.

### Questions
**(Q1)**: Given the primary weakness is unconditional generation, have the authors experimented with modifying the prior distribution to better match the model's log-normal structure (e.g., starting from a log-normal prior)

**(Q2)**: Do the authors have explanations for the behavior between the MC and non-MC samplers in Figure 4? Why are structural metrics better here compared to generation quality metrics? How should one decide between the two samplers?

**(Q3)**: What is $x_s$ and $\mu$ in the tasks outlined in Figure 3? How does the SDE behave when the tasks represents bridging two very semantically different distributions?

**(Q4)**: How sensitive is the model to the choice of the $\sigma_t$ schedule given that it now controls mean-reversion / a state dependent term in the diffusion term?

### Soundness
4

### Presentation
4

### Contribution
3
