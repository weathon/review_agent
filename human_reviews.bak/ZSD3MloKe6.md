# Alleviating Exposure Bias in Diffusion Models through Sampling with Shifted Time Steps

- Decision: Accept (poster)
- Scores: 6, 6, 6, 6

## Abstract
Diffusion Probabilistic Models (DPM) have shown remarkable efficacy in the synthesis of high-quality images. However, their inference process characteristically requires numerous, potentially hundreds, of iterative steps, which could exaggerate the problem of exposure bias due to the training and inference discrepancy. Previous work has attempted to mitigate this issue by perturbing inputs during training, which consequently mandates the retraining of the DPM. In this work, we conduct a systematic study of exposure bias in DPM and, intriguingly, we find that the exposure bias could be alleviated with a novel sampling method that we propose, without retraining the model. We empirically and theoretically show that, during inference, for each backward time step t and corresponding state ˆxt, there might exist another time step $t_s$ which exhibits superior coupling with $\hat{x}_t$. Based on this finding, we introduce a sampling method
named Time-Shift Sampler. Our framework can be seamlessly integrated to existing sampling algorithms, such as DDPM, DDIM and other high-order solvers, inducing merely minimal additional computations. Experimental results show our method brings significant and consistent improvements in FID scores on different datasets and sampling methods. For example, integrating Time-Shift Sampler to F-PNDM yields a FID=3.88, achieving 44.49% improvements as compared to F-PNDM, on CIFAR-10 with 10 sampling steps, which is more performant than the vanilla DDIM with 100 sampling steps. Our code is available at https://github.com/Mingxiao-Li/TS-DPM.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
To address the issue of exposure bias, this paper proposes a sampling method called the Time-Shift Sampler. Specifically, this method is based on the observation that within a window of size w around the time step t, there exists a better time step ts that corresponds to a closer match between the variance of the true samples and the predicted samples xt. Moreover, the feasibility of this method is theoretically validated by the authors. To validate the effectiveness of the proposed method, extensive experiments are conducted on multiple models and datasets, and the results strongly demonstrate its efficacy. Importantly, compared to other relevant papers addressing exposure bias, this article offers the advantage of not requiring retraining and incurring minimal additional costs.

### Strengths
1. This paper investigate an important problem which ignored by previous works, dubbed exposure bias, and propose an interesting method to remedy it.
2. The analysis for the exposure bias problem helps to understand the proposed method, Time-Shift Sampler.
3. Time-Shift Sampler does not require fine-tune the pre-trained diffusion models and incurs minimal additional costs, while effectively mitigating exposure bias. 
4. Moreover, Time-Shift Sampler enables to combine with various diffusion model, which demonstrates a good scalability.

### Weaknesses
1. Due to the conditions imposed by the theoretical derivation, amost 10% sampling time can not be simply ignored.
2. As mentioned 'seamlessly integrated to existing sampling algorithms', how about the performance combined with the DPM-solver [1], DEIS [2].
3. In my humble opinion, the analysis of the exposure bias problem is empirically not theoretically as mentioned in the 'contribution'.
4. There is no obvious advantage over the recent training-free sampling method.

[1] C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu. DPM-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787, 2022.
[2]  Q. Zhang and Y. Chen. Fast sampling of diffusion models with exponential integrator. International Conference on Learning Representations, 2023.

### Questions
1. Can this method combine with other training-free sampling methods, such as DPM-sovler, DEIS?
2. How is the performance on ImageNet?
3. Is there exits a analytical solution for the window size?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper examines the exposure bias of diffusion models and proposes a straightforward and efficient method to address it. The study presents clear experiments and motivation to elucidate both the exposure bias phenomenon and its solution.

### Strengths
1. This paper analyzes the exposure bias of diffusion models, presenting clear visual results and detailed analysis.
2. This paper presents a simple, effective, and training-free solution for exposure bias.
3. The proposed solution can be applied to both DDPM-like and DDIM-like methods, providing potential benefits for future acceleration work.

### Weaknesses
The selection of window sizes and cutoff values is primarily based on limited experience.

### Questions
How can we design a more effective strategy for selecting window sizes and cutoff values when dealing with random datasets and image sizes?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work studies the problem of exposure bias in diffusion models and proposes a training-free method to mitigate exposure bias during sampling. The paper first reasons why mitigating exposure bias in diffusion models could result in improved sampling by empirically demonstrating the accumulation of error at different time steps. To this end, a quantity $C(x_t, t)$, called input couple for trained DPM, is used to capture the discrepancy between ground truth and network predictions. The core idea is that there might be an alternate time step $t_s$ that might align better with the next state $\hat{x}_{t-1}$  predicted by the network. This assumption is empirically demonstrated for different datasets and for different choices of time steps. 

In order to find this alternate optimal time step $t_s$, this work derives the variance of this optimal tilmestep. This can then be used to empirically determine $t_s$ on-the-fly during sampling. This step can be seamlessly integrated into existing sampling methods like DDIM, DDPM, and PNDM. Further, this does not require any fine-tuning or training of diffusion models. However, this comes at the cost of some minor additional overhead in terms of sampling time. Overall, this method gives consistent improvements over baselines of DDIM, DDPM and PNDM in terms of FID scores.

### Strengths
1. The proposed method for alleviating exposure bias is training-free unlike previously proposed methods.
2. The proposed method results in consistent improved performance in terms of FID score compared to baselines of DDIM, DDPM, and PNDM sampling (See Table 1), as well as prior works like ADM-IP.
3. The primary contribution of this work which is empirical demonstration of the fact that correction of exposure bias can be done without retraining diffusion models is valuable.

### Weaknesses
1. The proposed method is training-free but introduces minor overhead in sampling time of diffusion models. Further, as the number of sampling steps increases, the method seems to be sensitive to the choice of hyperparameters like window size. At smaller number of sampling steps, the method is also a bit sensitive to cutoff time. (See Section 5.4)
2. The writing needs improvement as it is currently a bit ambiguous at certain places. (See 1. and 2. In questions below for further details). For instance, plotting details for Figure 3 are unclear from the appendix. Similarly, it is unclear to me how $var(x_{t-1})$ is computed in line 9 of Algorithm 3. The overall clarity of this paper will greatly improve if these paragraphs are rewritten by adding additional details. Similarly, mathematical expressions should be added at multiple places along with text for improved clarity. For instance, Figure 2, writing $var(x_t)$ is more informative than simply writing $x_t$ for the label of y-axis. In Figure 3, the mathematical form of error can be included instead of labelling y-axis as error.  
3. Certain parts of derivation of proof of Theorem 3.1 need further explanation. The proof assumes that for an image P, and pixels $p_i, p_j \in P$, $p_i \perp p_j$ if $i \neq j$ which is usually not true in practice as neighboring pixels in image usually have high correlation. It also assumes that each pixel in image $P$ follows distribution $\mathcal{N}(\mu_i, \sigma)$, but later it claims that $\mu_i$ is the mean of the distribution of $\hat{x}_{t-1}$ which looks incorrect as the mean of the latter distribution is a vector/tensor (as it is an image) while $\mu_i$ is a scalar as it is mean of pixel values.

### Questions
1. The explanation of details for plotting Figure 3 is unclear from the description given in Appendix B. Perhaps writing mathematical equations might make the idea more concise and clear. Also, for the purpose of rebuttal, could the authors include this expression here, or alternately provide an explanation of what the figure indicates? It is unclear to me why the error computation for this figure is split into two different stages. Any intuition/reasoning behind choosing the methodology of computing errors in Figure 3 is appreciated.
2. In DDPM sampling, we use line 4 in algorithm 2 to get the next sample $x_{t-1}$.  It is unclear to me how this $x_{t-1}$ is used to compute $var(x_{t-1})$ in line 9 of Algorithm 3. We cannot use the analytic closed form of variance as it won’t have errors/exposure bias from prediction in network. Thus it needs to be sample variance. In that case, to compute $var(x_{t-1})$, is $z \sim N(0, I)$ sampled multiple times and then sample $var(x_{t-1})$ computed? As other terms for sampling $x_{t-1}$ are fixed for a given $x_t$ (Line 4 in Algorithm 2), isn’t $z$ the only source of variance in this case? How many samples of $x_{t-1}$ are needed to get a reasonable estimate of this sample variance? Is it possible to add these details in the text that explains the algorithm?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes to select a suitable timestep for next step instead decreasing it as in most diffusion model sampling algorithms.

### Strengths
1. The paper is well-written and organized.

2. The analysis are sufficient and the findings in section 3. are interesting. 

3. According to the tables, the FIDs are improved with different baseline samplers. It is also good to see that the sampling time is not increased significantly as shown in figure 6.

### Weaknesses
1.  Other metric should also be included, such as precision and recall. Is there any reason why text-to-image performances are not added? It would be better if authors could include such results as the task is one of most important tasks of diffusion model. 

2. Could you also visualize the selecte timestep trajectory?  It also would better to have more analysis on the selected timesteps and around which timesteps are mostly important.

### Questions
as above

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
