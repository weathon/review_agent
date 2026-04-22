# Latent-to-Observable Score Correction for Probabilistic Time Series Imputation

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Missing data remains a key challenge in multivariate time series modeling, often degrading downstream performance. Recent score-based generative models show strong potential for high-quality imputations, yet most ignore original missing data during training, since ground truth is unavailable, resulting in biased score estimation. We theoretically analyze the effect of missingness on score-based modeling under the denoising diffusion probabilistic model (DDPM) framework. Our findings reveal that ignoring original missing patterns—especially under high missing rates or strong inter-variable correlations—can significantly distort the learned score function even at non-missing points. To overcome this, we propose the Hierarchical Score-Based Generative Model (HSGM) for probabilistic time series imputation. HSGM integrates latent-space and observation-space diffusion in a layer-wise refinement framework grounded in the chain rule of probability. A pretrained Variational Autoencoder (VAE) with normalizing flows captures complex latent distributions, while a continuous-time variational diffusion (VPSDE) operates in latent space. A cross-attention mechanism between the original and denoised latent states enhances the fidelity and resolution of the generative outputs, while an observation-space diffusion module further refines the final imputations. Experiments on four benchmark datasets show that HSGM achieves the best accurate imputations with tighter uncertainty estimates than existing methods, while effectively correcting score function bias, establishing a new state of the art in time series imputation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
To address the issue that existing score-based generative models ignore originally missing data during training, which leads to biased score estimation, this paper first provides a theoretical analysis of the impact of missingness under the DDPM framework. It then proposes HSGM, which integrates latent-space and observation-space diffusion within a layer-wise refinement framework grounded in the chain rule of probability. This design enables the model to handle originally missing data without requiring ground-truth supervision during training. Finally, extensive experiments are conducted to validate the effectiveness of HSGM.

### Strengths
- S1：This paper provides a comprehensive theoretical analysis.
- S2：The paper presents extensive experimental results and compares HSGM with a wide range of baseline methods.

### Weaknesses
- W1: The motivation is not sufficiently clear. It remains unclear why existing score-based models would perform worse under scenarios with high missing rates and strong inter-variable correlations when original missing data are ignored. It would be helpful to include experiments comparing the performance of existing score-based models with and without original missing data, particularly under datasets characterized by high missingness and strong inter-variable correlations. Such experiments would better clarify the motivation of HSGM.
- W2：The experimental setup is not clearly described. The settings of the ablation study variants are insufficiently specified, and there is no reasonable analysis explaining why Latent Diffusion or VAE performs better on the ETT dataset. In addition, more detailed ablation variants are needed. For example, the paper emphasizes the importance of the cross-attention mechanism, but no corresponding ablation variant is provided to support this conclusion.
- W3：The experimental analysis is incomplete. For instance, on the P2012 and MIMIC-IV datasets, HSGM shows significantly worse CRPS scores compared to the baselines. It would be important to analyze whether this is caused by certain characteristics of the datasets or other factors.
- W4：The model complexity is concerning. HSGM involves two diffusion stages and incorporates a cross-attention mechanism, yet lacks any analysis of computational complexity, model parameter count, runtime efficiency, or hyperparameter sensitivity. Such analyses are essential for this work.
- W5：The paper lacks necessary explanations of some notations, such as the meaning of N and F in line 136. There are also minor typographical issues, such as the misspelling of “synthetic” in Figure 3.

### Questions
- Q1：How does the performance of HSGM change with varying missing rates, and how does its robustness compare with the baselines?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Hierarchical Score-Based Generative Model (HSGM). Authors first observe that if we run DDPM on imputing missing data with zero or mean, there is intrinsic bias, which basically originates from the discrepancy between noisy distribution (probability path) seen during training and during inference. To mitigate this, HSGM propose to use another latent diffusion model that first imputes missing data, then run original generative model.

### Strengths
- The flow of the paper is great. They first provide theoretical insight, then propose the method that address the problem, and show that empirically it works.
- Empirical results seems good.

### Weaknesses
- I think proposition 4.1 and 4.2 is inappropriate for this context. 4.1 and 4.2 assumes that the original DDPM model is trained without missing values. These propositions are saying that if we train our model without missing values during training, but then if we have missing values during inference, there is a significant bias. This is totally true, but not interesting, because the condition information we give to the model is completely changed. This is not the case how imputation model is trained/used. Even CSDI paper used in this paper give missingness information to the model, so the model is trained to adapt its score function for missing values. If we add this additional condition to the model, then the analysis on section 4 should be completely re-written. (Correct me if I am misunderstanding)

### Questions
- Please address weakness. Since theoretical analysis is crucial point of this paper, if this part is addressed, I will re-evaluate and increase the score.
- Do we actually need expensive diffusion model to impute missing values? Please compare with using CSDI (or other baselines) with lighter imputation methods

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a hierarchical score-based diffusion model for probabilistic time series imputation. In this model, a pre-trained VAE is used to model the latent distribution, and a cross-attention module is used to enhance the data generation performance.

### Strengths
1) Relevant derivations regarding the proposed method are given.
2) The idea of using a pre-trained VAE to learn the latent distribution for the diffusion process seems reasonable. 
3) The idea of using cross-attention to improve the fidelity of the data sampling process seems feasible.

### Weaknesses
1)	Using a latent VAE and cross-attention to improve the diffusion model’s performance is not a very novel idea.
2)	Can the proposed method be accelerated like other types of diffusion models, e.g., DDIM and Rectified Flow? 
3)	The computational and time costs of the proposed method compared to other diffusion models are unclear.
4)	Using multi-sample averages from different initial noise can mitigate sampling bias of diffusion models. The author should also discuss this approach and compare it with the proposed method. 
5)	Minor: typo in Line 325, it should be “obtained”

### Questions
1)	The idea of debiasing does not make much sense as the sampling trajectories of the DDPM is stochastic. What is the unbiased estimation in the context？

2)	It’s not hard to prove that the bias is only related to the time-dependent variance as indicated in Eq (18). However, one of my main concerns is that the linear approximation in Eq (18) might not be accurate enough as the score function is learned from the data. 

3)	To let Eq (6) hold, it has to assume that the denoising process is absolutely accurate for x_t (which holds for the forward process with a pre-defined noise scheduler), however the score estimated by the neural network is only an approximation. Therefore, I have some doubt about Eq (6).

4)	Can the proposed method be used for other data types besides time series as well?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses missing data imputation in multivariate time series (MTS). While recent score-based generative models have shown promise for MTS imputation, most neglect the original missing data during training, leading to biased score estimation. 
The authors theoretically analyze this bias within the DDPM framework, demonstrating that ignoring missing patterns distorts the learned score function—even for non-missing data points. To overcome this, they propose HSGM, which combines latent- and observation-space diffusion using the chain rule of probability. Experiments on four benchmark datasets show that HSGM outperforms existing methods in imputation accuracy and uncertainty estimates, while effectively correcting score bias.

### Strengths
S1: The paper directly addresses the critical issue of ignoring original missing in score-based imputation.
S2: The paper rigorously analyzes the bias in the score function induced by missing data under the DDPM framework, offering a solid mathematical foundation for why existing methods fail and justifying the need for HSGM.
S3: The paper presents comprehensive experimental results across a variety of benchmark datasets, showcasing HSGM's superior performance compared to existing methods.

### Weaknesses
W1: The paper outlines the weaknesses of individual imputation models in isolation, but it does not attempt to group the literature into coherent categories. A taxonomy that first classifies existing approaches and then highlights the shared limitations within each category would provide readers with a clearer understanding.
W2: The Introduction does not clearly explain why the 'original missing effect' warrants attention, nor does it provide a rationale for choosing DDPM as the framework for theoretical analysis.
W3: Several components of the proposed method, such as VAE, CSDI, VPSDE, and cross-attention, are based on existing techniques, which limits the novelty of the method.
W4: The authors should further explore how their method benefits downstream tasks like prediction and classification in the Experiments.

### Questions
1.Why was linear interpolation chosen for handling original missing data? When 80 % of the values are fake, would the performance be seriously impacted?
2.The method described in the paper follows a multi-stage process (latent diffusion - VAE decoding - observation diffusion), with each stage feeding its output directly into the next. In this pipeline, is there a risk that errors from earlier stages could accumulate and amplify in later stages? If so, how might this affect the final imputation accuracy, and what measures have been taken to mitigate such error propagation?

### Soundness
3

### Presentation
3

### Contribution
3
