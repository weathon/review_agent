# Fast and Reliable Generation of EHR Time Series via Diffusion Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 5, 3

## Abstract
Electronic Health Records (EHRs) are rich sources of patient-level data, including laboratory tests, medications, and diagnoses, offering valuable resources for medical data analysis. However, concerns about privacy often restrict access to EHRs, hindering downstream analysis. Researchers have explored various methods for generating privacy-preserving EHR data. In this study, we introduce a new method for generating diverse and realistic synthetic EHR time-series data using Denoising Diffusion Probabilistic Models (DDPM). We conducted experiments on six datasets, comparing our proposed method with seven existing methods. Our results demonstrate that our approach significantly outperforms all existing methods in terms of data utility while requiring less training effort. Our approach also enhances downstream medical data analysis by providing diverse and realistic synthetic EHR data.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper developed a diffusion model, TimeDiff to generate synthetic EHR time-series data. Authors consider the generation of both numerical
(real-valued) and discrete time-series by combining both multinomial and Gaussian diffusions. Experiments on 6 datasets show the proposed method achieves better discriminative and predictive scores.

### Strengths
1. The paper studies an important problem, mixed-type EHR generation.

### Weaknesses
1. The authors claim that TIMEDIFF is the first to generate mixed type EHR. However, other works like [1,2,3,4] have done the same or similar things. The authors did not compare or discuss these works. And comparing one or two of them is important. 
2. Further, the proposed method might not be as new or unique as the authors suggest. It’s important to note that TIMEDIFF is a diffusion model by replacing the U-Net architecture. The change of loss function and nosing step is straightforward by incorporating previous works (multinomial diffusion). 
3. The authors say that TIMEDIFF is faster because it takes less time to train than GAN-based methods. However, when we look at how fast generative models work, we usually look at how quickly they can create samples (**sampling procedure**), not how quickly they can be trained. Diffusion models, which TIMEDIFF is based on, usually create high-quality samples but take a long time to do so. So, saying that TIMEDIFF is more efficient than GANs in the introduction is misleading. The authors should instead focus on comparing the speed of creating samples. 
4 Privacy evaluation is necessary as existing works do, like membership inference attack. 

## Reference 
1. Li et.al., 2023. Generating synthetic mixed-type longitudinal electronic health records for artificial intelligent applications
2. Ceritli et.al. 2022.  Synthesizing Mixed-type Electronic Health Records using Diffusion Models
3. Naseer1 et.al., 2023. ScoEHR: Generating Synthetic Electronic Health Records using Continuous-time Diffusion Models
4. Theodorou et.al., 2023. Synthesize high-dimensional longitudinal electronic health records via hierarchical autoregressive language model

### Questions
See weakness.

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors present an approach to generate synthetic EHR samples using denoting diffusion probabilistic models (DDPM). To admit both numerical and categorical data, they proposed a novel 2-stage method to generate samples using the diffusion model. Furthermore they compared against several baseline models for a number of tasks.

### Strengths
There are several key contributions in the paper as follows
- Synthetic samples for EHR is an immensely important topic that can potentially impact many aspects of AI for Health, including data availability and privacy preserving learning. The current SOTA method for synthetic EHR data is based of GAN. Seeing the promise of diffusion models for other domains, both in terms of performance and optimized training, it is thus quite exciting to see a working solution that can adapt to the nuances of EHR. The authors have explicitly considered several nuances such as mixture of numerical/categorical data and missing values. 
- The performances on several benchmark datasets are quite promising, especially in terms of being able to mimic the real world datasets
- The authors have tried to justify the importance of several sub components using ablation studies

### Weaknesses
There are several aspects which if addressed can improve the exposition of the paper. 
- The main aspect is that while the authors have performed various high level experimental evaluation, the paper is a bit under-analyzed, especially when considering the domain of healthcare. For example, it may be interesting to conduct sub-group analysis to understand reliability zones of the algorithm
- Another aspect that could be analyzed is some form of explainability analysis to understand the key driver of the learning. While the authors have presented results at a meta-level of categorical (multinomial) and numerical (gaussian) data modalities - it would be interesting to understand the modalities around health data dimensions such as diagnosis, drugs, and lab results.

Some other minor comments are as follows
- the presentation of the method can be substantially improved. While noting the page limit, the description of diffusion processes and the key contribution could be improved upon
- Some choices have not been explained in details. For example for the backbone network, the authors chose BiRNN. Were attention based models considered? 
- Also, the baselines, while many, should include a few of the more recent architectures (e.g. based on diffusion processes that granted may not address the categorical data well) and some classical ones e.g MedGaN

### Questions
There are few aspects which may need some clarification from the authors
- the Diffusion process presented assumes no interaction between numerical and categorical features. Is this choice justified? Have the authors considered investigating individual trajectories for validity of the samples? 
- The performance of in-hospital mortality task is rather low. Have the authors considered more advanced methods such as RNN/Attention models as modelers? In the same note, how was the data cohorted and the features selected for this task?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors adopt DDPM (Gaussian transition together with multinomial transition) for EHR generation. They adopt Time-conditional BRNN as the backbone together with Diffusion Step Embedding. They evaluate their methods on six datasets against seven baseline methods.

### Strengths
The experimental results are good. The authors adopt six criteria rather than only TSTR and similarity criteria by previous methods.

### Weaknesses
The contribution of this paper is more heuristic, i.e., Time-conditional BRNN with time embedding can achieve better performance for generating EHR data while with no theoretical guarantees.

### Questions
The author adopts the sample mean as the imputation methods for dealing with missing data. More advance techniques can be adopted, which might further improve the performance.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposed a diffusion probabilistic model for the generation of EHR time-series data, leveraging a combination of multinomial and Gaussian diffusion. By introducing this mixed diffusion approach specific to EHR time-series data, they have empirically demonstrated enhanced performance in comparison to other time-series generation methodologies, especially in terms of data utility.

### Strengths
- This is the first work to apply this mixed diffusion approach to EHR time-series data.
- The authors have demonstrated the model's performance not only on EHR data but also on non-EHR data, showcasing its applicability across diverse domains.

### Weaknesses
- The time series EHR synthesis studies from Kuo et al. (2023) and Yoon et al. (2023) were mentioned in the related works, but not included in the baseline section. It would be an imperative first step to improve the soundess of the paper to integrate these studies into the baseline to ensure a thorough comparative analysis, especially given the absence of synthetic models specifically designed for EHR synthesis in the current baseline candidates.
- The title suggests "Fast and reliable generation", yet the evaluation on this aspect seems somewhat limited. Merely measuring training time and claiming "fast generation" may be an overclaim, without addressing the sampling time.

### Questions
Please see the weaknesses

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
