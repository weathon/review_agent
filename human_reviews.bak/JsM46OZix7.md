# BiMix: Bivariate Data Mixing Law for Language Model Pretraining

- Decision: Reject
- Scores: 3, 5, 5

## Abstract
Large language models have demonstrated remarkable capabilities across various tasks, primarily attributed to the utilization of diversely sourced data. However, the impact of pretraining data composition on model performance remains poorly understood. This paper introduces BiMix, a novel bivariate data mixing law that models the joint scaling behavior of domain proportions and data volume in LLM pretraining. BiMix provides a systematic framework for understanding and optimizing data mixtures across diverse domains. Through extensive experiments on two large-scale datasets, we demonstrate BiMix's high accuracy in loss extrapolation (mean relative error $< 0.2\%$) and its generalization to unseen mixtures (R$^{2} > 0.97$). Optimization of domain proportions yields superior model performance compared to existing methods. Furthermore, we establish entropy-based measures as efficient proxies for data mixing, offering a computationally lightweight strategy. Our work contributes both theoretical insights into data mixing dynamics and practical tools for enhancing LLM training efficiency, paving the way for more effective scaling strategies in language model development.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces BiMix, a bivariate data mixing framework designed to optimize and understand the impact of multi-domain data in LLM pre-training. 
BiMix models the joint scaling behavior of domain proportions and data volume to improve data mixing in LLM pre-training. 
Key contributions include:
1. A mathematical framework that models domain proportions and data volume scaling to enhance interpretability in data mixing.
2. Experimental validation demonstrating BiMix's ability to predict and optimize model performance across diverse data mixtures, outperforming traditional approaches.
3. Introduction of entropy-based measures as efficient, computationally lightweight proxies for data mixing, facilitating faster and more effective pretraining.

### Strengths
1. The research topic on pre-training data mixture is timely and important for the LLM community. 

2. The evaluation is relatively comprehensive. Experiments on two large-scale datasets (The Pile and SlimPajama) demonstrated BiMix's high accuracy in loss extrapolation (mean relative error <0.2%) and its generalization to unseen mixtures (R2 >0.97). Optimizing data mixtures based on the proposed method yields superior model performance compared to existing methods, in terms of both convergence speed and downstream task performance.

3. The proposed entropy-based measures help to reduce the computational overhead of searching data mixtures.

### Weaknesses
1. The validness of the underlying domain-independent assumption. Similar to other recent papers that model the data mixture problem using scaling law like [1], there is also an implicit assumption of the proposed method: different domains are independent. In Eq. (3), no interaction term models the interactions between different domains. We would like to understand how each domain affects every other domain, this interaction might be highly complex and not local. 

2. Missing baselines. There does not seem to be a sufficient comparison between BiMix and other data selection methods, e.g. comparing with other token-level and sample-level automatic data selection methods [2, 3].

3. The claim of *theoretical insights into data mixing dynamics* seems to be an overclaim because there is no theoretical justification.

[1] Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance

[2] Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models

[3] DoGE: Domain Reweighting with Generalization Estimation

### Questions
1. The design choice of different entropy-based measures and how to derive the mixing proportions from them are hard to follow. 
Could the authors please elaborate more on this?

2. As shown in Figure 6, the downstream task performances of the model trained by different mixtures are fairly low, e.g. mostly under 10%. Are the models significantly better than random guesses and how is the variance of the downstream task performances?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces BIMIX, a data mixing law for better understanding and optimizing data mixtures across different domains.
The key contributions of the paper are:
1. They provides a scalable framework to model the relationship between domain proportions and training data quantity.
2. Through experiments on two large-scale datasets, Pile and SlimPajama, BIMIX predicts model loss with a mean relative error below 0.2%.
3. The authors introduce entropy-based measures as efficient and computationally lightweight proxies for data mixing, offering a practical alternative to resource-intensive optimization methods.

### Strengths
1. BIMIX is a novel methods, proposing a bivariate data mixing law with models domain proportions and training data quantity together. They also incorporates entropy-based measures as efficient proxies to fastly optimizate the training.
2. By validating BIMIX with validation error, the paper demonstrates robustness, presenting a comprehensive evaluation for practical effectiveness and applicability.
3. The paper is well-organized, effectively explaining complex ideas with clear definitions.

### Weaknesses
1. The downstream evaluation includes only PPL. This limits the understanding of how BIMIX-optimized mixtures perform on other important tasks, such as ARC, hellaswag, pica, openbookqa, boolq and so on.
2. While the authors aim to reduce computational costs, the paper lacks a detailed comparison of its efficiency. For example, how much GPU hours do we need in order to get the good mixture.
3.  The paper primarily contrasts BIMIX with DoReMi and baseline mixtures, without comparisons to other recent advancements, such as REGMIX[1] or Online Data Mixing[2].

1. RegMix: Data Mixture as Regression for Language Model Pre-training
2. Efficient Online Data Mixing For Language Model Pre-Training

### Questions
I have a little bit confused about the part 5.4 **ENTROPY MEASURES AS EFFICIENT MIXING PROXIES**, it seems that the autor trained
models on a series of entropy-driven data mixtures in order to collect the observational data points necessary for fitting the coefficients of BIMIX. After fitting the BIMIX, we can final get the final mixture for better pretraining. However, at the end of part 5.4, the author seems to show that entropy measures mixing proxy is already enough and do not talk much about the BIMIX.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper introduces a bivariate data mixing law for LLM pretraining, which disentangles models domain proportions and training steps and establishes a connection between domain validation loss and these variables. Empirical experiments validate this proposed scaling law. 

For the experiments, first, this paper assesses the extrapolated loss prediction over full training steps, showing minimal prediction error. Second, four entropy-based mixture candidates are trained to fit the mixing law, and these results are used to predict the loss for two other data mixtures (baseline and doremi) without actual training, demonstrating a strong fit. Third, once the scaling law is established, this paper performs constrained optimization to determine optimal domain proportions, outperforming baseline methods in three generative question-answering tasks. Finally, the entropy-based mixtures are shown to consistently perform better than the baseline.

### Strengths
- The disentanglement of domain proportions and training steps is novel and interesting.
- The experimental results in Sections 3.2, 5.1, and 5.2 are promising, indicating that the proposed formulation is both effective and sound.
- This paper is generally clear and easy-to-follow, addressing the significant problem of data mixing in LLM pretraining.

### Weaknesses
The primary concern with this paper lies in the empirical examination and application of the proposed data-mixing law. Section 5.2 suggests that the data-mixing law can estimate the impact (loss) of certain data mixture proportions without actual training. However, only two data mixture proportions (baseline and doremi) are tested, which may not provide sufficient evidence of general applicability.  Furthermore, these two proportions are highly similar (see Appendix D, where the correlation is approximately 0.9), which further limits the effectiveness of the verification. While I acknowledge the significant costs associated with additional testing, a more comprehensive validation of the proposed law would strengthen its credibility.

Section 5.3 examines the optimized proportion based on the scaling law but is limited to only three generative question-answering tasks, which is insufficient to conclusively demonstrate the optimality of this proportion. This aspect is crucial, as it represents a key use case for the proposed law.

Additionally, the presentation could be improved. Section 3 would benefit from reorganization to enhance clarity. In Sections 5.1 to 5.3, it would also be helpful to specify how many and which data points (i.e., at what training steps and with what proportion ratios) were used to fit the scaling law and predict specific points (i.e., at what training steps and with what proportion ratios). This information is crucial for assessing the practical utility of the scaling law.

### Questions
1. Could you clarify the logical flow between Sections 3.1 and 3.2? My understanding is that Equation (3) is initially proposed, followed by two empirical experiments in Section 3.2 aimed at verifying the proposed law. If this is correct, I suggest avoiding the term "defined" in Line 135, as it may cause confusion.
2. In Sections 5.1 to 5.3, could you specify **how many** and **which** (i.e., at what training steps and with what proportion ratios) data points were collected to *fit* the scaling law and *predict* **which** target data point (i.e., at what training steps and with what proportion ratios)? Presenting this information clearly would be essential to assess the scaling law's utility.
3. Based on my understanding, the four entropy-based data mixtures are used to fit the data mixing laws. 
- To fit a reliable function, it might be preferable to use more scattered-out data points rather than closely related ones.  Why not employ more diverse proportions rather than relying solely on entropy-based methods, which could be quite similar (as indicated by the correlations)?
- Could you clarify the purpose of Section 5.4? If the entropy-based methods are only used to gather observational data points, what is the significance of demonstrating their superiority over the baseline? Should this be considered an incidental result, or does it have a more central role in validating the proposed method?
4. The introduction (Line 59) claims improved convergence speed. Could you provide concrete evidence to support this?
5. What is the log perplexity of doremi? This result seems to be missing from both Section 5.3 and Figure 8(a) in Appendix D.
6. Beyond optimizing data proportions, what other practical use cases could the proposed law have? If this optimization is the primary use case, I would recommend placing greater emphasis on Section 5.3.

Minor suggestions (not affecting scores):
1. $s$ in Equation (2) can be explicitly defined.
2. It may be beneficial to compare recent related literature:

[1] Kang, Feiyang, Yifan Sun, Bingbing Wen, Si Chen, Dawn Song, Rafid Mahmood, and Ruoxi Jia. "Autoscale: Automatic prediction of compute-optimal data composition for training llms." arXiv preprint arXiv:2407.20177 (2024).

[2] Liu, Qian, Xiaosen Zheng, Niklas Muennighoff, Guangtao Zeng, Longxu Dou, Tianyu Pang, Jing Jiang, and Min Lin. "Regmix: Data mixture as regression for language model pre-training." arXiv preprint arXiv:2407.01492 (2024).

I am willing to raise the score if the authors can sufficiently address the above questions.

### Soundness
2

### Presentation
2

### Contribution
2
