# DPQuant: Efficient and Private Model Training via Dynamic Quantization Scheduling

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Differentially-Private SGD (DP-SGD) and its adaptive variant DP-Adam are powerful techniques to protect user privacy when using sensitive data to train neural networks. During training, converting model weights and activations into low-precision formats, i.e., quantization, can drastically reduce training times, energy consumption, and cost, and is thus a widely used technique. In this work, we demonstrate for the first time that quantization causes significantly higher accuracy degradation in DP training compared to regular SGD. We observe that this is caused by noise injection, which amplifies quantization variance, leading to disproportionately large accuracy degradation.
To address this challenge, we present DPQuant, a dynamic quantization framework that adaptively selects a changing subset of layers to quantize at each epoch. Our method combines two key ideas that effectively reduce quantization variance: (i) probabilistic sampling that rotates which layers are quantized every epoch, and (ii) loss-aware layer prioritization, which uses a differentially private loss sensitivity estimator to identify layers that can be quantized with minimal impact on model quality. This estimator consumes a negligible fraction of the overall privacy budget, preserving DP guarantees. Empirical evaluations on ResNet18, ResNet50, and DenseNet121 across a range of datasets demonstrate that DPQuant consistently outperforms static quantization baselines, achieving near Pareto-optimal accuracy-compute trade-offs and up to $2.21\times$ theoretical throughput improvements on low‑precision hardware, with less than 2% drop in validation accuracy. We further show that our framework extends to DP-Adam with similar gains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates dynamic quantization for differentially private SGD. The authors propose a specific method that converts specific layers into a low-precision format and design a selection criterion based on the difference between the quantized and non-quantized layers. They empirically validated the strength of their dynamic quantization approach across various settings, including different configurations and datasets.

### Strengths
•  The paper investigates the quantization of differentially private deep learning.

•  The authors provide some basic analysis of why DP training interacts poorly with quantization.

•  Based on their observations, the authors’ new training method enables faster training compared to FP32 training.

### Weaknesses
Please refer to the Questions section.

### Questions
•	In the related work section, the authors mention that all existing methods are orthogonal to this work. However, for readers who are not familiar with quantization, it is hard to understand the relationship between these methods and the authors' work. The authors need to improve their related work section. Maybe including more on DP network pruning or layer selection, such as [1]. 

[1] Differential Privacy Meets Neural Network Pruning, arXiv:2303.04612

•	The reviewer is not convinced why the authors shift their perspective from the L2 norm (for noise addition) to the L-inf norm (for analysis). Can the authors clarify this reason or explain why the L2 norm is not sufficient for their analysis? As far as I understand, Figure 1 does not support the reason for the L-inf analysis, and the empirical part uses the L2 norm.

•	Does the quantization method meet the properties in Proposition 1, such as the unbiased and scale-invariant properties?

•	If a low-precision format induces the DP accuracy drop, can using higher precision mitigate the accuracy drop in DP training?

•	When quantizing, is there any difference between floating-point and fixed-point quantization in terms of differential privacy?

•	The authors should carefully check the paper for readability. For example, the GTSRB dataset is not explained or cited when it is first used. And the indicator function in Eq. 1 should be clarified.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on differential privacy (DP) and the impact of quantization variance. The authors hypothesize that quantization variance leads to disproportionately large accuracy degradation in DP settings. To address this issue, they propose DPQUANT, a dynamic quantization framework that adaptively selects a changing subset of layers to quantize at each epoch. The main procedure consists of (i) probabilistic sampling of layers and (ii) loss-aware layer prioritization.

### Strengths
The mathematical analysis is clearly presented, and the theoretical treatment of differential privacy in the proposed algorithm, such as the post-processing of the update exponential moving average (EMA), appears sound. The empirical settings are well constructed, and the figures and tables are clear and easy to interpret. Compared with vanilla DP-SGD, the proposed method demonstrates meaningful performance improvements.

### Weaknesses
However, the related work and comparison baselines raise significant concerns. Quantization is a widely used technique across various applications, and it is natural to consider its potential benefits within differential privacy. Indeed, several prior works have explored the interplay between quantization and DP, including:
- Kang, Tianqu, et al. "The effect of quantization in federated learning: a rényi differential privacy perspective." 2024 IEEE International Mediterranean Conference on Communications and Networking (MeditCom). IEEE, 2024.
- Youn, Yeojoon, et al. "Randomized quantization is all you need for differential privacy in federated learning." arXiv preprint arXiv:2306.11913 (2023).
- Kim, Muah, Onur Günlü, and Rafael F. Schaefer. "Effects of quantization on federated learning with local differential privacy." GLOBECOM 2022-2022 IEEE Global Communications Conference. IEEE, 2022.
- Wang, Yongqiang, and Tamer Başar. "Quantization enabled privacy protection in decentralized stochastic optimization." IEEE Transactions on Automatic Control 68.7 (2022): 4038-4052.
-Xiong, Sijie, Anand D. Sarwate, and Narayan B. Mandayam. "Randomized requantization with local differential privacy." 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2016.
While some of these works focus on federated learning, the authors should still acknowledge them and discuss how their method relates to or differs from these approaches.

Beyond these, it is also important to consider recent advances in DP optimization methods beyond DP-SGD. Since DP-SGD is known to suffer from excessive noise and slow convergence, relying solely on it as a baseline is insufficient to demonstrate the full benefit of the proposed approach. Several improved variants could be relevant comparisons:
- Wang, Zihao, et al. "{DPAdapter}: Improving Differentially Private Deep Learning through Noise Tolerance Pre-training." 33rd USENIX Security Symposium (USENIX Security 24). 2024.
- Wei, Jianxin, et al. "Dpis: An enhanced mechanism for differentially private sgd with importance sampling." Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security. 2022.
- Park, Jinseong, et al. "Differentially private sharpness-aware training." International Conference on Machine Learning. PMLR, 2023.
- Denisov, Sergey, et al. "Improved differential privacy for sgd via optimal private linear operators on adaptive streams." Advances in Neural Information Processing Systems 35 (2022): 5910-5924.

In summary, while the proposed method is technically sound and the presentation is clear, the related work section and choice of baselines need substantial improvement to strengthen the novelty and empirical validity of this paper.

If these concerns are properly addressed, I would be pleased to raise my evaluation score.

### Questions
It is unclear whether the Exponential Moving Average (EMA) update plays an essential role in the proposed framework. While the paper mentions EMA as part of the post-processing step for maintaining differential privacy, there is no ablation study isolating its effect. It would strengthen the work to report performance both with and without EMA, or to justify why EMA is necessary for stability or privacy preservation in DPQUANT.
Furthermore, it would be helpful to discuss whether other optimization or averaging techniques (e.g., momentum-based methods, adaptive optimizers such as Adam or Adagrad, or other moving-average variants) could serve a similar role. Clarifying whether the proposed approach is tightly coupled with EMA or can generalize to other optimizers would provide deeper insight into the method’s robustness and applicability.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies methods to reduce quantization variance in DP-SGD, which slows down the convergence a lot. Technical efforts are dedicated to achieve good performance, maintain DP guarantee, and avoid extra computation. A number of experiments on computer vision and one language modeling are presented on relatively old architectures, showing a near Pareto-optimal utility-compute tradeoffs and about 2X theoretical speed improvement.

### Strengths
Overall the paper is clearly written and shows the originality in combining quantization and DP. The problem of DP training efficiency is an important one, highlighting the significance of this work. The experiments are careful and convincing, especially the ablation study. I enjoy reading the details in this paper, such as Pareto optimality and error bars.

### Weaknesses
1. Wrong positioning of mixed-precision training

The authors state "While effective for standard SGD, mixed-precision training degrades significantly under DP-SGD. To our knowledge, no prior work has explored mixed-precision training when differential privacy mechanisms are employed." Both sentences are wrong. Mixed-precision training with DP is comparable to DP-SGD with full-precision. See Appendix T of "https://arxiv.org/pdf/2110.05679" (no code though) and Section 3.4/Appendix C of "https://arxiv.org/pdf/2311.11822" (code is open-sourced). The caveat is the loss scaling when using fp16, and DP training with bf16 can just run without loss scaling.

2. Limitation to DP-SGD

I recommend the authors to analyze DP-Adam or DP-SignSGD, where the gradient norm is always \sqrt{d} and line 230 won't hold. How would you extend your analysis in this case? Also, DP-Adam and other adaptive optimizers are the main-stream for large models. Can your method work directly? What will be the performance of quantization empirically?

3. Limitation to training from scratch

DP pre-training is hard. I think this is why the authors stick to small and toy datasets. However, the methodology should be applicable to small-scale finetuning as well. Did the authors try that?

4. What about non-DP?

The methods in Section 5.1 and 5.2 should also improve non-DP quantization, even though additional care was taken to make it DP. Are these methods novel? Are there evidence these methods also work in non-DP scenario?

5. Not practical for throughput

Another minor weakness (and my review score does not depend on this point) is that the improvement of throughput is theoretical, not applicable in practice.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper observes that quantization variance is detrimental in DP-SGD. The authors propose two methods to reduce the variance, which achieve good Pareto-optimal performance when used in combination. Specifically, the loss-aware method takes care of DP guarantee. Some experiments on small datasets and models show a good utility-compute tradeoffs while leveraging the quantization speedup.

### Strengths
I think the identification of the effects of quantization variance is novel, and the solutions to reduce the variance is novel. Hence this work has fair significance in improving DP efficiency. The motivation of the solutions is reasonable, as variance in DP noise is already known as the key blocker in DP convergence like "Pre-training Differentially Private Models with Limited Public Data". Hence it is not surprising another type of variance from quantization also harms the convergence.

### Weaknesses
In practice, especially in NLP, models are optimized by adaptive optimizers like AdamW or Muon, whereas this work has a narrow focus on DP-SGD (maybe not even with momentum?) I would encourage the authors to put DP-AdamW in empirical experiments which should not require extra efforts.

In "Pre-training Differentially Private Models with Limited Public Data", it is claimed that variance only harms the early stage of training, e.g. in pretraining regime, to which this paper's experiments are limited. In fact, many existing works including "Large Language Models Can Be Strong Differentially Private Learners" have shown that DP finetuning is comparable to non-DP finetuning, suggesting variance is less detrimental in the finetuning regime. I wonder would DPQuant be less effective in this regime?

Also please use a different notation for policy. Right now both policy and probability is p.

### Questions
Can the authors add DP-AdamW at least for NLP experiments?

Does non-DP quantization works better in finetuning? like comparing fp4 with quantization to fp32 without quantization, will the gap in pretraining be larger than in finetuning?

How necessary is DP quantization in finetuning?

### Soundness
3

### Presentation
3

### Contribution
3
