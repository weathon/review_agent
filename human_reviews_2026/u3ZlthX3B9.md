# PreDiff: Leveraging Data Priors to Enhance Time Series Generation with Scarce Samples

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
The fundamental motivation for time series generation tasks lies in addressing the pervasive challenge of data scarcity. However, we have identified a critical limitation: existing time series generation models are prone to substantial performance degradation when trained on limited data. To tackle this issue, we propose a novel framework that integrates data priors to enhance the robustness and generalization of time series generation under data-scarce conditions. Our framework is structured around a two-stage pipeline: pre-training and fine-tuning. In the pre-training stage, the model is trained on synthetic time series datasets to learn data priors, which encode the fundamental statistical properties and temporal dynamics of time series data. Subsequently, during the fine-tuning stage, the model is refined using a small-scale target dataset to adapt to the specific distribution of the target domain. Extensive experimental evaluations demonstrate that our framework mitigates performance degradation caused by data scarcity, achieving state-of-the-art results in time series generation tasks. This work not only advances the field of time series modeling but also provides a scalable solution for real-world applications where data availability is often limited.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies time series generation models, particularly diffusion models, which suffer performance degradation when trained on scarce data. To address this, the authors propose PreDiff, a two-stage training framework. The first stage pre-trains a diffusion model on a large synthetic prior dataset), on the latter half of the denoising process. The second stage fine-tunes the model on the small target dataset), on the initial half of the process, with the parameters from the first stage frozen. The authors claim this method effectively leverages general priors to mitigate overfitting and achieves state-of-the-art results.

### Strengths
1. **Novel Training Heuristic**: The core idea of splitting the diffusion process at a point t0​ for pre-training and fine-tuning is a novel and clever heuristic. The intuition of learning varying time series data structure from a large prior and, then learning details from the target data is conceptually sound.

2. **Strong Empirical Results**: The ablation in Figure 3(a) is the most compelling part of the paper. It clearly demonstrates that the proposed two-stage split method outperforms simpler alternatives like "Pre-training Only," "Fine-tuning Only," and "Datasets Mixing," which validates the efficacy of the proposed training strategy over these baselines. Also, the tables 1 and 2 show that the proposed algorithm outperforms several baseline algorithms.

### Weaknesses
1. **Some Results Weakening the Motivation**: The paper's core premise is to resolve data scarcity. However, the authors state in Section 5.5 (and show in Table 5) that the method performs well when the prior data is close to the target data. This implies that one must already have access to a large, well-matched dataset or a synthetic generator that knows the target's core properties. I think this is a form of transfer learning from a known, similar source, which somewhat contradicts the "data scarcity" scenario where such well-matched priors are, by definition, unavailable.

2. **Lack of Technical Rigor and Theoretical Grounding**: 
 - In Section 3, the paper defines $\mu_\theta$​ as the model predicting the mean of the reverse process. However, the loss functions $L_{\text{pre}}$​ (Eq. 2) and $L_{\text{ft}}$​ (Eq. 3) train this model to predict the noise $\epsilon$ (i.e., $|| \mu_\theta (\cdot) -\epsilon||^2$. This is a bit confusion between the mean-predictor $\mu_\theta​$ and the noise-predictor $\epsilon_\theta$​, demonstrating a lack of technical precision.

 - Un-grounded Heuristic: The main contribution, the $t_0$​ split, is presented as an ad-hoc heuristic without any theoretical reasons. The paper claims $[t_0​,T]$ maps to "coarse structure" and $[0,t_0]$ to "fine details" but provides zero evidence for this assertion.

### Questions
1. Your results in Table 5 show that the prior seems to be "highly relevant" to the target for the proposed method to perform well. How do you argue this requirement with the paper's "data scarcity" motivation? I think a true scarce-data scenario implies such a large, well-matched prior is not available.

2. Can you please clarify the critical inconsistency in your method? Section 3 defines $\mu_\theta$​ as the mean-predictor (denoising process), but Equations 2 and 3 train it to target $\epsilon$​. Which is it? Also, please add $\epsilon$ to the expectation. 

3. The core claim is that $[t_0, T]$ learns "coarse" priors and $[0, t_0]$ learns "fine" details. What theoretical or empirical evidence (e.g., visualizations of samples at step $t_0$​) can you provide to support this assertion? Without this, I think $t_0$​ split appears to be just an un-grounded, dataset-specific hyperparameter.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses performance degradation in diffusion-based time series generation models under data scarcity. It proposes PreDiff, a two-stage framework: pre-training on a large data prior, synthetic or real, and fine-tuning on the target scarce dataset. Its claimed novelty is a specific "split-step" training strategy: pre-training focuses on later diffusion steps t_0 to T, global structure, while fine-tuning exclusively updates earlier steps 0 to t_0, fine details. Experiments aim to show improved generation quality under scarce data conditions.

### Strengths
1 Addresses the critical problem of TSG under data scarcity. The split-step training strategy, linking diffusion stages to transfer learning, is a conceptually distinct idea within this context.

2 Motivation is clear. The two-stage framework is presented logically. Experiments use standard benchmarks and show empirical benefits, particularly in severe data scarcity 10% data.

### Weaknesses
1 The central assumption linking diffusion steps to transferable features lacks strong theoretical support or broad empirical validation across diverse TS types/diffusion models presented here. Its effectiveness may be context-dependent.

2 Success hinges on a relevant, high-quality data prior X_prior. The paper offers little practical guidance on selecting or assessing prior suitability, posing a major barrier to reliable application and risking negative transfer.

3 Performance is likely sensitive to the split point t_0, yet guidance on its selection is minimal beyond empirical observation(Appendix I) . Lack of a principled selection method adds significant tuning difficulty.

4 As the pre-train/fine-tune paradigm is standard, the overall contribution relies heavily on the split-step strategy. If its universality is questionable, the novelty might be seen as incremental.

5 Comparison with ImagenFew highlights sensitivity to preprocessing, potentially affecting fairness and conclusions.

6 Reliance on potentially massive priors implies significant computational costs, limiting practicality, especially if priors need tailoring per domain.

### Questions
1 Can you provide stronger theoretical arguments or broader empirical evidence (e.g., across diverse datasets, different diffusion models) to support the universality of the hypothesis that pre-training high-noise steps and fine-tuning low-noise steps is an optimal transfer strategy for diffusion models?

2 How can a practitioner reliably select an effective data prior X_prior for a given scarce target dataset X_target? What happens if a truly relevant large prior is unavailable? Please elaborate on the risk and mitigation of negative transfer.

3 Given its likely sensitivity, how should t_0 be chosen in practice? Is there a risk that optimal t_0 heavily depends on the specific prior-target pair, requiring extensive tuning for each new application?

4 Considering the need for a massive relevant prior, the cost of pre-training, and the tuning required for t_0, how practical is PreDiff for real-world users facing data scarcity?

5 Could you clarify the impact of preprocessing differences and potentially provide results comparing PreDiff and ImagenFew under identical preprocessing settings to ensure fairness?

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
4

### Summary
In this study, the authors investigate the problem of time series generation under low-data regimes. In particular, they propose a two-step training procedure including (1) pre-training on synthetically generated data and (2) fine-tuning on target data. The experiments on four real-world datasets show that the proposed approach is applicable even when fine-tuned on 10% of the available target data. Ablation studies are conducted to assess the advantages of a two-stage training procedure and of pre-training on snythetic data.

### Strengths
1) The authors investigate a very interesting research question, trying to learn time series features purely from synthetic data.
2) The paper is well structured and easy to follow.
3) The authors evaluate their method on established benchmarks.
4) The authors conduct ablation studies to provide insights on the effectiveness of the proposed components.

### Weaknesses
1) The authors state that 'detailed configurations, hyperparameters, and implementation specifics for each baseline are meticulously documented in Appendix C' (see ll. 230-232), while they only provide basic information. 
2) The authors have mistakenly highlighted their method to achieve the best results in Table 6, while actually ImagenFew is superior. For instance, ED in the 70%, 40%, and 10% setup of Energy and ED and DTW in the 10% setup of Stocks. In light of this, the results of the work need to be treated with caution.
3) The authors state that 'When selecting data priors, we aim to choose those that are highly relevant to the data distribution of the target task' (see ll. 407-408). This suggests that pre-training is task-specific and does not achieve generalisable time series features, which would be desirable. 
4) The authors do not report their results across multiple seeds to guarantee robustness. 
5) The authors do not support reproducibility by making their code publicly available for evaluation. 
6) The authors do not discuss the limitations of their work.

### Questions
1) Is there a benefit of using synthetic data over real-world data from other domains than the target? 
2) Why is dataset mixing inferior to synthetic data only, as indicated by the results in Figure 3a? Does real-world data not increase the data diversity, which is beneficial to learn generalisable features?
3) Why is the proposed method performing substantially worse when applying a full-range training? 
4) Why is the proposed method performing worse when increasing the training samples of the Monash dataset from 100k to 10M, as indicated by the results in Table 5?
5) Finally, how does the proposed method advance the field of time series analysis? The authors state that 'one should prioritize priors whose distribution closely matches that of the target data' (see ll. 404-405). In light of this, it seems that models for time series generation are still task-specific. However, it would be desirable to have a single, task-agnostic model that can be pre-trained on synthetic data once and be applied to any downstream task.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
PreDiff introduces a two-stage diffusion based framework for time series generation under data scarcity. It first pretrains on synthetic priors  to capture general temporal structures, then fine-tunes on limited real data to adapt to specific domains. The method aims to mitigate degradation in diffusion-based time series generation when data is scarce.

### Strengths
* Data scarcity in time series is pervasive and underexplored in diffusion literature.
* The two-stage pretrain, finetune strategy is analogous to foundation model training in NLP/CV.
* Compared against 6 strong baselines 
* PreDiff outperforms baselines across multiple datasets and scarcity levels.
* The pseudo-code and diagrams are well organized and readable

### Weaknesses
*  Conceptually similar to “pretrained diffusion + fine-tuning,” which has analogues in vision and text domains.
*  No theoretical justification for why segmenting the diffusion process into [t0,T] and [0,t0] yields better transfer.
*  Effectiveness may rely on the quality of external priors rather than intrinsic model improvements.
*  The paper mentions varying priors but doesn’t deeply analyze how prior target similarity influences results.
*  Missing visual analysis. Few qualitative samples of generated time series are shown to demonstrate realism.

I believe these things are easy to add and can increase the value of the paper.

### Questions
* How sensitive is PreDiff to the choice of segmentation point t0?

* Can the method generalize to multimodal or irregularly sampled time series?

* Does pretraining on synthetic priors ever lead to overfitting or “prior bias” in domains with very different dynamics?

* what about many datasets present in UCR dataset?

### Soundness
3

### Presentation
3

### Contribution
3
