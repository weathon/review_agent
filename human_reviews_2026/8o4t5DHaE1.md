# Lost in the Non-convex Loss Landscape: How to Fine-tune the Large Time Series Model?

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 2

## Abstract
Recently, large time series models (LTSMs) have gained increasing attention due to their similarities to large language models, including flexible context length, scalability, and task generality, outperforming advanced task-specific models. However, prior studies indicate that pre-trained LTSMs may exhibit a poorly conditioned non-convex loss landscape, leading to limited trainability. As a result, direct fine-tuning tends to cause overfitting and suboptimal performance, sometimes even worse than training from scratch, substantially diminishing the benefits of pre-training.
To overcome this limitation, we propose Smoothed Full Fine-tuning (SFF), a novel fine-tuning technology. Specifically, we construct an auxiliary LTSM via random initialization to obtain a smoother loss landscape, and then linearly interpolate its weights with those of the pre-trained model to smooth the original landscape. This process improves trainability while preserving pre-trained knowledge, thereby enabling more effective downstream fine-tuning. From an optimization perspective, SFF perturbs sharp minima without significantly harming flat regions, facilitating escape from poor local basins toward smoother and more generalizable solutions. Extensive experiments on benchmark datasets demonstrate consistent improvements across eight representative LTSMs, including Timer, TimesFM, MOMENT, UniTS, MOIRAI, Chronos, TTMs, and Sundial, on diverse downstream tasks. The code is available at the link: \url{https://github.com/Meteor-Stars/SFF}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes the Smoothed Full Fine-tunning (SFF) to finetune LTSMs. SFF linearly interpolate the parameters of a pretrained LTSM with a randomly initialized auxiliary model, with the goal of smoothing sharp, non-convex regions of the loss landscape. The main contribution lies in the interpolation that perturbs the parameter of the model to reach flatter regions. Such perturbation towards flat minima is a well-known strategy for enhance generalization. Experimental results show the method can obtain better results than compared baselines and that SFF can perform fine-tuning with minimal memory and computational overhead.

### Strengths
- The authors identify a significant and practical problem, the poor fine-tuning performance of modern LTSMs due to the non-convexity of their loss landscape.  
- SFF is simple to implement, requiring only a one-time linear interpolation of weights before fine-tuning. It adds no computational or memory overhead during the actual fine-tuning process.

### Weaknesses
- Exploration of interpolation strategies. The paper states that $\alpha$ controls the proportion of pre-trained knowledge retained, and the experiments show a sensitivity to $\alpha$. However, the optimal $\alpha$ appears to be model-dependent. A more robust, perhaps adaptive, method for selecting or tuning $\alpha$ would significantly strengthen the work.

- The auxiliary model is defined as a randomly initialized LTSM. The paper claims this model has a smooth loss landscape, however is not clear why a randomly initialized model is guaranteed to be smoother than a pre-trained one, especially considering the vast differences in initialization schemes and model architectures. A more rigorous analysis or reference to why random initialization leads to flat minima would be beneficial.

- The theoretical analysis in Section 3.1 is relatively brief and qualitative, relying heavily on existing literature. I suggest the authors to present a more in depth analysis of the loss landscape for finetuning, such analysis would provide strong theoretical justification.

### Questions
Refer to the weakness part. 

- Can the authors provide a more detailed explanation or empirical evidence as to why a randomly initialized LTSM is guaranteed to have a smoother loss landscape than a pre-trained one?

### Soundness
3

### Presentation
3

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
The paper proposes Smoothed Full Fine-tuning (SFF), a lightweight method to improve fine-tuning of large time-series models by interpolating pretrained weights with a randomly initialized model before training. This simple step smooths sharp loss landscapes, enhancing optimization without extra computational cost. Tested on eight major LTSMs across forecasting and anomaly detection tasks, SFF consistently improves fine-tuning and often boosts zero-shot accuracy. The approach is practical and broadly effective, though the paper’s novelty is mainly within time-series models and lacks comparisons to established flatness-based fine-tuning methods.

### Strengths
- The method is simple and inexpensive. Smoothing is done with a one-shot linear interpolation using a randomly initialized copy before fine-tuning. It only needs a few lines of PyTorch and does not add compute or memory cost.
- The motivation is clear. The paper argues that pretraining can leave large time-series models stuck in sharp, non-convex regions, and interpolation helps escape these while keeping flat regions stable.
- The evaluation covers many settings. The authors test eight large time-series models across forecasting, anomaly detection, and imputation tasks.

### Weaknesses
- The novelty claim may be overstated. The idea of mixing weights for fine-tuning already exists in other domains, so the contribution should be limited to time-series foundation models.
- The baselines are limited. The comparisons use full fine-tuning and linear probing but omit other regularization or smoothing approaches such as SAM, SWA, Mixout, or L2-SP.
- Some zero-shot results decline after smoothing. Models like MOIRAI and Chronos perform worse on certain datasets.
- Interpolation with a random model might cause misalignment. It can disturb normalization or scale between layers, and the paper does not analyze these risks in detail.

### Questions
Please address the identified weaknesses and limitations noted above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper identifies a critical problem in the practical application of large time series models: pre-trained models often exhibit poor trainability when fine-tuned on downstream tasks. The authors attribute this to the models converging to sharp minima during pre-training, resulting in a non-convex loss landscape that leads to overfitting during fine-tuning—sometimes to the point of underperforming models trained from scratch. To address this, the paper proposes a simple and effective method called Smoothed Full Fine-tuning. SFF works by first creating an auxiliary, randomly initialized model, which possesses a smooth loss landscape  but no pre-trained knowledge. Before fine-tuning, SFF performs a single linear interpolation between the weights of the pre-trained model and this auxiliary model. The resulting "smoothed" model is shown to retain the valuable knowledge of the pre-trained model while inheriting the superior trainability of the random model , allowing it to escape sharp minima and find better, flatter basins. The authors provide extensive empirical validation, showing that SFF consistently improves the performance of eight different LTSMs  on forecasting and anomaly detection tasks compared to standard fine-tuning methods, without incurring any additional memory or computational overhead during the fine-tuning step.

### Strengths
The paper tackles a significant and practical limitation of pre-trained LTSMs. The observation that direct fine-tuning can lead to overfitting and even perform worse than training from scratch is a crucial finding that motivates the need for better fine-tuning strategies. The problem is clearly illustrated using loss landscape visualizations.

The proposed SFF method is exceptionally simple, consisting of a single linear interpolation of model weights before fine-tuning begins. This makes it easy to implement and, importantly, it adds no additional memory or computational overhead to the actual fine-tuning process, making it a highly practical solution.

In a particularly strong finding, the smoothing process by itself is shown to improve zero-shot forecasting performance, suggesting it guides the model to a better, more generalizable basin in the loss landscape even without fine-tuning.

### Weaknesses
The theoretical motivation in Section 3.1 is more of a high-level, intuitive argument rather than a formal analysis. While the explanation is plausible and well-supported by citations to related work, the paper does not provide a rigorous theoretical proof for why this specific form of interpolation is an optimal or principled way to achieve this smoothing
 
The interpolation coefficient $\alpha$ is a new, critical hyperparameter introduced by SFF. The paper shows SFF is robust, outperforming FF across a wide range of $\alpha$ values However, the paper also shows that the optimal $\alpha$ for zero-shot performance differs from the optimal range for fine-tuning. The paper provides limited guidance on how $\alpha$ should be selected for a new model or dataset, other than selecting from a predefined set.

The method relies on constructing an "auxiliary LTSM" through random initialization of the same architecture. The paper does not explore or justify this specific choice. It is unclear if a different, or perhaps simpler, randomly initialized model could achieve a similar or even better smoothing effect.

### Questions
Listed in weakness

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
This paper investigates the fine-tuning challenges of large time series models and identify non-convex loss landscapes in pre-trained LTSM is the key root cause that lead to poor trainability and overfitting during fine-tuning stage. To alleviate the challenge, the authors propose smoothed full fine-tuning, which linearly interpolates the weights of a pre-trained LTSM with a randomly initialized auxiliary model to smooth the loss landscape, to improve trainability while preserve pre-trained knowledge. The method evaluation on forecasting, imputation, and anomaly detection tasks using eight existing LTSMs across multiple benchmark datasets shows consistent improvements over baselines (i.e., full fine-tuning, training from scratch, linear probing, and linear probing then full fine-tuning) with no added computational overhead.

### Strengths
1. The paper studies an interesting issue that pre-trained LTSMs may have sharp loss landscapes that hinder fine-tuning with empirical evidence and visualization.

2. The proposed method is simple yet effective, requiring only a one-time weight interpolation before fine-tuning.

3. The experiment is comprehensive, covering wide range of LTSM architectures and tasks.

### Weaknesses
1. Lack of novelty: The core idea of linear weight interpolation to smooth landscapes is not a new idea. Weight averaging/interpolation [1] has been widely studied in model merging [2], continual learning [3], yet none of these have been discussed in related work. Although the paper claims to be "the first" for LTSMs, this domain-specific application does not justify novelty, which seems more like a repackaging of existing optimization tricks with application on LTSMs.

2. Lack of theoretical justification: The paper attempts to explore the challenge of model fine-tuning from optimization theory perspective. However, the discussion in Section 3.1 seems hand-wavy. Specifically, the claim, "interpolation perturbs sharp minima without harming flat regions" is intuitive but lacks rigor. Providing some theoretical proofs or connection to sharpness-aware minimization can better provide theoretical merit to the readers.

3. Baselines for comparison: None of the parameter-efficient methods like LoRA, QLoRA are included, these are standard for fine-tuning large models. Also, there is no ablation on other smoothing techniques such as Gaussian noise or label smoothing. Without those baselines, it is hard to evaluate the contribution of this paper as the claim of paper is to tackle the "fine-tuning" problem of time series foundation model.
 
[1] Vlaar, Tiffany J., and Jonathan Frankle. "What can linear interpolation of neural network loss landscapes tell us?." International Conference on Machine Learning. PMLR, 2022.

[2] Wortsman, Mitchell, et al. "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time." International conference on machine learning. PMLR, 2022.

[3] Kozal, Jędrzej, et al. "Continual learning with weight interpolation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024.

### Questions
1. How does SFF compare to simply adding random noise to pre-trained weights (e.g., Gaussian perturbation)?
2. How sensitive is SFF to the random init distribution?

### Soundness
3

### Presentation
3

### Contribution
2
