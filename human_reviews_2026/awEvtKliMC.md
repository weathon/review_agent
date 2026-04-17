# PairFlow: Closed-Form Source-Target Coupling for Few-Step Generation in Discrete Flow Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
We introduce $\texttt{PairFlow}$, a lightweight preprocessing step for training Discrete Flow Models (DFMs) to achieve few-step sampling without requiring a pretrained teacher. DFMs have recently emerged as a new class of generative models for discrete data, offering strong performance. However, they suffer from slow sampling due to their iterative nature. Existing acceleration methods largely depend on finetuning, which introduces substantial additional training overhead. $\texttt{PairFlow}$ addresses this issue with a lightweight preprocessing step. Inspired by ReFlow and its extension to DFMs, we train DFMs from coupled samples of source and target distributions, without requiring any pretrained teacher.
At the core of our approach is a closed-form inversion for DFMs, which allows efficient construction of paired source–target samples. Despite its extremely low cost, taking only up to 1.7\% of the compute needed for full model training, $\texttt{PairFlow}$ matches or even surpasses the performance of two-stage training involving finetuning. Furthermore, models trained with our framework provide stronger base models for subsequent distillation, yielding further acceleration after finetuning. Experiments on molecular data as well as binary and RGB images demonstrate the broad applicability and effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PAIRFLOW, a lightweight preprocessing method that accelerates sampling in Discrete Flow Models (DFMs) without relying on pretrained teachers or costly finetuning. DFMs, though powerful for modeling discrete data such as molecules and images, typically require many iterative steps for generation. PAIRFLOW addresses this by deriving closed-form forward and backward velocity fields that allow efficient pairing of source (prior) and target (data) samples directly in discrete spaces, in terms of the Hamming distance. This pairing enables the model to learn “straighter” probability paths, drastically reducing the number of sampling steps needed. Despite its minimal computational overhead (≤1.7% of total training cost), PAIRFLOW achieves or surpasses the performance of state-of-the-art distillation-based methods like ReDi and Discrete Consistency Distillation, across both molecular (QM9, ZINC-250k) and image (MNIST-Binary, CIFAR-10) benchmarks. Furthermore, models trained with PAIRFLOW serve as stronger bases for later distillation, achieving additional speed and quality gains, suggesting a general strategy for efficient few-step generation in discrete generative models.

### Strengths
- The authors show that the method works well, especially in training small text-to-image models. It can also be combined successfully with distillation methods such as DCD and ReDI.

- Despite the weaknesses I point out below, the method can be a cheap way to boost pre-training of DFMs, and is relevant to the community.

### Weaknesses
- The concept of straightness of probability paths is central to the paper, but the explanation in Section 3.2 is lacking and leaves the questions with many questions as to what it means to straighten probability paths. This explains my low presentation score. Happy to raise my score if this is addressed.

- As the authors point out in Section 6, the methods seems to perform better for relatively low dimensional data. This is not surprising, because the construction of the backward trajectories relies on the training dataset having good coverage over the distribution measured in the Hamming distance, which can be hard when the dimension is very high. 

- The authors try their method on datasets of tens to low hundreds of thousands of samples. Since evaluating the backward velocity field involves a sum over all training samples, I wonder how scalable the method is when the training set is much larger, which may be needed for good performance in higher dimensional settings (related to the previous point). Can the authors comment on this?

### Questions
- Can the proposed method be extended to masked priors? It doesn’t look like it can. Masked priors are relevant because they are the best performing methods for some applications.

- The proposed method is close in spirit to the works in continuous diffusion that use empirical estimates of the score function (with different goals). This kind of approaches are fundamentally cursed by dimension, and although they may provide some gains, are hard and expensive to scale to very high dimensions and very large training datasets. Can the authors comment on this?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents PairFlow, a novel and highly efficient preprocessing method for training Discrete Flow Models (DFMs) that enables high-quality generation with very few sampling steps. Inspired by ReFlow and ReDi, the authors uses coupled source and target distributions for training. The core innovation is the derivation of closed-form forward and backward velocity fields for DFMs, which allows for the direct construction of optimized source-target data pairs from a collection of samples from the target distribution, without needing a pretrained teacher model.

### Strengths
1. The paper is well-motivated and easy to follow.
2. The derivation of the closed-form velocity field of DFM is a significant theoretical contribution. 
3. The proposed method has a huge improvement in computation complexity.

### Weaknesses
1. The continuous flow experiments in Appendix E.3 suggest that the advantage of closed-form pairing may diminish with increasing data dimensionality. A brief discussion and further experiments on the scalability of the discrete PairFlow method to very large vocabularies and sequence lengths (Image datasets of higher resolution or language modeling) would be beneficial for setting expectations for future applications.
2. Typos in Line 127-128: the codomain of $p_t(\cdot)$ and $v_t(\cdot)$.

### Questions
PairFlow now focus on uniform source distribution. What's the difficulties of (or reasons of not) considering a more general class of source distributions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
PairFlow introduces a pre-processing step for training discrete flow models to enable few-step sampling without the requirement of a pre-trained teacher model. The approach achieves this through around a closed-form inversion step for discrete flow models, enabling the development of source-target pairs which can be used training PairFlow.

### Strengths
The approach is mathematically interesting, and offers significant practical utility, as it eliminates the need for a pre-trained teacher for distillation while accelerating inference through few-step sampling in discrete settings.

### Weaknesses
Deriving the closed-form velocity field although possible requires summing over all the training data, which can become significant for large systems. Further, it is possible that using the analytical vector field to determine source-target pairs leads to a distilled model that overfits to the data due to the way the analytical field is obtained. It would be useful to demonstrate that this in fact does not occur. There is some evidence to suggest that this may be happening (novelty of molecules on the molecular datasets is the worst compared to other baselines).

### Questions
- Using the closed-form velocity field to generate source-target pairs may lead to the distilled model being more likely to overfit to the training data. This has been observed in continuous settings by Bertrand et al. (2025). Proving that this isn't the case would be useful across the considered datasets (my concerns around the low novelty for QM9 validate these concerns). 
- Do additional Reflow steps (deriving a new analytical vector field on reflow samples from each distilled model) applied to subsequent trained models improve performance across metrics? In other words, does the concept of straightening the paths still apply in the discrete setting with multiple iterations of distilled models? 
- How does the approach perform on more complex systems, like ImageNet? Since it's observed that the continuous approach fails for higher-dimensional systems, does this also hold for the discrete setting? 
- How does the approach perform compared to continuous flow matching methods? A contrast on the already-shown image baselines, e.g., MNIST, CIFAR10, etc., would be useful.

### Soundness
3

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
5

### Summary
This paper proposes a new training method to accelerate discrete flow matching sampling, achieving good results on various tasks, especially CIFAR-10 generation.

The method, called PairFlow, is conceptually similar to ReFlow (Rectified Flow), which is commonly used in the continuous domain. However, PairFlow differs from ReFlow in that it starts from the true data domain and moves backward toward the noise domain using a backward velocity field.

In general, this paper aims to generalize the results obtained in the continuous domain—such as the closed-form velocity formulation and the ReFlow training approach—back to the discrete domain.

### Strengths
- Starting from real data, back to noise. This is just a little bit change of the reflow. But I think it is a cool idea. Intuitively, if your reflow model is not trained or sampled well, then the quality of your sampled-images maybe much worse than that of real data. But starting from clean image, you can avoid this.
- this paper has some smart ideas to get the theoretical results, for example, how to get the closed-form formula of forward velocity (A.1). Although the proof about the closed-formula of backward velocity has some flaws, but the direction of that proof (A.2) I think it's correct.

### Weaknesses
The main weakness of this paper is dued to its proof of the closed-form formular of the backward velocity.

- line 770 ~ line 792, has so many typos. Those typos make the proof unreadable, though although I can grasp the approach the authors intended to use.
- I do some calculations, and find some part results are right, but the proof process is flawed. There are numerous algebraic mistakes (e.g., signs flipped, missing ±1 terms, and other careless errors) that makes people suspect the correctness of the results.

### Questions
Could you please check and rewrite the proof in Appendix A.2? I think the overall approach is sound, but there are too many careless errors. If you can make it accurate and typo-free, I will raise my score.

### Soundness
2

### Presentation
3

### Contribution
4
