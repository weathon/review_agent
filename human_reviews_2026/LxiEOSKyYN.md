# Conformal Reliability: A New Evaluation Metric for Conditional Generation

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Conditional generative models have recently achieved remarkable success in various applications. However, a suitable metric for evaluating the reliability of these models, which takes into account their inherent uncertainty, is still lacking. Existing metrics, which typically assess a single output, may fail to capture the variability or potential risks in generation. In this paper, we propose a novel evaluation metric called \emph{reliability score} based on conformal prediction, which measures the worst-case performance within the prediction set at a pre-specified confidence level. However, computing this score is challenging due to the high-dimensional nature of the output space and the nonconvexity of both the metric function and the prediction set.
To efficiently compute this score, we introduce Conformal ReLiability (CReL), a framework that can \textbf{(i)} construct the prediction set with desired coverage; and \textbf{(ii)} accurately optimize the reliability score within the constructed prediction set. We provide theoretical results on coverage and demonstrate empirically that our method produces more informative prediction sets than existing approaches. Experiments on synthetic data and on the image-to-text and text-to-image tasks further demonstrate the interpretability of our new metric, and the validity and effectiveness of our computational framework.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose two methods: (i) A method that constructs conformal prediction sets in the latent space of a generative model and (ii) an evaluation metric that takes into account the uncertainty of a conditional generative model. The motivation for (i) is that generating conformal prediction sets in high dimensions leads to overly conservative prediction sets. The motivation for (ii) is that common metrics based on models such as CLIP do not capture the output variability of generative models (though, I believe that such naive methods are still unbiased if we sample from the model (i.e., we do not take the mode of the generative model)). The methods are demonstrated on synthetic data and an image-to-text task.

### Strengths
1. After reading the entire manuscript, the proposed methods are clear to me.

2. The experimental evaluation is strong.

3. The motivation is clear and justified.

### Weaknesses
1. My main concern regards soundness: It is not clear to me how it is justified to use conformal prediction to compute this evaluation metric (CReL). Conformal prediction is about risk control, i.e., providing probabilistic guarantees with respect to prediction sets. However, it is not clear to me why such guarantees are useful in any way if we end up using the prediction set only to generate some score. The only justification I could dream up is if the score inherits guarantees from conformal prediction, meaning that we can make statements such as: *"The probability CReL assesses model A to be better than model B, even though B is better than A, is smaller than $\alpha$"*. However, the authors do not show that a guarantee of this kind holds, so the application of conformal prediction is not sound, in the current state of the manuscript.

2. There exist quite a few works by now that construct conformal prediction sets in the latent space of a generative model [1, 2, 3] and it is not clear to me what the benefit of the proposed method is over the existing ones (only [2] is discussed). It seems to me that the convexity property (Proposition 3.5) also holds for [1] and perhaps also for [3].

3. It seems the paper is actually about two topics (see (i) and (ii) in my summary), and the connection between these is somewhat loose. Especially in the abstract and the introduction, it is not clear to me what to expect from the manuscript. It would be great if the authors could highlight these two topics ((i) and (ii)) and their connection more clearly in the abstract and introduction.

I believe that soundness (point 1) is the most critical criterion for acceptance and in the current state, the manuscript is not sound. I am willing to increase my score if the authors can convince me that using conformal prediction in this setting is justified.

[1] Fang, Zhenhan, Aixin Tan, and Jian Huang. "CONTRA: Conformal prediction region via normalizing flow transformation." The Thirteenth International Conference on Learning Representations. 2025.

[2] Feldman, Shai, Stephen Bates, and Yaniv Romano. "Calibrated multiple-output quantile regression with representation learning." Journal of Machine Learning Research 24.24 (2023): 1-48.

[3] Sankaranarayanan, Swami, et al. "Semantic uncertainty intervals for disentangled latent spaces." Advances in Neural Information Processing Systems. 2022.

### Questions
* The abstract is unclear to me: It starts of with a evaluation metric and then it seems that the proposed method is to efficiently compute the evaluation metric: *"To efficiently compute this score, we introduce Conformal ReLiability (CReL)"*. However, then it goes on to say *" framework that can (i) construct the prediction set with desired coverage; and (ii) accurately optimize the reliability score"*, indicating that this is a conformal prediction method that optimizes, and not computes, the evaluation score. I would suggest re-writing the abstract.

* I have the same difficulty in understanding the main contributions at the end of the introduction as I have understanding the abstract. I believe it would make sense to clarify how (i) the computation of a score; and (ii) the computation of prediction sets go together. To me (and perhaps other readers), it is not immediately clear. Would it be possible to go more into this?

* *"Additionally, we empirically find that the prediction set given
by our procedure has much smaller or comparable size to other methods"*. This sentence seems a bit contradictory to me.

* I believe that equation (1) has a typo: It should read $\hat{Y}_{n+1}$ everywhere and not $\hat{Y}$.

* In *"Step 2: Fitting the DQR model."*, it is not clear what is fitted. From the context, I would guess that it is the quantile model $f_\beta$. It would be helpful if it could be stated explicitly in the text.

* I believe that there is a typo in the assumption of Theorem 3.3: It reads $\mathcal{Dec}(\mathcal{E}(\hat{Y}, x), x) =_d \mathbb{P}(\hat{Y} | X)$, but I believe it should be $\mathcal{Dec}(\mathcal{E}(\hat{Y}, x), x) =_d \hat{Y} | X$.

* In the proof of Theorem 3.3, the final remark is *"We complete the proof."*. I believe it should rather say something like *"This completes the proof."*.

### Soundness
1

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
3

### Summary
Feldman, Bates, and Romano proposed an algorithm that calculates a "worst-case reliability score" for conditional samples. The algorithm is meant to estimate the distance of a sample from the "ground truth" among the set of most likely samples that cover a (1 - alpha) fraction of the conditional probability mass region.

The main contribution of this paper is an efficiency-motivated modification of that algorithm: The region in question is defined so that it can be efficiently "calibrated" via convex optimization (unlike in the previous work where it required brute-force search).

### Strengths
The paper does a good job of explaining a fairly complicated algorithm. The contributions are clear. The experiments indicate that the algorithm faithfully calculates regions with the correct conditional probability mass. It gives some illustrative examples in which the scores calculated by their approach are more conservative than scores based on single generation like CLIP.

The efficiency improvement over the work of Feldman et al. is well-supported.

### Weaknesses
I am not sure how well the claim that "CReL effectively identifies misalignments" (Section 4.1) is supported. The difference in scores on the examples shown is quite small. 

The theoretical guarantees (Section 3.2) have to do with the indistinguishability of training data and test data. They have little to do with the algorithm. They explain why the coverage sets have the desired probabity, but then so would sets of much lower complexity (e.g. a halfspace). They do not explain what is the advantage in modelling the calibrated regions as intersections of halfspaces.

### Questions
No questions

### Soundness
2

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
This paper introduces Conformal ReLiability (CReL), a framework for evaluating conditional generative models through a reliability score that measures the worst-case metric value within a calibrated prediction set at confidence level. The method first maps outputs into a learned latent space, constructs a directional-quantile region (DQR), applies conformal calibration to enforce marginal coverage, and then computes the worst-case metric over this convex calibrated set using projected gradient descent with linear-program projections. Empirically, CReL is validated on synthetic data and an image-to-text benchmark (BLIP / GIT on MS-COCO), showing that the proposed metric can reorder model rankings compared with single-sample CLIP or BERT scores and better capture semantic or visual errors.​

### Strengths
- Addresses Evaluation Gap: This paper introduces a new "reliability score" to assess worst-case performance within a prediction set. It moves past unreliable single-sample metrics.

- Sound Theory: The paper is based on a solid theoretical foundation that includes formal coverage guarantees (Theorem 3.3). Its formulation makes the optimization problem easy to solve (Proposition 3.5).

- Flexible Application: The score provides practical flexibility and works with many existing similarity metrics, such as CLIP-SIM, BERT-SIM, SSIM, and FID.

- Strong Empirical Signal: It shows a more informative evaluation signal by re-ranking models based on reliability (Table 2). It also identifies semantic errors overlooked by standard metrics (Figure 4) and creates more compact, adaptive prediction sets (Table 1, Figure 2).

- Computationally Efficient: The method demonstrates significant computational efficiency. The calibration runtime scales nearly linearly, greatly surpassing the exponential scaling of earlier grid-based methods (Figure 7, Appendix E).

### Weaknesses
- Motivation-Evaluation Mismatch: The paper highlights the importance of reliability in high-stakes applications like drug discovery and autonomous systems. However, it only validates its claims with a low-stakes image captioning task.

- Limited evaluation scope: Only one real task (image→text) is reported; no cross-domain or scaling results.​

- Clarity / notation: Frequent symbol changes between $X$, $Y$, $\hat{Y}$, and $GT$ with inconsistent definitions, making the paper challenging to follow. A single running example or unified notation scheme would improve readability.

- Fragile Theoretical Guarantee: The main coverage guarantee (Theorem 3.3) depends on the unrealistic assumption of a perfect latent generative model (LGM). The ablation study (Table 3) reveals that the metric is very sensitive to the LGM's hyperparameters. A slight change to the VAE's KL weight leads to nearly a five-fold increase in the prediction set area, raising concerns about the metric's stability.

- Missing related work and discussion. See for example [1-3] below, which are related to conditional generation evaluation. 

[1]. Rectifying Conformity Scores for Better Conditional Coverage. Plassier et al., 
[2]. Probabilistic Conformal Prediction Using Conditional Random Samples. Wang et al.,
[3]. Evaluation metrics for conditional image generation. Benny et al.,

### Questions
- Can you show calibration or coverage plots across different confidence levels ($\alpha$) to
confirm that the empirical coverage matches the target?​
- Could you briefly discuss the computational cost (time and memory) of computing the CReL
score, compared to a standard single-sample metric like CLIP or BERT similarity?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a reliability score to evaluate conditional generative models, capturing the worst-case performance within a prediction set at a specified confidence level. To handle high-dimensional outputs and nonconvex similarity metrics, the authors propose Conformal ReLiability (CReL), a latent-space framework combining directional quantile regression (DQR) and conformal calibration.

### Strengths
- The paper tackles an important question of evaluation for conditional generation.

- Latent-space formulation improves tractability and scalability for high-dimensional outputs.

- Empirical results show improved calibration and more meaningful evaluation compared to standard metrics (CLIP, BERT).

### Weaknesses
My primary concerns regarding this paper are the significance of its contribution and its reliance on training a latent generative model. While the paper provides theoretical guarantees, the theoretical contribution itself is limited, and the core idea is not novel. Furthermore, the need to train a latent generative model may introduce bias into the evaluation, as discussed in [1].

My additional comments are as follows:

- Related works: Several recent works on evaluating conditional generative models appear to be missing, such as the Conditional-Vendi score [2], HEIM [3], and the Scendi score [4]. Including a comparison with these methods would strengthen the manuscript.

- The experiments are limited to the image-to-text domain. The performance of the proposed metrics on other modalities, particularly for text-to-image models, remains unexamined.

- Choice of similarity metrics and hyperparameters could significantly affect outcomes; sensitivity analysis is limited.

---

[1] Stein et. al, "Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models", NeurIPS 2023

[2] Jalali et. al, "An Information-Theoretic Approach to Diversity Evaluation of Prompt-based Generative Models", ICCV 2025

[3] Lee et. al, "Holistic Evaluation of Text-to-Image Models", NeurIPS 2023

[4] Ospanov et. al, "Scendi Score: Prompt-Aware Diversity Evaluation via Schur Complement of CLIP Embeddings", ICCV 2025

### Questions
- Why is calibration needed after DQR, and how is it performed?
- How does CReL identify misalignments in the image-to-text task that standard metrics miss?
- Why does CReL scale better than grid-based approaches for high-dimensional calibration?

### Soundness
3

### Presentation
3

### Contribution
2
