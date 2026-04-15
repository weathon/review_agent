# Memorization and the Orders of Loss: A Learning Dynamics Perspective

- Decision: Reject
- Scores: 3, 3, 3, 8

## Abstract
Deep learning has become the de facto approach in nearly all learning tasks.
It has been observed that deep models tend to memorize and sometimes overfit data, which can lead to compromises in performance, privacy, and other critical metrics.
In this paper, we explore the theoretical foundations that connect memorization to various orders of sample loss, i.e., sample loss, sample loss gradient, and sample loss curvature, focusing on learning dynamics to understand what and how these models memorize. 
To this end, we introduce two proxies for memorization: Cumulative Sample Loss (CSL) and Cumulative Sample Gradient (CSG).
CSL represents the accumulated loss of a sample throughout training, while CSG is the gradient with respect to the input, aggregated over the training process. CSL and CSG exhibit remarkable similarity to stability-based memorization, as evidenced by considerably high cosine similarity scores. We delve into the theory behind these results, demonstrating that CSL and CSG represent the bounds for stability-based memorization and learning time. Additionally, we extend this framework to  include sample loss curvature and connect the three orders, namely, 
sample loss, sample loss gradient, and sample loss curvature, to learning time and memorization. 
The proposed proxy, CSL, is four orders of magnitude less computationally expensive than the stability-based method and can be obtained with zero additional overhead during training. We demonstrate the practical utility of the proposed proxies in identifying mislabeled samples and detecting duplicates where our metric achieves state-of-the-art performance. Thus, this paper provides a new tool for analyzing data as it scales in size, making it an important resource in practical applications.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This paper presents a novel approach to analyzing memorization in deep learning models by introducing Cumulative Sample Loss (CSL) and Cumulative Sample Gradient (CSG) as proxies. The authors theoretically and empirically demonstrate the relationship of these proxies to stability-based memorization and learning time, showing that CSL and CSG are computationally efficient alternatives. The framework is validated on a range of computer vision tasks, including the detection of mislabeled and duplicate samples, underscoring its potential for practical dataset curation.

### Strengths
**S1.** The submission tackles the important problem of memorization in deep learning, which has practical and theoretical relevance, particularly for privacy and interpretability.

**S2.** The proposed CSL and CSG metrics significantly reduce computational demands compared to previous stability-based memorization metrics.

**S3.** The experimental evaluation is diverse, demonstrating that CSL and CSG can effectively identify mislabeled and duplicate samples, which is useful for curating and refining datasets.

### Weaknesses
**W1.** The theoretical claims in this paper depend on several restrictive assumptions—including bounded loss, bounded expected stochastic gradient moments, and stability—that are challenging to ensure in practice and are particularly questionable in deep learning context. For example, a quadratic loss (or MSE) for a linear model does not satisfy these assumptions, as it is unbounded. Additionally, Theorem 4.6 relies on the $\mu$-PL condition, a slight relaxation of strong convexity, which, when combined with bounded gradient norm assumptions, results in an basically empty class of functions.

**W2.** The paper’s mathematical notation is dense and occasionally ambiguous, which may limit accessibility and readability. Key terminology (e.g., “loss estimation variance,” “expected learning time,” $\vec{w}^*$, and $\alpha$-Ball) is introduced without adequate definition, and the dependence of bounds on various constants makes the theorems challenging to follow. Additionally, some of these constants can be arbitrarily large, rendering certain theoretical results less insightful. 

The proposed sample learning condition (4) is very questionable to me as the stochastic gradient norm can be arbitrary large even at the optimum for convex models.

**W3.** Some claims regarding the linear relationships between the metrics are not adequately justified, as the theoretical results mostly provide upper bounds in worst-case scenarios rather than precise relationships. The statement that loss 
> Loss serves as **the most compute-efficient** proxy for measuring memorization

seems overly strong and requires refinement for accuracy.

#### Minor comments
- Consider providing a clearer explanation of the memorization “metric,” as it is challenging to understand on first reading.

### Questions
1. Why is the bound on the stochastic gradient norm (Equation 18) defined for the third power? This is unconventional.

2. Does the SGD optimization apply to a finite sum (empirical risk minimization) or a stochastic problem?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes two new ways to detect memorization (defined by the leave-one-out score definition proposed in Feldman 2020), which is cumulative loss and cumulative gradient norm of a data point. Theoretical justifications are provided and the empirical results look promising.

### Strengths
Trending topic. Memorization detection is important, but it is often inefficient to compute the memorization score defined by the counterfactual leave-one-out score. The paper proposes lightweight method that seems to strongly correlated with the memorization score. 

The paper is well-organized and very pleasant to read. Theorems are always followed by proof sketch and implications.

### Weaknesses
While the theoretical justifications are provided, I am not sure how meaningful are they. For instance, for Theorem 4.1, the bound includes term $T_{max} L/E[\ell(w_0)] >= T_max$, but isn't $T_{z_i} \le T_{max}$ as that's the max model training iterations? I also don't understand where the T_{ref} comes from. 

Some notations are weird. Does $\nabla_X$ means the gradient over input feature or parameter? 

While the theorems states a bunch of inequalities between different quantities, I am not sure whether those inequalities can lead to the conclusion that these quantities are linearly correlated with each other.

### Questions
See weaknesses.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper investigates memorization in deep neural networks through the lens of learning dynamics. The authors propose two computationally efficient metrics - Cumulative Sample Loss (CSL) and Cumulative Sample Gradient (CSG) - to measure memorization effects during training. They establish theoretical bounds connecting these metrics to learning time and stability-based memorization. Through experiments on CIFAR-100 and ImageNet, they demonstrate these metrics can effectively detect memorized examples and are computationally efficient. The practical applications include detecting mislabeled examples and duplicates in datasets, where their approach achieves state-of-the-art performance while being 4 orders of magnitude faster than previous methods.

### Strengths
Strengths:
1. Clear, well-defined problem and the approach is novel from what I. know.
2. The writing is clear and the problem is well motivated 
3. This is a significant problem to solve since there are multiple approaches to measuring memorization but none that works fast and consistently.

### Weaknesses
Weaknesses:
1. My main concern with the paper lies with the empirical evaluation strategy. The authors validate their proposed metrics (CSL/CSG) against Feldman's [1] subsampled influence scores rather than true leave-one-out training. This is problematic because recent work by Basu et al[2] and  Bae et al.[3] has shown that influence functions do not correlate well with actual leave-one-out training for large neural networks. This approximation gets worse for larger networks. Without validation against true leave-one-out training, we cannot verify if their metric actually captures memorization or merely correlates with Feldman's influence scores. Essentially, it would show correlation between two approximations rather than validation against ground truth.
2. The theoretical framework assumes that SGD is trained to convergence which isn't really done in most practical settings. On the theoretical side, on its own, this is fine to prove some theorems (with the caveat that they are idealized). But related to the above point, Bae et al. have shown that exactly this property (among others) causes influence functions to not correlate with leave one out retraining. So there is a gap here between the theoretical properties being proven and the practical aspects. In my opinion, such a gap is okay if there is a novel technical contribution on the theoretical side but most proofs seem to follow: Start with SGD convergence assumptions, Use telescoping sums to relate changes across iterations, Apply basic inequalities (Lipschitz, bounded gradients etc.), Arrive at bounds that relate different quantities. Maybe I'm missing something? 
3. On the theoretical side, there is an $\alpha$-adjacency assumption used in the proofs: 

  3.1. Requires existence of data points within $\alpha$-balls of each other

  3.2. Authors claim this holds as $\alpha$ is unrestricted, but this makes the assumption essentially meaningless - if $\alpha$ can be arbitrarily large, what does the bound tell us? Ideally, I would have like some empirical validation that real datasets satisfy this property for meaningful $\alpha$ values as well. 

4. The authors use arbitrary thresholds (0.25 for memorization, 0.15 for influence) across all datasets.  I did not see a justification for why these specific values were chosen or an analysis of how results change with different thresholds.
5. No error bars on key results, also. only a single architecture (resNet-18) is used in all experiments.
6. The $\mu$-PL condition is not defined anywhere (at least int he main paper). I looked through some of the proofs and I assume it is the Polyak-Lojasiewicz condition but it should be clarified.


[1] Feldman et al. What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation

[2] Basu et al. Influence Functions in Deep Learning Are Fragile

[3] Bae et al. If Influence Functions are the Answer, Then What is the Question?

### Questions
How can one be sure if we can detect memorization with a new, proposed score if we don't train from scratch without that example?

On the theory front, what is the novel technical contribution?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper explores the memorization phenomenon in deep learning by presenting a theoretical framework that links learning time, memorization, and three orders of loss: sample loss, sample gradient, and sample curvature. The authors introduce two proxies, Cumulative Sample Loss (CSL) and Cumulative Sample Gradient (CSG), which exhibit high cosine similarity with established memorization method yet are significantly less computationally expensive. Experiments on deep vision models validate their theory and demonstrate that these proxies achieve state-of-the-art performance in identifying mislabeled or duplicate samples.

### Strengths
Originality & Significance:
1. This paper presents a novel theoretical framework that establishes relationships among learning time, stability-based memorization, and memorization proxies: cumulative sample loss, cumulative sample gradient, and input loss curvature. Specifically, the authors demonstrate that (1) learning time exhibits a linear relationship with the three orders of loss, (2) stability-based memorization also follows this linear relationship with these metrics, and (3) learning time and memorization are linearly related.
2. The proposed proxies for memorization, Cumulative Sample Loss (CSL) and Cumulative Sample Gradient (CSG), are significantly more computationally efficient than existing methods. Specifically, the CSL proxy is 4 orders of magnitude less computationally expensive than stability-based memorization (Feldman \& Zhang, 2020) and approximately $14\times$ less costly than input loss curvature (Garg et al., 2024).

Quality:
The theoretical framework is well-supported by detailed assumptions and rigorous proofs. Experiments across multiple deep vision models and datasets validate both the theory and the effectiveness of the proposed proxies for memorization.

Clarity:
Overall, the paper is well-written and well-organized. The "Takeaways" and "Interpreting Theory" sections are particularly effective, as they help clarify the theoretical contributions and experimental results.

### Weaknesses
1. Although the theoretical framework is rigorous, it relies on several assumptions, particularly the bounded loss function and the $\lambda$-proximal sample condition, which may not always hold in practice.

2. The experiments are primarily focused on deep vision models, limiting the scope of validation. Testing on other model types or domains, such as natural language processing, would enhance the generalizability of the findings.

### Questions
1. The definition of learning time in the main theorems is unclear. It is implicitly mentioned in the sample learning condition (Equation 4), which suggests that learning time depends on a threshold $\tau$ (although in proofs, $\tau$ is chosen as an upper bound in terms of $T$). A more explicit definition or explanation would enhance understanding.
2. In the definition of $\lambda$-proximal, does $T_{ref}$ refer to the learning time of the reference sample $\vec{z}_{\text{ref}}$?

### Soundness
3

### Presentation
3

### Contribution
3
