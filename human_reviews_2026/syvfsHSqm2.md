# Dual Randomized Smoothing: Beyond Global Noise Variance

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Randomized Smoothing (RS) is a prominent technique for certifying the robustness of neural networks against adversarial perturbations. With RS, achieving high accuracy at small radii requires a small noise variance, while achieving high accuracy at large radii requires a large noise variance. However, the global noise variance used in the standard RS formulation leads to a fundamental limitation: there exists no global noise variance that simultaneously achieves strong performance at both small and large radii. To break through the global variance limitation, we propose a dual RS framework which enables input-dependent noise variances. To achieve that, we first prove that RS remains valid with input-dependent noise variances, provided the variance is locally constant around each input. Building on this result, we introduce two components which form our dual RS framework: (i) a variance estimator first predicts an optimal noise variance for each input, (ii) this estimated variance is then used by a standard RS classifier. The variance estimator is independently smoothed via RS to ensure local constancy, enabling flexible design. We also introduce efficient training strategies to iteratively optimize the two components involved in the framework. Extensive experiments on the CIFAR-10 dataset demonstrate that our dual RS method provides strong performance for both small and large radii—unattainable with global noise variance—while incurring only a 60\% computational overhead at inference. Moreover, it consistently outperforms prior input-dependent noise approaches across most radii, with particularly large gains at radii 0.5, 0.75, and 1.0, achieving relative improvements of 15.6\%, 20.0\%, and 15.7\%, respectively. On ImageNet, dual RS remains effective across all radii, with 8.6\%, 17.1\% and 9.1\% performance advantages at radii 0.5, 1.0 and 1.5 respectively. Additionally, the proposed dual RS framework naturally provides a routing perspective for certified robustness, improving the accuracy-robustness trade-off with off-the-shelf expert RS models.  Our code is available at https://github.com/eth-sri/Dual-Randomized-Smoothing.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel Randomized Smoothing (RS) method called Dual Randomized Smoothing. It dynamically estimates an optimal noise variance for each input and then apply it to the RS classifier, which shows stronger performance than prior input-dependent RS.

### Strengths
1. This paper introduces a dynamically method to find the optimal $\sigma(x)$ for each input $x$ to maximize the verifiable robustness radius $R$, obtaining the stronger robustness. And it is supported by theory.
2. They show the strong performance compared with SOTA input-dependent RS method. And show the comprehensive ablations study such as alternating fine-tuning, expert classifiers, and the trade-off between accuracy and robustness.

### Weaknesses
1. For each type of data, specific processing for training data is required. Only a relatively small dataset (CIFAR-10) is shown, and there is no research on generalization (for ImageNet). And the generation of training data takes a huge amount of time, reaching an extremely long duration of 703 hours just for CIFAR-10 alone.
2. The presentation of empirical accuracy lacks clarity. And all the results are almost presented in the form of figures. The important results are clearer in the form of tables. (Refer to the Multiscale [1] that they compared with.)

[1] Multi-scale diffusion denoised smoothing, NeurIPS 2023.

### Questions
1. Could you show the trade-off between the amount of training data? Such as 20000, 10000 rather than 50000 in whole CIFAR-10 (Just randomly separate it from the dataset you have already processed). Thereby reducing the processing time for the training data.
2. Could you adjust the presentation method of the results based on weakness 2?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper extends randomized smoothing (RS) to allow input-dependent noise variances, addressing the accuracy–robustness trade-off inherent in global noise settings. The authors theoretically prove that RS remains valid under locally constant variance, and propose a “dual RS” framework with a variance estimator and a standard RS classifier, showing empirical improvements on CIFAR-10. While the idea is clearly presented and theoretically sound, the paper’s novelty and empirical depth are limited, reducing its potential impact.

### Strengths
1. The paper identifies a well-known limitation of traditional RS and provides a theoretically justified extension to input-dependent noise while maintaining certification validity.

2. The decomposition into a variance estimator and a classifier, with the option to interpret it as a routing system among expert models, is conceptually clean and practical.

### Weaknesses
1. The core idea, using input-dependent noise variance in randomized smoothing, has already been discussed in several prior works. The theoretical extension to “locally constant variance” is incremental rather than fundamentally new.

2. The paper repeatedly claims distinctions from prior approaches in different sections (sections 4 and 5), but these differences are scattered and qualitative. I would like to suggest that the authors add a clear summary table comparing key assumptions, theoretical guarantees, and computational overhead across existing input-dependent RS methods (e.g., Data-Dependent RS, Multiscale RS, Adaptive RS, etc.) to make the claimed advances more transparent.

3. The experiments are limited to CIFAR-10 only, without ablation on larger or diverse datasets (e.g., ImageNet, CIFAR-100). Moreover, many results are shown in curves without concrete numerical tables, making it hard to assess statistical significance.

### Questions
I have no further questions. Please refer to the weaknesses.

### Soundness
3

### Presentation
2

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
This paper proposes Dual Randomized Smoothing (Dual RS), a framework that generalizes randomized smoothing (RS) to input-dependent noise variances while maintaining valid robustness guarantees. The key theoretical result (Theorem 4.1–4.2) shows that RS remains certifiably correct when the variance function $\sigma(x)$ is locally constant within the certified region. Building on this, the authors introduce a two-stage “dual” framework: a smoothed variance estimator predicts $\sigma(x)$, and a standard RS classifier uses this estimate for certification. The resulting method achieves stronger accuracy–robustness trade-offs on CIFAR-10 compared to both standard and multiscale RS baselines.

### Strengths
The theoretical contribution is clear, rigorous, and well-motivated. Previous input-dependent RS methods (e.g., Súkeník et al., 2022; Alfarra et al., 2022) were conceptually appealing but failed to provide valid certification due to the dependence of $\sigma(x)$ on the evaluation point. Here, the authors convincingly fix this flaw by proving that local constancy of $\sigma(x)$ is sufficient for correctness. The proof is clean, self-contained, and does not rely on unreviewed external results. I found the argument based on Lipschitz continuity much more convincing than earlier Neyman–Pearson-based approaches. The routing interpretation in Section 5.3 is also insightful and connects RS to the mixture-of-experts design.

### Weaknesses
The framework introduces double certification: one for the classifier and one for the variance estimator. While theoretically sound, this adds non-trivial complexity and sampling cost. More importantly, ensuring or certifying local constancy of $\sigma(x)$ can be difficult as the input space scales up. In high-dimensional domains such as ImageNet, verifying that $\sigma(x)$ is approximately constant in a local neighborhood is challenging, and the accuracy of the second-stage certification will heavily depend on how stable the variance estimator itself is. I would like to see a discussion (or experiment) analyzing the behavior of the variance estimator in such large-scale settings.

### Questions
See weakness

Can you please experiment on ImageNet? If results scale to this dataset, I will upgrade my score.

### Soundness
4

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
4

### Summary
This paper introduces Dual Randomized Smoothing (Dual RS), a novel framework that addresses the fundamental limitation of standard Randomized Smoothing (RS), which uses a global noise variance across all inputs. The authors prove that RS certification remains valid with input-dependent noise variances, provided the noise variances are locally constant within the certified region. Based on this theoretical foundation, they propose a two-stage framework: (1) a variance estimator that predicts optimal noise variance for each input, and (2) a classifier that uses the predicted variance for certification. The authors also introduce a new procedure for jointly training the prediction of $\sigma$ and the base classifier. Finally, the authors also introduce a MoE setting for RS. The method is evaluated on CIFAR-10 and shows significant improvements over existing approaches.

### Strengths
The paper provides a rigorous theoretical foundation by proving that RS certification remains valid with locally constant noise variances (Theorems 4.1 and 4.2). This generalizes the original RS framework and opens new possibilities for adaptive certification methods.

The paper also provides a new training methodology using soft labels based on certified radius quality rather than hard labels for variance estimation. The proposed iterative training scheme, which alternates between learning the variance estimator given a classifier and classifier optimization given a variance estimator, is also interesting.  

The Mixture of Experts generalization of the certification procedure also provides a novel method for improving certified accuracy.

The experiments in the paper demonstrate consistent improvements over state-of-the-art methods with only 60% computational overhead compared to standard RS. The paper provides a detailed analysis of design choices, including the effects of hyperparameters for consistency loss, training iterations, and variants of the loss function.

### Weaknesses
The evaluation is restricted to CIFAR-10. The paper would benefit from experiments on larger datasets (e.g., ImageNet) and across other domains to demonstrate generalizability.

The training process requires substantial computational resources (1517 GPU hours total, with 703 hours just for building the optimal variance dataset). This high cost may limit practical adoption. As the framework relies on a discrete set of candidate variances $\Sigma = \{0.25, 0.5, 1.0\}$, there should be ablation studies examining how the choice, number, and granularity of $\Sigma$ affect performance.

### Questions
Please refer to the Weaknesses section above.

### Soundness
4

### Presentation
4

### Contribution
4
