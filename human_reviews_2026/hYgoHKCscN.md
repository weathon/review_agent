# Hybrid Query Strategy with Diversity-Weighted Metropolis–Adjusted Langevin Algorithm

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Although deep learning has achieved remarkable success in various fields, most of these advances typically rely on a large-scale well-annotated dataset. However, in real-world applications, collecting labeled data is often expensive and timeconsuming. Active Learning (AL) has emerged as a promising solution to mitigate labeling costs by selectively querying the most informative instances for annotation. In particular, hybrid AL methods have been gaining attention by integrating multiple acquisition criteria such as uncertainty, diversity, or representativeness as a joint function. In this paper, we propose a novel hybrid active learning method named Diversity-Weighted Metropolis-Adjusted Langevin Algorithm (DW-MALA). Our method precisely approximates the data distribution by leveraging gradient-based Langevin dynamics, and selects instances from high-density regions using a representativeness score derived from density estimates. Simultaneously, a diversity score is incorporated by measuring the distance to the nearest labeled instance, which also ensures coverage of low-density regions. The quantitative and qualitative analyses demonstrate the effectiveness of DW-MALA in selecting diverse and representative samples under a limited labeling budget, compared to the baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new AL approach that selects samples based on both diversity and data-distribution representativeness. Instead of relying only on uncertainty or distance-based heuristics, the approach uses Metropolis-Adjusted Langevin Algorithm (MALA) to approximate the underlying data density in the feature space, and identify points that better reflect the overall structure of the unlabeled data pool. Experiments on classification, imbalanced settings, object detection, and domain-shift segmentation tasks/data situations show competitive performance, especially in early AL rounds.

### Strengths
This paper presents a technically interesting AL method by incorporating Langevin-based density estimation into the AL sample selection process, this method aims to capture data representativeness. The idea of leveraging MALA in this context is interesting, it is a creative combination of sampling-based density modeling with hybrid acquisition functions. 
The experimental evaluation is comprehensive across multiple datasets and task types, and the results support the method’s strengths, particularly in early AL rounds. The presentation is generally clear and easy to follow. In terms of significance, the method tries to address the limitation of existing diversity-based AL approaches and demonstrates practical gains.

### Weaknesses
I have some concerns:
1. The method is evaluated only under fixed random initialization; it does not study the effect of smaller or imbalanced seed sets, nor self-supervised warm-starts. However, the proposed method relies heavily on the quality of early learned features to perform MALA sampling. If initial embeddings are weak or even biased distribution, the estimates would drift and amplify bias across AL rounds.
2. The method considers the representativeness and diversity but doesn't include uncertainty, but uncertainty is crucial for boundary exploration in many AL situations. Although DWUS was tested, it even underperforms, and the paper does not analyze why or explore alternative integration ways. 
3. This paper provided sensitivity studies show the method is not too sensitive to $\alpha$ or $M$. However, other hyperparameters like Langevin and KDE hyperparameters (e.g., step size, noise scale, kernel bandwidth) are not examined.

### Questions
1. The improvements on semantic segmentation tasks are marginal. Could the author explain more about why the proposed method does not perform competitively on such dense prediction tasks? 

2. Figure 13 suggests the embedding space is weak early and improves gradually, yet DW-MALA yields the largest gains in early rounds. Could the authors elaborate on why density-driven sampling performs best specifically when feature quality is lowest?

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
3

### Summary
This paper introduces DWMALA, an active learning strategy that combines two principles known in the active learning community which are representativeness and diversity when selecting samples for annotation. The proposal achieves representativeness by using the Metropolis-Adjusted Langevin Algorithm to sample candidate samples then weights the candidate samples using Kernel Density Estimation. The diversity is accounted for by using the minimum Euclidean distance between the candidate sample and the already labeled samples. The experiments on balanced and imbalanced datasets along with challenging tasks like object detection and semantic segmentation show improvements over other active learning strategies.

### Strengths
(S1) The integration of representativeness and diversity is clear, and the trade-off between the two components is systematically analyzed.
(S2) The method is evaluated across diverse settings and tasks, including classification, imbalanced data, object detection, and semantic segmentation, demonstrating broad applicability.
(S3) The paper includes thorough ablation studies and sensitivity analyses, providing good insight into the effect of design choices, particularly the α hyperparameter.

### Weaknesses
(W1) The use of a physical model used to model a dataset distribution is not motivated well.  
(W2) The choice of the energy potential U(x) is not justified well. Why $$U(x) =\frac{1}{2} k x^2$$ ?
(W3) It is not mentioned how many runs are used for each experiment. 
(W4) The improvements provided by the method are in many cases marginal.  
(W5) The writing of the Ablation study section is a bit confusing. What is being held and what is used?

### Questions
(Q1) Please state how you will describe the motivation behind using a physical model to model the distribution of the datasets in the method section.
(Q2) Please explain how you will show statistically significance results in a revision of the manuscript. (Maybe it works well with imbalanced datasets but not good enough with balanced dataset)
(Q3) Please state how you would rewrite the ablation study section.

(Please specify how you will improve the paper as we reviewers are not interested in just private education but in improvements of the manuscript).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an active learning method based on Metropolis adjusted Langevin dynamics. Based on the estimated data distribution and its gradient flow, it draws reasonable samples from which it measures representativeness of candidate samples of active learning. In addition, the authors also adds diversity weighting based on distance in feature space of a classifier. They compare the proposed method on several active learning methods on some popular benchmark datasets for image classification as well as object detection and semantic segmentation tasks.

### Strengths
1. The paper proposes a novel active learning method based on actual density estimation of data distribution.
2. It is fairly easy to read the paper although some details are missing.
3. The proposed method was demonstrated on several tasks for image recognition such as classification, object detection and semantic segmentation.

### Weaknesses
1. The derivation of Langevin dynamics in Eq.(11) is based on some simplifications such as $U(x) = \frac{1}{2}kx^2$, $\beta=1$ and $\gamma=1$.
2. It seems like the target distribution $\pi$ is estimated as KDE with a Gaussian kernel. I do not think this is assumption is practically useful especially in high-dimensional space. 
3. The paper says “most of the previous methods have focused on designing heuristic metrics” but I am not clear if the proposed DW-MALA is based on heuristics especially in terms of weighting using $\alpha$. Is there any theoretical justification of why the weighting has to be that way?
4. It is missing many highly relevant literature.
* Representation-based
    * Guy Hacohen, Avihu Dekel, and Daphna Weinshall. “Active learning on a budget: opposite strategies suit high and low budgets”, ICML 2022.
    * Wonho Bae, Junhyug Noh, and Danica J Sutherland. “Generalized coverage for more robust low-budget active learning”, ECCV 2024.
* Hybrid methods
    * Amin Parvaneh, Ehsan Abbasnejad, Damien Teney, Gholamreza Reza Haffari, Anton Van Den Hengel, and Javen Qinfeng Shi. “Active learning by feature mixing”, CVPR 2022.
    * Yichen Xie, Han Lu, Junchi Yan, Xiaokang Yang, Masayoshi Tomizuka, and Wei Zhan. “Active finetuning: exploiting annotation budget in the pretraining-finetuning paradigm”, CVPR 2023.
    * Guy Hacohen and Daphna Weinshall. “How to select which active learning strategy is best suited for your specific problem and budget”, NeurIPS 2024.
    * Wonho Bae, Gabriel L. Oliveira, Danica J. Sutherland, “Uncertainty herding: one active learning method for all label budgets”, ICLR 2025.
5. The experiments were conducted only on small size datasets: CIFAR10, 100 and Tiny Imagenet. It is unsure how the proposed method scales up to larger datasets like ImageNet.
6. Sensitivity analysis on $\alpha$ in Section 4.2.3 shows the differences across settings are marginal. It sounds like it does not really matter to just use either representativeness or diversity measure. Given that diversity measure is nothing new, it is not clear if the proposed method adds any significant improvement over the diversity baseline.

### Questions
1. According to the code shared by the authors, this is how Gaussian mixture samples are drawn. But, I do not understand how this “GMM” has any meaning without fitting GMM parameters. Aren’t they simply prior of GMM not posterior, which means there is no information from data? 

```
def sample_from_gmm(num_samples, latent_dim='', num_modes=''):
    means = np.random.randn(num_modes, latent_dim) * 2.0
    cov = np.array([np.eye(latent_dim) for _ in range(num_modes)])
    samples = []
    for _ in range(num_samples):
        k = np.random.randint(0, num_modes)
        sample = np.random.multivariate_normal(means[k], cov[k])
        samples.append(sample)
    return np.array(samples)
```

2. How was wall-clock time measured for time complexity in Section 4.1.4? I do not particularly understand the fact that Random took 12.17sec. Even for Entropy 45.64 seems pretty high when the classifier is VGG.

### Soundness
2

### Presentation
2

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
The manuscript proposes a Langevin-based active sampling method that balances diversity and representativeness when querying data. Compared to deterministic approaches (e.g., DWDS), it introduces a stochastic mechanism intended to better handle multi-modal data distributions. The method is well-motivated, and experiments on image classification and detection show improvements over several baselines.

### Strengths
- Clear presentation of background and method.

- Thorough comparisons with many baselines; ablations on different representativeness measures and α-sensitivity indicate robustness.

- UMAP visualizations help illustrate the sampling behavior. (It would be helpful to add visualizations for more baselines and couple them with brief case studies, for instance, showcasing samples with high representativeness.)

- Runtime comparisons suggest DW-MALA is competitive with other non-trivial AL methods.

### Weaknesses
- The manuscript does not specify a tractable target density π, so it is unclear how ∇log π is computed/approximated in Eq. (11). This creates a derivation–implementation disconnect.

- Active learning performance can vary significantly across domains. The current evaluation is limited; extending to other tasks (e.g., NLP) would strengthen claims about robustness/generalizability.

- Presentation polish: Table 2 uses symbols (X / O / Δ) that are not explained; the corresponding discussion in the main text is also sparse.

- Minor: Since DWDS is a key baseline, including it directly in Fig. 2 would make trends easier to compare. Distinct line styles/markers (and a color-blind-friendly palette) could keep the figure readable even with more curves.

### Questions
- (W1) Without a tractable π, how is ∇log π obtained for Eq. (11)?

- Could the approach benefit from integrating deep neural architectures (e.g., Transformers) or diffusion models (given the sampling connection)? If so, what are the expected benefits and main challenges?

### Soundness
3

### Presentation
3

### Contribution
3
