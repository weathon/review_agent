# Estimating Semantic Alphabet Size for LLM Uncertainty Quantification

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 2, 6

## Abstract
Many black-box techniques for quantifying the uncertainty of large language models (LLMs) rely on repeated LLM sampling, which can be computationally expensive. Therefore, practical applicability demands reliable estimation from few samples. Semantic entropy (SE) is a popular sample-based uncertainty estimator with a discrete formulation attractive for the black-box setting. Recent extensions of SE exhibit improved LLM hallucination detection, but do so with less interpretable methods that admit additional hyperparameters. For this reason, we revisit the canonical discrete semantic entropy (DSE) estimator, finding that it underestimates the ``true'' semantic entropy, as expected from theory. We propose a modified semantic alphabet size estimator, and illustrate that using it to adjust DSE for sample coverage results in more accurate SE estimation in our setting of interest. Furthermore, we find that two semantic alphabet size estimators, including our proposed, flag incorrect LLM responses as well or better than many top-performing alternatives, with the added benefit of remaining highly interpretable.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
I really like this paper. It takes a fairly simple premise: sampling-based estimators of the clustering distribution will tend to miss the tail, leading to systemic underestimation of semantic entropy. But: We have robust estimators for the size of the tail. So, we can get modified estimators of either the number of semantic clusters, or the semantic entropy. And, these modified estimators lead to improved performance.

### Strengths
* It’s a simple idea... in a good way. It identifies a clear problem, finds parallels in the species sampling literature for heavy tailed distributions, and proposes a simple, interpretable solution.
* It leads to improvement. 
* The evaluation is very thoughtful (one of the best discussions of how to evaluate UQ methods I have seen)
* The paper is clear and well-written

### Weaknesses
* It’s not strictly improving on SoTA... but it is compatible with SoTA using a much simpler and more interpretable paradigm
* The idea of using a Good-Turing estimator has been explored in related problems in the literature, as acknowledged by the authors.

### Questions
* It’s interesting that you find that the adjusted number of sets outperforms the adjusted semantic entropy. It would be nice to see how the “ground truth” number of sets compares with “ground truth” semantic entropy... if in this setting ground truth number of sets outperforms ground truth semantic entropy, it might suggest something interesting about miscalibration of the LLM. 
* I’d like to see (maybe in the discussion) discussion of how one might extend this to white-box SE estimation (which will tend to suffer from the same problem, albeit to a lesser extent).

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Goal:
- This paper addresses a specific issue in discrete semantic entropy (DSE), a widely used black-box UQ method.  
- The authors show that DSE tends to underestimate uncertainty when the response sample size is small, because it often fails to capture all unseen semantic clusters, therefore underestimating the number of distinct meaning clusters in its outputs (**semantic alphabet size**).

Empirical Validation:
- To verify this bias, the authors compare DSE values from small-sample settings against 100-sample semantic entropy (treated as ground-truth uncertainty).
- Across all model–dataset pairs, DSE consistently produces lower uncertainty estimates as sample size decreases.

Method:
To better estimate the semantic alphabet size, the paper considers 2 existing approaches:
1. A Good–Turing coverage estimator, adapted from population ecology, which corrects for unseen categories in finite samples
2. A spectral method based on eigenvalue decomposition (Lin et al.), which estimates alphabet size from the number of non-zero eigenvalues of a semantic similarity matrix.
The authors then propose a hybrid estimator that combines these two strategies: 
- when Good–Turing coverage becomes unreliable (e.g., each cluster appears only once), the method defaults to the spectral estimate; 
- otherwise, it uses a weighted combination to mitigate underestimation bias.
Using this hybrid alphabet size, they further define a hybrid semantic entropy estimator, extending classic ecological entropy formulations to the semantic domain.

Evaluation:
1. Following standard UQ evaluation protocols, the study benchmarks ten uncertainty estimators across four LLM families and four datasets.
2. Because raw AUROC scores can vary with model and dataset, the authors adopt a Bradley–Terry latent strength model to aggregate pairwise AUROC comparisons into an overall ranking with confidence intervals—an extension of the win-rate framework proposed by Nikitin et al. (2024).
Overall, results show that the hybrid semantic alphabet size estimator performs most consistently across settings, while the hybrid semantic entropy estimator reduces but does not fully eliminate the negative bias in DSE.

### Strengths
1. The problem itself is important.
2. The solution is simple.
3. The evaluation considers the weakness of AUROC and proposes the Bradley-Terry latent strength scores, which make the evaluation more reliable. 
4. The paper is easy to understand and well-structured. I like its simplicity and enjoy reading it.

### Weaknesses
1. Limited methodological novelty and analysis.
- The proposed hybrid estimator mainly combines two existing techniques 
   -  the Good–Turing coverage estimator from population ecology
   - a spectral eigenvalue-based method
I am not saying that combining prior ideas is not good, and my point is that it can be done with a deeper analysis of each individual methods and the hybrid operation can be done more adaptively. 
- For example, the Good–Turing approach is well established for ecological sampling, yet the sampling distribution of LLM-generated responses can be very different. It remains unclear why this method should perform reliably in the LLM setting or under what conditions it might fail. 
- Similarly, the hybrid design could be more principled — for instance, by characterizing when each component estimator is more accurate, rather than simply prioritizing the larger estimate under certain cases. This max operation inherently assumes that both the spectral-based method and Good-Turing coverage-based methods are also underestimating the ground truth semantic cluster size. But is this assumption true?

2. Inconsistent motivation and results.
The empirical validation (Figure 2 and table 1) assumes that semantic entropy computed with 100 samples represents the ground truth. However, this assumption itself is not rigorously justified. Moreover, the paper’s motivation is to improve DSE-based uncertainty estimation, yet the final results show that DSE-derived methods (including the hybrid entropy estimator) still lag behind KLE and the semantic alphabet size estimator in overall performance (always ranking from 7-10). This weakens the narrative that improving DSE necessarily leads to better uncertainty quantification.

### Questions
1. Equation (8) appears to be missing the definition or notation for $p_i$. 
2. I have multiple questions regarding the experiment results (Figure 3): 
- Why was white-box semantic entropy not included in the comparison, since you also consider the PE? 
- In addition, both hybrid semantic entropy ($\hat{H}_\text{hybrid}$) and discrete semantic entropy (DSE) show notably lower rankings (around 7–10). Could you elaborate on why these methods perform substantially worse than others? Conceptually, both $\hat{H}_\text{hybrid}$ and DSE should still be affected by the same underestimation bias that motivates your paper. If the semantic alphabet size were estimated more accurately, shouldn’t this also mitigate the underestimation problem in DSE to some extent?

### Soundness
3

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
3

### Summary
This paper addresses the problem of underestimating semantic entropy (SE) as an uncertainty estimate for black-box LLMs. The authors propose an interpretable semantic alphabet size estimator which, when used to adjust the discrete SE estimator for sample coverage, yields more accurate uncertainty estimation.

### Strengths
- The proposed semantic alphabet size estimator effectively mitigates the underestimation bias of discrete semantic entropy (DSE) in the few-sample regime.
- The method is simple and interpretable, avoiding the complexity and hyperparameter dependence of recent SE extensions.
- Empirically, it improves UQ performance and matches or exceeds the performance of SOTA approaches in the back-box setting.

### Weaknesses
- My main concern is that the proposed method is largely an adaptation of existing estimators to the SE setting. While the idea is insightful, it constitutes an incremental improvement over Farquhar et al. (2024) rather than a fundamentally new theoretical contribution.
- The paper treats semantic clusters as fixed and does not analyze sensitivity to the clustering procedure (e.g., entailment thresholds or NLI model biases). Since clustering quality directly affects entropy estimates, this omission is a potential confounder that limits the scope of the contribution.
- Experiments are restricted to relatively small models (<10B parameters), which raises questions about generalization to larger, frontier LLMs. Moreover, the paper does not compare against computationally cheaper uncertainty measures that avoid expensive sampling (e.g., G-NLL [1]) and which outperform SE and DSE with a single sample.
Minor point: The paper inconsistently uses the abbreviations (like SE and DSE), alternating between the abbreviated and full forms without clear rationale, which makes it harder to follow.

---

[1] Lukas Aichberger, Kajetan Schweighofer, and Sepp Hochreiter. Rethinking uncertainty estimation in natural language generation. arXiv preprint arXiv:2412.15176, 2024.

[2] Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 2024.

### Questions
- How quickly does the “white-box” SE estimate converge to the "true" SE as the number of samples increases? Including this in Figure 2 would be interesting.
- Are 100 samples sufficient to approximate the true distribution over semantic clusters? An ablation with larger sample sizes (e.g., 1 k) could strengthen the claim that this is a valid reference.
- How sensitive are the results to the semantic clustering method (e.g., choice of NLI model, thresholding, or entailment aggregation)?
- The hybrid estimator uses a max(·) rule combining two estimators. Why is this choice theoretically justified versus, say, a weighted or Bayesian combination?

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
Authors propose a better (less biased) estimator for semantic-alphabet-size in the semantic-entropy-related methods for UQ estimation in NLG tasks. 
They show empirically that their semantic-alphabet-size is competitive with some of the SOTA methods.

### Strengths
S1. The contribution is principled and to my best judgement technically correct.

S2. The empirical performance of the proposed method is strong.

S3. I find the empirical evaluation methodology sound.

### Weaknesses
W1. The evaluation could use more datasets (despite being sound methodologically).   
There are codebases available which would make the process rather straightforward, given the method proposed is rather simple to implement once the Semantic-Clustering is computed: https://github.com/AlexanderVNikitin/kernel-language-entropy. + maybe SimpleQA?(https://openai.com/index/introducing-simpleqa/)

W2. Not sure how "significant" the contribution will prove to be, but I hate to judge this criterium, so please treat this complaint as secondary. If this was judged by TMLR criteria, my overall score would be an "accept".

### Questions
Q1. Would it possible to create a Figure-2-like plot but for semantic-alphabet-size estimators?

Q2. Nitpick: could you please change the font in Fig 1 to a Serif font?

### Soundness
3

### Presentation
3

### Contribution
2
