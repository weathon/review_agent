# Beyond Levels and Continuity: A New Statistical Method for DNNs Robustness Evaluation

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3, 5

## Abstract
Evaluating the robustness of deep neural networks (DNNs) is crucial in safety-critical areas, driving research into methods that accurately measure and enhance their resilience against adversarial attacks, specifically from a statistical perspective due to scalability issues faced by deterministic methods. Existing approaches based on independent sampling usually fail to directly capture such instances due to their rarity. Hence in this work, we treat the existence of adversarial examples as a rare event, and propose an innovative statistical framework for assessing the adversarial robustness of DNNs, called REPP.  Our approach redefines the problem of calculating the occurrence of adversarial examples as the exponential of the mixture of a Poisson random variable and some potential geometric random variables. We adopt the point process to develop a Minimum Variance Unbiased Estimator (MVUE) to accurately estimate the likelihood of encountering adversarial examples, with an upper bound of the true probability with high confidence. Unlike existing rare-event methods based on Multi-level Splitting, REPP does not require the inherent level concept or the continuity condition of the cumulative distribution function (CDF) within DNNs. This adaptation allows for practical application across both computer vision and natural language processing tasks. Experimental results demonstrate that our method is more flexible and effective, offering a more reliable robustness evaluation than existing statistical approaches.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work addresses the challenge of evaluating adversarial robustness in deep neural networks (DNNs), particularly for large models where traditional white-box verification methods are infeasible. The authors introduce a novel statistical framework, REPP, which redefines the probability of encountering adversarial examples as a Poisson-based problem, enabling the development of a Minimum Variance Unbiased Estimator (MVUE) that improves estimation accuracy. Unlike previous approaches, REPP does not require continuity or level conditions, making it broadly applicable and effective across domains like computer vision and natural language processing, with demonstrated success on large models including ViT classifiers and LLMs.

### Strengths
1. The paper introduces REPP, a unique statistical framework that redefines the probability of adversarial examples using a Poisson-based approach, allowing for more precise and unbiased estimation of adversarial risk.

2. Unlike traditional methods that rely on continuity or level-based conditions, REPP is adaptable to both continuous and discontinuous outputs, making it broadly applicable across diverse domains, including natural language processing and computer vision.

3. REPP reduces the number of queries and simulations required.

### Weaknesses
1. The motivation behind the proposed method is unclear. If a service provider wants to evaluate their model's robustness, they could simply use white-box evaluation methods. Conversely, if a user with only black-box access to the model API attempts to assess robustness, they would need to make millions of queries, potentially incurring significant costs.

2. The experimental evaluation is limited. The experiments are conducted only on the MNIST dataset and 200 ImageNet images, and the single baseline used is outdated (published in 2018). To better demonstrate the proposed method's effectiveness, the authors should include a broader range of datasets and comparison baselines, including both black-box adversarial attacks and basic white-box attacks, such as FGSM.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes a statistical method to assess the likelihood of an adversarial example existing in a given perturbation set. In particular, the problem is formulated as a rare-event estimation problem and, compared to previous work, allows for discrete input domains. The method is evaluated for continuous perturbation sets e.g., in the image domain and for discrete perturbation models in the NLP domain.

### Strengths
* Understanding the likelihood of adversarial examples occurring is an interesting problem and considering how to extend existing work to deal with discrete input domains is a natural continuation
* The poison process modeling framework used seems to be a valid approach to solve the problem
* The method is applicable to semantically more interesting perturbation sets such as e.g., geometric transformations for images or adversarial suffixes for LLMs

### Weaknesses
1. The main claim of the paper does not match the contribution. In particular, the paper claims to develop a minimum variance unbiased estimator (MVUE) to accurately estimate the likelihood of encountering adversarial examples. However, the paper "only" uses the MVUE developed by Walter (2015), which does not require any adaptation in the derivation and can be applied to arbitrary random variables. The novelty is applying the existing estimator to a random variable describing whether there exists an adversarial example. In particular nearly the complete Methods Section 4, i.e., Definitions 1, 2, 3, 4, Theorems 1, 2, 3, Propositions 1 are (close to) direct copies from Walter (2015) or standard results as mentioned by Walter (2015), but this is only indicated for Definition 3 and Theorem 3. While the authors do cite Walter (2015) before Section 4, it was not clear to me while reading that these results are already existing.
2. The focus of presentation in the paper seems odd. I do think it is very valid to explore an existing statistical estimator for the problem studied in the paper, however, in the presentation of the method (i.e., the methods section), I would expect a description of the statistical estimator and how/which adaptions are necessary for the particular ML application as done e.g. by Webb et al. (2019) regarding AMLS, and not a derivation of the (existing) statistical method itself in a theorem-proof style (without making clear that these are standard/existing results as mentioned above). E.g., the complexity discussion, pseudocode, and description how to actually leverage the method to estimate the probability for adversarial examples is moved to the appendix and only briefly mentioned at the beginning of Section 5.
3. The provided experimental details seem limited for reproducibility. 
   * No code is attached to the submission.
   * There is no reproducibility statement.
   * F.1 and F.3 just state "We mainly follow the same settings in the AMLS paper (Webb et al. 2018).". I would expect to state concretely if one follows the same settings and if only mainly, state where one deviates. Also, there is an "as" missing.
   * I think providing more details and/or pseudocode for the employed Gipps sampling (5.2.1.) or Metropolis Hasting (5.1) would help reproducibility. Mention the MCML sampling used in 5.2.2.
   * I'm in general missing details how sampling is performed from the perturbation sets (i.e., Line 1 in Algorithm 1)
4. There seem to be small technical errors and inaccuracies in language/definitions. 
   * The arrival times $(T_n)_{n\ge 0}$ of the Poisson process are claimed to be a homogeneous Poisson process with parameter 1. Indeed, the proof only shows that the inter-arrival times follow an exponential distribution leading to the counting process being a Poisson process. The arrival times of a Poisson process are continuous and distributed like the Gamma distribution. 
   * I'm confused about the hypothesis test setting. The hypothesis seem not clearly defined, but the text rather defines the terms *robustness violation* and *probabilistically certified robust*. 
   * Also, in $\mathcal{H}_1$ it states "[...] and the estimated probability $\hat{\mathcal{I}}$ [...] satisfying [...]", but a condition on $\hat{\mathcal{I}}$ is never stated and the probability statement regarding the confidence only holds for $N \rightarrow \infty$ not achievable in practice.
   * In the main body, it is never stated that $(T_n)_{n\ge 0}$ are the arrival times, which was confusing to me among others, as the distribution was claimed to be Poisson.
   * Theorem 2: $M_y$ follows a Poisson law with parameter $t_y = - \log P[Y \ge y]$ and not $t_y = - \log \mathcal{I}$
   * Theorem 3: "strict inequality random walks" are never introduced in the paper, thus reading "non-strict inequality random walks" feels confusing.
   * Provide domain definitions for input / labels / outputs at the beginning of Section 3.
5. In general, the paper together with the points above, feels unpolished. 
   * In the beginning of Section 5 the authors make an argument using the variance of the estimator. But at this point of the paper, the variance of the estimator is not discussed, only later in Line 376 there is a short reference to the appendix discussing the variance.
   * There are some sentences that feel "lost" e.g., Line 122/23 ", where a base classifier ... [...]."
   * Appendix A regarding background knowledge on the Poisson Process is never cited. It would have helped me to reference it at the start of Section 4. 
   * There are many grammatical issues and I recommend running a spell checker such as Grammarly. 
6. The Conclusion is just a summary of the paper, I'm missing a discussion on potential limitations, open problems, and potential ways forward.

### Questions
1. GeoRobust is claimed to be complete given sufficient computational resources. Are the results in Table 1 using GeoRobust referring to complete or incomplete certification? In line 421/22 it is claimed that REPP outperforms the optimal result from GeoRobust. But if GeoRobust's optimal result is a complete certificate, this means that REPP achieving higher certified robustness is evidence of the unsoundness of REPP. 
2. Related, is it correct that REPP cannot provide a sound certificate? I would expect a discussion on the soundness as this can be a limitation compared to deterministic verification approaches or probabilistic approaches such as randomized smoothing.
3. I'm excited about the applicability to more complex adversarial perturbation sets. E.g., it would be interesting to certify against common corruptions (Hendrycks et al. 2019) as included e.g. in the Robust Bench benchmark. Have I understood correctly, that the only requirement is that we have to be able to sample i.i.d. from the perturbation set?
4. For me, it is not 100% clear how previous work, e.g. by Webb et al. (2019) is not applicable to more complex adversarial perturbation sets. Taking not a fixed translation, but e.g. an arbitrary translation sampled between no translation and a translation up to a certain point. Would then AMLS still be applicable as we include all translations up to a certain point, making the input & output distribution continous again? I think a general discussion on the applicability to more complex (arbitrary) perturbation models and how previous work is limited in that respect would be helpful.
5. I'm missing the parameter $N$ for REPP in Section 5.2.2. Also, can you explain the context that you arrive at 25000 tokens? 
6. The estimated log probability in Section 5.2.2 are many orders of magnitutde below the permissible probability level for the upper bound. Thus, I wonder how large is the upper bound compared to the estimate? What does it mean if the upper bound is so loose? Further, I wonder, how telling the resulting log-probability really is (especially if so small as reported in Table 2). Here, the perturbation space is tremendously huge and what matters for reliability is probably how easy it is for an adversary to reach a subset in the perturbation space through optimization or some other guiding principle, rather than the size of the adversarial subset per se. 
7. Given the confidence interval is only approximate. How valid is it/how confident can we be for finite $N$? Or even small $N$ as proposed in Section 5 for verification?
8. What is the computational complexity of Algorithm 1 and what are the practical runtimes for the algorithm in the reported experiments in the case that either an adversarial example exists, or in the case that none exists.
9. Could you elaborate a bit more on the argument why for verification a smaller $N$ suffices? 
10. Why does the number of calls for estimating the probability $\mathcal{I}$ scale with $\mathcal{O}(\log(1/\mathcal{I}))$?

Minor things:
* Line 174: The concept of a level is not introduced, but plays an important role in discussing the related work and contribution.
* Line 312: It reads as if $\hat{\mathcal{I}}$ has no effect on the discontinous case, but the discontinous case has no effect on  $\hat{\mathcal{I}}$
* Line 312: Second sentence writes $\hat{\mathcal{I}}$ [...] special version of $\hat{\mathcal{I}}$"
* Line 520: 10 and not 6 behaviors

**References**  
Walter, "Rare event simulation and splitting for discontinuous random variables", ESAIM: Probability and Statistics 2015  
Webb et al., "A Statistical Approach to Assessing Neural Network Robustness", ICLR 2019  
Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations", ICLR 2019

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper focuses on the problem of calculating the occurrence of adversarial examples. The problem is modeled as the exponential of the mixture of a Poisson random variable and some potential geometric random variables. A Minimum Variance Unbiased Estimator (MVUE) is developed to accurately estimate the likelihood of encountering adversarial examples. Different from previous works, the proposed method does not require the continuity assumption or the concept of levels. Experiments are conducted on both CV and NLP areas.

### Strengths
- Experiments have covered both CV and NLP areas.

### Weaknesses
- Motivation is unclear and usefulness is doubtful. The estimated adversarial occurrence seems extremely rare and I wonder how useful this information is. For example, Table 2 shows the log probabilities for adversarial occurrence are lower than -50, implying that the probabilities for adversarial occurrence are basically always zero. How are model developers or users supposed to use these numbers?
- The paper is hard to follow.
- - Preliminaries are not well covered. For example, not requiring the level concept is claimed as one of the main benefits of the proposed method beyond the existing work AMLS. However, neither the AMLS nor the meaning of "level" is introduced. As a result, it's hard to assess the novelty of this paper.
- - Notations are confusing. For example, lowercase $y_0$ is the ground truth label, while uppercase $Y$ is the logit margin and $Y_0$ seems like a random walk state. Another instance is $\mu(x)$, instead of $\mu(x_0)$, denotes the input domain of $x_0$.
- - What is $t_j$ in Equation 12?
- Typos:
- - For equation 2 integral part, I suppose it should be $1_{s(x)\geq0}$?

### Questions
Please refer to those in the weaknesses section.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents a novel statistical method for evaluating the robustness of DNNs through a framework called REPP, which enhances the Adaptive Multi-Level Splitting (AMLS) approach. They treat adversarial instances as rare events and estimates their probability as the exponential of the mixture of a Poisson random variable and some potential geometric random variables.

### Strengths
- Motivation for statistical methods looks good- pointing out the limitations and impracticality of deterministic methods effectively.
- The framework demonstrates substantial time efficiency compared to baseline methods.
- It is applicable across a wide range of domains.

### Weaknesses
- A bit difficult to interpret Table-1. For certified Acc' which result is better? REPP does not have the best numbers.
- No baseline for results in Table 2.

### Questions
- Could the results be compared with Randomized Smoothing (another statistical verification technique) or other certified accuracy methods, like direct certification via IBP, on commonly used datasets for adversarial robustness, such as CIFAR-10 or Tiny IMN? (or a justification on why that is not needed)

### Soundness
3

### Presentation
3

### Contribution
2
