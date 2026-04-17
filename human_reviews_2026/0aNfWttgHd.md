# Conformal Prediction Adaptive to Unknown Subpopulation Shifts

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Conformal prediction is widely used to equip black-box machine learning models with uncertainty quantification, offering formal coverage guarantees under exchangeable data. However, these guarantees fail when faced with subpopulation shifts, where the test environment contains a different mix of subpopulations than the calibration data.  
In this work, we focus on *unknown* subpopulation shifts where we are not given group-information i.e. the subpopulation labels of datapoints have to be inferred. 
We propose new methods that provably adapt conformal prediction to such shifts, ensuring valid coverage without explicit knowledge of subpopulation structure. 
While existing methods in similar setups assume perfect subpopulation labels, our framework explicitly relaxes this requirement and characterizes conditions where formal coverage guarantees remain feasible. 
Further, our algorithms scale to high-dimensional settings and remain practical in realistic machine learning tasks. Extensive experiments on vision (with vision transformers) and language (with large language models) benchmarks demonstrate that our methods reliably maintain coverage and effectively control risks in scenarios where standard conformal prediction fails.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper addresses the challenge of applying conformal prediction (CP) under unknown subpopulation shifts, where the proportions of latent subgroups differ between calibration and test data but group labels are unavailable. The authors propose three algorithms that adapt CP to such shifts: Algorithm 1 (Weighted CP with Domain Classifier) uses a domain classifier to estimate subpopulation probabilities and reweights calibration scores; Algorithm 2 (Multiaccuracy-based CP) relaxes the assumption by requiring only multiaccuracy of the classifier, making the method more practical; and Algorithm 3 (Similarity-based CP) handles settings without any domain classifier by using embedding similarity to weigh calibration examples.

Empirical evaluations on vision benchmarks (ImageNet subpopulation splits) and LLM hallucination detection show that the proposed methods maintain tight coverage across distribution shifts, outperforming standard and group-conditional CP methods.

### Strengths
1. The paper tackles the underexplored setting of unknown subpopulation shifts where group labels are unavailable, which is an important step beyond existing CP methods assuming known domains or exchangeability.

2. The authors provide formal coverage guarantees under varying assumptions (Bayes-optimal, multicalibrated, multiaccurate classifiers). 

3. Experiments cover both vision (ImageNet-based BREEDS subpopulation shifts) and language (LLM hallucination detection).

### Weaknesses
1. The coverage guarantees hinge on the domain classifier being multicalibrated or multiaccurate—properties that are difficult to ensure in high-dimensional practice. While the authors reference empirical evidence of approximate multicalibration, more rigorous discussion of real-world feasibility is needed.

2. The experiments primarily compare against variants of CP (standard, max, and conditional calibration). It would be informative to compare against general **distribution-shift calibration** methods (e.g., density-ratio-based or covariate shift adaptation techniques).

3. Algorithm 3 introduces parameters $\beta$ and $\sigma$, but their sensitivity and impact on performance are *not* analyzed.

4. Although the BREEDS benchmark simulates subpopulation shifts, all domains are derived from ImageNet classes. How about cross-dataset or real-world domain adaptation settings?

5. How robust are the proposed methods to misspecification of the domain classifier? For example, if they systematically bias certain subpopulations rather than random errors?


6. Concerning algorithms, 

6a. The explanation/clarification of algorithms in main text are missing, making it hard to follow the ideas. 

6b. The statements in algorithm are **BROKEN** and **NOT** self-contained. Take Algorithm 1 as an example: 

(i) How to "calculate score $s_i^k$", based on some equations or score function $S$ or other?

(ii) $\hat{\lambda}$ and $\hat{q}_{\alpha}$ appear abruptly. 

(iii) What is the relationship bewtween $\hat{\lambda}$ and $\hat{\lambda}_k$, given the latter is the $k$-th entry of $c$?

(iv) What is the definition of $J$?


7. Besides algorithms, other writing and presentation issues makes it challenging to follow without constant reference to prior sections.

What are the definition of "A1", "A2", and "A3"? If they refer to three algorithms, respectively, should "oracle" be "A1"?

Further, I personally do not think Algorithm 1 could be referred as "orcale".

### Questions
see above.

### Soundness
2

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
This paper proposes new conformal prediction methods that maintain valid uncertainty coverage under unknown subpopulation shifts, where test data distribution differs from calibration data. The authors develop algorithms that use subpopulation structure through domain classifiers or similarity measures.

### Strengths
1- The introduction effectively motivates the problem and provides a thorough overview of previous methods

2- The paper is well written

### Weaknesses
Please check the questions.

### Questions
1- how was the parameter $\beta$ was selected?

2- In the conclusion, the paper mentions experiments with synthetic data. Could the authors clarify where synthetic data was used and for what purpose?

3- If I understand correctly, in Figure 2 the average coverage of the unweighted conformal prediction method appears above the desired $1−\alpha$. If this interpretation is correct, could the authors explain why over-coverage occurs here?

4- Please include discussion or analysis of the computational complexity of the proposed algorithms, especially in comparison to standard conformal prediction

5- The empirical section need to report the average prediction set size of the proposed methods versus existing baselines

6- Although the algorithms are claimed to scale to high-dimensional tasks, no runtime or memory comparisons are provided to substantiate the claim.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper is addressing the subpopulation shift for conformal prediction. This problem was addressed prior by Tibshirani et al 2020 under the term covariate shift, however there it is assumed that. the distribution of calibration and test are known. Here the authors first show that the existing approach drastically fail for imperfect estimators, and then provide a series of methods by estimating the subpopulation distribution. Their hierarchy of algorithms are tailored for various levels subgroup prediction accuracy.

### Strengths
Besides the interesting and applicable problem, the authors break down the problem in different levels of knowledge about the subpopulation. This allows to choose upon the task and the environment.

Their experimental results cover a wide range of setups from image to language which is a plus. This shows that their method is applicable and not only abstract.

Despite the minor flaws, the writing is good, and the paper is easy to follow.

### Weaknesses
**Unclear statement about the subgroups.** I could not understand whether the authors are assuming that the subpopulations are known, and discrete? And if the assignment for such subgroups are given at least over the training data. This is important to be clarified in all algorithms and theorems. This is an important flaw since for instance in Algorithm 1, and 2 the classifier c is assumed to be trained or at least trainable.

**Strong assumptions in Section 3.** Although not made clear, the assumptions noted in Definition 3.2, and 3.4 are very strong. If we were able to train optimal Bayes classifier, or multi-calibrated classifier, then wasn't it easier to achieve subgroup conditional, or even conditional coverage at first place? If so then the subpopulation shift would be meaningless as APS is already capable of providing conditional and hence subpopulation conditional coverage. 

However the last point is (to the best of my understanding) very strong, I would still think the paper is acceptable due to the results in Section 4.

### Questions
1. In algorithm 3, sigma is a function, but you are treating it as a scaler and divide a number by divide it. I can not parse the algorithm. What does setting the score to infinity mean here?
2. Can you elaborate more on Definition 3.4? What is the expectation over? 
3. Can you make some examples about when an optimal Bayes classifier, and a multicalibrated classifier is even possible to have?
4. I am not sure but should the guarantee in theorem 3.3, 3.4, and 3.5 be conditional to the X coming from any subpopulation? 
5. Does the theorem 3.1 reduce to Mondrian Conformal Prediction?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles conformal prediction under subpopulation shifts when group labels aren't available. The authors propose using domain classifiers to weight calibration data adaptively, with theoretical guarantees under multicalibration and multiaccuracy assumptions. They also introduce a similarity-based approach when no domain information exists. Experiments on ImageNet variants and LLM hallucination detection show reduced coverage variance across test environments compared to standard conformal prediction.

### Strengths
1.	This work addresses a relevant and underexplored problem of adapting conformal prediction to unknown subpopulation shifts.
2.	The theoretical results are clear and intuitive.
3.	This paper provides a clear motivation.
4.	The proposed approach is relevant to a wide range of tasks, from ImageNet to LLM hallucinations.

### Weaknesses
1.	The theoretical guarantee of Theorem 3.3 relies on a strong assumption of having perfect or multicalibrated domain classifiers.  
2.	This work does not analyze what happens when the domain classifier is not multiaccurate and provides no empirical or theoretical results for this case.
3.	Algorithm 3, which handles the most realistic setting with no domain labels, is purely heuristic and lacks theoretical or empirical motivation.
4.	This paper should compare the proposed approach to other existing methods, such as Robust/max CP and group conditional CP, or the method proposed by Cherian et al. (2024) for LLM validity control.

### Questions
1.	What is the “standard LLM uncertainty estimation method” in line 475?
2.	In Algorithm 3, how were the parameters $\beta$ and $\sigma$ chosen, and how stable are the results under different values?
3.	The experiments consider 15-26 domains. How does the method scale to 100+ domains? Does the domain classifier turn harder to train, and does the coverage rate have a higher variance?

### Soundness
3

### Presentation
3

### Contribution
2
