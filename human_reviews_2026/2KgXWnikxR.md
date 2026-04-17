# Probabilistic Robustness for Free? Revisiting Training via a Benchmark

- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
Deep learning models are notoriously vulnerable to imperceptible perturbations. Most existing research centers on adversarial robustness (AR), which evaluates models under worst-case scenarios by examining the existence of deterministic adversarial examples (AEs). In contrast, probabilistic robustness (PR) adopts a statistical perspective, measuring the probability that predictions remain correct under stochastic perturbations. While PR is widely regarded as a practical complement to AR, dedicated training methods for improving PR are still relatively underexplored, albeit with emerging progress. Among the few PR-targeted training methods, we identify three limitations: i) non‑comparable evaluation protocols;  ii) limited comparisons to strong AT baselines despite anecdotal PR gains from AT, and; iii) no unified framework to compare the generalization of these methods. Thus, we introduce $\mathtt{PRBench}$, the first benchmark dedicated to evaluating improvements in PR achieved by different robustness training methods. $\mathtt{PRBench}$ empirically compares most common AT and PR-targeted training methods using a comprehensive set of metrics, including clean accuracy, PR and AR performance, training efficiency, and generalization error (GE). We also provide theoretical analysis on the GE of PR performance across different training methods. Main findings revealed by $\mathtt{PRBench}$ include: AT methods are more versatile than PR-targeted training methods in terms of improving both AR and PR performance across diverse hyperparameter settings, while PR-targeted training methods consistently yield lower GE and higher clean accuracy. A leaderboard comprising 222 trained models across 7 datasets and 10 model architectures is publicly available at  https://tmpspace.github.io/PRBenchLeaderboard/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the PRBench, a unified evaluation benchmark for adversarial robustness and probabilistic robustness.

The authors compared multiple robust learning methods across several model architectures, concluding that robustness generalizes from the adversarial setting to the probabilistic setting but not vice versa, while models trained for probabilistic robustness see smaller generalization gaps.

Additionally, the authors presented theoretical analyses on robust generalization and robust training objective smoothness.

### Strengths
This paper is valuable for understanding the relationship between adversarial and probabilistic robustness, providing important insight for building real-world dependable deep learning systems.

### Weaknesses
The paper presents some exciting theoretical analyses and a valuable benchmark. However, their connections are not entirely clear. Specifically, Theorem 1 seems disjoint from the rest of the paper, and Theorem 2 appears in a completely different section. I believe the paper would become clearer with the following modifications:
- Add some discussions about how Theorem 1 is related to the creation of the PRBench.
- Reorganize Theorems 1 and 2 into a theoretical analysis section before the discussion of the benchmarking results.

The description "some function is smoothness" appears repetitively in the paper, making it somewhat unnatural to read. For example, Definition 3 says "$f$ is $\\beta$-smoothness". I suggest changing such descriptions to "some function is smooth", like $f$ is $\\beta$-smooth" in places like Definition 3, Theorem 1, and Theorem 2.

### Questions
The authors mentioned that "For the two PR metrics, $| \mathcal{D} |$ denotes the number of test samples that are correctly classified by the model, whereas for AR, $| \mathcal{D} |$ refers to the total number of test samples. Why do we have this discrepancy?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces PRBench, which compares the adversarial robustness, probabilistic robustness, and generalization error of a variety of adversarial and PR-targeted training methods. While adversarial training outperforms PR-targeted training in probabilistic robustness in most cases, it also yields a higher generalization error. This paper involves upper bounds on the expected stability for both training methods, which implies that adversarial training theoretically leads to a higher generalization error.

### Strengths
1. The author conducts extensive experiments to demonstrate that AT outperforms PR-targeted training in adversarial and probabilistic robustness but results in a higher generalization error. The evidence derived from various model architectures and standard datasets is consistent and compelling. 

2. The author comprehensively illustrates the multi-fold trade off among (PR, AR), GE, and efficiency, which may illuminate future research to appropriately handle them.

### Weaknesses
1. There are some confused descriptions in the relationship among p, f, and L, for instance, in Eq. 1, p is a softmax function outside the model f, but in Thm.1, the project p is an inner composition of the model f. Besides, the author seems mistakenly using the model f as the loss function L in Proofs located in appendix F. 5.  

2. The author overlooks a critical assumption for Lemma 2 that p_i>0 must hold when for l_i=1. If l_i=1 and p_i=0, the gradient of the cross-entropy loss would be infinity, the loss is not Lipschitz in this case. Indeed, cross-entropy loss is deemed as a non-Lipschitz one due to such potential infinite discontinuities. I suggest the author to impose some constraint on the value of p_i, for example, if x is classified correctly, the author could obtain a lower bound 1/|Y|  of p_i in this case and can further derive a tighter Lipschitz constant since p and l are in the same direction, which could be beneficial to the subsequent proofs.

3. The author unreasonably omits the supremum of the loss function in Line 134 in Proof 7. The author should explain this operation since the supremum of the cross-entropy function is obviously infinity, so the expected stability in Lemma 6 is seemly unbounded, which contradicts to current result of Lemma 6.

4. The author adopts the standard Lipschitz and smoothness assumptions for the model f in the theoretical analysis. However, the architectures employed in the experiments do not strictly satisfy these conditions. This discrepancy raises the question of whether the empirical observations on GE can be interpreted by theoretical results.

5. Although the claim “AT outperforms PR-targeted training in adversarial and probabilistic robustness but results in a higher generalization error.” is substantially guaranteed by experiments, the author doesn’t display any convinced explanation on that. I think there may exist some intrinsic relationship between adversarial and probabilistic robustness that has not been exploited.

### Questions
In addition to my comments in the above "weakness" section, I also believe that the paper would have notable contribution if the author could further provide some persuasive explanations on the relationship between adversarial and probabilistic robustness (as mentioned in Weakness #5.).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the relationship between two key types of model robustness: Adversarial Robustness (AR), which measures resilience against worst-case perturbations, and Probabilistic Robustness (PR), which measures the statistical likelihood of correct predictions under stochastic perturbations. The most important contribution is PRBench, a comprehensive benchmark designed to evaluate and compare various robustness training methods, with a huge number of empirical baseline results involving multiple methods, datasets, architectures, and metrics.

### Strengths
- The distinction and relationship between AR and PR are of great importance to the community. The claim that standard AT is a superior method for achieving both AR and PR is a convincing finding.

- I appreciate the extensive workload and great efforts in building this benchmark. I believe it would have a significant influence on this field and be very useful for the following studies.

- The paper is well-written, also providing a solid theoretical analysis to explain the observed differences in generalization error, which makes the empirical findings more convincing.

### Weaknesses
- I'm a little confused by the title and conclusion that PR comes "for free". The results clearly show that this "free" PR is paid for with a significant drop in clean accuracy. So isn't this a complex, multi-objective trade-off, instead of a "free lunch"?

- The PR-targeted methods evaluated are relatively simple. The finding is more accurately "strong AT is better than simple RT," which is less surprising. Further including more advanced PR-targeted techniques might be an effective refinement.

### Questions
- Would it be more accurate to frame the paper's findings as an analysis of two different trade-off frontiers: one (AT) that trades accuracy for high AR+PR, and another (RT) that trades AR for high accuracy+PR?

- The AT-PR method seems to provide the best Pareto-optimal solution across Acc, AR, and PR, with its only drawback being computational time. Does this not motivate further research into efficient hybrid training methods, rather than supporting the claim that PR-targeted methods are of "limited practical need"?

- What is the intuition behind the success of the new KL-PGD variant? Why would using a KL-based objective to find the adversary, but a standard CE-loss to train on it, be a more effective strategy than using a consistent loss for both (like in TRADES)?

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
4

### Summary
This paper introduces PRBench, the first comprehensive benchmark designed to evaluate probabilistic robustness across diverse robustness training methods systematically. While prior work has primarily focused on adversarial robustness under worst-case perturbations, this work shifts the focus to the evaluation of probabilistic robustness. PRBench addresses these issues through a well-structured benchmark with 222 trained models, 7 datasets, and 10 architectures. Most importantly, all models are accessible via the leaderboard.

### Strengths
This paper provides a comprehensive and systematic analysis of probabilistic robustness (PR) across a wide variety of datasets and model architectures. Specifically, the benchmark includes diverse datasets such as CIFAR-10, CIFAR-100, MNIST, SVHN, and Tiny-ImageNet, and evaluates a broad set of models including DeiT-S, DeiT-T, ResNet-18, ResNet-34, Simple-CNN, VGG-19, ViT-B, ViT-S, and WRN-28-10.

The paper further makes a novel contribution by introducing PRBench, the first dedicated framework for evaluating and comparing PR-focused training methods. This work addresses a significant gap in robustness research, traditionally dominated by adversarial robustness.

The empirical findings are particularly insightful, and the authors also include a theoretical analysis that provides a strong conceptual foundation that deepens the understanding of the empirical results. The release of a public leaderboard and benchmark suite further enhances its impact by enabling standardized evaluation and facilitating reproducible research. Overall, the paper is well-motivated, clearly presented, and offers both theoretical and practical contributions that are likely to influence future work on adversarial robustness.

### Weaknesses
(Minor issue only)

The paper defines GE as “the difference between the natural and empirical risk.” I suggest including its mathematical formulation, similar to Equation (4), for clarity and consistency.

While abbreviations such as AT and GE are widely used, the paper employs too many acronyms (e.g., AT, PR, GE, RT, …), which can hinder readability. I recommend adding a table summarizing all abbreviations for convenience.

Several prior works are closely related to this paper in terms of GE and robust overfitting, but are currently omitted. I recommend citing and discussing the following:

- Jiang, Yiding, et al. “Fantastic Generalization Measures and Where to Find Them.” ICLR, 2020.

- Kim, Hoki, et al. “Fantastic Robustness Measures: The Secrets of Robust Generalization.” NeurIPS 36 (2023): 48793–48818.

These papers are conceptually and experimentally relevant, especially in how they represent and benchmark generalization and robustness. Please include them and discuss the relationship and differences with your work.

I would be happy to recommend strong acceptance if the minor issues are properly addressed.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
