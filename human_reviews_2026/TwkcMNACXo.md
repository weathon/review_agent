# f-INE: A Hypothesis Testing Framework for Estimating Influence under Training Randomness

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 8

## Abstract
Influence estimation methods promise to explain and debug machine learning by estimating the impact of individual samples on the final model. Yet, existing methods collapse under training randomness: the same example may appear critical in one run and irrelevant in the next. Such instability undermines their use in data curation or cleanup since it is unclear if we indeed deleted/kept the correct datapoints. To overcome this, we introduce *f-influence* -- a new influence estimation framework grounded in hypothesis testing that explicitly accounts for training randomness, and establish desirable properties that make it suitable for reliable influence estimation.
We also design a highly efficient algorithm *f*-*IN*fluence *E*stimation (**f-INE**) that computes f-influence in a **in a single training run**. Finally, we scale up f-INE to estimate influence of instruction tuning data on Llama 3.1 8B and show it can reliably detect poisoned samples that steer model opinions, demonstrating its utility for data cleanup and attributing model behavior.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper the authors propose a novel data influence estimation algorithm which is an improvement on top of the existing TracIn (Training Data Influence by Tracing Gradient Descent) approach. The proposed extension is less vulnerable to random training runs, and is motivated by privacy auditing and hypothesis testing theories. The paper estimates the statistical significance of data influence by quantifying the distinguishability of two distributions: a distribution of influence scores when the sample participates in training over multiple random training trials and a distribution when the sample did not participate in the training process. The paper proposes to look at the tradeoff curves between Type I and Type II errors as a means to quantify the distinguishability of those two distributions.
It draws parallels between their definition and Gaussian Differential Privacy, claiming that the tradeoff can be fully characterized by a single variable, mean.
The paper shows that the proposed approach is more stable and is effective for MNIST's miss-labeling task and train-test influence for LLM’s data poisoning tasks.

### Strengths
+ The paper is well and clearly written without major ambiguities.
+ The paper studies an important problem, randomness in the influence estimation. It proposes an interesting approach motivated by the Differential Privacy theory and explains the theoretical intuition behind it.

### Weaknesses
+ See weaknesses 
+ In Figure 7 the error bar for F-INF looks pretty large. If  F-INF is more stable than LESS why does it have wider std ?

### Questions
+ The experimental evaluation part is a bit lightweight
  + MNIST is a very toy model and LLM experiments are good but cover a narrow set of applications -  bias in a specific category.
It is unclear how well the approach performs for a wider range of datasets or models. E.g. LESS performs the evaluation on a wider range of downstream tasks and datasets (bbh, tinyqa and mmlu). In addition to that, TRAK evaluates their approach on fact checking dataset. It would be good to report results on a wider range of tasks and datasets similar to those baseline approaches.
+ The algorithm description is a bit unclear
	+ E.g. Algorithm1: lines 9 and 10 the right side of the assignment seem to be identical,
               likely there is a typo there. Similarly line 4 and line 5 perform the same sampling (D \ S ) ? 
+ It is unclear how S is selected in Algorithm listing 1 and whether Algorithm 1 is called in a loop for random subsets of S. Is it guaranteed that each example in S will be encountered only once in S ?
+ It seems that instead of retraining the model multiple times the authors leverage training epochs but this can be a very noisy approximation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
hey This paper identifies the sensitivity of influence estimation methods to training randomness (e.g., data ordering) as a critical flaw. It proposes f-INE, a new framework grounded in hypothesis testing, to provide an influence estimate that is robust to this randomness. The authors apply this method to mislabeled data detection on MNIST and to attributing behavior in an LLM.

While the technical idea is interesting, the paper is built on a flawed premise that mischaracterizes the research landscape. It ignores two distinct, highly relevant lines of established research that either (1) value the training-run-specific influence the authors dismiss as a "flaw" or (2) have already provided solutions for the exact problem the authors claim to be solving. The empirical results are invalid and do not support the claims.

### Strengths
The technical framework itself, which uses hypothesis testing to quantify the distinguishability between training runs, has some merit. This approach could potentially be valuable for other problems, even if it is misapplied here.

The layout of the manuscript looks comfortable and the manuscript is easy to read.

### Weaknesses
1. The paper's central premise—that robustness to training data ordering is universally desirable and that sensitivity to it is a "flaw"—is incorrect. There is an established line of research (e.g., [Data Shapley in One Training Run]) dedicated to quantifying a sample's contribution within a specific training run. This research finds that a sample's influence varies significantly at different training stages, guilding decisions on what data to train on during different stages.

2. The paper aims to find a data-centric utility measure that is robust to training randomness. Yet it fails to cite or discuss the large, established field of "data valuation" which also applies to this problem. Methods such as Data Shapley (https://arxiv.org/abs/1902.10275 ), LAVA (https://arxiv.org/abs/2305.00054 ) provide metrics for data utility from a pure data perspective, free from the randomness of any single training run. Data Banzhaf (https://arxiv.org/abs/2205.15466) is specifically designed to maximize robustness to training variations. The omission of this entire body of work is a critical flaw.

3. The experiments are insufficient to support the paper's claims. In the mislabeled MNIST experiment, the baselines appear to be incorrectly chosen or implemented. The results show that f-INE performs similarly to other methods, failing to demonstrate its purported superiority. The second experiment (LLM attribution) is conducted on a sample size (training on 50 examples, testing on 10) so tiny that it is impossible to derive any scientifically valid conclusions. Furthermore, this experiment only compares f-INE to LESS. Citing LESS as the sole baseline is questionable, as it was proposed for large-scale data selection, not the small-scale setting tested here.

### Questions
Can the authors explain a bit on how should we interpret the results from the MNIST experiment where all the methods perform similarly?

### Soundness
2

### Presentation
1

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
This paper introduces F-INE, a framework to estimate data influence that accounts for training randomness. This is important and interesting problem, and how the authors addressed this is also interesting. The core idea is to reframe influence as a hypothesis testing problem, which the authors connect to membership inference. They show this f-influence converges to a single scalar ($G_{\mu}$) for composed algorithms like SGD. The paper proposes an efficient, single-pass algorithm (f-INE) to estimate this $G_{\mu}$. Empirically, F-INE is shown to be significantly more consistent and robust than baselines, while demonstrating superior utility in detecting mislabeled data (MNIST) and poisoned instructions in Llama-3.1-8B.

### Strengths
1. The paper's focus on tackling training randomness is an important and necessary direction for the field. This is a well-known, critical flaw in existing influence estimation methods, and the paper addresses it head-on.

2. The theoretical connection established between influence estimation and hypothesis testing (specifically, membership inference and f-DP) is interesting. This provides a principled and statistically grounded foundation for the proposed method.

3. The empirical results are impressive. The high consistency score on the MNIST/MLP setup and the demonstration of superior stability and utility on the Llama-3.1-8B scale validate the method's practical advantages over existing SOTA baselines.

### Weaknesses
1. My main concern is the limited scope of the mislabeled data detection experiments. While the MNIST/MLP results are promising for showing consistency, this is a relatively simple setting. The claim of the method's general utility for data cleanup would be strengthened by evaluating it on more complex vision datasets like CIFAR-10/100 or Tiny-ImageNet, and on different architectures (e.g., ResNet family). I wonder if the strong performance and stability guarantees hold in these more challenging, higher-dimensional settings.

2. Another concern is the separation of the utility and consistency metrics. While both are important, they don't fully answer the key practical question: how consistently are the correct (ground-truth) items identified? These two metrics need to be simultaneously measured, perhaps as a conditional metric. The paper would be stronger if it included a unified metric. This would directly measure if the method is reliably useful.

### Questions
Please see the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, estimating the influence of a training data point is reframed as a binary hypothesis test between distributions. The paper shows that instead of treating influence as a deterministic property of a fixed training path, one should take a distributional view that explicitly incorporates training-time randomness (randomness stemming from factors such as optimization, weight initialization, data shuffling, etc, ). They formalize f-inf as the distinguishability between model outputs trained with and without a given data subset. The key idea is that strict total ordering, during ML training for neural nets, is underspecified unless one fixes a testing trade-off. Consequently, they define f-influence using trade-off functions and show that, for a learning procedure that composes, these functions asympotitically tend towards a notion that they term: single scalar Gaussian Influence. The f-INE, is a single-run algorithm that makes this estimate by collecting gradient-similarity statistics with and without a target subset of inputs and then computing test errors across thresholds. Their experiments show that improved stability to randomness compared to baselines and alternatives like TRAK, LESS, etc.

### Strengths
- Important and Timely problem: instability of influence scores across random seeds is an important and mostly understudied problem in the literature. This work puts this at center stage and provides an algorithm for attacking this issue.
- Interesting Application of Differential Privacy Ideas: While the connection between influence and differential privacy is not new. The formulation here is quite directly connected and based on ideas from differential privacy. It seems that the key change moving away from a worst-case setting that differential privacy typically requires.
- This approach avoids multi or retraining.
- Compelling LLM Llama-3.1-8B  case study: On Llama-3.1-8B instruction tuning with poisoned, f-INE achieves higher recall of poisoned items and lower coefficient of variation across seeds than LESS (a different approach).

### Weaknesses
- This work requires a careful reading of the Dong et. al. paper on f-Differential Privacy. In fact, I would say one main point of this work is to apply the insights from that paper to influence estimation. However, I am not an expert on differential privacy, so I don't know the amount of innovation involved in translating this insight here. 

- Decision guidance: Provide a short “operationalization” section: given an estimated $\mu$ how should data curators set Type-I/II budgets to choose deletions under randomness?

Minor issues: 
- Need a gap between last sentence of first page and the first sentence of the page 2. Put a space between lines 067 and 068.

### Questions
- Decision guidance: Provide a short “operationalization” section: given an estimated $\mu$ how should data curators set Type-I/II budgets to choose deletions under randomness?
- How sensitive are your results to batch size and the projection dimension (for LoRA gradients), and the number/placement of checkpoints? For typical setting, how would you suggest one set these? 
- It is somewhat unclear to me whether the negative influence values have a different behavior and subsets with positive influence. I assume because these are distributional tests that there is no difference in implementation of the algorithm between these two groups.

### Soundness
4

### Presentation
3

### Contribution
3
