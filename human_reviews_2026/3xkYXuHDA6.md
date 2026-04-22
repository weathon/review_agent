# Optimizing Canaries for Privacy Auditing with Metagradient Descent

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
In this work we study black-box privacy auditing, where the goal is to lower bound the privacy parameter of a differentially private learning algorithm using only the algorithm’s outputs (i.e., final trained model). For DP-SGD (the most successful method for training differentially private deep learning models), the canonical auditing approach uses membership inference—an auditor comes with a small set of special “canary” examples, inserts a random subset of them into the training set, and then tries to discern which of their canaries were included in the training set (typically via a membership inference attack). The auditor’s success rate then provides a lower bound on the privacy parameters of the learning algorithm. Our main contribution is a method for optimizing the auditor’s canary set to improve privacy auditing, leveraging recent work on metagradient optimization (Engstrom et al., 2025). Our empirical evaluation demonstrates that in certain instances, using such optimized canaries can improve empirical lower bounds for differentially private image classification models by several times when compared to canaries proposed in prior work. Furthermore, we demonstrate that our method is DP-SGD agnostic and efficient: canaries optimized for non-private SGD with a small model architecture remain effective when auditing larger models trained with DP-SGD.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Differential privacy (DP) is a privacy measure of a mechanism - a randomized function from a set of data points to some output. Its privacy is measured by the ability of an adversary to distinguish between two possible input datasets differing in a single element (referred to as neighbors), based on a single output of the mechanism. The better the privacy, the close is the success rate of the most sophisticated analyst to a random guess.
This framing as an hypothesis testing naturally lends itself to an empirical lower bound scheme; select two neighboring datasets, repeatedly run the mechanism on one one of the two datasets selected at random, and guess the identity of the used dataset based on the mechanism's output. The more accurate the guesser, the lower the privacy. Recently, a more efficient method has emerged, where rather than repeating many time the experiment to determine a change in a single element, the mechanism is called once and the auditing attempts to identify the participation of many elements, but the general idea remains.

Naturally, such auditing can only provide a lower bound, since DP is a worst-case property, bounding the privacy loss of *any* neighboring datasets against *any* guesser. As a result, a tight lower bounds requires selecting "easy to distinguish" elements (referred to as canaries), which increase the success rate of the guesser. This work considers a new method for curating such canaries using meta-gradient optimization, where a hyper parameter of the optimization process (e.g., learning rate) is optimized via GD, by propagating the gradient through the entire learning process. The authors propose a somewhat simplified attack which is differentiable, and show that using it top curate gradients improves the auditing results.

### Strengths
* This work addresses a known challenge, achieving a tighter lower bound in DP auditing by optimizing the choice of canary(es).
* It achieves this goal using the natural and novel (in this context, to the best of my knowledge) method of meta-gradients.
* The authors empirically back their idea by several experiments.
* The presentation is clear and detailed.

### Weaknesses
While the idea of using meta-gradients for canaries optimization seems promising, the implementation considered in this work seems over simplified, and the empirical comparison does not support the claims of meaningful improvement. Concretely, the Surrogate objective function (Eq. 3) captures a naive auditing method which relies only of the change in loss, which is one of the weakest auditing techniques even in the context of black-box auditing. Furthermore, the authors only compare their results to two of the weakest methods for canaries curation - random choice or elements from the underlying distribution or random elements with random labels. The existing literature contains many advanced canary curation methods, some of which were mentioned in the related work section, and without proper comparison it is impossible to asses the contribution of this work.

### Questions
Can the authors address my main concern, lack of comparison to other canary curation methods discussed in the literature, such as the ones mentioned by the authors and in [1] (the black-box case)?

[1] Nasr, Milad, et al. "Adversary instantiation: Lower bounds for differentially private machine learning." 2021 IEEE Symposium on security and privacy (SP). IEEE, 2021.

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
4

### Summary
This paper studies black-box auditing of DP-SGD and proposes a canary-generation method based on metagradient optimization to strengthen empirical privacy lower bounds. The approach uses a proxy (non-private) model to optimize canaries and then applies these optimized canaries to audit models trained via DP-SGD following existing one-run procedures (Steinke et al. 2023 and Mahloujifar et al. 2024). They empirically show on CIFAR10 that metagradient optimized canaries result in stronger attacks than random canaries under a fixed privacy budget.

### Strengths
- The paper is well-presented and clearly situates itself within the literature of DP (black-box) auditing and canary-based attacks.
- The explicit focus on last-iterate black-box auditing is an important and practically relevant problem and the idea of using optimized canaries to audit in this setting is an open problem.

### Weaknesses
- Empirical results are limited as only a single dataset (CIFAR10) is used alongside one non-DP experiment and one DP setting (with epsilon=8). It remains unclear how these gains using optimized canaries generalize to other datasets or differing privacy budgets, particularly high-privacy regimes.
- The paper claims efficiency of the canary optimization approach (using REPLAY) but does not quantify these overheads empirically or otherwise.
- This idea of using optimized canaries seems a straightforward extension of attacks in alternative settings (white-box auditing).

### Questions
1. The main contribution of the paper does not seem to be the optimized canary approach since this has been used before in prior work [1,2] but in adapting this to a black-box setting via a separate surrogate model/loss. However, there does not seem to be much explanation or detail as to why and when this proxy model approach would be effective vs. using random canaries.
2. It is not clear how the proxy model is trained and how much of an effect this has on the DP-SGD auditing process. Could you give more information on the precise training setup that is being used in your experiments? 
3. One of the contributions is leveraging REPLAY to provide efficient metagradient computations. However, there are no results that highlight the overhead of crafting these optimized canaries. The tradeoffs between this and random canaries is not made clear i.e., why bother with optimized canaries when random canaries could be good enough for the amount of compute needed? How practical is the suggested attack?
4. There are no results on differing privacy budgets or different datasets which limits how strong a conclusion to takeaway on the current set of results. Does your approach work well for auditing other datasets and model architectures?
5. How do the learned canaries look? Are they diverse or do they collapse to similar patterns as random noise? How are canaries initialized?
6. The paper dedicates a substantial portion of space to reviewing prior work and setup (~6 pages), leaving only a little room for the proposed method and the experimental results. It would be beneficial to provide more space for the technical details and additional experiments.

[1] Nasr, Milad, et al. "Tight auditing of differentially private machine learning." 32nd USENIX Security Symposium (USENIX Security 23). 2023.

[2] Maddock, Samuel, Alexandre Sablayrolles, and Pierre Stock. "Canife: Crafting canaries for empirical privacy measurement in federated learning." arXiv preprint arXiv:2210.02912 (2022).

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
The paper describes a method to discover a strong set of canary examples to use for empirical privacy auditing, via optimizing (with metagradient descent) a function that approximates the distinguishability of canaries *included* or *excluded* from training. Experiments demonstrate that these canaries are significantly strong (produce higher empirical epsilons) than random canaries. Each step of metagradient descent requires a model training run, so the technique would be infeasible on a large architecture, but they show that canaries optimized using a smaller architecture are also strong when ported to a larger architecture.

### Strengths
The core idea is original and promising. Although the method is too expensive to use on large architectures, the experiments show that it is sufficient to optimize the canary set with a small architecture and port them over. The writing is very clear and precise.

### Weaknesses
I have two main concerns regarding soundness and contribution.

First, the claim that the method can be used on a small architecture and generalized to a larger one is critical for the method to be of real utility, and therefore it needs more evidence. Could you experiment with more model sizes, or generalizing to a completely different architecture like ViT? Maybe at least one other data modality, like audio?

Second, I think a simple baseline that is important to compare to would be maximizing the gradient norm of the canary (on the initial model parameters, or perhaps after some training on real data). This would be computationally much lighter than metagradient optimization through the entire training run, and might produce a set of canaries that are nearly as strong as the ones you find.

### Questions
Do you have more evidence that it works to optimize canaries on one model and port them to another?

Could you compare to simply maximizing the norm of the canary gradient, or do you have a convincing argument why this will not work?

Could you add a topline to figure 2, showing the maximum possible empirical epsilons producible by that auditing method with that number of canaries? It would be interesting to see if you are hitting that max.

Note: there's some ambiguity in the literature about what constitutes a "black box" audit. For some, that would imply only allowing the adversary API access, not giving it even the final model checkpoint, or ability to compute a loss. In reality there is a spectrum of auditing scenarios depending on the threat model, so it may be best to retire the "black" vs "white" distinction.

Typo: $i$ vs $t$ in Alg 5

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the crucial problem of deriving tighter empirical lower bounds for the privacy parameter (ε) in black-box privacy auditing of differentially private learning algorithms. The authors propose a novel method to optimize canary examples using metagradient descent. This optimization is achieved by iteratively updating the canaries using a surrogate model trained on the CIFAR-10 dataset to minimize a surrogate objective function. This function is defined as the loss gap between canaries included in the training set and those held out. The optimized canaries are then deployed to solve the problem of underestimating privacy loss with standard auditing techniques. Through experiments, the authors demonstrate that their method yields higher and more accurate empirical ε values compared to baseline canaries, particularly in the challenging DP finetuning scenario.

### Strengths
S1. This paper proposes a novel and effective approach to improve privacy auditing by actively optimizing canary examples. This significantly advances the SOTA in auditing effectiveness, enabling auditors to uncover more accurate privacy leakage estimates.

S2. The optimized canaries prove effective for auditing models trained both from scratch and via DP-finetuning, which establishes the method's utility in different practical settings of differentially private model deployment.

S3. This paper includes detailed algorithms for both the metagradient optimization and the auditing procedures, which ensures the reproducibility of the proposed method.

### Weaknesses
W1. The metagradient optimization process involves training a surrogate model multiple times within each meta-iteration. This paper does not discuss the computational overhead of this optimization phase, which could be a practical barrier for scenario with limited resources.

W2. The empirical evaluation is exclusively confined to image classification tasks using CIFAR-10. The generalizability of this canary optimization approach to other data modalities or different machine learning tasks remain unexplored. And this paper lacks discussion or an ablation study on the sensitivity of the results to different surrogate model architectures for the canaries optimization.

### Questions
Overall, this paper is well-structured. I have the following concerns:
1. While the paper compares against random and mislabeled canaries, are there other canary optimization techniques that could serve as stronger baselines for comparison? (e.g., the briefly mentioned Jagielski et al. and Nasr et al. in the related work)

2. This paper states that initial canaries are sampled from CIFAR-10. Could the authors elaborate on the impact of this initial sampling? How would initializing the canaries with entirely random noise affect the optimization process and their final effectiveness?

3. This paper uses a fixed canary set size of m=1000. While effective for the tested scenarios, the scalability of the metagradient optimization to a significantly larger number of canaries remains unexplored. It would be beneficial to understand if there is a practical limit to the number of canaries that can be effectively optimized.

4. This paper mainly presents quantitative results. A deeper analysis or visualization of the optimized canaries would provide mechanistic insights into why they are so effective. For example, showing how their appearance or features differ from regular canaries.

### Soundness
3

### Presentation
3

### Contribution
3
