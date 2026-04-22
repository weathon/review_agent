# Real-Time Evaluation for Novel Class Discovery at Test Time

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
We introduce Test-Time Discovery (TTD), a real-time evaluation protocol for novel class discovery under sequential test-time conditions. Unlike post-hoc NCD evaluation, which assesses clustering only after the full test set is processed, TTD requires models to classify known categories and discover novel ones in real time as samples arrive. To address this setting, we propose a training-free Hash Memory (HM) method. HM encodes feature norm and direction into semantic-aware hash codes, enabling Locality-Sensitive Hashing (LSH) for efficient retrieval and consistent reuse of discovered classes. A global-to-local strategy combines prototypes for stable known-class predictions with memory-based reasoning for flexible novel discovery. A lightweight self-correction mechanism further improves reliability by removing mislabeled entries from early discoveries. Experiments on diverse benchmarks show that HM achieves more accurate and stable real-time discovery than NCD and TTT methods, while maintaining performance on known classes. Our code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes the task of test-time discovery, i.e. generalized category discovery (classifying known categories and identify new ones) where at test time data is processed online, making clustering more difficult. To tackle this task the paper proposes a hash memory method that uses locality-sensitive hashing and a global-to-local strategy for efficient retrieval and novel class formation, with a lightweight self-correction mechanism.

### Strengths
- The focus on a online testing setting for novel class discovery is interesting and makes the problem more suitable for different scenarios
***

- The experiments exploring the effect of the number of unknown classes and different test-time distributions are appreciated. this analysis adds some empirical depth beyond a single evaluation setup.

### Weaknesses
- relation to prior work and conceptual clarity
- The paper does not sufficiently situate its contribution within prior work. It is unclear whether similar approaches have already been explored in the GCD or related literature. The related work sections and problem definition settings do attempt to do this but it remains unclear
-  For instance, the intro states "In TTD, the model must not only classify known categories but also decide in real time whether a sample belongs to a previously discovered class or should initiate a new one.". This is exactly what the problem of generalized category discovery addresses. This is also not mentioned when GCD is described in the related work. It is made clearer in the comparison in Table 1 but should be clear from the start.
- The relation to PHE (Zheng et al., 2024) is unclear. That work also claims an online or real-time discovery setting, in contrast to how it is represented in Table 1. It is also missing from the related work despite clear conceptual overlap.
- PHE is compared to in the experiments, however the conceptual difference should be made clear
***

- comparisons to state-of-the-art
- related to the above point since the main difference between GCD and the proposed task is the online testing it seems that GCD methods should be compared to, at least in the post evaluation setting
- it would also strengthen the papers claims to compare to GCD methods in the online setting by iteratively testing as the test set expands to demonstrate that GCD methods are either much worse in this setting or too computationally expensive.
***

- experimental design and fairness
- The choice of datasets and the 7:3 known–unknown class ratio is not well justified. Given the similarity to GCD or on the fly category discovery (OCD), it would be more convincing to adopt standard datasets and splits used in priors works. Some of the datasets are the same but it isn't clear why tiny-imagenet is chosen over imagenet-100 and why Stanford cars isn't used. Prior works also tend to use a 1:1 ratio of known to unknown classes.
- The new evaluation metrics introduced in the paper seem to obscure rather than clarify the relationship between GCD/OCD and the proposed setting. While some new evaluation is reasonable, several metrics appear to be renamed or slightly altered versions of existing GCD/OCD metrics, making interpretation difficult.
***

- ablation study
- The ablation study is limited, evaluating only the presence of the hash memory and self-correction components. Key elements such as the graph-augmented retrieval or the global-local hybrid strategy are not analyzed.
- The reported ablation results are unconvincing. While cluster agreement improves, both true-label agreement and known-class accuracy (KA) decline. With the issue mentioned above, its hard to interpret these metrics and weight which is more important and evaluate whether the proposed components are actually benefitting the model. 
***

- presentation
- The paper relies heavily on acronyms, which makes the results and reasoning difficult to follow in places.

### Questions
- How does your approach relate to PHE (Zheng et al., 2024), which also performs online or real-time discovery? The distinction is unclear from both the related work and Table 1.
***

- How does the proposed test-time discovery (TTD) setting fundamentally differ from Generalized Category Discovery (GCD) beyond the sequential or streaming data assumption? Why are GCD baselines not included in the post evaluation comparisons?
***

- Why were the chosen datasets and the 7:3 known–unknown ratio used? Would the method still perform well on standard GCD datasets and splits?
***

- Several of the proposed evaluation metrics seem to be renamed or modified versions of standard ones. Can you clarify the link between existing metrics and new metrics and justify why new terminology was introduced instead of extending existing metrics?
***

- The ablation omits analysis of several important components (e.g., graph-augmented retrieval, global-local hybrid strategy). Could you provide further justification or additional experiments?
***

- In Table 7, cluster agreement improves but true-label agreement and KA decrease. What does this trade-off imply about the quality of the discovered clusters?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Test-Time Discovery (TTD), a new protocol for real-time novel class discovery that requires models to simultaneously classify known categories and identify emerging ones under sequential test-time constraints. The authors propose a training-free Hash Memory (HM) framework that combines semantic-aware hashing of feature norms and directions, a cooperative inference strategy integrating global prototypes and memory-based reasoning, and a self-correction mechanism to refine mislabeled samples. Experimental results across multiple benchmarks show that HM achieves more accurate and stable real-time discovery than prior NCD and TTT methods while maintaining strong performance on known classes.

### Strengths
- First of all, the paper introduces Test-Time Discovery (TTD), a sequential real-time evaluation protocol that bridges the gap in existing NCD approaches by jointly measuring classification and discovery.
- The proposed Hash Memory combines semantic-aware hashing, graph-augmented retrieval, and self-correction into a simple yet effective training-free design for real-time novel class discovery. Especially, a hybrid strategy integrates a global prototype classifier for confident known samples with an LSH-based local voting mechanism for ambiguous or novel ones, ensuring stability on known classes while enabling flexible discovery.
- Comprehensive experiments analyze real-time vs. post-hoc metrics (KA, KF, TA, CA), parameter sensitivity, and robustness under varied memory sizes, sample orders, and data distributions, consistently outperforming NCD and TTT baselines across benchmarks.

### Weaknesses
- A first concern is an inherent ambiguity in the term real-time evaluation. It is unclear whether it refers to quantitative latency and memory overheads for hashing, neighbor retrieval, and SC updates.
- The system relies on multiple heuristic mechanisms, making it difficult to clearly attribute performance gains to individual modules and to isolate their true effectiveness within the overall framework. 
- The provided results are limited to relatively short sequential test streams with a small number of discovered classes, leaving it unclear how the method would perform under long-term discovery or class reoccurrence which are central challenges in realistic continual open-world scenarios.

### Questions
- How does it estimate or calibrate prediction confidence to prevent early pseudo-label errors from propagating?
- What is the computational and memory complexity of the hash-based graph retrieval and self-correction steps as the number of discovered classes grows, and can the method remain real-time in longer streams?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper deals with the problem of discovering novel classes in real-time test scenarios. Along with discovering novel samples, the task is also to maintain the performance of known classes. To tackle the challenge of online discovery, the proposed work maintains a hash memory of feature norm and direction. A new class is discovered by querying the buckets. For robustness against noisy data, the top-k neighbors of a bucket are augmented while applying discovery process. For the final decision, global prototypes and hash memory are leveraged to ensure performance in both known and novel classes. Finally, to purify misassigned samples for novel discovered classes, a memory self-correction mechanism is applied. Experiments are conducted in 4 datasets.

### Strengths
- Real-time discovery of novel classes is a practical setting of NCD, as introduced by the paper. The problem of TTD is well-motivated by highlighting rediscoveries of novel classes from existing solutions that focus on non-real-time-based postdoc evaluation.
- The paper gives a good overview of the related works regarding NCD and TTT.
- The proposed method is practical for real-time discovery problem.

### Weaknesses
- Will the feature norm and directions of the first identified sample of a novel class be representative to identify all the future samples of the same class? As we see more samples, do we need aggregation?
- Some of the details in the paper are missing. For example: How to construct the dynamic graph? What is the frequency of updating the graph?
- The notation k is used for multiple purposes. It is recommended to use distinct notations for specific purposes.
- What is the value of ε, α? How to determine the values?
- If the graph is not used, how does it affect the performance?  Also, how does k in the top-k of the graph neighborhood impact the performance?
- How to determine the optimal size of stored samples in the self-correction module?

### Questions
Please refer to the weakness section

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
4

### Summary
The paper introduces Test-Time Discovery (TTD), a real-time evaluation framework for novel class discovery, along with a training-free Hash Memory (HM) method that integrates semantic-aware hashing, a global-to-local prototype-to-LSH classifier, and a lightweight self-correction mechanism. TTD focuses on per-sample decision-making during streaming evaluation rather than relying on post-hoc clustering. Experiments on CIFAR100D, CUB-200D, Tiny-ImageNetD, and AircraftD demonstrate consistent improvements on new real-time metrics (TA and CA) while maintaining stability on known classes.

### Strengths
- The TTD protocol effectively reveals practical issues often overlooked in post-hoc NCD and clearly identifies three key challenges.
- The method is appealingly simple and efficient at test time. Hash codes represent both feature norm and direction, LSH buckets are reused to avoid redundant rediscovery, and known-class predictions rely on a fallback prototype classifier.
- Overall, the paper is well written and easy to follow.

### Weaknesses
- The semantic-aware hash employs random projections and a norm bin, which is practical but offers limited conceptual novelty compared with existing prototypical hashing or LSH-based retrieval methods. Please provide a clearer positioning against closely related online discovery and hashing approaches to highlight the contribution.
- The approach relies on several design choices, such as κ for norm binning, the number of random directions, the number of bucket-graph neighbors (k), memory size (K), EMA factor (α), and the self-correction cadence. The current sensitivity analysis is limited; robustness claims would be more convincing with systematic parameter sweeps and a clearer examination of latency–memory trade-offs under streaming conditions.

### Questions
- How were κ (norm discretization), number of random directions n, bucket neighbor count k, and memory size K chosen? Please provide ranges, validation protocols, and wall-clock latency/throughput under streaming loads.
- Have you tried larger-scale open-world streams (e.g., ImageNet-derived streams with >1k categories) or non-vision modalities?
- Dataset split descriptions are useful; a quick table in the main text (with known/unknown counts and stream length) would aid readability.

### Soundness
3

### Presentation
2

### Contribution
3
