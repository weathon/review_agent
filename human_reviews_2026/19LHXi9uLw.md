# The Lie of the Average: How Class Incremental Learning Evaluation Deceives You?

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4, 6

## Abstract
Class Incremental Learning (CIL) requires models to continuously learn new classes without forgetting previously learned ones, while maintaining stable performance across all possible class sequences. In real-world settings, the order in which classes arrive is diverse and unpredictable, and model performance can vary substantially across different sequences. Yet mainstream evaluation protocols calculate mean and variance from only a small set of randomly sampled sequences. Our theoretical analysis and empirical results demonstrate that this sampling strategy fails to capture the full performance range, resulting in biased mean estimates and a severe underestimation of the true variance in the performance distribution. We therefore contend that a robust CIL evaluation protocol should accurately characterize and estimate the entire performance distribution. To this end, we introduce the concept of extreme sequences and provide theoretical justification for their crucial role in the reliable evaluation of CIL. Moreover, we observe a consistent positive correlation between inter-task similarity and model performance, a relation that can be leveraged to guide the search for extreme sequences. Building on these insights, we propose **EDGE** (Extreme case–based Distribution \& Generalization Evaluation), an evaluation protocol that adaptively identifies and samples extreme class sequences using inter-task similarity, offering a closer approximation of the ground-truth performance distribution. Extensive experiments demonstrate that EDGE effectively captures performance extremes and yields more accurate estimates of distributional boundaries, providing actionable insights for model selection and robustness checking. Our code is available at https://github.com/AIGNLAI/EDGE.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper is about fixing misleading evaluation in class incremental learning (CIL). The authors propose EDGE, an extreme-case-based protocol that samples hardest, easiest, and random class sequences guided by inter-task similarity computed from CLIP text embeddings.

### Strengths
1. The paper reframes CIL evaluation from mean/variance over a few random orders to estimating the full performance distribution.
2. EDGE remains effective across datasets (CIFAR‑100, CUB‑200, ImageNet‑R), backbones (ResNet, ViT), and CLIP encoder sizes.
3. This paper is fluently written and easy to read.

### Weaknesses
1. Missing baselines.  RS is treated as a weak baseline (3 sequences with fixed seeds), while more principled sampling exists (stratified, Latin hypercube, coverage maximization, Bayesian optimization).

### Questions
In the EDGE protocol design section, it mentions that the extreme sequence generation algorithm uses CLIP text encoding to construct class similarity matrices. Could you provide more details on how this process works and how it contributes to generating high and low similarity sequences? Additionally, what are some potential limitations or challenges associated with using CLIP for this purpose?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper critiques the evaluation protocol in Class Incremental Learning (CIL), which relies on averaging results over a few random class-order sequences. The authors theoretically and empirically demonstrate that such Random Sampling (RS) severely underestimates variance and produces biased means due to the factorial explosion of possible class sequences. To address this, they propose EDGE (Extreme case–based Distribution & Generalization Evaluation), a new evaluation protocol that identifies easy and hard cases using inter-task similarity derived from CLIP text embeddings. They claim that, evaluating models on these extreme and median sequences better approximates the true performance distribution.

### Strengths
1. Challenges a long-standing but under-scrutinized assumption in CIL benchmarking, introducing a valuable distributional perspective.
2. Provides clear theorems quantifying the inadequacy of random sampling.
3. Demonstrates consistency across model types (pre-trained and non-pre-trained) and multiple datasets.

### Weaknesses
1. Lack of real-world validation: Although the authors repeatedly motivate their work through real-world continual learning scenarios (e.g., autonomous driving, uncontrolled class emergence), all experiments are performed on simple disjoint data streams. They do not include any real-world or realistic task-order settings, such as those in Real-World Continual Learning[1] or I-Blurry[2]/Si-Blurry[3] setups, which explicitly simulate gradual or overlapping task transitions.
2. Dependence on CLIP embeddings: The approach assumes that CLIP’s text-space similarity correlates with inter-task difficulty, which may not hold in all domains (e.g., non-visual or specialized data).
3. Edge cases unexplored: The paper could discuss scenarios where inter-task similarity fails to predict performance variance (e.g., highly regularized models).
4. Overlooked fairness implications: Focusing on extreme sequences might overemphasize worst-case behavior, potentially penalizing stable models that perform consistently across moderate task orders.

citations: 
[1]Online Class-Incremental Learning For Real-World Food Image Classification (Siddeshwar Raghavan, Jiangpeng He, Fengqing Zhu)
[2] Online Continual Learning on Class Incremental Blurry Task Configuration with Anytime Inference (Hyunseo Koh, Dahyun Kim, Jung-Woo Ha, Jonghyun Choi) 
[3]Online Class Incremental Learning on Stochastic Blurry Task Boundary via Mask and Visual Prompt Tuning (Jun-Yeong Moon, Keon-Hee Park, Jung Uk Kim, Gyeong-Moon Park)

### Questions
1. How robust is the inter-task similarity heuristic across modalities or non-visual datasets? Could EDGE generalize beyond CLIP-based text embeddings?
2. For large-scale settings where enumerating even three sequences per dataset is costly, how would EDGE be adapted or approximated?
3. Does emphasizing extreme sequences risk over-penalizing models optimized for stability rather than adaptability?

I am willing to change my review if the concerns are addressed.

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
This paper critiques the current evaluation methodology used in class-incremental learning (CIL), where a small number of randomly sampled class orders are used to estimate model performance. The authors argue that this common practice does not accurately approximate the true expected performance across all possible class sequences. To address this, the paper proposes an improved evaluation approach designed to more reliably estimate true model performance.

### Strengths
- This paper investigates an evaluation issue in CIL that is often overlooked in the literature, which is an interesting direction for the CL community.

- The paper is clearly written and well structured, with both theoretical and empirical analysis supporting the proposed evaluation method.

- Highlighting the impact of class ordering contributes valuable insight toward more trustworthy benchmarking in CIL.

### Weaknesses
- The motivation of the paper raises concerns. The introduction uses real-world scenarios (e.g., self-driving cars) where the learner does not have control over class arrival order. However, the proposed method assumes class ordering can be controlled or enumerated for evaluation, which conflicts with the motivation.
- Although the common RS (Random Sampling) evaluation does not estimate the true model accuracy, it does enable fair comparison among methods if all algorithms are evaluated using the same sampled sequences. Since CIL benchmarks are used for comparison of CIL algorithms rather than real-world deployment, the evaluation of existing literature is not entirely unreasonable.
- A key practical question would be whether a method that performs well on difficult sequences also performs well on easy sequences. A more actionable direction may be benchmarking both cases and identifying algorithms that consistently perform well regardless of class order.
- The existence of hard vs. easy class sequences is already known in CIL. Thus, the novelty of the paper is mainly limited to proposing a new evaluation metric. Without demonstrating an algorithm that improves under the new metric, the contribution may feel incomplete.
- The experiments show limited benefit in the practical application of ranking existing CL algorithms (referring to Table 2).  
  - Example: CIFAR-100 (hard sequence): both True and RS rankings identify iCaRL as best. Here, EDGE does not contribute any new information in terms of ranking existing CL algorithms
  - Example: ImageNet-R (easy sequence): True says iCaRL as best, but RS and EDGE disagree. Here, neither estimator consistently reflects the true ranking.

### Questions
- For experiments using pre-trained models, what dataset is used to pre-train the backbone?
- Why are RS and EDGE results very close in Table 1? Does this imply that EDGE only offers minor improvements over RS in most realistic cases?
- Can methods that perform well on difficult sequences also automatically perform well on easy sequences? If not, what may be the reason behind it?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on the evaluation protocol of continual learning. Under the standard evaluation protocol, Random Sampling (RS) tends to be unfair, as it emphasizes the mean rather than the deviation. Consequently, the EDGE protocol is proposed for improved sampling of task sequences. First, the author identifies that the difficulty of the continual learning (CL) sequence is highly correlated with inter-task similarity. Therefore, EDGE samples "easy" and "hard" sequences based on the similarity calculated using the CLIP model. Experimental results show that EDGE provides a better estimation of the performance range of CL methods.

### Strengths
1. The proposed relationship between task similarity and difficulty is intuitive and interesting.
2. Under the existing setup (3 runs for each method), the proposed EDGE can be much stronger compared with RS.
3. The source code is partially included in the submission.

### Weaknesses
1. It is true that recent continual learning (CL) papers consistently use Random Sampling (RS) in their evaluations. However, the experimental setting of this manuscript assumes that only three runs are used for the evaluation, which does not fully simulate a realistic Class Incremental Learning (CIL) evaluation protocol. According to basic statistical theory, the estimation provided by the RS protocol will be much closer to reality if more runs are employed. As this paper focuses on the fairness of the evaluation protocol, it is essential to consider scenarios where the number of runs exceeds three for a factual and equitable comparison, and to examine how performance differences scale with an increased number of runs.
2. EDGE fails to account for cases where analytic continual learning methods are evaluated, such as GACL [A] and RanPAC. The performance of these methods is usually minimally impacted by task order. This may explain the similarity in RS evaluation for RanPAC and the EDGE evaluation, despite the fact that RanPAC fine-tunes the backbone using the training samples.
3. As an evaluation protocol, EDGE only takes final accuracy into consideration. However, it is well known that final accuracy is not the sole important factor. To evaluate the dynamics of the CL learner, metrics such as anytime inference [B] and plasticity/forgetting metrics are also vital, yet these are unfortunately not discussed within the proposed metrics.
4. Methods such as ConvPrompt [C] and SEMA [D] take inter-task similarity into account. Including such methods in the benchmarking would be beneficial.

References:

    [A] Zhuang, Huiping, et al. "GACL: Exemplar-free generalized analytic continual learning." Advances in Neural Information Processing Systems 37 (2024): 83024-83047.
    [B] Koh, Hyunseo, et al. "Online continual learning on class incremental blurry task configuration with anytime inference." arXiv preprint arXiv:2110.10031 (2021).
    [C] Roy, Anurag, et al. "Convolutional prompting meets language models for continual learning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
    [D] Wang, Huiyi, et al. "Self-expansion of pre-trained models with mixture of adapters for continual learning." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
For the paper to be deemed acceptable, I believe the following question is crucial:
1. How do Random Sampling (RS) and EDGE perform with an increased number of sampled runs?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper discusses the evaluation protocol of the class incremental learning and shows that the widely used random sampling protocol fails to capture the full distribution of model performance. To solve this problem, the authors propose an evaluation framework, EDGE, that finds extreme class sequences based on inter-task similarity computed via CLIP text embeddings. Both theoretical analysis and empirical experiments demonstrate that EDGE offers a more accurate estimation of performance distribution on CIFAR-100, CUB-200, and ImageNet-R, EDGE better captures upper/lower bounds and reduces distribution distance versus RS, remaining robust across different model backbones and CLIP encoder sizes.

### Strengths
- This paper addresses the limitations of evaluation protocols in CIL, a critical area gaining increasing attention.
- The theoretical analysis is clear and robust, highlighting the fundamental limitations of random sampling (RS) protocols.
- The proposed method is straightforward and easy to implement. Utilizing CLIP-based label embeddings to evaluate inter-task similarity appears appropriate for most tasks.
- The comparison of RS and EDGE is analyzed from various perspectives, emphasizing the limitations of random sampling.
- Both non-pretrained and pretrained CIL methods were assessed using two evaluation protocols (random sampling & EDGE), offering valuable insights for researchers.

### Weaknesses
- Using CLIP text embeddings for class similarity in non-natural image datasets, like medical imaging or remote sensing, may lead to inaccuracies due to potential misalignment between label semantics and visual similarity.

- The comparison relies solely on random sampling, without addressing other potential baseline evaluation methods. 

- The assumption that inter-task semantic similarity perfectly correlates with task difficulty may not be universally valid, particularly in areas where semantic similarity fails to encompass the full complexity of tasks.

- The application scope of this evaluation protocol requires further discussion. Specifically, how crucial is it to capture performance distribution and absolute accuracy compared to solely focusing on relative model ranking? Is this type of evaluation necessary for all continual learning scenarios, or is it primarily relevant for certain deployment-critical applications?

### Questions
1) The paper emphasizes that RS leads to inaccurate variance estimation and poor coverage of extreme cases, and that EDGE improves the estimation of the performance distribution. While this is well motivated, it raises a fundamental question about what the community should prioritize in continual learning evaluation. In many continual learning scenarios, our focus may be more on relative model ranking rather than on absolute accuracy metrics.

2) The proposed framework relies on the assumption that semantic similarity between class labels, derived from CLIP text embeddings, correlates with inter-task similarity and reflects task difficulty. How robust is this assumption beyond natural image datasets? In domains such as medical imaging or synthetic datasets, label names may inadequately represent visual or distributional similarities between tasks.

3) The framework may be vulnerable to data contamination, especially since both the evaluated continual learning (CL) methods and the CLIP model used for task similarity computation are pre-trained on extensive web-scale datasets. Have the authors considered how data contamination—resulting from either model pre-training or benchmark overlap with CLIP—could affect task similarity estimates and, in turn, impact the reliability of the EDGE protocol?

4) When evaluating continual learning methods that are pre-trained (e.g., on CLIP or similar datasets), it's important to note that if the downstream datasets overlap with the pre-training corpus, task difficulty may be misrepresented. Tasks with greater overlap might seem "easier" not due to actual similarity, but because of data leakage from the pre-training phase.

5) How does RS improve with larger sample sizes? Specifically, how many RS sequences are required for RS to match EDGE in distribution approximation or ranking stability?

### Soundness
3

### Presentation
3

### Contribution
3
