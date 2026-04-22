# Neural Mutual Information Estimation in Real Time via Pre-trained Hypernetworks

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 6, 4

## Abstract
Measuring statistical dependency between high-dimensional random variables is
fundamental to data science and machine learning. Neural mutual information
(MI) estimators offer a promising avenue, but they typically require costly test-
time iterative optimization for each new dataset, making them impractical for
real-time applications. We present *FlashMI*, a pretrained, foundation model-like
architecture that eliminates this bottleneck by directly inferring MI in a single
forward pass. Pretrained on large-scale synthetic data covering diverse distributions
and dependency structures, *FlashMI* learns to identify distributional patterns and
predict MI directly from the input dataset. Comprehensive experiments demonstrate
that *FlashMI* matches state-of-the-art neural estimators in accuracy while achieving
100× speedup, can seamlessly handle varying dimensions and sample sizes through
a single unified model, and generalizes zero-shot to real-world tasks, including
CLIP embedding analysis and motion trajectory modeling. By reformulating
MI estimation from an optimization problem to a direct inference task, *FlashMI*
establishes a practical foundation for real-time dependency analysis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes FlashMI, a pretrained neural architecture for real-time mutual information (MI) estimation. Traditional neural MI estimators (e.g., MINE, InfoNCE) require dataset-specific iterative optimization, which makes them impractical for large-scale or streaming scenarios.
FlashMI reformulates MI estimation as a direct inference task instead of an optimization problem. A hypernetwork generates critic parameters in a single forward pass, enabling efficient estimation.
The model adopts a dual-path architecture (joint/marginal branches) with cross-attention mechanisms consistent with the Donsker–Varadhan formulation. FlashMI is pretrained on large-scale synthetic distributions covering diverse dependency and marginal patterns, enabling zero-shot generalization to unseen data.

### Strengths
1.The main contribution is conceptually elegant—the paper redefines MI estimation as a one-step inference problem instead of an optimization task. This paradigm shift offers both theoretical insight and practical value for scalable dependency estimation.
2.he dual-path hypernetwork with cross-attention effectively models relationships between joint and marginal distributions. The inclusion of a noise-padding mechanism enables flexible handling of different input dimensions, enhancing robustness and adaptability.
3.FlashMI achieves substantial computational gains without sacrificing estimation accuracy, making it particularly appealing for real-time and streaming applications.

### Weaknesses
1.Limited evaluation on high-dimensional data:
Current experiments focus on low- to mid-dimensional inputs (up to ~20D). The paper mentions potential scaling via slicing or fine-tuning, but provides no empirical evidence or analysis of high-dimensional performance trends.
2.Dependence on synthetic pretraining:
The model’s success heavily depends on the diversity of its synthetic pretraining data. However, the paper does not quantify the variety or coverage of these distributions, making it difficult to assess robustness across real-world domains.

### Questions
1.Is the critic parameter generation theoretically guaranteed to approximate the optimal critic, or is it purely empirical?
2.How are the synthetic pretraining distributions designed to capture real-world dependency patterns? Are there metrics to measure their diversity or coverage?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a core bottleneck of existing neural mutual information (MI) estimators: the need for costly and time-consuming iterative optimization for each new dataset at test time. To overcome this, the authors propose FlashMI, a pre-trained, foundation-model-like architecture. FlashMI reformulates MI estimation from an "optimization problem" to an "inference problem." At its core is a dual-path, attention-based Hypernetwork. This hypernetwork takes the entire dataset (as a sequence of samples) as input and directly generates the optimal parameters for the "critic network" from the Donsker-Varadhan (DV) formulation in a single forward pass. FlashMI is pre-trained on large-scale, diverse synthetic data (covering various distributions and dependency structures), allowing it to learn general distributional patterns. The model flexibly handles varying input dimensions (via noise padding) and sample sizes (via attention). Extensive experiments demonstrate that FlashMI matches state-of-the-art (SOTA) neural estimators in accuracy while achieving over 100x speedup. Furthermore, it successfully generalizes in a zero-shot manner to real-world applications, such as CLIP embedding analysis and motion trajectory modeling, showing its significant potential as a tool for real-time dependency analysis.

### Strengths
1.	Exceptional Efficiency: The key strength is the 100x+ speedup by replacing per-dataset optimization with a single forward pass. This makes real-time neural MI estimation practical.
2.	Novel Architecture and Generalization: The dual-path hypernetwork is an innovative design that handles variable inputs. Its effectiveness is proven by the strong zero-shot generalization from synthetic training to real-world tasks (e.g., CLIP, motion data), which is a significant result.
3.	Comprehensive Evaluation: The experimental design is rigorous, testing against optimization-based, pre-training-based, and traditional MI estimators.

### Weaknesses
1.	The method still relies on slicing for high-dimensional data (e.g., 512-dim), which is an approximation and limits the core method's direct applicability in such settings.
2.	The model's impressive inference speed comes at the cost of a very high pre-training budget (hardware and time). Its generalization is also entirely dependent on the diversity of the synthetic pre-training data.

### Questions
- In the CLIP experiment (512-dim), the caption for Figure 4 mentions "5-sliced MI using 25 random projections". How does this work exactly? Does this mean $k=5$ (as in the k-sliced MI definition) or $S=25$ (the number of projections)?

-  What is the practical value of the max dimension $D$ mentioned on page 6? (Appendix A.1, Alg 1 seems to imply $d_{max}=8$). Please clarify the practical $D$ used and how k-slicing works with it.

- Figure 3 shows that performance (AUC) drops noticeably for small sample sizes (e.g., $n < 400$). How does FlashMI's robustness in this low-sample regime compare to MINE (which can optimize specifically for that small dataset)? Does the hypernetwork need to "see" a sufficient number of samples to accurately infer the distribution's properties?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new mutual information (MI) estimation method based on neural networks. More specifically, the proposed method aims to compute MI in real time. The main bottleneck in this problem is the time required for MI estimation. To address this issue, the authors propose using a frozen model that is separately trained on synthetic data. After training with synthetic data, mutual information is estimated using the DV representation. Through experiments, the paper shows that the proposed method compares favorably with existing approaches.

The idea of using synthetic data is interesting. However, a similar approach could be implemented simply by extending the MINE method with synthetic data. Moreover, it lacks to comparing to important previous work. Therefore, I feel that the novelty may lie more in the model architecture than in the overall framework itself. This point should be carefully verified.

### Strengths
1. The idea of using synthetic data for pretraining would be interesting.
2. The proposed method is much faster than existing methods.

### Weaknesses
1. DV based representation learning for mutual information is not new. For example, the following paper has already worked on MI based representation learning. 

   Neural Methods for Point-wise Dependency Estimation, NeurIPS 2020.

2. Although the synthetic data pre-training is interesting, the approach is used in the computer vision community. Using it for mutual information is new, but it is not significantly novel.

### Questions
1. Regarding independence testing, can the proposed method control the false positive rate?

2. It seems possible to train MINE with synthetic data and then use the trained model to estimate MI, similar to the proposed method. Would the proposed approach still outperform MINE in this setting?

3. Similar to Q4, I feel that the novelty of this paper mainly lies in the proposed model architecture. Could the authors provide an ablation study to support this claim?

### Soundness
3

### Presentation
3

### Contribution
2
