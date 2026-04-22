# ZeroSiam: An Efficient Asymmetry for Test-Time Entropy Optimization without Collapse

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 8, 2, 4

## Abstract
Test-time entropy minimization helps adapt a model to novel environments and incentivize its reasoning capability, unleashing the model's potential during inference by allowing it to evolve and improve in real-time using its own predictions, achieving promising performance. However, pure entropy minimization can favor non-generalizable shortcuts, such as inflating the logit norm and driving all predictions to a dominant class to reduce entropy, risking collapsed solutions (e.g., constant one-hot outputs) that trivially minimize the objective without meaningful learning. In this paper, we reveal asymmetry as a key mechanism for collapse prevention and introduce ZeroSiam--an efficient asymmetric Siamese architecture tailored for test-time entropy minimization. ZeroSiam prevents collapse through asymmetric divergence alignment, efficiently achieved by a learnable predictor and a stop-gradient operator before the classifier. We provide empirical and theoretical evidence that ZeroSiam not only prevents collapse, but also regularizes biased learning signals, enhancing performance even when no collapse occurs.  Despite its simplicity, extensive results show that ZeroSiam performs more stably over prior methods using negligible overhead, demonstrating efficacy on both vision adaptation and large language model reasoning tasks across challenging test scenarios and diverse models, including particularly collapse-prone tiny models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Entropy minimization in test-time adaptation often leads collapse into one-hot outputs without learning meaningful features. To alleviate this issue, the paper proposes to employ asymmetric siamese architecture inspired by negative-free self-supervised learning such as BYOL. More specifically, it simply adds a shallow linear layers of which outputs are trained to minimize the divergence with the target branch designed to have no gradients. The effectiveness of the method was demonstrated with extensive experimental results on both image and natural language reasoning benchmark datasets.

### Strengths
1. The proposed method is simple yet effective.
2. It provides some theoretical support that shows convergence behavior.
3. It was validated on extensive experiments for both image and natural language datasets compared to several state-of-the-art methods.
4. Various ablation studies were provided.

### Weaknesses
1. It is not clear why it is beneficial to have large deviation from identity mapping and what exactly imbalance ratio is in Figure 2(a).
2. Why is the inflation of logit L2 norm bad (according to Observation 2) in Figure 2(c)? Is there any theoretical connection between increase of logit norm and generalization ability?
3. What exactly is center dominance in Figure 2(d)? How is it computed? How is it connected to generalization ability in the target domain? 
4. It seems like the experiments were run with one random seed. It is important to have standard deviation to compare significance of the difference between different methods.

### Questions
1. How small is $L_f$ in practice? Wouldn’t it be arbitrary large for the worst case scenario? Also, I don’t quite understand the meaning of “Given that only the affine parameters of the normalization layers are optimized, this assumption is well justified.”
2. Does the theorem hold when we update only on one sample? Or does it hold for any arbitrary number of inputs with random order?

### Soundness
3

### Presentation
2

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
This paper proposes ZeroSiam, a method inspired by asymmetric self-supervised learning to address the collapse problem of entropy minimization-based approaches, where all predictions degenerate into a single one-hot vector regardless of the input. By introducing an asymmetric structure, ZeroSiam eliminates the need for additional augmentations and enhances efficiency through a single forward pass, effectively preventing model collapse.

### Strengths
- The paper presents extensive experimental results across a wide range of benchmarks. In particular, the experiments on the blind spot adaptation effectively demonstrate the limitations of entropy minimization methods when faced with incorrect initial predictions, thereby strengthening the motivation and necessity for the proposed method.
- The paper addresses an important issue in test-time adaptation by improving inference time through a single forward pass without any augmentation. The utilization of all samples and the reduction in inference time enhance the practicality of the method for real-world applications.
- The visualization of model collapse in entropy minimization (Figure 2) is clear, and the visualization of recovery possibility (Figure D in Sup. material) is intriguing and provides additional insight.

### Weaknesses
- Although the proposed method is shown to be applicable under various learning rates, the comparison may not be entirely fair since it also updates affine parameters similar to previos methods but applies a higher learning rate. For instance, in Table 5, it is unclear how the learning rate for LoRA was determined, as the optimal value for the proposed method could differ due to the influence of the projector module.
- While the efficiency achieved through a single forward pass is acknowledged, there are concerns regarding the novelty of the approach compared to prior works that also draw inspiration from self-supervised learning and apply it to TTA. For example,\
[1] Self-Bootstrapping for Versatile Test-Time Adaptation, ICML 2025\
[2] Ranked Entropy Minimization for Continual Test-Time Adaptation, ICML 2025\
[3] Test-Time Ensemble via Linear Mode Connectivity: A Path to Better Adaptation, ICLR 2025\
Furthermore, [1] employed the self-supervised learning and [2, 3] addressed the model collpase problem. 
- Some typos should be revised. For example, u^r in eq(2) and u^o in eq (3) are not same to softmax(u_r) and softmax(u_o) (line 190), resepctively.

### Questions
- In Figure 4, the sensitivity analysis of the learning rate is conducted using ViT-Base, where collapse does not occur seriously, making it difficult to fairly assess the robustness of learning rates across different approaches. 
- Moreover, different learning rates are applied to the predictor depending on the architecture, and the hyperparameters of existing methods are not fully optimized, which complicates fair comparison. If the predictor’s learning rate is treated as a tunable hyperparameter, wouldn’t it also be possible to alleviate the collapse problem in other entropy minimization methods by appropriately adjusting parameters such as the entropy threshold or learning rate?

### Soundness
4

### Presentation
3

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
This paper addresses the problem of "model collapse" in test-time entropy minimization, where models adapting during inference can degenerate to predicting a single dominant class. The authors propose ZeroSiam, a lightweight, asymmetric Siamese architecture inspired by self-supervised learning methods like SimSiam. The key idea is to process a feature from a single encoder pass through two branches: an "online" branch, which includes a learnable predictor before the classifier, and a "target" branch, which uses a stop-gradient on the original classifier output. The model is then trained to minimize the entropy of the online branch while simultaneously aligning its output distribution to the stop-gradient target branch. The authors claim this approach efficiently prevents collapse without requiring augmentations or extra backbone passes5. This claim is supported by theoretical analysis and extensive empirical results showing stable performance on vision-based test-time adaptation (TTA) tasks.

### Strengths
S1. The primary strength of this paper is the asymmetric, predictor-stop-gradient architecture (inspired by negative-free SSL) to the problem of test-time entropy minimization. While the components (predictor, stop-gradient) are not new, adapting this mechanism to stabilize a single-branch, entropy-driven objective is an architectural contribution.

S2. The authors demonstrate the method's effectiveness beyond standard vision TTA by applying it to incentivize reasoning in large language models.

S3. The organization and the writing of the paper are good. The motivation of the proposed method is natural. The paper is easy for reader to follow.

### Weaknesses
W1. The most significant weakness of this paper is the misrepresentation of the primary baseline, TENT. The original TENT paper and its open-source implementation are designed for online adaptation over a limited number of epochs (e.g., 1-5). It is well-understood that adapting indefinitely on an unlabeled, out-of-distribution data stream is inherently unstable. This paper, however, appears to run TENT for an excessive number of steps to induce the very collapse it aims to solve. This creates a "strawman" argument. The plots showing TENT's accuracy dropping to near-zero are a consequence of this unfair experimental setup, not necessarily an inherent flaw of the TENT method under its intended operating conditions. This experimental choice undermines the validity of the central empirical claims.


W2. The paper's motivation states that "minimizing entropy alone does not ensure that the corresponding supervised loss... is also reduced". This is a trivial observation. The lack of ground-truth labels is the fundamental challenge of all unsupervised test-time adaptation. This framing slightly overstates the conceptual contribution by presenting a foundational premise of the subfield as a problem that this work is uniquely identifying.

W3. The alignment weight $\alpha$ is a critical hyperparameter that balances the original entropy objective with the new alignment regularizer. The paper fixes this to 1, claiming it works well without tuning24. While Table F provides a sensitivity analysis, this trade-off seems under-explored. It is likely that the optimal balance between exploration (entropy minimization) and stability (alignment) depends on the severity of the domain shift, which is hard to know beforehand.

### Questions
Q1. Could the authors provide a direct comparison (e.g., for Table 3 or 4) where both TENT and ZeroSiam are run for only a small, fixed number of epochs (e.g., 1, 3, and 5) over the entire test set? 

Q2.  Does the alignment loss $D(p^o || sg[p^r])$ also grow indefinitely, eventually overpowering the entropy term $H(p^o)$? A plot of the individual loss components over time would be very insightful.

### Soundness
2

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
ZeroSiam introduces a lightweight, low‑cost asymmetric architecture for test‑time adaptation by inserting a single linear predictor before the classifier and pairing it with a stop‑gradient target branch, enabling adaptation with only one encoder pass. Its test‑time loss function minimizes entropy on the online branch while aligning it to the stop‑grad branch via a symmetric‑KL term, which discourages shortcut solutions such as logit‑norm inflation and single‑class collapse. Extensive experiments across diverse backbones (ResNet, ViT, ConvNeXt, Swin) show consistent gains under mixed corruptions, label shifts, and batch‑size‑1 streams. This paper also shows improvement in LLM reasoning experimentally.

### Strengths
- This paper delivers stable improvements over strong TTA baselines under mixed corruptions, online label shifts, batch‑size‑1, and even blind‑spot adaptation. Furthermore, it includes LLM mathematical reasoning underscoring concrete experimental cases.

 - The network modification is minimal, instead it requiresa single linear predictor plus a stop‑grad branch. In contrast, it does not depend on augmentations or extra encoder passes. Thus, Its runtime equals to baselines.

### Weaknesses
- This paper lacks a comprehensive understanding of representation adaptation to test-time scalings and the preservation of pre-training knowledge. The proposed loss function, which minimizes the difference from pre-trained knowledge, is driven by KL Divergence. While the paper asserts that $\alpha = 1.0$ is applicable across all experimental cases, I believe that a deeper understanding and explanation of representations are crucial for forming more concrete and reasonable evidence, rather than elaborating numeric numbers across experimental cases.

- I believe that the theoretical analysis presented in this paper doesn’t align with the empirical contribution. The analysis focuses on entropy minimization and KL Divergence separately. Specifically, it fails to accomodate which dynamic occurs during the training process when two terms are considered at the same time. When applied to test-time adaptation experiments, these concepts serve as loss functions simultaneously. Consequently, I believe that this analysis hinders the accurate understanding of the proposed loss function.

### Questions
- I was wondering if ZeroSiam achieves consistant performance when it negates predictor $h$ and employ only classifier $g$. It would be more favorable to highlight the proposed loss function without any modification of networks.

### Soundness
2

### Presentation
3

### Contribution
2
