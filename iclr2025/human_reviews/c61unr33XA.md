## Human Reviewer 1

### Summary
This paper explores dataset distillation methods for self-supervised learning (SSL). The authors demonstrate that the MTT method cannot be directly applied to SSL due to high trajectory variance, both theoretically and empirically. To address this, they propose a solution leveraging knowledge distillation (KD) to reduce the length and variance of SSL trajectories.

### Strengths
- Self-supervised dataset distillation is an important yet underdeveloped area with wide-ranging applications.
- The empirical and theoretical evidence provided (Theorem 4.1 and Figure 1) for high gradient and trajectory variance of MTT in SSL settings is convincing and insightful.

### Weaknesses
- The performance improvements over random subset and high-loss subset are marginal in many cases, especially on the larger TinyImageNet dataset (Table 3). This raises concerns about the method’s practical value, as it requires significant computational resources to distill synthetic data while yielding minimal improvement.
- The absence of KRR-ST results in Tables 4, 5, 6, and 7 limits the ability to assess the proposed method's effectiveness, given that KRR-ST is a closely related baseline.
- The MKDT method appears to add only a knowledge distillation process to MTT, where a student model mimics the SSL teacher model’s representations. The synthetic data is then learned by matching the student model’s trajectory, rather than the teacher model’s trajectory, as in MTT. It is unclear why introducing a student model as an intermediary reduces trajectory variance, or what specific role the student network plays throughout the data distillation process. The authors are encouraged to provide further discussion or insights into this mechanism.

### Questions
- Line 295: Incorrect format in “[41] trains student ….”
- The MKDT method uses ResNet-18 as the teacher model and ConvNet as the student model. What is the backbone network used for KRR-ST in Tables 2 and 3?
- Presenting some distilled images would help demonstrate the proposed method’s effectiveness and its advantage over the baseline KRR-ST.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
- This paper propose a data distiallation method for self-supervised pretraining. This methods MKDT is a based on knowledge distiallation method, which is mainly motivated by the study on the gradient's variance of different loss function.

### Strengths
- The first, as this paper claims, effective data distillation method for SSL pretraining is proposed.
- Outstanding experimental performance on provided datasets, \textit{e.g.}, CIFAR-10, 100/Tiny ImageNet.

### Weaknesses
- Not matched theory and implementation. The toy case in theoretical analyses is too simple and the gap between analyses and implementation is too large.
    1. The linear model is provided in the main paper only. However, even in linear probe experiments, the tested model is non-linear. Moreover, in the literature of the related ones (\textit{e.g.}, sharp-aware generalization, bias-variance trade-off, Out-of-Distribution Generalization), analyses about gradient's variance on different loss design are usually conducted on non-linear case (\textit{e.g.}, 2-layer network with non-linear Activation function, such as ReLU in early years).[1,2,3]
    2. The derivative and presentation of its proof is too complicated however the logic is actually trival (i.e., simply unfold the deviation of the gradient of SL and SSL, and then compare each term by \textit{easy to see is larger}).
- Pretraining is important nowadays. However, the model architecture is too small and simple, for which pretraining is not as important as the large models (\textit{e.g.}, LLMs), which is not discussed in the main paper.
    1. In the assumption of the theorem, citation [4] about sparse code is on multi-modal pretraining, but similar architecture is not mentioned in the rest of the paper.
- Experiments about generalization is confusing.
    1. In the distillation setup, it claims that the teacher model is ResNet-18 and the student model is ConvNet. However, the models in the experiments about generalization to Larger architecture are ResNet-18 and ResNet-10. The models are not larger at all, and ResNet-L (L>18) are available in the provided code.
- Typos and inconsistent prensentation, \textit{e.g.}, synhronous in Line 215, pre-training and pretraining.

[1] Estimating Example Difficulty using Variance of Gradients. CVPR 2022
[2] Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization. ICML 2022
[3] Rethinking Bias-Variance Trade-off for Generalization of Neural Networks. ICML 2022
[4] Data-efficient contrastive language-image pretraining: Prioritizing data quality over quantity. AISTATS 2024

### Questions
It seems like that the logic of the paper is: the gradient variance of SSL is larger than the one of SL, thus the knowledge distillation is needed to lower the variance due to the relationship between gradient and trajectory matching. The questions are:
- It's due to the high variance of the SSL gradient, why not show and print the empirical gradient variance directly?
- How is the straightforward relationship between model performance and gradient variance, if albation study on trajectory sampling is considered?

I first give 6 here, actually, I tend to give 5.5. therefore, some important concerns should be discussed during the rebuttal period.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper introduces one of the first approaches to dataset distillation for self-supervised learning (SSL) pre-training, addressing the challenge of generating compact, synthetic datasets that can efficiently pre-train deep networks without labeled data. By leveraging a teacher-student framework, the authors demonstrate that using knowledge distillation (KD) significantly reduces the variance in training trajectories, a common issue in SSL due to high variance in gradient updates. This approach, termed "Matching Knowledge Distillation Trajectories" (MKDT), trains a smaller student model to match the embeddings of a larger teacher model, resulting in a low-variance objective that allows effective dataset distillation for SSL. Experimental results indicate that MKDT outperforms prior methods by up to 13% across various downstream tasks with limited labeled data, underscoring its potential for memory- and compute-efficient SSL pre-training​

### Strengths
1. This work is among the first to address dataset distillation for self-supervised pre-training, providing an innovative approach to generate compact, synthetic datasets that enable efficient pre-training without labeled data.
2. The theoretical motivation for introducing a teacher-student learning model is compelling. By using knowledge distillation, the method effectively reduces the high variance in gradients commonly seen in self-supervised learning objectives, improving upon naive trajectory matching approaches. While I have not throughly checked the the mathematical proofs, but nonetheless, the proofs intuitively seem convincing.
3. The related work section is particularly well-constructed, offering a thorough exploration of the dataset distillation literature and providing valuable insights into the challenges and developments in both supervised and self-supervised settings, positioning this work within the broader context of current research.

### Weaknesses
1. **Limited Discussion of Alternative Distillation Paradigms**: The paper presents knowledge distillation (KD) as a novel proxy for enhancing trajectory matching in SSL dataset distillation. However, the authors omit a comparison with other established dataset distillation methods, such as gradient matching or distribution matching. The Related Work section briefly acknowledges these methods but dismisses them based on label dependency. This lack of comparative analysis leaves a gap: could these paradigms outperform trajectory matching if adapted to SSL? Addressing this gap would strengthen the paper by either justifying the choice of trajectory matching for SSL or empirically showing why KD-based approaches are superior.

2. **High Variance in SSL Gradients and Potential Regularization Techniques**: A core motivation for the KD approach is the high variance in SSL gradients, which makes naive trajectory matching ineffective. Yet, the authors do not explore whether regularization techniques, such as Sharpness-Aware Minimization (SAM), could mitigate this variance issue. SAM, which reduces oscillations in model updates by flattening the loss landscape, could be a promising candidate to stabilize SSL gradients. An empirical analysis incorporating such techniques would clarify whether the KD setup is necessary or if regularization alone could suffice.

3. **Clarification and Takeaways in Figure 1**: Figure 1 illustrates the challenges of applying MTT to SSL, emphasizing issues like high gradient variance and chaotic updates to synthetic images. However, the figure captions lack concise takeaways summarizing the implications of each plot. Clearer captions explaining the significance of variance trends and how they justify the proposed KD approach would improve the paper's readability and impact.

**Minor Weakness (Formatting and Page Limit)**: The submission uses a numerical citation format, which does not adhere to the ICLR 2025 requirements of (Author, Year) format. Given the 10-page length, updating the citation style within the limit should be manageable.

**Reference**:  
[1] Minimizing the Accumulated Trajectory Error to Improve Dataset Distillation, CVPR 2023.

### Questions
1. In Figure 1, what are the primary insights that should be drawn from each sub-figure? Specifically, could the authors clarify how each metric in Figure 1 (e.g., variance in weights, distillation loss) directly impacts the effectiveness of the KD-based trajectory matching approach?

2. The paper argues for the superiority of trajectory matching in the SSL setting. Could the authors elaborate on any specific challenges or limitations of gradient matching and distribution matching that make them unsuitable for SSL dataset distillation, as compared to trajectory matching? (this question ties to one of the points in the weakness section)

3. Regarding the high-variance problem in SSL gradients, has there been an investigation into alternative approaches, such as sharpness-aware minimization, to reduce gradient variance? What led to the decision to focus solely on KD rather than considering variance-reducing techniques as a complementary approach? (also raised in the weakness section)

4. In Table 2, could the authors clarify the relationship between initialization choice (e.g., high-loss vs. random subsets) and performance? Is there a grounded rationale behind the preference for high-loss subset initialization?

5. Could the authors provide more context on how the choice of SSL algorithm (e.g., SimCLR vs. Barlow Twins) affects the efficacy of the MKDT method? Is there a fundamental reason one algorithm would be preferred over another in the KD-based trajectory matching framework? Which method among SimCLR and BarlowTwins as faster convergence while learning the distilled samples? Can this method also be extended to other SSL methods such as masked reconstruction?

6. Lastly, in the methodology, could the authors clarify how they determined the optimal values for hyperparameters like the number of distillation steps, K expert trajectories, and distillation loss thresholds?

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
The authors of the paper deal with applying dataset distillation on unlabeled dataset as part of a self-supervised pre-training. A fundamental observation presented in the paper is that optimizing for dataset distillation while using SSL objective is hard to convergence because of a high gradient variance in the batch. This observation was demonstrated empirically and theoretically under some assumptions. To mitigate the high gradient variance, the authors proposed to use Matching Training Trajectories following knowledge distillation, which leads to a much smooth gradients, allowing a stable dataset distillation optimization and accuracy improvements on several downstream tasks.

### Strengths
- In my view, the proposed approach effectively transforms the problem of data distillation in an unlabeled setting (SSL objective) into a supervised approach using knowledge distillation loss.

- The concept of enhancing self-supervised approaches by incorporating dataset distillation is interesting.

- The approach presented in the paper is simple and clearly explained.

- The theoretical and experimental motivations regarding high variance gradients appear reasonable and valid.

### Weaknesses
- Line 75: “… datasets distilled with ConvNets can transfer to other larger architectures such as ResNets.” Note that ResNets are also a type of Convolutional network (ConvNet).

- What about considering larger (and more modern) architectures? For example, using larger ResNet architectures instead of just 3-4 convolutional layers.

- Do the authors consider connections to data-free knowledge distillation (DFKD)?

- The authors' theoretical derivation is based on very strict assumptions. Statements such as "as confirmed theoretically above" (line 242) may be overstated.

- The paper uses the Barlow Twins SSL objective. Did the authors try other SSL approaches? It would be useful to know if the observations are generalizable and if the accuracy improvements hold across different SSL algorithms.

- Experiments: the authors only compare linear probes on 1% and 5% of the downstream labeled data. Why not present results for additional subset sizes? Especially when comparing to SAS, which is a data pruning method where the main focus is on higher subset sizes (e.g., SAS provides results for 20% subset size and above). This raises the question of whether the comparison to SAS is fair.

### Questions
- My main question is more general, concerning the concept of dataset distillation (DD). Despite over four years of research on DD, progress appears to be limited, with most work focused on small datasets and controlled settings. I would appreciate the authors' insights on the future potential and development of dataset distillation.

- I think the authors may consider presenting some connections to the field of data-free knowledge distillation. What do the authors think?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
5