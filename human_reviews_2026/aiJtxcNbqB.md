# Decoupling Shared and Modality-Specific Subspaces in Multimodal Learning via Low-Rank Representation Fine-Tuning

- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Multimodal data in machine learning promises to improve generalization and performance on complex tasks. However, training multimodal models requires extensive paired datasets, can be computationally expensive, and lacks transparency by entangling shared and modality-specific signals in ways that hinder interpretability and control. In this work, we introduce MultiLoReFT: a low-rank representation fine-tuning framework for multimodal learning using pretrained unimodal models. Our approach extends low-rank representation finetuning to the multimodal setting and learns interpretable projection subspaces that decouple shared and modality-specific information. MultiLoReFT adaptively learns the rank of each subspace to best capture complementary contributions of each modality with minimal trainable parameters.  Our method offers an efficient and scalable solution to adapting pretrained representations for multimodal reasoning, enabling interpretable fine-tuning across both synthetic and real-world benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes MultiLoReFT, an efficient low-rank finetuning framework that disentangles shared and modality-specific information for multimodal representation learning. This offers more interpretability and control over the representation learning. The proposed approach is compared with two other methods APOLLO and DRIM-U and has shown to be more effective in decoupling the representation on both controlled, synthetic tasks and real-world tasks CremaD and Flickr30.

### Strengths
- The paper is well-motivated by the research question of decoupling shared and modality-specific information in multimodal representation learning and offers an efficient approach that gives more interpretability and control over the representation learned;
- The method, synthetic data generation, and experiment setups are well-documented;
- The design of components of loss that concern with the independence, orthogonality, and mutual information is novel;
- Ablation of the proposed method on controlled, synthetic tasks and real-world tasks with different fusion schemes, along with the PCA visualization of the learned representations, show the effectiveness of the proposed method in decoupling the shared and modality-specific representations;

### Weaknesses
- The paper mentions 2 key questions at the beginning of Section 4 Experiments that the reviewer agrees are very important to be addressed: 1. MultiLoReFT effectively captures the shared and modality-specific information and gives an interpretable representation showing how these information are distributed across the components; 2. The decoupled representation from MultiLoReFT can effectively improve multimodal representations to better perform at downstream tasks. While Section 5 Results gives an in-depth study to answer 1, question 2 is not addressed. In particular, evaluations of the decoupled representations used jointly in downstream tasks (with proper ablations on model architecture, scales, tasks preferred) are missing, which has significantly undermined the utility and motivation of the work (note this is different from table 2, which evaluates the effectiveness of individual components with knowing which component works the best a priori and can only indicate the effectiveness of the framework in decoupling representation);
- One highlight / contribution of MultiLoReFT is its parameter-efficiency. While it's clear from the context that this approach does not require training the encoders, there lacks an explicit section discussing / comparing the compute between MultiLoReFT and other existing approaches;
- The writing can be further improved: in particular,
1. The results section is a little unorganized in terms of its presentation. Preferably, there should be a one-sentence summary highlighting the main takeaway from the discussed results;
2. $\textbf{z}_{m_i}$ are not labeled clearly for table 2 to indicate which captures video, audio, or text, image in different tasks;
3. It's unclear what metrics are evaluated in table 3 and why the lower scores indicate stronger predicative power;
4. Inconsistent naming of the method name: some are in small caps MultiLoReFT while others are the regular text MultiLoReFT;

### Questions
- The authors should work on including more evaluations of the learned, decoupled representations in downstream tasks: would the more separated representations allow the model to better perform at various multimodal predictions? It's understandable that this is not the focus of the paper, but the paper should include a minimal ablation on the effect of model architecture, scale, modalities/tasks on the effectiveness of MultiLoReFT. Otherwise, it is not clear why it's important to learn a more effectively decoupled representation for multimodal learning except for better interpretability;
- Why is Simulation II only has the M1 prediction task and it's evaluated in Accuracy as opposed to MSE in Simulation I M1? What's the difference? Similarly, why is only the Language task evaluated for Flickr30 and is there a task that can be dominantly predicted using $\textbf{z}_{m_2}$?

### Soundness
2

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
4

### Summary
This paper proposes MultiLoReFT: a low-rank representation fine-tuning framework for multimodal learning based on pre-trained unimodal models. It extends low-rank representation fine-tuning to multimodal scenarios and learns interpretable projection subspaces that separate shared and modality-specific information. MultiLoReFT adaptively learns the rank of each subspace to best capture the complementary contributions of each modality with a minimum number of trainable parameters

### Strengths
1. The motivation is clear and targeted, addressing core pain points in multimodal learning
2. The method design is innovative and logically coherent, with clear advantages in efficiency and interpretability
3. The experimental verification is sufficient and rigorous, supporting conclusions effectively

### Weaknesses
1. The model used in the experiment doesn't seem very large, so efficient fine-tuning doesn't seem necessary.

2. What about the performance when scaling to larger models like llava?

3. The performance gain of MultiLoReFT in real datasets, such as cramed, seems negligible (0.6%). The MultiLoReFT seems to be useless

4. More importantly, the performance of MultiLoReFT on the CRAMED dataset is even worse than that of a model trained directly with ResNet-18 as the backbone[1]. So what is the point of using such a complex model (DINO) for fine-tuning?


[1] Xiaokang Peng. Balanced Multimodal Learning via On-the-fly Gradient Modulation Xiaokang

### Questions
see the weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors presented MultiLoReFT, a method for low-rank multimodal fine-tuning. The proposed method includes a lightweight architecture that takes in unimodal representation vectors, and transform them into low-rank subspaces while performing nonlinear transformations within the subspace. The authors devised a 3-component training objective that enforces independence between shared/modality-specific representations, the mutual information, and orthogonality between low-rank transformations. In addition, pruning was applied to rank adaptation. The proposed method was evaluated against 2 baseline finetuning methods (DRIM-U and APOLLO) on 2 simulated datasets as well as 2 real datasets, and the evaluations focused on assessing the disentangling of shared and modality-specific signals. Ablation studies was conducted on pruning and staged training.

### Strengths
1. The proposed method is lightweight and have a relatively low number of training parameters.

2. The evaluations show that the proposed method better disentangles shared information and modality-specific information in the generated representations.

3. The inclusion of algorithm blocks on page 5 made the overall methodology a lot more clear and easy to follow.

### Weaknesses
1. Limited evaluation on real multimodal datasets: other than the synthetic simulations, the paper only contains experiments on 2 real datasets, out of which only one dataset's main task (CremaD emotion prediction) performance was compared to baselines. The main reason of applying multimodal fine-tuning, in most occasions, is to improve model performance over the main task, but the proposed method only showed very marginal gain in CremaD's emotion prediction over a simple late fusion, and no main multimodal classification task was evaluated over Flickr30K.

2. The proposed method also does not seem to consistently show larger deltas in the simulation experiments. The authors claims that APOLLO's larger gap over M2 is due to its overall worse performance, but DRIM-U seem to have same delta on M2 MSE and a much larger delta on M1 MSE for simulation 1. 

3. The proposed method seems restricted to 2-modal settings only and does not seem to be generalizable to tasks with more than 2 modalities or tasks with unimodal representations as sequence of vectors rather than single vectors.

4. There is no ablation on some of the most important design choices. For example, there is no ablation on the objectives, so it is unclear whether all 3 objectives are necessary for the proposed method; there is no ablation on doing GradNorm vs fixed objective weighting as hyperparameters. It is also very hard to interpret the results in the existing ablation studies in Table 5, as the results for the full method is not present in the same table.

5. The writing quality is not the best. There are some undefined variables in equation 7: (a) $\tau$ is not officially defined. Do you treat $\tau$ as a fixed constant or a trainable temperature parameter (as in CLIP)? (b) N is not officially defined. Since j is used to indicate modality, maybe N should just be 2? Also, there are typos like "sdsptive" in line 201

### Questions
For Table 3a, how exactly are the classification performance obtained? Do you fit another linear layer or MLP on top of the fine-tuned representations ($z_s$)?

In Tables 1 and 2, performance with $z_s$ is reported; however, in Algorithm 1, it only shows how $z_{s1}$ and $z_{s2}$ are obtained. How do you obtain $z_s$ from $z_{s1}$ and $z_{s2}$? Also, Eq(1) states that  $z_{s1}$ and $z_{s2}$ are always equal. Is that true? If so, why is that true?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses an important challenge in multimodal machine learning: disentangling the respective contributions of inter- and intra-modal interactions, framed as shared and modality-specific components. The authors propose MultiLoReFT, a low-rank representation fine-tuning framework that facilitates efficient and interpretable multimodal learning by leveraging pretrained unimodal models. The approach extends low-rank adaptation techniques (LoReFT) to the multimodal domain, introducing projection subspaces that separate shared and modality-specific information through independence and orthogonality constraints.

The experimental section includes both controlled simulations and large-scale datasets, and the results suggest that the proposed method achieves a better disentanglement of modality contributions compared to the chosenbaselines. Nonetheless, the paper would benefit from more detailed analyses, such as ablation studies on the contribution of individual loss terms—and additional clarification on certain aspects of the experimental setup.

Overall, the paper presents an interesting and promising direction for interpretable multimodal learning. However, some methodological choices could be better justified, and the experimental validation could be strengthened to fully support the claimed advantages of the approach.

### Strengths
1. **Interesting and timely topic of disentanglement.**
The paper addresses an important problem in multimodal machine learning: the disentanglement of shared versus modality-specific representations. This topic is both theoretically significant and practically relevant, as achieving clearer separation between modality interactions can lead to better interpretability, and transferability of multimodal systems.

2. **Well-structured and diverse experimental setting.**
The experimental design is carefully structured to evaluate the proposed framework in both controlled and realistic conditions. The inclusion of simulated datasets allows for a clear and interpretable assessment of the method’s disentanglement capabilities under known ground-truth conditions. Complementing this with experiments on larger-scale, real-world datasets demonstrates the  practical potential of the approach.

3. **Parameter efficiency and flexibility.**
The framework is designed to be computationally efficient by introducing only a limited number of additional parameters through adaptive low-rank subspaces. Moreover, the approach can, in principle, be applied on top of _any_ pretrained unimodal model, which makes it flexible across modalities and backbones. However, while this adaptability is promising, the paper provides limited empirical evidence on how well the framework generalizes across different pretrained representations or scales to substantially larger models.

### Weaknesses
1. **Insufficient justification for key constraints and design choices.**
The introduction of independence and orthogonality constraints is intuitively motivated but not sufficiently justified from a theoretical or empirical standpoint. It remains unclear why these specific constraints are optimal for disentangling shared and modality-specific representations, and how sensitive the results are to their exact formulation. 

2. **Limited contextualization within related multimodal literature.**
The paper would benefit from a clearer discussion situating the proposed approach within the broader landscape of multimodal representation learning. In particular, several recent methods explicitly aim to model shared and modality-specific information through mutual information (MI)–based formulations, such as FactorCL [1], or through partial information decomposition (PID)–based frameworks, such as CoMM [2]. These families of models also address the problem of disentangling inter- and intra-modal contributions, yet their conceptual relationship with MultiLoReFT is not discussed, and they are not included among the baselines. A more explicit discussion, highlighting similarities, differences, and potential complementarities, along with empirical comparisons, would help clarify the novelty and distinct contribution of MultiLoReFT in relation to existing multimodal approaches.

[1] Liang, P. P., Deng, Z., Ma, M. Q., Zou, J. Y., Morency, L. P., & Salakhutdinov, R. (2023). Factorized contrastive learning: Going beyond multi-view redundancy. Advances in Neural Information Processing Systems, 36, 32971-32998.

[2] Dufumier, B., Castillo-Navarro, J., Tuia, D., & Thiran, J. P. (2025). What to align in multimodal contrastive learning?. International Conference on Learning Representations.


3. **Unclear experimental design and training procedure.**
The description of the training pipeline lacks clarity. It is not explicitly stated whether the model is subsequently fine-tuned for specific downstream tasks after the self-supervised stage, and if so, whether modality fusion occurs during or after fine-tuning. Additionally, the loss configuration across different training stages is not clearly explained — it is uncertain whether the same loss functions are applied throughout, what the “selected convergence criteria” refer to, and whether any ablation was performed on these procedural choices.
Similarly, the role and definition of the cross-modal mutual information loss (l. 183) would benefit from clearer explanation and justification.

4. **Missing ablation studies and analytical evaluations.**
The paper lacks ablation studies that would help disentangle the contribution of each component in the overall loss function. Understanding how each loss term (e.g., independence, orthogonality, mutual information) impacts the final representation quality and disentanglement would significantly clarify the method’s internal dynamics.

5. **Restricted evaluation to two modalities.**
The proposed method is evaluated only on bimodal setups, which limits the generality of the conclusions. Moreover, the formulation does not appear to scale naturally beyond two modalities. Extending the approach to more modalities seeems cumbersome, as the number of pairwise interactions and constraints would likely grow combinatorially. A discussion or preliminary experiment addressing this scalability aspect would strengthen the paper’s claims of general applicability.

6. **Practical limitations in real-world settings.**
In practical multimodal scenarios, the dominant source of information relevant to a given task is typically unknown. The proposed method seems to assume that the relative importance of each modality is known (in all experiments, the dominant feature is underscored), but this assumption may not hold in real applications. It remains unclear how the model would perform when the task-relevant modality is not dominant or when the useful information is distributed unevenly across modalities. Clarifying how the framework handles such cases would help assess its robustness and practical usability.

7. Related to the previous point, Table 3a appears to be the only experiment in which the dominant feature is not known beforehand. However, the description of this experiment lacks clarity. The authors refer to the prediction as being based on “fused” features but do not specify how this fusion is performed or which features are actually used. A more detailed explanation of the fusion mechanism is needed to properly assess the experimental setup and interpret the reported results.

8. **Fairness of comparisons.**
The fairness and consistency of experimental comparisons are not fully transparent. It is not clear whether all baselines were trained under comparable settings, especially regarding the use of pretrained encoders and the amount of supervision. For instance, l.314 mentions that DRIM-U was adapted to a self-supervised variant, without any details on how this affects its performance.

9. **Limited analysis of dependence on encoder quality.**
Since the method relies on pretrained unimodal encoders, it would be interesting to study how performance scales with encoder quality. It remains unclear whether the disentanglement and downstream performance would improve linearly (or saturate) with stronger unimodal representations. This sensitivity analysis is missing.

10. **Potential inconsistencies.**
L. 426 seems to contradict the formulation of the loss function. How can fine-tuning improve individual modality representation if the independence loss was specifically designed to prevent leakage of information. This suggests possible inconsistencies in the implementation or interpretation.

### Questions
- Could you elaborate on the motivation for enforcing both independence and orthogonality constraints?

- How does MultiLoReFT conceptually and empirically differ from mutual information–based approaches such as FactorCL [1] or partial information decomposition–based models like CoMM [2]? They could also be included in the comparisons as baselines.

- Can you clarify the overall training procedure? Specifically, after the self-supervised stage, is the model fine-tuned for specific downstream tasks?

- Are the same loss functions used throughout all stages of training, or are they modified across phases?

- Could you provide more details on the definition and role of the cross-modal mutual information loss (l.183)?

- What exactly is meant by the “selected convergence criteria”?

- How would the complexity (in terms of number of constraints or pairwise interactions) scale with the number of modalities? Have you considered any strategies to mitigate potential combinatorial growth?

- How would the framework behave in real-world cases where the dominant or most informative modality is not known a priori?

- Could you clarify how the “fused” features are constructed in Table 3a? What mechanism is used (e.g., concatenation, learned fusion, or averaging)? How does this fusion process affect interpretability and performance?

- How sensitive is the method to the quality of the pretrained unimodal encoders? Does the disentanglement or downstream performance improve consistently with stronger encoders?

- Line 426 seems contradictory to the loss formulation: if the independence loss prevents information leakage, how can fine-tuning improve single-modality representations?

### Soundness
3

### Presentation
2

### Contribution
2
