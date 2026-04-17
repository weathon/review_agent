# PULSE: Projection-based Unlearning via Linear Speedy Entropy Maximization

- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Machine unlearning enables selective erasure of knowledge associated with specific data points from trained models without retraining from scratch. However, existing unlearning approaches face significant limitations: they typically degrade performance on remaining data, require access to the original retain dataset to maintain the model utility, and incur high computational costs. To address these challenges, we propose PULSE (Projection-based Unlearning via Linear Speedy Entropy Maximization), a novel retain-data-free unlearning method. PULSE jointly learns a projection matrix alongside the model backbone during training. During the unlearning phase, PULSE freezes the model backbone and trains a forget-specific projection matrix that maximizes confidence on the data to be forgotten. By subtracting this forget-specific matrix from the original projection, PULSE transforms confident predictions into targeted uncertainty, effectively achieving forgetting. Unlike existing methods that modify model outputs to enforce forgetting, PULSE operates directly on representation space. Extensive experiments on standard benchmarks show that PULSE achieves near-perfect forget accuracy while preserving retain accuracy and runs 10–20× faster while being memory efficient thus, establishing a new paradigm for efficient machine unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
PULSE learns a projection matrix during model training (i.e. not a post-hoc unlearning method) and then trains a forget-specific projection matrix during unlearning that maximises confidence on the forget set. This matrix is then subtracted to induce unlearning.

### Strengths
The approach of working in the representation space instead of output space for unlearning is of interest.

### Weaknesses
W1 While PULSE does not need retain data, its need for learning a projection matrix during training seems much more restrictive. Not being applicable to already trained models is a severe restriction compared to established SOTA methods.

W2 Wrong claim: SSD also does not need to store retain data after computing the diagonal of the FIM once (which is less intrusive than learning a projection during model training). “[]D can be calculated at any point after training before unlearning and only needs to be computed once, allowing for the training set to be discarded and only []D stored.”

W3 Unfair compute comparisons, as PULSE does a majority of the computation before unlearning when learning the projection during model training. This should also be accounted for in other methods (e.g., the retain set importance calculation of SSD should then be ignored and only the forget set calculation should be counted).

W4 Insufficient evaluation metrics that are present in the papers you benchmark against. Accuracy alone does not assess unlearning sufficiently. See Streisand effect (Chundawat) and other related literature. No usage of MIA or U-LIRA metrics leads to inconclusive results. NOTE: the results in 4.6 with MIA do not state on what kind of task MIA is calculated and as discussed in W6, PULSE clearly does not lead to a model that behaves like a retrained model in subclass settings. Please report MIA for each row in Tables 1-3.

W5 Limited and outdated selection of SOTA unlearning methods

W6 The forget accuracy results for PULSE in table 3 (subclass) show that the unlearned model achieved by PULSE is completely different from a retrained model (e.g. 79% acc vs 0% ac on baby sub-class) while other methods are similar to retrain at 79/66. This indicates to me that PULSE would lead to terrible MIA or U-LIRA results due to its over aggressive unlearning leading to 0 values across the board in the sub-class setting for Df.

W7 Please list the hyperparameters used for the benchmarked methods. Results seem a bit off compared to the original papers.

### Questions
Please see weaknesses.

### Soundness
1

### Presentation
2

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
This paper presents yet another machine unlearning algorithm by including a projection layer between encoding and prediction layers such that capture the property of forget set in the prediction forward path. Buy removing it through projection subtraction, the proposed method can achieve unlearning purpose. The experimental results more or less demonstrated the effectiveness of the proposed approach compared to limited number of baseline approaches.

### Strengths
1. The proposed idea is simple and easy to follow.

### Weaknesses
1. The criticism of existing methods for their reliance on the retain dataset is not well justified. In practice, the restriction on using the forget set is typically regulatory (e.g., due to data privacy laws), but the use of a limited retain dataset is generally considered acceptable. Labeling this as a “significant limitation” is therefore unconvincing. Moreover, when the authors claim that “existing methods typically degrade performance on remaining data,” it is unclear how the proposed method mitigates this issue. In fact, according to the reported experiments, the proposed approach seems to perform even worse in this regard.  

2. The proposed idea appears preliminary and lacks sufficient theoretical and empirical support of its design. Many existing unlearning algorithms [1] follow a similar two-step pattern: (1) identifying properties as projection of the forget set and (2) removing them from input data via simple vector subtraction. This paper essentially adopts the same mechanism but much more simpler than existing ones. While the authors place the projection between the encoding and prediction layers, whereas prior work are of different place or advanced why of identifying the gap between encodings, the conceptual difference is minimal. The contribution therefore feels incremental rather than innovative. But that is the the main concern I have; the concern is that how the authors believe the vector projection can lead to better unlearning without analyzing the orthogonality of P_L and P_unlearn. Note, this is aligns to where the authors criticized the existing approach on "they typically degrade performance on remaining data". Please show how the proposed methods solved/mitigated the problem. 

3. The experimental design is overly simplified. The paper only compares the proposed method with Bad Teacher and SSD, omitting several more recent and competitive unlearning approaches. Without broader benchmarking, it is difficult to assess the effectiveness or practical advantages of the proposed method.

[1] Sun, Changchang, et al. "Forget Vectors at Play: Universal Input Perturbations Driving Machine Unlearning in Image Classification." arXiv preprint arXiv:2412.16780 (2024).
[2] Seo, Seonguk, Dongwan Kim, and Bohyung Han. "Revisiting machine unlearning with dimensional alignment." 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025.
[3] Shah, Vedant, et al. "Unlearning via sparse representations." arXiv preprint arXiv:2311.15268 (2023).

### Questions
The questions are included in the above comments.

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
4

### Summary
The paper proposes a machine unlearning algorithm that employs  Linear Speedy Entropy Maximization to unlearn the forget set while ensuring the unlearning process is retain-data-free. THis method learns a projection matrix alongside the model backbone during
training and then substracts this forget-specific matrix from the original projection to increase the uncertainty of the model on the forget set.

### Strengths
Very happy they included the sequential results. 
The umap was very clear.
The idea of subclass unlearning based on the CifarSuper20 is interesting.

### Weaknesses
poor writing quality: $\rightarrow$ for example: "The theoretical elegance of our approach lies"

informal description of the scientific concepts such as "PULSE achieves effective unlearning by achieving near perfect forget set accuracy and preserving model utility" $\rightarrow$ what is considered to be near perfect forget accuracy?

Lack of Statistical Evaluation: The reported results appear to lack statistical evaluation—no mean or standard deviation values are provided


The main objective is that it cannot really claim to be retain-data free. In order to prepare the projection matrix, you need the retain dataset no matter what. when the time comes to unlearn you only need the forget set but this is not an out-of-box solution that never needs the forget set.
 
The table is not statistically significant and has 2 in some places for an unclear reason.

### Questions
why the authors think that random unlearning is impractical, couldn't a social media user want all their images removed from an algorithm and that forget set would include pictures of people, pets, places, food, etc?

### Soundness
2

### Presentation
1

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
This paper proposes PULSE, a “retain-data-free” unlearning method that inserts a learnable projection layer
between the feature extractor and the classifier during initial training. During unlearning, the backbone and
head are frozen; a forget-specific projection is optimized on the forget set via an entropy objective, and the
final projection is an interpolation between the trained projection and the forget-specific one. Experiments
are conducted only on small-scale to medium-scale vision datasets (CIFAR/STL variants), claiming strong
forgetting with limited retention degradation in less runtime.

### Strengths
The proposed PULSE addresses a retrain-free setting that only requires a forget set. The method is
computationally cheaper compared to existing methods.

### Weaknesses
There are some isseu with the current version of papaer some of them are listed as follows:
-- The method assumes the target model must have a dedicated projection matrix PL. How practical is the utility of methods in real-world
Scenarios where models don’t have PL? Since projection is part of the core forward path to the classifier,
inserting it after deployment would change the model’s behavior and therefore would require retraining.
These limitations significantly narrow down the deployability of the PULSE in the real world.

--The intuition remains unclear why unlearning can be achieved by just linearly
interpolating two projection matrices. The paper does not state under what condition they have implicitly
assumed that the contribution of the forget set can be linearly isolated and can be cleanly subtracted. There
is no theoretical or strong empirical justification showing how subtraction will behave when the forget set is
large or under highly semantically entangled retain classes. The authors must discuss the required properties of PL, for better unlearning. 

---The paper claims to achieve “representation-space forgetting” by editing a projection matrix P placed immediately before a linear classifier head W. However, in the absence of any explicit structural constraint on P (e.g., orthogonality, low-rank, non-linear gating, or multi-use elsewhere in the model), the combination of P and W is algebraically just a single linear map
 $$f(x) = W(P (ϕ(x)))$$
Thus, modifying P  is equivalent to modifying an effective head W′ = W P In this form, the method
reduces to standard head reparameterization (i.e., editing the classifier head), which is already standard
practice in unlearning. It remains unclear what is novel in terms of mechanism.

--The paper does not convincingly show that the unlearning effect is localized only to the forget set.
Because the method applies a single global linear edit before the head, editing P will affect all classes,
especially similar retain classes under high overlapping.

--Even though the forgetting-retention tradeoff hinges entirely on interpolation parameter α, the method
provides no principled mechanism for selecting or adapting it across unlearning scenarios. While authors state that $$\alpha \in [0.8, 0.9]$$ they do not specify whether it is fixed or varied across experiments, or
how it is selected.

--- The paper utilizes the general update rule $$P_{UL}(\alpha) =
\alpha P_L + (1 − \alpha) \tilde{P}(F)$$ from eq. 5, which always uses only the current forget set F and the original PL.
This implies that once you forget a set F1, the method does not update or store a new “base” projection.
When you later forget F2, you go back to the original PL and mix only with P˜(F2). So the effect of
forgetting F1, in principle, can be overwritten. Then how does PULSE preserve earlier removals?

--The paper acknowledges that the process of random unlearning is not done as it is not useful, but for the validation of the proposed method, random and mixed learning should be performed.
--Instance learning is useful in many applications; I suggest that authors should validate their method on this setup.

---I believe the Membership Inference Attack (MIA) is a very important metric for unlearning; the authors should evaluate their method using MIA across all datasets.

---In Table 1, what does “similar class unlearning” mean? It should be discussed in the paper.

---What is the purpose of P mentioned in Question 4?

---What does $p^{(i)}_c$ represent in Equation (3)?

---In Table 3, why is the standard deviation reported only for the proposed method? What is the number of runs used?

---Figure 1 is not referenced in the text.

### Questions
-- For the better comparison, the  results for the baseline that directly re-tunes the head on the forget set  are required to  compared the
Results with PULSE?

--For sequential setting, after forgetting a set F1 and then a set F2, how do you produce a single final model
that has forgotten F1 ∪ F2? Do you have an operator to merge P˜(F1) and P˜(F2) into one projection
PUL that behaves like unlearning F1 ∪ F2 in one shot? How do you ensure that the first deletion (on F1)
is not undone when you subsequently unlearn F2?

More question please look the weakness section.

### Soundness
2

### Presentation
3

### Contribution
1
