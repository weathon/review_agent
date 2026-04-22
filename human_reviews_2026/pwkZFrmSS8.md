# Silent Neighbors, Loud Secrets: Privacy Leakage from Nearby Classes in Unlearned Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
In this paper, we reveal a significant shortcoming in class unlearning evaluations: overlooking the underlying class geometry can cause privacy leakage. We further propose a simple yet effective solution to mitigate this issue.
We introduce a membership-inference attack via nearest neighbors (MIA-NN) that uses the probabilities the model assigns to neighboring classes to detect unlearned samples. Our experiments show that existing unlearning methods are vulnerable to MIA-NN across multiple datasets. We then propose a new fine-tuning objective that mitigates this privacy leakage by approximating, for forget-class inputs, the distribution over the remaining classes that a retrained-from-scratch model would produce. To construct this approximation, we estimate inter-class similarity and tilt the target model’s distribution accordingly. The resulting Tilted ReWeighting (TRW) distribution serves as the desired distribution during fine-tuning. We also show that across multiple benchmarks, TRW matches or surpasses existing unlearning methods on prior unlearning metrics. More specifically, on CIFAR-10, it reduces the gap with retrained models by $19\%$ and $46\%$ for U-LiRA and MIA-NN scores, accordingly, compared to the SOTA method for each category.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper argues that standard class-unlearning evaluations overlook class geometry, leaving leakage that a new attack (MIA-NN) can expose. The authors proposed TRW to mitigate the privacy leakage caused by MIA-NN and did extensive experiments to support their claims.

### Strengths
1. The nearest-neighbor leakage perspective is simple and intuitive.

2. TRW is an output-space objective, easy to implement, and adds little overhead relative to fine-tuning.

### Weaknesses
1. The MIA via nearest neighbors attack is not clearly substantiated in the main text. For an attack, at least the threat model, capability of each role and the attacking goal should be defined clearly. In this paper, it seems that all things are conducted by the ML server.


2. What MIA-NN really measures. As defined, MIA-NN trains per-class discriminators on retrain (and then on the unlearned model) and computes their accuracy on forget-class test samples to quantify a gap to retrain. This is not a per-example membership decision about whether a specific sample was in the original training set; it is a distributional test of whether forget-class behavior matches retrain. The paper sometimes presents MIA-NN as a “membership inference attack” without clearly distinguishing this from standard per-sample MIAs.


3. The proposed Tilted ReWeighting (TRW) objective adjusts the output distribution proportionally to class similarity, but this only ensures proportional alignment. It does not guarantee the decision boundary geometry or higher-order distributional structure. As a result, while the marginal behavior may resemble retraining, the actual local decision regions may diverge.

4. The paper's writing and methodological details are not clear, making it hard to follow:
 - In Section 3.4, the constant c (from the definition of set A) is not explained clearly (between Eq. 1 and Eq. 2, no marker). How to determine the value of expected similarity c, and why the paper set β as 10?
 - The theoretical motivation for using an exponential term (exp) in the tilting factor in Eq. 2, rather than a simpler linear weighting, is not discussed in the main text. The proof in the appendix shows this is a result of KL-divergence minimization, this should be mentioned after Eq.2 at least 1 sentence.
 - A method named TRW-2R appears in multiple tables (e.g., Table 2, 3) and often performs differently than TRW. However, this method is never defined or explained in the paper. This is a significant omission that needs to be addressed.

5. For paper citing, authors could use \citep{} to replace \cite{}.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a new membership inference attack on the class unlearning task and then proposes a novel unlearning method with design robustness against membership inference attacks. The author made an observation that the retrained model demonstrates consistent misclassification of some retained classes when tested on the unlearning class. Based on the observation, the author designed MIA-NN, which exploits the probability assigned to the closest forget set to achieve a stronger MIA. The paper proposes Tilted Re-Weighting(TRW) to achieve secure unlearning under MIA. TRW redistributes the probability among the remaining classes to achieve a consistent prediction with a retrained model.

### Strengths
1. The proposed unlearning method is clear and insightful. 
2. Strong theory foundation with clear mathematical explanation. 
3. Comprehensive benchmark method comparison.

### Weaknesses
1. The table description is inaccurate and causes confusion due to the lack of notation in the table. Specifically, Table 1 is described as follows: " Higher values indicate better unlearning; however, the paper also describes that the gap between the Acc_i and Acc^Mu_rn is used to measure the unlearning effectiveness. In the experiment section, the paper proposes that the MIA score of the unlearned model is expected to match that of the retraining model. However, the table does not provide the MIA score of any retraining model or the difference between the unlearned model and the retrained model. These problems pose difficulties in understanding the results of the paper. 
2. In Table 2 and Table 4, the method TRW-2R outperformed TRW unlearning  introduced in the paper. In Appendix B.5, the author also mentioned the proposed TRW-2R as the fastest among other baseline methods. However, there's a lack of the implementation and theoretical details of the methodTRW-2R.  The authors do not explicitly specify what additional components or improvements has been made to distinguish TRW-2R from TRW. Without the details, it’s hard to understand and verify the reason for the experiment improvement. 
3. The discussion of the method is limited to class unlearning on classical vision models. This presents a dual limitation for extending this method to either other types of models(for example, GNN and LLM) or type of unlearning (for example, per-example unlearning). I would appreciate some discussion of the possibility of extending this method to other models, such as ViT.

### Questions
One of the important insights from this paper is the vulnerability of the unlearned model after class unlearning, and a corresponding attack method, MIA-NN, has been proposed to exploit it. I am curious why MIA-NN is not part of the evaluation matrix(except for B.7 with an ablation study focus on the effect of beta).

### Soundness
3

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
The paper offers a crisp diagnostic (MIA-NN) and a pragmatic fix (TRW) that together materially advance evaluation and practice of class unlearning. While the core ideas are simple, they are impactful and well-validated. Clarifications on the attack’s practicality and broader stress-testing would further solidify the case for acceptance.

### Strengths
1. The paper pinpoints a blind spot in current class-unlearning evaluation—evaluations ignore how retrained models systematically misclassify forgotten-class samples toward semantically similar retained classes. This motivates a new attack (MIA-NN) that probes leakage via the nearest neighbor of the forgotten class and exposes failures of many SOTA methods.
2. The proposed Tilted ReWeighting (TRW) modifies the fine-tuning objective by zeroing the forget label and tilting the remaining class distribution using inter-class similarities derived from logit weights; the resulting target distribution can be seen as an information projection with a linear moment constraint. It is lightweight (drop-in during fine-tuning) and conceptually clean.

### Weaknesses
1. TRW hinges on a particular class-similarity score (cosine in a PCA-projected logit space with a sharp softmax temperature). While intuitive, this is heuristic and sample-independent; performance sensitivity to the choice of similarity, PCA dimension, temperature, and $\beta$ needs deeper analysis beyond brief ablations.
2. The attack identifies a “nearest neighbor” via statistics from multiple retrained models and trains an SVM on logits. Although the paper claims the attack does not assume access to training data, the practicality of assembling enough scratch-retrained references (and the knowledge required) deserves clearer discussion and a black-box-only variant analysis.

### Questions
1. How many scratch-retrained models are required for MIA-NN to be reliable, and under what access (black-box probabilities only vs. logits vs. labels)? Can you report MIA-NN performance under strictly black-box probability queries and with one or zero reference retrains (e.g., using public checkpoints as surrogates)?
2. How sensitive is TRW to (i) PCA dimension, (ii) softmax temperature over similarities, (iii) cosine vs. centroid-distance vs. feature-space CKA similarities, and (iv) the tilt parameter $\beta$  (beyond the brief ablation)? Please include a grid showing ACCr/ACCf/MIA/U-LiRA vs. these hyperparameters.

### Soundness
3

### Presentation
3

### Contribution
3
