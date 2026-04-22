# Neural Decoding through Multi-subject Class-conditional Hyperalignment

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Understanding brain dynamics in multi-subject studies is challenging, as each individual exhibits unique neural patterns. Such variability complicates the identification of shared task-related dynamics without carefully accounting for meaningful individual differences. Typical analyses involve fitting subject-specific models separately and aggregating results post hoc. This approach, however, precludes the possibility of information sharing across the models. Hyperalignment methods resolve this by mapping subject-specific responses into a shared latent representational space, but typically require a secondary dataset to learn these mappings by exposing all subjects to an identical, rich and evocative stimulus, such as watching an exciting movie. These datasets are costly to collect and understandably infeasible in nonhuman studies. An alignment method for multi-subject studies that can be applied directly to the primary dataset would be of immense value. To this end, we introduce the Multi-Subject Class-Conditional Hyperalignment ($\mathbf{MuSCH}$) model which learns aligned latent embeddings of multi-subject data by leveraging class labels available from the experimental protocol of the primary dataset itself. $\mathbf{MuSCH}$ trains subject-specific encoder networks using a novel Supervised Contrastive Learning framework which simultaneously makes same-class embeddings similar and different-class embeddings dissimilar across subjects. Using both simulation studies and a real memory experiment, we demonstrate how principled information sharing improves the performance of a downstream neural decoding task. Furthermore, by modulating signal strength in the simulated dataset, we show that classification improvements are especially pronounced in regimes with weak signals, a situation commonly encountered in neuroscience investigations. $\mathbf{MuSCH}$ obviates traditional hyperalignment's onerous prerequisite of a secondary alignment dataset, extending the promise of a single robust and generalizable model to any labeled, multi-subject dataset where subject-specific distortions prevent a joint analysis.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors sued supervised contrastive learning and by formulating positive pairs from data across different subjects that share classes. Through this loss objective, a separate encoder was trained for each subject to hyperalign the latent space of neural data across subjects. The proposed method is tested in simulated data and rats' hippocampal spike data performing odor memory task.

### Strengths
* The problem is well defined and the limitation of the previous methods are well described.

* The work demonstrates successful implementation and training of supervised contrastive learning that uses across-subject pairs to encourage hyperalingment.

### Weaknesses
* The definition of equation 1 is not used or proven. It is unclear that the mathematical ground or any empirical evidence that the loss in equation 2 will guarantee (or likely converge to) that the solution will converge to the right hand side of equation 1. As an arbitrary neural network can express an arbitrary function, grounding evidence is crucial that demonstrates the hyperalignment definition in eq 1 can be achieved. 

* Mixed results in ablation experiment: putting aside the groundness of the proposed method, the decoding results denoted in Table 1 show that including the proposed training method yields degraded performance from the w/o proposed method (“Aligned-Single”), for many cases. This may imply the proposed loss is not very well compatible or suboptimal in hyperalignment purpose.

* The model evaluation is significantly limited in terms of recruiting baseline models. Only MMVM-VAE is tested. There are other hyperalignment methods listed in the related works. (Some of them need timely aligned stimuli but the dataset can be easily processed to provide such by chunking time span of each category, and dynamic time warping can be used.) Moreover, how the authors trained the model using the dataset they are testing is not elaborated. If they used a pretrained model, it should be made sure that the training/testing distribution exactly matches to provide a fair comparison. 

* The proposed methods still require class information and cannot be applied to complex stimuli or behaviors which are hard to define discrete categories.This significantly limits the utility of the method.

* The approach lack novelty that it simply reorganize positive/negative pairs to be recruited across subjects. Due to the above caveats in lack of ground and limited evaluation, this lack of technical novelty is yet to be resolved.

### Questions
See weaknesses.

### Soundness
2

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The study introduces a supervised contrastive learning framework for class-conditional functional alignment across subjects that does not have the secondary dataset requirement in hyperalignment method. The suggested method jointly class-conditionally-clusters and encodes samples in a lower-dimensional space with supervised contrastive learning, prior to classification.

### Strengths
- The paper proposes a novel supervised contrastive method for hyperalignment problem. 
- The application domain is novel, introducing new challenges for the field.

### Weaknesses
- Regarding the optimization of the subject-specific encoders, the stopping criterion is not explained in the study. The optimization of encoders and their utility is data dependent, hence a metric is needed to quantify convergence for reproducibility.
- Since the evaluation is dependent on the chosen decoder architecture, an ablation of network parameters in Section 5 is missing and required for isolating alignment performance from decoder capacity. 
- The decoder that is used to assess how well the suggested method works, is a nonlinear neural network, while the encoders are linear networks. Since a nonlinear neural network gradient is input-dependent, due to saturation mode of the sigmoidal function, its performance is not a reliable identifier of how well the data is separable. In an ablation study, adding the results of a linear decoder is required to more directly measure the improvement in linear separability. 
- The positioning of the study could be strengthened by discussing closely related studies on supervised functional alignment implementations in Canonical Correlation Analysis (CCA) studies, specifically Maxvar-Generalized-CCA and variational-CCA implementations.  
- An overall method figure for Section 3 is missing, making it harder to grasp the core contribution of the paper. 
- T is introduced without a definition (Line 134-135) that stands for the number of frames in movie-watching task.

### Questions
Q: Could you please specify the convergence criteria used for optimizing the subject-specific encoders? For example, was it based on early stopping using a validation loss, a fixed number of epochs, or a performance threshold? Was the target subject held-out data used in the assessment of convergence? Clarifying this is essential for assessing the robustness of the training procedure and for ensuring the results are reproducible.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MuSCH (Multi-Subject Class-conditional Hyperalignment), a method for aligning neural data across multiple subjects using contrastive learning. This eliminates the need for secondary alignment datasets required by traditional hyperalignment methods by leveraging class labels (supervised) from the primary experimental dataset.

### Strengths
- The paper tackles a limitation of existing hyperalignment methods—the requirement for expensive, time-consuming secondary datasets with time-locked stimuli. This is especially problematic for animal studies where data collection is hard

- Reproducibility: The author used public dataset and shared with code.

### Weaknesses
1. Supervised contrastive learning is a good way to train the model, however, recent works in self-supervised learning (foundation model) in neural data also show extremely well performance for cross-subject/animal/session results. Compare with one of state-of-the-art decoding methods in this field, e.g. POYO+ (ICLR2025), NEDS (ICML2025).

2. Scalability Concerns. What about scale up the method training with 10, 50 animals? Does the N different encoder network still works well?  Computation might be costly?

3. Only one real dataset result. I'm not sure about model's generalization ability.

### Questions
- What's the representation of neural data looks like? Are there any clustering effects for subjects or similar tasks?

- Can you show some further analysis on the representations?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces MuSCH (Multi-Subject Class-Conditional Hyperalignment), a method for aligning neural data across subjects using Supervised Contrastive Learning based on class labels instead of time-synchronized stimuli. This approach eliminates the need for secondary alignment datasets and enables cross-subject neural decoding in nonhuman studies. The method is evaluated on both simulated datasets and real hippocampal spike data from rats performing an odor sequence-memory task. Results show that MuSCH improves downstream decoding accuracy.

### Strengths
Strengths:
1. The use of Supervised Contrastive Learning (SupCon) for multi-subject alignment is new in neuroscience applications.
2. The method is clearly formulated and mathematically consistent.

### Weaknesses
Weaknesses:
1. The problem of aligning neural representations across subjects is not new, as several existing methods, such as [1][2][3], already address or could feasibly address multi-subject representation learning under different assumptions. However, the authors do not sufficiently discuss these related approaches or clearly delineate how their method conceptually differs from them.
2. The authors do not compare MuSCH with closely related multi-subject or multi-session decoding models.


  
References:  
[1] Azabou, Mehdi, et al. "A unified, scalable framework for neural population decoding." Advances in Neural Information Processing Systems 36 (2023): 44937-44956.   
[2] Zhang, Yizi, et al. "Towards a" universal translator" for neural dynamics at single-cell, single-spike resolution." Advances in Neural Information Processing Systems 37 (2024): 80495-80521.  
[3] Zhang, Yizi, et al. "Neural encoding and decoding at scale." arXiv preprint arXiv:2504.08201 (2025).

### Questions
Questions:
1. How are positive pairs and negative pairs exactly sampled within a batch?  
2. Why the encoders are frozen before training the decoder, rather than training the whole model end to end? It’s unclear how this design choice affects the adaptability of the learned representations and overall decoding performance.

### Soundness
2

### Presentation
2

### Contribution
2
