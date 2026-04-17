# CE-SSL: Computation-Efficient Semi-Supervised Learning for ECG-based Cardiovascular Diseases Detection

- Decision: Reject
- Scores: 6, 6, 4

## Abstract
The label scarcity problem is the main challenge that hinders the wide application of deep learning systems in automatic cardiovascular diseases (CVDs) detection using electrocardiography (ECG). Tuning pre-trained models alleviates this problem by transferring knowledge learned from large datasets to downstream small datasets. However, bottlenecks in computational efficiency and detection performance limit its clinical applications. It is difficult to improve the detection performance without significantly sacrificing the computational efficiency during model training. Here, we propose a computation-efficient semi-supervised learning paradigm (CE-SSL) for robust and computation-efficient CVDs detection using ECG. It enables a robust adaptation of pre-trained models on downstream datasets with limited supervision and high computational efficiency. First, a random-deactivation technique is developed to achieve robust and fast low-rank adaptation of pre-trained weights. Subsequently, we propose a one-shot rank allocation module to determine the optimal ranks for the update matrices of the pre-trained weights. Finally, a lightweight semi-supervised learning pipeline is introduced to enhance model performance by leveraging labeled and unlabeled data with high computational efficiency. Extensive experiments on four downstream datasets demonstrate that CE-SSL not only outperforms the state-of-the-art methods in multi-label CVDs detection but also consumes fewer GPU footprints, training time, and parameter storage space. As such, this paradigm provides an effective solution for achieving high computational efficiency and robust detection performance in the clinical applications of pre-trained models under limited supervision.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper proposes CE-SSL, a computation-efficient semi-supervised learning framework for CVD detection from ECG under limited labeled data. It combines random-deactivation, one-shot rank allocation, and a lightweight semi-supervised pipeline to adapt pre-trained models efficiently. Experiments on four datasets show that CE-SSL achieves better detection performance than state-of-the-art methods while using less computation and storage, making it practical for clinical applications.

### Strengths
* The paper tackles the practical trade-off between detection performance and computational efficiency in ECG-based CVD detection under limited labels.

* Integrates low-rank adaptation, one-shot rank allocation, and semi-supervised BN into a single framework, which could reduce training costs compared to traditional SSL methods.

* Each component (RD-LoRA, one-shot rank allocation, lightweight SSL) is described with equations and reasoning. The logic of reducing computation while trying to maintain performance is consistent.

* Experiments on multiple downstream datasets show reductions in GPU usage, training time, and parameter size, which aligns with the goal of computational efficiency.

### Weaknesses
* Complexity vs. gain: The random-deactivation and one-shot rank allocation add methodological complexity. It’s not entirely clear if the performance gain justifies the added complexity, especially for smaller backbones.

* Semi-supervised BN limits: Using unlabeled data only in BN layers may not exploit all information in the unlabeled data. The claim of comparable performance to SOTA might depend heavily on dataset characteristics.

### Questions
1. How sensitive is the one-shot rank allocation to the initial gradient estimation? Could this approximation fail in practice?

2. Is there any trade-off in inference speed due to merging low-rank matrices in the testing stage?

3. Could the semi-supervised BN approach handle situations where labeled and unlabeled distributions differ significantly?

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
This paper proposes CE-SSL, a computation-efficient semi-supervised learning framework for ECG-based cardiovascular disease detection. The method combines random-deactivation LoRA (RD-LoRA), one-shot rank allocation, and a lightweight semi-supervised BatchNorm module to achieve efficient fine-tuning of large ECG models with limited labeled data. CE-SSL aims to reduce the computational cost of semi-supervised learning while maintaining strong performance. Experiments on four ECG datasets show consistent gains in Fβ=2 with 30–70% lower training cost compared to standard SSL approaches. The framework is conceptually clear, empirically validated, and practically relevant for resource-constrained medical AI applications.

### Strengths
1. The paper tackles a well-defined and realistic challenge—efficiently adapting large ECG models under limited labeled data and computation—which is an important problem for real-world clinical applications.
2. The proposed framework integrates stochastic LoRA activation, one-shot rank allocation, and semi-supervised BN in a logically consistent way. The design is simple, interpretable, and effectively reduces computational overhead without compromising accuracy.
3. Evaluations across four ECG datasets demonstrate solid and reproducible gains in Fβ=2 with substantial reductions in computation and memory cost. The results are consistent and adequately support the efficiency claims.
4. The framework is easy to implement, uses public datasets, and aligns well with deployment needs in constrained biomedical environments, making it practically useful beyond academic settings.

### Weaknesses
1. The derivation of RD-LoRA assumes independence between the random gate and the low-rank parameters, which is not strictly valid during training. While this approximation is common in stochastic-depth literature and works empirically, it should be explicitly acknowledged rather than presented as a rigorous proof.
2. The paper only removes the semi-supervised BN in ablations, leaving unclear how much each individual component (RD-LoRA, rank allocation, SSL-BN) contributes to the final performance or whether their synergy is essential.
3. The reported efficiency gain partly arises from a simpler semi-supervised design compared with heavier methods like FixMatch or FlexMatch. A more balanced discussion of this difference would make the efficiency claim more convincing.

### Questions
1. Could the authors clarify how the random gating variable δ interacts with the low-rank matrices during training? Since δ affects which adapters receive gradients, A and B are not fully independent of δ. Explaining this as an approximation (similar to stochastic depth) would help align the theory with practical implementation.
2. It would strengthen the paper to report results for variants that retain only one of the three modules (e.g., RD-LoRA only, SSL-BN only). Such analysis could help confirm whether the performance–efficiency trade-off truly relies on their joint design.
3. The improvement from SSL-BN appears modest. Can the authors provide direct comparisons between standard BN (using only labeled data) and SSL-BN (using both labeled and unlabeled data) to quantify the real contribution of unlabeled samples?
4. Have the authors tested whether CE-SSL generalizes across centers or device domains (e.g., training on PTB-XL and testing on Chapman)? Demonstrating cross-distribution robustness would further support the framework’s practical value.

### Soundness
3

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
4

### Summary
This paper introduces CE-SSL, a computation efficient semi-supervised method for fine-tuning pre-trained models to address label scarcity and computational limitation in downstream ECG datasets for cardiovascular disease (CVD) detection. CE-SSL combines three main methods: (I) Random deactivation of layers' low-rank matrices which is very similar to classic dropout, but applied to full layers compared to neurons in layer (see Eq.2); (II) one-shot rank allocation, considering only two possible ranks (see Eqn.10) and (III) A lightweight semi-supervised learning using both labeled and unlabeled data to update BN layers in a semi-supervised manner. These improve generalization and stability under label scarcity while reducing memory and computation costs. Experiments on 4 public ECG datasets show that CE-SSL outperforms some SOTA semi-supervised and parameter-efficient baselines in terms of multi-label classification accuracy, while reducing training time, GPU memory footprint, and parameter storage needs. 

Overall, the work is quite solid in providing extensive empirical evaluation with some ablation (e.g., on $p$, dropout rate or rank $r$), yet theoretical and methodological contributions are limited. All components of the method have been done before, including random deactivation (not much different than classic dropout); one-shot rank allocation, and semi-supervised tuning; some specifics could be different but ideas are there. Also, the straightforward argument in App C.1 and its conclusion (final network is an ensemble of all possible sub-networks during model training) are not too surprising given the sampling technique. I believe this work is useful for the "Health informatics" community, but I am afraid it could be of limited interest to ICLR community.

### Strengths
- Extensive validation with large backbone models against 6 baselines, and on 4 downstream datasets, while also include some ablation. 
- Flexibility with initial rank $r$ and top-k important layers, allowing for accuracy-computation tradeoff.
- Avoids pseudo-labeling & consistency training to reduce complexity and potential error-propagation from unlabeled data.

### Weaknesses
- I believe the main weakness of this work is its limited scope and interest to ICLR community. I enumerate some more suggestions below: 
- I suggest including direct comparisons with fully supervised models trained on the same downstream datasets with limited labels. This will highlight how much performance gain is attributed to the semi-supervised pre-trained method. 
- Better justification/analysis for the use of a single gradient pass for rank estimation could benefit the paper (Is Taylor approximation sufficiently accurate? Could you ablate the approximation to deduce its effect? 
- Other ablations could include comparisons between "no rank allocation", "random allocation", and "one-shot allocation" to isolate the effect of the latter. 
- Benchmarking against newer ECG foundation models (e.g., large-scale pretrained ECG transformers or contrastive frameworks) would position CE-SSL more clearly within the current state of the field. 
-  To quantify the quality of pre-trained representations by CE-SSL, one can use linear probing. Also, it It could be helpful to evaluate CE-SSL using various backbone models (e.g., transformer-based or hybrid ECG encoders).

### Questions
See weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
