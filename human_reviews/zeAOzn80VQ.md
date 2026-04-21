# Dataset Ownership Verification in Contrastive Pre-trained Models

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 5, 8, 8

## Abstract
High-quality open-source datasets, which necessitate substantial efforts for curation, has become the primary catalyst for the swift progress of deep learning. Concurrently, protecting these datasets is paramount for the well-being of the data owner. Dataset ownership verification emerges as a crucial method in this domain, but existing approaches are often limited to supervised models and cannot be directly extended to increasingly popular unsupervised pre-trained models. In this work, we propose the first dataset ownership verification method tailored specifically for self-supervised pre-trained models by contrastive learning. Its primary objective is to ascertain whether a suspicious black-box backbone has been pre-trained on a specific unlabeled dataset, aiding dataset owners in upholding their rights. The proposed approach is motivated by our empirical insights that when models are trained with the target dataset, the unary and binary instance relationships within the embedding space exhibit significant variations compared to models trained without the target dataset. We validate the efficacy of this approach across multiple contrastive pre-trained models including SimCLR, BYOL, SimSiam, MOCO v3, and DINO. The results demonstrate that our method rejects the null hypothesis with a $p$-value markedly below $0.05$, surpassing all previous methodologies. Our code is available at https://github.com/xieyc99/DOV4CL.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this paper, the authors investigate the challenge of protecting high-quality open-source datasets from misuse by individuals who lack ownership verification. To address this issue, they propose a method that exploits a natural property of contrastive learning: the distinct distances between seen and unseen examples. Specifically, they demonstrate that these distances often appear significantly larger for unseen examples compared to their seen counterparts. The authors conduct an extensive set of experiments with various models and datasets to evaluate the efficacy of their proposed approach, which yields substantial improvements over existing baselines.

### Strengths
1. The research topic is important.
2. The authors conduct many experiments.
3. The performance is strong compared with baselines.

### Weaknesses
1. While the authors present distance metrics for dsus and dsdw, I believe it would be beneficial to include some visualizations.
2. Contrastive learning is currently a hot research area in computer vision, but the proposed methods appear to be limited to it, which may restrict their broader applicability.
3. The distances between examples are influenced by many factors beyond seen and unseen examples, including generalization capabilities and augmentations. I have concerns that the results presented may not fully support the claims.

### Questions
1. Does the proposed method work with the CLIP model, which also utilizes contrastive learning for pre-training? 
2. Are the findings influenced by the number of training classes and examples? For instance, if Msus has 10k classes and Msdw has only 10, do the proposed methods work as expected?
3. How sensitive is our method to the choice of augmentations? Specifically, we consider scenarios where new augmentation techniques are introduced during evaluation but were not present during training of Msus. In such cases, the distances between outputs from Msus may increase, even when it used public data?

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
4

### Summary
This work proposes the first dataset ownership verification method specifically for self-supervised pre-trained models using contrastive learning. The paper identifies significant variations in unary and binary instance relationships within embedding spaces when models are trained with specific datasets, compared to those trained without them. It introduces the concept of "contrastive relationship gap," a novel technique for verifying dataset ownership in contrastive pre-trained models. Extensive experiments demonstrate the approach's effectiveness, with a p-value significantly below 0.05, surpassing previous methods.

### Strengths
1. The paper introduces a novel method for dataset ownership verification (DOV) specifically tailored for contrastive pre-trained models, addressing a critical need in data rights protection.
2. The paper introduces the concept of "contrastive relationship gap," providing a clear technical approach to differentiate the model's performance on training and non-training datasets.
3. The method has been validated across multiple contrastive pre-trained models, including SimCLR, BYOL, SimSiam, MoCo v3, and DINO, demonstrating its broad applicability.
4. Experimental results show that the method can significantly outperform previous methodologies with a high probability of rejecting the null hypothesis (p-value well below 0.05).

### Weaknesses
1. As paper illustrated in limitations and conclusion, the method is primarily effective for encoders pre-trained with contrastive learning and may not perform well with other self-supervised learning pre-training methods like Masked Image Modeling (MIM).
2. The method lacks comparisons with enough baselines in the experimental section to clearly show its superiority.
3. CONTRASTIVE RELATIONSHIP GAP Part is difficult and too mathematical to understand. The authors could improve the writing to make it easier for the reader to understand.

### Questions
1. Could the authors provide both theoretical and experimental insights into the role of comparative learning within your approach?
2. Authors only compare the method with two baselines: DI4SSL and EncoderMI. Could the authors compare more recent related works?
3. Table.5 is missing EncoderMI time cost. Could author afford more information about time cost of other baselines?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a method for protecting datasets from infringement in contrastive learning scenarios by introducing two relationships: unary and binary. The unary relationship assesses the clustering ability of representations for augmentations of a known sample, while the binary relationship evaluates the separation between representations of two known samples. Using these properties, the authors define distance metrics to evaluate protected training data (known to both the suspect and defender) and secret data (unseen by the suspect but known to the defender). Since suspect models are overfitted to training data and have not seen the secret data, they exhibit different behavior for these two data types, which the method uses to identify suspect models.

### Strengths
This paper addresses an important and novel problem—dataset copyright protection in contrastive learning. The authors provide a comprehensive range of experiments, and the proposed method consistently demonstrates outstanding results across all tested settings.

### Weaknesses
I have several concerns: 


1. The proposed unary and binary relationships align with the goals of contrastive learning, which promotes close representations for variants of the same sample and separation for different samples. The authors rely on overfitting to training data for verification, but as contrastive learning improves, this approach may be less effective. Enhanced contrastive learning might eventually generalize representations, clustering representations from single-sample into a single point, and representations obtained from $N$ distinct samples into $N$ separate points, even for unseen data. Though this is an idealized scenario, this aligns with the direction of contrastive learning research, so verification should be robust to and able to coexist with advances in contrastive learning.


2. The proposed method is similar to verification approaches in supervised learning that use differences in confidence scores or accuracies between training and test data. As Guo et al. [1] demonstrated, task performance differences between seen and unseen data are well-documented, but they are not commonly used for verification due to two primary reasons:
1) Future research is expected to reduce overfitting and improve generality, as noted in Comment 1.
2) An adversary could argue that similar samples exist in their training data by chance, complicating proof that observed low p-values are due to protected data.

To address these, verification metrics should be designed independently of task performance, ensuring that evidence cannot naturally occur by chance. Given this, the proposed method may lack admissible evidence of dataset infringement.

[1] Chuan Guo et al. On Calibration of Modern Neural Networks

3. Given the challenges noted above, many dataset protection methods in supervised learning use backdoor attacks or data poisoning. There are also backdoor attack studies specific to contrastive learning, such as Zhang et al. [2] and Carlini et al. [3]. The authors, however, only compare their method to a model protection technique and a unary-only method (e.g., EncoderMI). I suggest adding comparisons with established backdoor and data poisoning methods for contrastive learning.

[2] Zhang et al. Data Poisoning-based Backdoor Attacks to Contrastive Learning

[3] Carlini et al. Poisoning and Backdooring Contrastive Learning


4. Section 4.5.2 is critical, as contrastive learning is often used as a pretraining method, and adversaries are more likely to release fine-tuned models. Thus, verification post-fine-tuning is essential. However, this section only states that experiments were conducted, without presenting results in the main text. It references "Table 7 in Appendix A.6," which are outside the main manuscript. Ideally, essential content should be included in the main text, with the appendix for supplementary details. Additionally, details on the experiments are missing from the appendix, and Table 7 should include downstream performance results, as small learning rates could affect the reported outcomes. Additionally, Figure 3 occupies too much space; it would be better to reduce its size and include more analysis results directly in the main manuscript.


5. The authors state that they "focus on the black-box setting where defenders have no information about other training configurations (e.g., loss function and model architecture) and can only access the model via Encoder as a Service (EaaS)" and that "defenders can only retrieve feature vectors via the model API." However, Section 4.5.2 notes, "we can only use the predicted probability vectors of the input samples," which seems inconsistent. In a true black-box setting, I would expect only the predicted class ID, not output logits or probability vectors, to be available.



6. The analysis related to the amount of $D_{alt}$ in Figure 4 is essential but lacks explanation in Section 4.4.2. There is no clarification on how the authors control the ratio, whether by increasing $D_{alt}$ or reducing $D_{pub}$, or on what each point in Figure 4 represents. Since the change in log(p) for $D_{pub}$ suggests a controlled amount of $D_{pub}$, it may not be appropriate. With a fixed $D_{pub}$, only the amount of $D_{alt}$ should be adjusted.

### Questions
See above

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a novel method for dataset ownership verification (DOV) specifically tailored for contrastive pre-trained models in self-supervised learning. The method utilizes two observations about contrastive learning: the unary and binary relationships in the embedding space of models trained on a specific dataset. These observations are exploited through a contrastive relationship gap metric, calculated between the suspected model and a shadow model pre-trained without the dataset in question. Comprehensive experiments across datasets and models demonstrate that the method effectively detects unauthorized dataset usage, outperforming baseline techniques.

### Strengths
1. **Innovative Approach:** The method uniquely applies to self-supervised models by leveraging characteristics of contrastive learning, filling a gap in current DOV methods that primarily target supervised learning.
2. **Black-box Applicability:** The approach is suitable for black-box scenarios, which is practical and aligned with real-world applications where full model access is unavailable. The approach demonstrates robust performance across different datasets (e.g., CIFAR, ImageNet) and architectures, indicating generalizability.
3. **Effective Performance:** Results show high sensitivity, specificity, and AUROC scores, suggesting that the proposed metric reliably distinguishes between legitimate and unauthorized dataset use.
4. **Efficiency:** Compared to alternatives, the method is computationally efficient, which enhances its applicability to large datasets like ImageNet.
5. **Thorough evaluation:** The paper is very comprehensive, they make specific claims and justify them with solid experiments and results.

### Weaknesses
1. **Dependency on Feature Representation Access:** The method requires access to feature representations, which might not be feasible in all practical scenarios, as many services limit this access for security reasons.

2. **Limited Application to Non-Contrastive Pre-Trained Models:** The method’s effectiveness is constrained to contrastive learning. Other prevalent pre-training strategies, such as masked image modeling (MIM), are not effectively addressed, potentially limiting applicability. However, the authors make a clear claim and explain this as a limitation, thus I think the weaknesses are ok given this is one of the earlier works.

### Questions
1. Could early stopping or other techniques be leveraged to reduce the effectiveness of this detection method? A brief section on possible attack approaches in the appendix may be considered.

The paper is quite extensive very thorough! :)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel method for verifying dataset ownership in self-supervised contrastive learning models, protecting curated datasets from unauthorized use. The proposed approach leverages the "contrastive relationship gap," capturing distinct similarities in representations for models trained on a defender's dataset versus unrelated data. Through a three-step process, the method identifies unauthorized usage efficiently and with high accuracy. Experimental results on several datasets and models show that this approach outperforms existing methods in accuracy, computational efficiency, and robustness, even under privacy-preserving settings like DP-SGD, making it a promising tool for dataset security in machine learning.

### Strengths
- The paper presents a unique dataset ownership verification (DOV) method specifically tailored for self-supervised contrastive learning models. This is a valuable addition to the field, as existing DOV methods are generally focused on supervised or non-contrastive learning models, leaving a gap that this paper addresses.
-The authors conduct extensive experiments across multiple datasets (CIFAR10, CIFAR100, SVHN, ImageNet variants) and contrastive learning architectures (SimCLR, BYOL, MoCo, DINO), demonstrating the method's effectiveness and generalizability. This robust experimental setup strengthens the validity of the proposed approach.
- By requiring only a small subset of the defender’s data for verification, the proposed method is more computationally efficient than baseline methods like D4SSL, which require access to the entire dataset, making it suitable for large-scale applications.

### Weaknesses
- While the proposed method offers a novel approach to dataset ownership verification, its applicability is limited to contrastive learning models. Many self-supervised learning models use objectives other than contrastive learning, so expanding the method’s scope could enhance its impact. However, this limitation is relatively minor.
- In line 488, the authors state that "the private training method does not affect our verification results," but this claim is based on experiments using only DP-SGD with a high privacy budget (epsilon=50). To support this claim, it would be beneficial to test the method under stronger privacy settings or with alternative privacy-preserving techniques, such as differentially private generative models.

### Questions
- How effective is the approach when the suspect model undergoes adaptation, such as fine-tuning on a different dataset or with altered weights? Is the method still robust under such conditions, or are there specific scenarios where fine-tuning could mask the original dataset’s influence on the model?
- How sensitive is the proposed DOV method to variations in the suspect model’s hyperparameters or pre-training dataset characteristics (e.g., domain-specific data)?

### Soundness
3

### Presentation
3

### Contribution
2
