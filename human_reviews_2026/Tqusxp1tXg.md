# Distributed PATE and CaPC on a DIET: Private Knowledge Transfer without Public Data or Private Inference

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
The PATE algorithm is one of the canonical approaches to private machine learning. It leverages a private dataset to label a public dataset, enabling knowledge transfer from teachers to a student model under differential privacy (DP) guarantees. However, PATE's reliance on public data from the same distribution as the private data poses a fundamental limitation, particularly in domains such as healthcare and finance, where in-distribution public data is typically unavailable. In this work, we propose DIET-PATE which overcomes this limitation. Therefore, it combines programmatically generated data and data-free knowledge distillation. Our experiments demonstrate that DIET-PATE closely matches the performance of standard PATE, despite the absence of in-distribution public data. Furthermore, we show that our approach seamlessly extends to distributed collaborative learning with CaPC. In this setting, only PATE-based learning can be used to provide DP guarantees, as teacher models are trained by different entities and exchange knowledge solely via labels. By eliminating the need for in-distribution data during knowledge transfer, our method removes CaPC’s reliance on private inference with encrypted data, substantially reducing computational overhead and, for the first time, enabling the use of more complex models and learning tasks. Moreover, leveraging programmatically generated data allows parties in CaPC to jointly train a global model, rather than just improving local ones, thereby achieving significantly higher utility. These advances extend the practicality of distributed private learning with PATE and CaPC to sensitive and complex domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DIET-PATE, which enables PATE-style private knowledge transfer without relying on public in-distribution data by leveraging programmatically generated data and data-free knowledge distillation. It further extends the approach to a distributed setting, replacing expensive encrypted inference with standard inference and enabling a shared student model.

### Strengths
1. The paper addresses a practical limitation of PATE by removing the dependence on public in-distribution data. The extension to distributed learning improves efficiency by eliminating costly encrypted inference.
2. The paper is clearly written with a well-structured methodology description, making the technical workflow easy to follow.

### Weaknesses
1. The method stays close to the classic teacher-student paradigm for privacy. It mainly combines known techniques rather than introducing a clear new algorithmic idea.
2. The efficiency analysis does not account for the computational and engineering cost of generating the synthetic data at scale.
3. While the introduction mentions finance and other sensitive domains, the experiments use very basic benchmarks like MNIST (TissueMNIST, while biomedical, is still a standardized image-classification benchmark). This weakens claims about generalization to real-world settings.

### Questions
Have the authors evaluated or estimated the computational and engineering cost of generating synthetic data at scale, and how would this overhead impact the claimed efficiency gains in practical deployments?

Can the authors demonstrate the practicality of their approach on more realistic and domain-relevant datasets to support the generalization claims?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces DIET-PATE, which removes the dependence of PATE on in-distribution public data by combining programmatically generated synthetic data and data-free knowledge distillation. Teacher and student models are initialized on synthetic data, teachers fine-tune on private subsets, and knowledge transfer is performed on synthetic samples using the current batch normalization statistics to mitigate distribution shift. The paper also proposes DIET-CaPC, which integrates DIET-PATE into the collaborative CaPC framework to replace encrypted private inference with efficient plaintext synthetic inference, thereby enabling joint student training across parties.

### Strengths
- The work addresses the limitation of PATE, which is a dependence on public data, therefore broadening applicability to domains like healthcare. 
- A good engineering contribution by extending the setting into a collaborative setting.

### Weaknesses
- In Sec 3, the paper states that the current BN statistics must always be used during inference, but it's unclear how this works for single-sample inference, where BN statistics may be unstable. 
- The transition from teacher-student to collaborating parties is confusing to the reader. The mapping between entities is unclear and should be introduced more gradually and consistently. 
- The method's success heavily depends on the quality of the generated data. If the synthetic data distribution is too far from private data, transfer may fail; this limitation is not clearly discussed. 
- Despite claims of enabling learning in complex, sensitive domains, experiments use simple datasets (MNIST, CIFAR-10, TissueMNIST). These are insufficient and unconvincing to validate the claims. 
- TissueMNIST results underperform DP-SGD and lack a discussion of practical relevance for medical tasks. 
- Table 2 is redundant, and it is not necessary to include it; the gap is expected, and a textual summary would suffice. 
- One other important point is about DIET-CaPC. While integrating it into a collaborative learning (CL) algorithm is a natural extension, the paper treats CaPC as the only CL framework. This is a narrow view, as a broad landscape of CL paradigms exists, which could equally benefit from the proposed mechanism. A stronger paper would evaluate or at least discuss why CaPC was chosen over other frameworks, and how DIET-PATE could integrate with or outperform more scalable, widely used methods. As written, the work doesn't convincingly argue its generality.

### Questions
Please refer to the above weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes DIET-PATE, a framework designed to address the limitation of the PATE framework's reliance on in-distribution public data. It achieves private knowledge transfer by leveraging programmatically generated data and data-free knowledge distillation. Furthermore, this approach is extended to distributed collaborative learning with CaPC, eliminating the need for costly private inference on encrypted data. Experimental results demonstrate DIET-PATE's effectiveness.

### Strengths
1. The paper is well-structured, presenting a clear delineation of the proposed methodology, experimental setup, and results. 
2. The proposed DIET-CaPC circumvents the prohibitive cost of private inference in CaPC by replacing encrypted private queries with programmatically generated data. This design is novel and well-executed, enhancing both engineering feasibility and practical applicability.

### Weaknesses
1. In Section 4.2 (DIET-CaPC), the paper claims that "the shared student model can then also be used to label new locally obtained data and improve teacher i, without incurring any additional privacy cost." I'm not really doubtful of the reasonableness of the claims, but I think the paper should quantify and qualify them better.

2. How should the relationship between DIET-CaPC and the original CaPC be understood? It appears to prioritize training efficiency and model shareability over personalized learning for querying parties. Should this be viewed as a shift in objectives rather than a technical extension?

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes DIET-PATE, a variant of PATE that removes the need for public data drawn from the same distribution as the private dataset. The key idea is to (1) pretrain both teachers and student on large procedurally generated image datasets, and (2) apply data-free knowledge distillation with adapted batch-normalization statistics to reduce the covariate shift between synthetic and private data. The authors extend the approach to a distributed collaborative setting, DIET-CaPC, which integrates DIET-PATE into the CaPC framework and replaces expensive private inference on encrypted data with standard inference on synthetic queries, enabling the training of a single shared student. Experiments on MNIST, CIFAR-10, and TissueMNIST show that DIET-PATE approaches or exceeds standard PATE under comparable privacy budgets when in-distribution public data is absent, and that DIET-CaPC yields substantial efficiency gains and better privacy–utility trade-offs than CaPC.

### Strengths
1. Addresses a real limitation of PATE and CaPC.
The paper tackles a well-known practical bottleneck of PATE: the dependence on public in-distribution data, which is often unrealistic in sensitive domains such as healthcare or finance. Likewise, DIET-CaPC directly targets CaPC’s main pain points—slow private inference and fragmented privacy budgets leading to only modest local improvements. The problem formulation and motivation are clear and compelling.

2. Extension to distributed collaboration:
DIET-CaPC is a meaningful extension, e.g., replacing encrypted private queries with synthetic queries removes the need for HE/MPC-based private inference and allows larger models compared to the CryptoNet-style networks used in CaPC. The protocol description (teachers, student party, privacy guardian) is clear, and the empirical comparison shows both speedups and more favorable privacy–utility trade-offs. 

3. Extensive evaluation and ablations: The experimental section is fairly thorough.

4. Clarity and reproducibility:
The paper is overall well-written and easy to follow. Implementation details are described in the appendix with specificity to help reproduce the experiments.

### Weaknesses
1. Incremental methodological novelty: It seems like DIET-PATE is an integration of PATE, programmatically generated pretraining, and data-free KD with BN “current statistics.” Each component is taken from prior work; the main novelty lies in combining them to remove PATE’s reliance on public in-distribution data and in porting this combination into CaPC. While this integration is practically useful, the methodological contribution may be seen as somewhat incremental for a top-tier venue, especially since there is no new privacy analysis, theoretical insight, or substantially new algorithmic mechanism beyond the pipeline design.

2. Limited experimental scope and scalability:
The evaluation focuses on image classification tasks with relatively standard-sized models (ResNet10/18) and modest-resolution data (MNIST, CIFAR-10, TissueMNIST). It remains unclear how DIET-PATE behaves with larger architectures or more complex modalities (e.g., high-resolution images) where pretraining and synthetic generation might be more challenging. Given that the paper emphasizes improved scalability for DIET-CaPC, additional experiments or a more in-depth discussion of scaling bottlenecks would be valuable.  ￼

3. Lack of security/robustness analysis:
The work is framed around privacy, but does not investigate robustness to adversarial manipulation of the synthetic data or the protocol. For example, what happens if the synthetic generation process is biased or partially corrupted, or if an adversary contributes malicious teachers in DIET-CaPC? Even a discussion of such risks and possible mitigations would enhance the paper’s practical relevance for sensitive domains.

4. Intuitive design choices:
In DIET-PATE, teachers fine-tune only the last layer on private data, and the student also fine-tunes a small fraction of parameters; the paper shows empirically that this helps alignment and performance, but these choices are not deeply analyzed.

### Questions
Please refer to 'Weaknesses'.

### Soundness
3

### Presentation
3

### Contribution
2
