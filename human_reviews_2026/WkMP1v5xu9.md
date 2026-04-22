# Preserving Representation In Continual Learning via Feature-Preserving Fine-Tuning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
In real-world applications, deep learning models must continually adapt to sequentially arriving tasks without access to previous data. Although pre-trained foundation models show generalisation and zero-shot abilities, fine-tuning them in a continual learning setting often leads to representation degradation. In this study, we firstly systematically evaluate several recent feature-preserving fine-tuning methods (L2-SP, FTP, WiseFT and ImpReg) in continual learning scenario using a large scale pre-trained foundation model. We further explore the 
effectiveness of full fine-tuning (FullFT) versus parameter-efficient fine-tuning (PEFT) and propose a novel two-stage fine-tuning strategy, 
PEFT+Cons, designed to balance stability and plasticity by combining PEFT with task-specific knowledge consolidation. Extensive experiments on the CIFAR-100 and ImageNet-R benchmark datasets demonstrate that our proposed PEFT+Cons approach effectively prevents representation forgetting while enhancing task-specific knowledge retention.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the challenge of representation degradation in continual learning (CL) for pre-trained foundation models. The authors systematically evaluate several feature-preserving fine-tuning methods in a class-incremental learning setting using a CLIP ResNet-50 model. It proposes a two-stage strategy, PEFT+Cons, which integrates PEFT with task-specific knowledge consolidation to better balance stability and plasticity.

### Strengths
This paper explored the challenges of representation degradation in continual learning with pre-trained foundation models. It proposed a two-stage fine tuning strategy that combines PEFT with task-specific knowledge consolidation, and provide insights into its effectiveness in mitigating representational forgetting.

### Weaknesses
The innovation of this paper is limited. This paper primarily provides an empirical exploration of how combining FullFT or PEFT with different feature-preserving methods affects the maintenance of representational capacity in continual learning. The analysis of the underlying reasons is relatively empirical and lacks an investigation into the fundamental causes; the analysis is somewhat superficial and not sufficiently in-depth. The proposed method also appears to be a straightforward combination of FullFT or PEFT with various feature-preserving techniques, raising significant doubts regarding its novelty.

Furthermore, the method proposed in this paper is not compared with other approaches based on pre-trained models. The datasets and continual learning scenarios used for validation are too limited, making the experimental support insufficient. In the Related Work section, a comprehensive analysis of current continual learning methods based on pre-trained models should be provided, but the discussion of such methods in this section is not thorough.

### Questions
See Weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the representation degradation problem in continual learning of large pre-trained models, focusing on how fine-tuning strategies affect the preservation of generalizable representations. They find that while feature-preserving methods mitigate catastrophic forgetting under FullFT, representation drift still occurs. In contrast, PEFT substantially reduces forgetting but at the cost of lower task-specific accuracy. To address this trade-off, the paper proposes PEFT+Cons, a novel two-stage fine-tuning strategy that first performs PEFT to stabilise pre-trained features, then applies feature-preserving FullFT for task-specific consolidation.

### Strengths
1. The introduction of representation-level metrics (RF, UTA, FinalGLP) extends evaluation beyond accuracy, offering more interpretable and transferable measures for representation stability.
2. The two-stage PEFT+Cons procedure is intuitive, reproducible, and well-motivated by observed limitations of both FullFT and PEFT.
3. The authors’ interpretation of the attention pooling block as a naturally stable representation aggregator provides useful intuition that may inspire architectural research in continual learning.

### Weaknesses
1. The study’s conclusions are based solely on CLIP-ResNet-50 backbones and vision-only tasks. It remains uncertain whether the same representational preservation trends would hold for other backbones and other tasks.
2. While the empirical findings clearly demonstrate the stability benefits of feature-preserving fine-tuning, the paper provides no theoretical framework explaining why these methods maintain representational similarity in continual settings.

### Questions
1. Can the consolidation stage be scheduled adaptively based on a forgetting metric rather than fixed intervals?
2. Could combining PEFT+Cons with replay-based methods further improve the stability–plasticity trade-off?

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
This paper presents a framework for preserving representations in continual learning, addressing the degradation of pre-trained foundation models when fine-tuned sequentially. The authors systematically evaluate several feature-preserving fine-tuning (FPFT) techniques (L2-SP, FTP, WiseFT, and ImpReg) in class-incremental learning with CLIP-ResNet-50. They highlight that naive full fine-tuning (FullFT) causes severe representational forgetting, while parameter-efficient fine-tuning (PEFT), which only updates the attention pooling block, greatly alleviates this issue but limits adaptability to new tasks. To balance stability and plasticity, the paper introduces PEFT+Cons, a two-stage fine-tuning strategy that first performs PEFT to stabilize pre-trained features, then applies feature-preserving FullFT to consolidate task-specific knowledge. This approach enables both robust representation retention and effective adaptation. Experiments on Split CIFAR-100 and Split ImageNet-R demonstrate that PEFT+Cons with FTP achieves the best trade-off, substantially improving final task accuracy and representational robustness while maintaining strong generalization to unseen tasks. Overall, the study provides new insights into how feature-preserving regularization and modular fine-tuning interact in foundation models. It underscores the importance of designing fine-tuning strategies that protect pre-trained representations while supporting continual adaptation, marking a step toward more stable and generalizable continual learning with vision-language foundations.

### Strengths
1.The paper is clearly written, and its overall structure flows logically from the underlying motivation and problem formulation to the methodological design and comprehensive experimental validation.

2.The experimental design is rigorous and goes beyond conventional evaluations based solely on classification accuracy. Instead, it employs a comprehensive set of representation-level metrics—such as Representational Forgetting (RF), Unseen Task Accuracy (AvgUTA), and Global Linear Probe Accuracy (FinalGLP)—thereby providing a more nuanced and convincing assessment of the model’s effectiveness.

3.Building on comprehensive experimental analyses, the paper introduces PEFT+Cons, a novel two-stage fine-tuning strategy designed to achieve a balanced trade-off between stability and plasticity. In this approach, Stage 1 employs PEFT to preserve robust representations and maintain generalization, while Stage 2 applies a constrained FullFT to consolidate task-specific knowledge. Empirical results demonstrate that FTP with this strategy consistently outperforms other fine-tuning methods across multiple evaluation metrics.

4.The paper carefully characterizes the behaviors of both FullFT and PEFT strategies through well-designed experiments, offering valuable insights into their respective strengths and limitations. This analysis provides a clear rationale for why the proposed PEFT+Cons method achieves better performance under continual learning scenarios.

### Weaknesses
1.The empirical evaluation uses only two datasets (SplitCIFAR-100 and SplitImageNet-R) and a single backbone (CLIP-ResNet-50). While these are reasonable starting points, the paper does not demonstrate whether conclusions generalize to (a) other backbone families (e.g., ViTs or larger CLIP variants), (b) longer task sequences or different class granularities, and (c) warm start setting[1].

2.The work attributes PEFT’s resistance to forgetting to properties of the attention pooling block, yet does not examine alternative PEFT designs or architectural modifications that might further improve plasticity without FullFT (e.g., small adapter modules, layer-wise low-rank updates, or selective unfreezing). It would be beneficial for the authors to conduct additional experiments to investigate the effectiveness of PEFT+Cons with other modules which are commonly used in peft PTM methods[2,3].

3.The paper reports that the combination of FTP with PEFT+Cons yields the best overall performance, yet provides limited mechanistic explanation for why FTP consistently outperforms other feature-preserving approaches within this two-stage framework. Moreover, the performance of other methods under the PEFT+Cons strategy is generally lower on the FTA, AvgUTA, and FinalGLP metrics compared to their results under the PEFT strategy without consolidation. A deeper analysis of these results would offer valuable insights and strengthen the authors’ conclusions.

4.The experiments presented in this paper show that the FTP combined with the PEFT+Cons strategy achieves the best performance on two benchmark datasets compared with other fine-tuning methods. However, this evidence alone is not fully convincing. The applicability and generality of the proposed approach could be further strengthened by evaluating its performance with additional continual learning methods [4–7] (e.g., LwF, EWC, iCaRL, ZSCL). 

[1] Elastic feature consolidation for cold start exemplar-free incremental learning.
[2] Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning.
[3] LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning.
[4] Learning without Forgetting
[5] Overcoming catastrophic forgetting in neural networks
[6] iCaRL: Incremental Classifier and Representation Learning
[7] Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models

### Questions
1.Why was CLIP-ResNet50 pre-trained on ImageNet-1K chosen as the backbone? Given that CLIP models are already trained on large-scale and diverse datasets, the necessity of additional pretraining on ImageNet-1K is unclear. Furthermore, evaluating the proposed method with different backbone architectures would further validate its generality and strengthen the evidence for its effectiveness.

2.All experiments appear to have been conducted using a single random seed, raising concerns that the chosen hyperparameter settings may be overfitted to this specific configuration. Evaluating the method across multiple random seeds would provide a more reliable assessment of its robustness and generalization.

3.As shown in Table 1, the RF of the naive method combined with PEFT strategy is lower than that of L2-SP, FTP, and WiSE-FT on the benchmark datasets. What accounts for the reduced forgetting observed in the Naïve method? Moreover, the final GLP of the naive method is comparable to that of other approaches while requiring significantly less computational cost. What factors contribute to these results?

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
3

### Summary
This paper investigates the problem of representation degradation in continual learning with pre-trained foundation models. The authors systematically evaluate four recent feature-preserving fine-tuning approaches (L2-SP, FTP, WiseFT, and ImpReg) under class-incremental continual learning using a CLIP-ResNet-50 backbone on Split CIFAR-100 and ImageNet-R. They compare full fine-tuning (FullFT) and parameter-efficient fine-tuning (PEFT) and introduce a two-stage method (PEFT+Cons) that combines PEFT with feature-preserving consolidation. Results indicate that PEFT+Cons, particularly when paired with FTP, provides improved balance between stability and plasticity, reducing representational forgetting without substantially constraining task-specific adaptation.

### Strengths
1. The systematic comparison of FullFT and PEFT strategies reveals nuanced insights into when and why representational forgetting occurs, using well-chosen metrics.
2. Demonstrates up to over FTA points over the next-best method 
3. Figures provide a clear illustration of the improvement induced by the proposed approach.

### Weaknesses
Limited Novelty in Core Mechanisms: The PEFT+Cons procedure, while well-executed and carefully evaluated, essentially combines PEFT and existing feature-preserving FullFT approaches in a sequential pipeline. It does not introduce fundamentally new algorithms or theoretical insights into representation preservation. The novelty largely resides in the two-stage orchestration, not in a new method or principle.

### Questions
Can the authors justify the novelty of the proposed approach over the simple recombination of existing approaches from the same domain?

### Soundness
2

### Presentation
2

### Contribution
2
