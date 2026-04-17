# Towards Adversarially Robust CLIP: A Hierarchical Model Fusion Method Using Optimal Transport

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
In recent years, multimodal models such as CLIP have achieved impressive performance but remain vulnerable to adversarial perturbations. Although adversarial training can enhance robustness, it often leads to overfitting toward specific attack types. One solution for improving generalization is to integrate multiple diverse and adversarially trained submodels, but this strategy could incur high test-time cost. To achieve a promising tradeoff between robust generalization and efficiency, we consider to design an optimal transport (OT) based model fusion method, which is called ``HOT-CLIP (Hierarchical Optimal Transport Fusion for CLIP)". Although several OT based model fusion methods have been proposed before, they cannot be easily adapted to solve our problem, since they may suffer the issues like parameter misalignment when dealing with highly diverse and multimodal submodels. Our proposed method constructs diverse submodels by varying both attack methods and textual prompts, and integrates them via a hierarchical two-level OT fusion method. The intra-attack fusion first aligns and merges models within the same attack family, and the inter-attack fusion subsequently combines these aligned models across different attacks. Through this carefully crafted fusion strategy, HOT-CLIP can significantly improve the accuracy for alignment and reduce the total occupied memory. More importantly, the obtained robust visual encoder can be deployed without additional inference-time cost. In our experiments, the results on multiple vision-language tasks demonstrate that HOT-CLIP can greatly enhance the model's adversarial robustness while maintaining competitive clean accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the problem of adversarial robustness in multimodal models, particularly CLIP. While CLIP achieves strong performance on vision-language tasks, it remains highly vulnerable to adversarial perturbations. To address this, the authors propose HOT-CLIP, a Hierarchical Optimal Transport–based model fusion framework that enhances robustness without adding inference overhead. The method first constructs diverse adversarially trained CLIP sub-models using different attack strategies and prompt templates. Then, it performs a two-level hierarchical fusion—language-level and vision-level—using optimal transport (OT) to align and merge model parameters effectively. This hierarchical OT fusion improves alignment among heterogeneous models, achieving better adversarial robustness while maintaining clean accuracy.

### Strengths
1.	The proposed two-stage framework—comprising sub-model generation and hierarchical optimal transport (OT) fusion—is conceptually clear and technically sound. 
2.	Unlike conventional ensemble-based defenses, HOT-CLIP does not introduce any additional computational overhead during inference. The fused model maintains the efficiency of a single model, which makes the method attractive for real-world deployment of large-scale vision-language systems such as CLIP.
3.	The paper provides extensive experimental validation across multiple multimodal tasks, including zero-shot classification, image captioning, and visual question answering. The results consistently demonstrate that the proposed method improves adversarial robustness while preserving clean accuracy, supporting the method’s general effectiveness.

### Weaknesses
1.	Although the hierarchical structure helps control memory usage, the overall pipeline involves multiple rounds of adversarial training and OT optimization. This increases implementation complexity and computational cost, which may limit practical adoption.
2.	All experiments are conducted on CLIP and its variants. The absence of results on other vision-language architectures (e.g., BLIP) leaves open the question of whether HOT-CLIP generalizes beyond the CLIP family.
3.	The paper primarily focuses on empirical evidence. It lacks a deeper theoretical explanation of how optimal transport specifically contributes to parameter alignment and robustness enhancement. A stronger theoretical foundation would improve the paper’s impact and clarity.
4.	The evaluation is restricted to linf-bounded attacks (2/255, 4/255), without considering other perturbation types such as l2 norm

### Questions
I prefer to give borderline(score 5). Please see the weakness. The problem formulation is sound, and the challenge is well-motivated. However, the solution is mainly an engineering-level improvement built on existing fusion and alignment techniques, with limited theoretical support for its claimed effectiveness.

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
3

### Summary
This paper proposes HOT-CLIP (Hierarchical Optimal Transport Fusion for CLIP), a method to enhance adversarial robustness of vision-language models. The approach first trains diverse submodels using different adversarial attacks (FGSM, PGD, MIM) and textual prompts, then fuses them using a two-level hierarchical optimal transport method. The first level (intra-attack) fuses models within the same attack family but different prompts, while the second level (inter-attack) combines these fused models across different attacks. Experiments on image classification, VQA, and image captioning demonstrate improvements in adversarial robustness while maintaining competitive clean accuracy.

### Strengths
+ First work to systematically apply optimal transport-based model fusion to adversarial robustness in multimodal models, addressing a well-motivated problem.
+ Experiments span multiple tasks (classification, VQA, captioning) and datasets, with consistent improvements shown across different perturbation budgets.
+ The hierarchical fusion strategy reduces memory requirements from O(|A||T|·U) to O(max{|A|,|T|}·U), making the approach more deployable.
+ Strong empirical results: Relative improvements of ~2.6% (classification), ~20.4% (VQA), and ~16.5% (captioning) in robust accuracy over baselines.

### Weaknesses
+ The paper can use more theoretical analysis or deeper insights into when and why the geometric alignment via OT succeeds for adversarially diverse models.
+ While inference is efficient, training requires multiple submodels. 
+ AutoAttack is the only adversarial evaluation method used.

### Questions
+ Why does hierarchical fusion work better than direct fusion? 
+ What is the total training time/cost compared to baselines? Is this practical for larger models?
+ The method shows noticeable drops in clean accuracy compared to vanilla CLIP (74.9→69.9 on ImageNet). Is this tradeoff acceptable?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a new framework HOT-CLIP to enhance the adversarial robustness of large vision-language models like CLIP. Standard adversarial training often overfits to specific attack types and that ensembling multiple adversarially trained submodels improves robustness but is computationally expensive. To address this trade-off, they introduce HOT-CLIP, which constructs diverse CLIP submodels by varying both attack methods and textual prompts, and then fuses them using a two-stage hierarchical optimal transport method. Experiments on zero-shot image classification, image captioning, and visual question answering show that HOT-CLIP substantially improves adversarial robustness while maintaining competitive performance on clean data.

### Strengths
1. The proposed hierarchical strategy effectively alleviates neuron misalignment issues that often arise when fusing diverse models, outperforming naive averaging, or single-level OT methods.
2. Although multiple adversarial sub-models are required during training, only a single fused model is needed for inference, significantly improving efficiency.
3. Comprehensive experiments demonstrate that the proposed method remains robust across multiple tasks and attacks while maintaining competitive performance on clean data.
4. The fused visual encoder can be directly used in multimodal LLMs such as LLaVA-1.5 and OpenFlamingo.

### Weaknesses
1. The method has high complexity, as demonstrated in Appendix C.1, Hierarchical OT fusion iteratively computes the cross-layer transfer matrix, aligns neurons, and averages the aligned weights to obtain the fused representation, which limits its practical value.
2. Although inference remains effective, this method requires training multiple adversarial sub-models under various attacks and prompts, resulting in significant computational and resource costs during the training process.
3. The application seems limited as only with CLIP. The authors demonstrate the application of a fusion visual encoder to LLaVA-1.5 and OpenFlamingo, but its transferability to other multimodal architectures remains undiscussed.

### Questions
1. It would be helpful if the authors could provide analysis of computational resources, e.g., training time, GPU memory, etc., to better evaluate the practicality of the proposed method.
2. Does this method still apply to different LVLMs, and whether additional fine-tuning required? Or can the proposed fusion be directly applied to a purely visual encoder?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes HOT-CLIP, a hierarchical optimal transport (OT) based fusion framework to improve the adversarial robustness of multimodal models such as CLIP. The method first performs intra-attack fusion to align submodels within the same attack type, then inter-attack fusion to combine across attack families. The goal is to achieve a balance between robustness and efficiency without increasing inference-time cost. Experiments on several vision–language tasks show improvements in adversarial robustness while maintaining clean accuracy.

### Strengths
1. The hierarchical two-level OT fusion is clearly described.

2. The empirical results are clearly reported and show consistent improvement on benchmarks.

### Weaknesses
1. There is a lack of novelty in the proposed work. The proposed method is largely a direct application of existing optimal transport (OT) fusion techniques, with limited methodological innovation or new theoretical contribution.

2. There is no theoretical justification. The paper lacks formal analysis or theoretical guarantees explaining why the hierarchical OT fusion improves robustness or parameter alignment.

3. The computational analysis is missing. There is no discussion or experiment on computational cost, including the memory and runtime implications of the hierarchical fusion.

4. The insight is limited. The results, while positive, do not provide deeper understanding of why or when the method works, reducing the paper’s scientific value.

### Questions
1. Could the authors include a discussion or measurement of training and fusion cost to support the claim of efficiency?

2. What is the theoretical motivation for using OT over simpler averaging or linear fusion methods?

3. How sensitive is the hierarchical OT fusion to the choice of submodels or the diversity of attacks?

### Soundness
2

### Presentation
2

### Contribution
2
