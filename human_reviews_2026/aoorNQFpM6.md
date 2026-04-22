# VER: Vision Expert Transformer for Robot Learning via Foundation Distillation and Dynamic Routing

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 8

## Abstract
Pretrained vision foundation models (VFMs) advance robotic learning via rich visual representations, yet individual VFMs typically excel only in specific domains, limiting generality across tasks. Distilling multiple VFMs into a unified representation can mitigate this limitation but often yields inflexible task-specific feature selection and requires costly full retraining to incorporate robot-domain knowledge.
We propose VER, a Vision Expert transformer for Robot learning. During pretraining, VER distills multiple VFMs into a vision expert library. We then fine-tune only a lightweight routing network (fewer than 0.4% of parameters) to dynamically select task-relevant experts from the pretrained library for downstream robot tasks. We further introduce Patchwise Expert Routing with Curriculum Top-K Annealing to improve both flexibility and precision of dynamic expert selection. Moreover, VER supports parameter-efficient finetuning for scalable expert utilization and robot-domain knowledge integration. Across 17 diverse robotic tasks and multiple policy heads, VER achieves state-of-the-art performance. We find that VER reduces large-norm outliers in task-irrelevant regions (e.g., background) and concentrates on task-critical regions. More visualizations and codes are available in https://yixiaowang7.github.io/ver_page/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces VER (Vision Expert Transformer for Robot Learning), an approach that consolidates Mixture-of-Experts (MoE) for downstream control tasks. The authors introduced a two stage approach, a routing network to select task relevant experts and a curriculum top-k annealing to improve the dynamic expert selection. The paper reports state-of-the-art performance on multiple robotics benchmarks.

### Strengths
- The paper demonstrates strong empirical results across a large and diverse set of benchmarks
- VER is an important practical contribution to real-world robot deployment and scalability

### Weaknesses
- The proposed mechanism is an important contribution to robotic learning, nevertheless the papers lacks theoretical justification and several design choices seem heuristic engineered. Further principled analysis and justification are needed. 
 
- The ablation study is insufficient to justify several hyper-parameter choices and to justify the complexity of the proposed technique. My major concern here is it fails to isolate the performance gains attributed to each novel mechanism versus the underlying MoE architecture or increased model capacity.

### Questions
- What are the theoretical guarantees or formal analyses regarding how the annealing schedule in CTA impacts expert load balancing or the stability/convergence of the MoE router?

- Beyond empirical performance, what is the formal algorithmic advantage of PER over a standard token-wise or layer-wise routing scheme in the context of dense visual representations?

- The distillation loss uses mutual-information regularization. Please elaborate on the theoretical motivation for this specific regularization in this context.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces VER (Vision Expert Transformer), a novel architecture for robot learning. It addresses the limited generalization of single Vision Foundation Models and the inflexibility of distilling multiple VFMs into a static representation. VER's core contribution is a two-stage framework: 1. Pretraining: Knowledge from multiple VFMs is distilled into a "Vision Expert Library" (VEL) based on a Mixture-of-Experts (MoE) architecture. 2. Fine-tuning: The pretrained experts are frozen, and only a lightweight "Robot Router" is trained to dynamically select task-relevant experts. To enhance expert selection, the authors propose "Patchwise Expert Routing" (PER) and "Curriculum Top-K Annealing" (CTA). CTA starts training with all experts active and gradually anneals the number of active experts to prevent premature convergence. Experiments show that VER achieves state-of-the-art performance.

### Strengths
- This work addresses a practical and important problem: efficiently leveraging multiple powerful VFMs for robotics. The ability to adapt to new tasks by fine-tuning only <0.4% of parameters is highly valuable for resource-constrained robotic systems.
- The work creatively combines multi-teacher VFM distillation with an MoE architecture.
- The paper is well-organized. The experimental evaluation is thorough and robust.

### Weaknesses
I'm sorry, but I'm not very familiar with this field. Please see the Questions section for details.

### Questions
1. The CTA schedule depends on the hyperparameter $S$. How was $S$ determined in your experiments? Is the method sensitive to this choice? I may have missed the relevant ablation studies. Additionally, are there any empirical conclusions that help in selecting this hyperparameter?
2. Could you explain the main differences between VER and the baseline Theia? Table 7 shows that Theia does not use the MoE architecture, while VER has 135% more parameters than Theia. Would it be feasible to replace Theia's network architecture with the MoE architecture for comparison?
3. Could you please elaborate on the training procedure for the "Train-from-Scratch" (TFS) experts in Table 5? Specifically, for the (#DFM=6, #TFS=1) setting, how is this 7th expert initialized and trained? Additionally, would fine-tuning all parameters across 6 DFMs yield better performance? Or introduce a LoRA module for fine-tuning?

### Soundness
2

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
3

### Summary
This paper investigates how to distill a vision encoder from existing vision foundation models for downstream robotic tasks. Unlike existing works that distill a monolithic encoder, this paper utilizes MoE to enable a more diverse, flexible and adaptive feature extraction process. Experimental results show that the proposed method outperforms baseline methods, especially Theia, on a broad range of tasks.

### Strengths
1. The paper is clearly motivated and presented. 
2. The proposed method is evaluated on a broad range of tasks and shows clear advantage compared to baseline methods. 
3. The ablation analysis and visualization are thorough and illustrative to demonstrate how the visual representation differs across different methods and the importance of different design choices.

### Weaknesses
My main concern about this paper is that its novely seems not very significant, more like just replacing existing distillation models for visual encoder in robotic models with MoE. The empirical gain is significant, but I think if we care about absolute performance, it may be better to compare with a broader range of methods for robotic control, such as VLAs that are fine-tuned on the same datasets. For other points, see my questions below.

### Questions
1. It would be good to add a qualitative analysis on the failure reasons of different methods. 
2. Do you fine-tune the vision backbone of the baselines when learning the policy? If yes, do you fine-tune all their parameters? Or use some parameter-efficient methods like LoRA?
3. The problem setting and training pipeline are different, but how does your method compare with VLA methods that first pretrain on large-scale robotic data, then fine-tune on specific downstream tasks? Can your method be extended to replace the visual encoders used in VLA? If yes, does it perform better or not?
4. What is the size of your model at the three different parameter scale? Do you use a smaller MLP dimension in MoE so that the total parameter number is similar to that of Theia? As from Figure 9 it seems that your method has a smilar number of total parameters as Theia, but from Table 4 it seems that your method has a similar number of activated parameters (K=2) as Theia. 
5. What will happen if you put more layers in your ViT as MoE layers?
6. As Theia is an important baseline you compare to, I think it's better to give a more detailed introduction on how Theia works for consistency of the paper, so we better know how your method differs from it.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the limitation that single vision foundation models (VFMs) cannot meet the diverse requirements of robotic tasks, and existing multi-model distillation methods dilute model-specific capabilities while lacking task adaptability. The proposed VER framework distills multiple VFMs into a Vision Expert Library and employs a dynamic routing mechanism to select the most relevant visual experts for different robotic tasks. The first core innovation is Patchwise Expert Routing combined with Curriculum Top-K Annealing (PER+CTA), which achieves fine-grained patch-level expert selection and prevents early training collapse through curriculum-based K-value annealing. The second core innovation involves fine-tuning only a lightweight Router (<0.4% parameters) while freezing all pretrained experts, enabling parameter-efficient task adaptation and flexible integration of trainable experts to incorporate robot-domain knowledge. Experimental validation across 17 robotic tasks achieves state-of-the-art performance (74.7% average success rate) and outperforms existing methods across multiple policy heads (ViLT, Flow-Matching, Diffusion). A key finding is that dynamic routing suppresses large-norm outliers in task-irrelevant regions and concentrates attention on task-critical areas (such as target object poses), generating more compact and discriminative visual features.

### Strengths
Originality and Problem Formulation: The paper presents a novel paradigm shift from static unified representations to dynamic expert selection for robot learning, introducing the innovative combination of vision foundation model distillation with Mixture-of-Experts (MoE) architecture and the creative Curriculum Top-K Annealing mechanism that addresses the early collapse problem in lightweight router training.

Quality and Experimental Rigor: The work demonstrates exceptional experimental rigor with comprehensive evaluation across 17 diverse robotic tasks spanning multiple benchmarks (Franka Kitchen, Meta-World, Adroit, LIBERO), different policy heads (ViLT, Flow-Matching, Diffusion), and real-world experiments, complemented by thorough ablation studies and insightful patch-level feature analysis (entropy, mutual information) that reveal the underlying mechanism of how dynamic routing concentrates attention on task-critical regions.

Clarity and Significance: The paper is well-structured with clear motivation and methodology, providing strong evidence that VER achieves state-of-the-art performance (74.7% average success rate) while maintaining computational efficiency (only <0.4% trainable parameters during downstream tasks, identical inference time to baselines), and the framework's extensibility—enabling flexible Top-K adjustment for compute scaling and seamless integration of task-specific trainable experts—offers significant practical value for scalable robot learning systems.


In addition, the experimental results in Figures 6 and 8 (1) are one of the reasons I gave you a high score.

### Weaknesses
Arbitrary choices lacking design principles: The paper fixes L=6 experts and K=2 active experts without providing any ablation studies to justify these hyperparameter choices (e.g., why not L=3,K=1 or L=9,K=3), and the "coincidence" that L=6 is exactly twice the number of three teachers suggests a potential pre-assigned expert-teacher correspondence, contradicting the paper's claim of "dynamic allocation."
Missing critical architecture search: The paper fails to explore the joint impact of L and K (e.g., performance comparison of L=3,K=1 vs L=6,K=2 vs L=12,K=4 under the same computational budget), making it impossible to determine whether the performance gains stem from the inherent advantages of the MoE architecture or merely from accidentally selecting a configuration that works for this specific task set, severely undermining the method's generalizability and reproducibility. But these are not big issues.

### Questions
Can you explore Inference Speed and Efficiency?

I recommend acceptance, but cannot confirm yet whether a higher score can be given.

### Soundness
4

### Presentation
4

### Contribution
3
