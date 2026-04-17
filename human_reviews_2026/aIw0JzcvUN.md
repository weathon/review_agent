# UniPrompt-CL: Sustainable Continual Learning in Medical AI with Unified Prompt Pools

- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Although modern AI models achieve state-of-the-art performance with large-scale datasets, strict ethical and institutional constraints in medicine make centralised learning nearly impossible. Institutions must therefore rely on local data, but traditional training methods quickly overfit new samples and suffer from catastrophic forgetting, making continual learning (CL) essential. While CL has advanced in the field of natural images, prompt-based continual learning (PCL) remains largely unexplored in the context of medical applications. We present UniPrompt-CL, the first PCL framework designed specifically for healthcare. Preliminary experiments show that existing PCL approaches perform poorly on medical datasets, which motivates our hypothesis that the prompt pool design needs to be more effective. UniPrompt-CL introduces a unified prompt pool with minimal expansion and a novel regularisation term, reducing computation while balancing stability and plasticity. On three diabetic retinopathy datasets (APTOS, DDR and DRD), UniPrompt-CL improves accuracy by at least 10% and the F1 score by 9 points compared to previous methods, while reducing the cost of inference. Additionally, it achieves superior performance on continual learning evaluation metrics. These results demonstrate that UniPrompt-CL lays the foundation for sustainable medical AI, enabling consistently high performance in distributed healthcare environments. To ensure reproducibility, the code and all training configurations can be found in this repository.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a continual learning approach with an application in learning from images from different hospitals (in particular ophthalmology). The paper is easy to read and the method is motivated well. Some good improvements are shown especially over the method by Kim.

### Strengths
- For the specific setting discussed (multiple hospitals that cannot share data) the methods are suited
- Methods are clearly described 
- There is an improvement over the existing methods for this specific task

### Weaknesses
- The paper is very incremental from Kim 2025 (or 2024?). 
- Although the paper aims to be quite general in several parts of the descriptions the motivation for doing this is quite specific and also the experiments are focused on a very specific application and dataset.  This would be suited for a a medically oriented conference but for a venue as ICLR you want methods to generalize beyond the specific application. That might be possible in this case but that point cannot be convincingly made with the current paper.

Minor things:
- In table 4 there is a mismatch between the ECCV2024 and the statement after that (2025).
- Title talks about sustainability but that is not something that is really addressed in the paper

### Questions
- What are the precise technical extensions made to the method by Kim?
- What other applications do you envision for the proposed methods?

### Soundness
3

### Presentation
3

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
This paper identifies the suboptimal performance and high computational cost of existing prompt-based continual learning (PCL) methods when applied to the medical domain. To address these limitations, the authors propose **UniPrompt-CL**, a novel framework specifically designed for continual learning in healthcare.

The core of this method is a unified prompt pool, where prompts are shared across all layers of the model. At each layer, a query derived from the [CLS] token is used to select and combine prompts from this shared pool to generate a layer-specific instruction. This design is motivated by the authors' hypothesis that medical images necessitate a focus on capturing fine-grained details, as opposed to the more varied and broadly dispersed features found in natural images.

Consequently, the proposed method achieves state-of-the-art (SOTA) performance on medical CL benchmarks while significantly improving computational efficiency by requiring only a single inference pass, unlike prior approaches that often necessitate two.

### Strengths
1.  **Addresses a Significant Problem with a Well-Reasoned Motivation**
    * The paper is grounded in a compelling and well-reasoned rationale: that medical images require a fundamentally different continual learning strategy than natural images.
    * The motivation is built on the clear distinction between the standardized nature and fine-grained analytical needs of medical data versus the high variability of natural images, providing a strong foundation for the work.
***
2.  **Proposes a Useful Conceptual Framework for Domain-Specific PCL**
    * The work introduces a valuable **"dispersed vs. clustered" framework** to analyze and explain the behavior of prompts on different data domains.
    * This conceptual lens provides a useful and insightful perspective for the research community to evaluate and design future PCL methods tailored for specialized domains.
***
3.  **Demonstrates Clear and Quantifiable Computational Efficiency Gains**
    * The proposed method has a clear practical advantage by achieving superior performance with only a **single inference pass**, directly addressing a major limitation of prior SOTA methods.
    * This efficiency gain is explicitly quantified, resulting in a **~33% reduction in FLOPs** compared to the top-performing baseline (44.17 GFLOPs vs. 66.42 GFLOPs), as shown in Table 2. This is a significant improvement for real-world applications.

### Weaknesses
1.  **Lack of Clarity in Methodology Compromises Reproducibility**
    * The manuscript's most significant weakness is the ambiguous and incomplete description of the training procedure, which hinders the reader's ability to understand and reproduce the work. Specifically, the procedure for training the model on each successive continual learning (CL) stage is ambiguous.
    * The description in Section 4.2 is particularly confusing and logically contradictory: *"we first construct and train the integrated prompt pool as described in Section 4.1, and then freeze the prompt weights corresponding to the previous stages"*. This implies that prompts are trained *before* being frozen, which is counter-intuitive for a CL process. It is unclear if each new task involves one or two distinct training phases.
    * Furthermore, key implementation details for baselines are missing. To fairly assess the proposed method's parameter efficiency, the number of learnable parameters for the compared baseline methods should be reported.

***

2.  **Insufficient Empirical Support for the Central Causal Claim**
    * The paper's central claim is that the architectural design of the **"unified prompt pool"** is the inherent cause for learning the **"clustered"** prompt representations supposedly ideal for medical data, as shown in Figure 2.a.
    * However, this causal link is not rigorously substantiated. The observed prompt distribution could be a result of **confounding variables**, such as specific hyperparameter choices (e.g., learning rate, optimizer, regularization strength), rather than a direct and inevitable consequence of the architectural design itself. The paper does not provide experiments to decouple these effects.


***

3.  **Limited Experimental Scope Restricts Generalizability Claims**
    * The authors frame their method as a framework for "healthcare" and "medical AI", suggesting broad applicability.
    * However, the empirical validation is confined to a **single task (classification) on a single modality (fundus photography for diabetic retinopathy)** across three similar datasets. This narrow experimental scope does not sufficiently support the broad claims of generalizability across the medical domain. The authors themselves acknowledge this limitation in the future work section.

***

4. **Inconsistent Reporting of Core Continual Learning Metrics**
    * The manuscript's treatment of core continual learning (CL) metrics lacks internal consistency. Given that the work addresses catastrophic forgetting, the direct measurement of this phenomenon is a critical component of the evaluation.
    * In Section 5, the authors explicitly state their decision to omit standard forgetting metrics like Backward Transfer (BWT) and Average Forgetting (AvgF), citing the "limited-size medical datasets" as the reason.
    * This statement is directly contradicted in a later part of the paper. In Section 7 and Table 4, the authors present and analyze results for these exact metrics (BWT and AvgF) as part of their ablation study.
    * Such a direct contradiction between different sections of the paper affects the clarity and perceived rigor of the experimental evaluation.

### Questions
#### **Regarding the Methodology**

1. **On the Training Procedure for Each CL Stage:**
    The current description of the training process is ambiguous, which may pose a challenge for reproducibility.
    * Could the authors please provide a detailed, step-by-step description or, ideally, an algorithmic pseudo-code of the training procedure for a single continual learning (CL) stage?
    * Specifically, clarification is requested for the statement in Section 4.2: *"we first construct and train the integrated prompt pool... and then freeze the prompt weights corresponding to the previous stages."* Could the authors clarify if this implies a two-phase training process for each new task, and elaborate on the exact sequence of operations?

2. **On the Prompt Pool Design and Comparison to Prior Art:**
    The 'unified prompt pool' is presented as a key innovation. However, its relationship with similar concepts in prior work could be further clarified.
    * Methods like Coda-Prompt, L2P and DualPrompt also appear to use a form of unified, layer-shared prompt pool. Could the authors first elaborate on the key architectural and methodological differences between UniPrompt-CL and Coda-Prompt?

3.  **On the Substantiation of the "Dispersed vs. Clustered" Claim:**
    The paper's motivation hinges on the claim that baseline methods learn "dispersed" prompts, while the proposed method learns "clustered" ones.
    * Was this conclusion based on a systematic analysis across various experimental conditions, or primarily on the visual observation presented in Figure 2.a?
    * More importantly, how did the authors ensure that this observed difference is a direct consequence of the architectural design (unified vs. independent pools) rather than an artifact of hyperparameter settings? For instance, could the "dispersed" nature of baseline prompts be altered if their hyperparameters were tuned specifically for the medical datasets? A more thorough analysis to decouple these effects would greatly strengthen this central claim.

***

#### **Regarding the Experiments**

4.  **On the Scope of Generalizability Claims:**
    The paper suggests the method is designed broadly for "healthcare" and "medical AI."
    * Given that the experiments are conducted exclusively on diabetic retinopathy classification, could the authors comment on the method's expected generalizability to other medical tasks (e.g., segmentation, detection) and modalities (e.g., CT, MRI)? This would help to properly contextualize the scope of the contributions.

5.  **On the Comprehensive Evaluation of Catastrophic Forgetting:**
    Catastrophic forgetting is a critical metric in any continual learning evaluation.
    * The main results in Table 1 report final accuracy, while forgetting metrics (BWT, AvgF) are only provided for a subset of methods in the ablation study. To provide a more comprehensive and necessary comparison of the method's SOTA performance in mitigating forgetting, would it be possible for the authors to provide the BWT and AvgF results for **all** baseline methods evaluated in Table 1?

6.  **On the Efficiency Comparison with All Baselines:**
    The paper rightly emphasizes computational efficiency as a key advantage.
    * To allow for a complete assessment, could the authors provide a table comparing both the **inference FLOPs** and the **total number of learnable parameters** for **all** baseline methods listed in Table 1, not just OS-Prompt++?

7.  **On Data Processing for Reproducibility:**
    * Could the authors please specify in the appendix how the original 5 classes of the APTOS dataset were mapped to the 3 classes used in the experiments? This detail would greatly aid in the full reproducibility of the results.

### Soundness
1

### Presentation
2

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
This paper proposes UniPrompt-CL, a framework designed to tackle catastrophic forgetting in medical AI by combining a strong backbone with efficient prompt learning. The method is evaluated on three diabetic retinopathy datasets, outperforming existing state-of-the-art PCL methods. The work also highlights the unique challenges posed by medical data, as their standardized nature impacts the effectiveness of traditional PCL methods. By incorporating an integrated prompt pool and lightweight expansion, UniPrompt-CL effectively preserves prior knowledge and captures fine-grained features, ensuring robust and stable performance.

### Strengths
1. A reasonable method is proposed to tackle an important problem, i.e., prompt-based continual learning in medical domain.
2. The experimental results show clear performance improvements over several prompt-based CL methods.
3. The writing is generally well.

### Weaknesses
1. The paper states that standard Prompt-based Continual Learning methods fail on medical data because they are designed for the broad feature space of natural images, whereas medical images require more fine-grained distinctions. However, this claim is supported by limited and largely qualitative evidence, primarily a single t-SNE visualization.  The analysis lacks depth and fails to investigate the fundamental reasons for this performance gap.

2. The comparison lacks other CL families (e.g., regularization, rehearsal, LoRA-based, etc).

3. The paper's title and abstract make broad claims about advancing "Medical AI," yet the experiments are confined to diabetic retinopathy datasets.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new prompt-based continual learning method termed UniPrompt-CL, motivated by data sharing constraints across institutions in medical imaging settings. The method uses a unified prompt pool that shares prompts across all transformer layers, combined with minimal prompt expansion and a new regularization term. They empirically evaluate their method on a sequence of diabetic retinopathy tasks (APTOS, DDR, DRD) in the domain-incremental learning setting, demonstrating that they outperform previous prompt-based approaches for the considered tasks.

### Strengths
1. Clear motivation. The paper articulates well why CL techniques are critical for medical settings under data-sharing restrictions and how current PCL frameworks may fail in this domain.
2. The method is computationally efficient compared to dual-inference methods.
3. The method outperforms the considered PCL baselines for the considered diabetic retinopathy task.

### Weaknesses
1. The main weakness is limited empirical evaluation. The paper only conducts evaluation on one benchmark. To strengthen the empirical claims, the authors should consider evaluating the method on more tasks in the medical domain and ideally for longer sequences (more than 3 domains).  [1] Table 6 and [2] DermCL contain tasks in the domain incremental learning setting, which are publicly available. 
2.  The main motivation of this work is for continual learning methods in medical settings. However, it lacks discussion of current continual learning methods in the medical domain. [3][4][5] specifically account for the domain incremental setting.

[1] Continual Learning in Medical Image Analysis: A Comprehensive Review [2] Expert Routing with Synthetic Data for Continual Learning [3] Feature Transformers: Privacy Preserving Lifelong Learners for Medical Imaging. [4] Conditional diffusion replay for continual learning in medical settings. [5] Generalizable Continual Classification of Medical Images.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2
