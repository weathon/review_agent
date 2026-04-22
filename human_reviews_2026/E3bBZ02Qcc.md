# Rethinking Continual Learning with Progressive Neural Collapse

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 6, 4

## Abstract
Continual Learning (CL) seeks to build an agent that can continuously learn a sequence of tasks, where a key challenge, namely Catastrophic Forgetting, persists due to the potential knowledge interference among different tasks. On the other hand, deep neural networks (DNNs) are shown to converge to a terminal state termed Neural Collapse during training, where all class prototypes geometrically form a static simplex equiangular tight frame (ETF). These maximally and equally separated class prototypes make the ETF an ideal target for model learning in CL to mitigate knowledge interference. Thus inspired, several studies have emerged very recently to leverage a fixed global ETF in CL, which however suffers from key drawbacks, such as *impracticability* and *limited performance*. To address these challenges and fully unlock the potential of ETF in CL, we propose **Progressive Neural Collapse (ProNC)**, a novel framework that completely removes the need of a fixed global ETF in CL. Specifically, ProNC progressively expands the ETF target in a principled way by adding new class prototypes as vertices for new tasks, ensuring maximal separability across all encountered classes with minimal shifts from the previous ETF. We next develop a new CL framework by plugging ProNC into commonly used CL algorithm designs, where distillation is further leveraged to balance between target shifting for old classes and target aligning for new classes. Extensive experiments show that our approach significantly outperforms related baselines while maintaining superior flexibility, simplicity, and efficiency. Our code is available at https://github.com/yourname/ProNC.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces Progressive Neural Collapse (ProNC), a continual learning framework inspired by *Neural Collapse*. This phenomenon describes how deep networks, at convergence, produce class features that collapse to a single point for each class creating simplex equiangular tight frames (ETFs), creating orthogonal class prototypes and maximizing separation between classes. ProNC proposes to progressively expand the ETF as new tasks arrive, maintaining geometric consistency and feature separability without prior knowledge of all class counts. The framework integrates three loss components: cross-entropy for supervision, an alignment loss enforcing ETF-based feature geometry, and a distillation loss preserving past representations (to mitigate forgetting). The method is evaluated on standard benchmarks (Seq-CIFAR10/100 and Seq-TinyImageNet) under Class-IL and Task-IL setups, reporting substantial performance gains across datasets and memory budgets.

### Strengths
The idea of progressively adapting the ETF target during continual learning without knowing the number of total classes in advance is novel and addresses the shortcomings of fixed ETF methods for CL.

The paper is built on a convincing motivation. The reasoning is coherent and carefully developed, making the overall argument both logical and easy to follow.

### Weaknesses
### **Major Weaknesses**

1. **Questionable baseline performance values**: in Table 1, several baseline results (Co$^2$L, CILA, MNC$^3$L, STAR) are notably lower than those reported in their original papers (where they surpass the results from the proposed ProNC). This discrepancy indicates possible reproduction or configuration issues, undermining the fairness and credibility of the comparison and invalidating the paper’s main “state-of-the-art” claim.

2. **Missing baselines**: some important methodologies are missing in the experimental evaluation:
   - XDER (Boschini et al., TPAMI 2022, also cited by the authors);
   - GCR (Tiwari et al., CVPR 2022, also cited by the authors);
   - LODE (Liang and Yi, NeurIPS 2023).

3. **Limited novelty relative to NCT**: NCT (Yang et al., 2023, also cited by the authors and reported in the main table) already introduced ETFs in class-incremental learning. As far as I can tell, the main innovation of the proposed method compared to NCT is the removal of the "known number of classes in advance" assumption. While this is an interesting modification, it alone does not seem sufficient to constitute a truly novel methodology.

4. **Limited significance of ablation studies**: most ablation experiments in Figure 2 examine only a very narrow range of values (from about 2e-1 at best to 1e-2 in some cases) and consider just one dataset. Such a limited scope restricts the interpretability of the results, as the observed variations may not represent a consistent behavior across the full cosine similarity range (–1 to 1).

### **Minor Weaknesses**
5. Clarity issue in Section 3.1 (point 2): the explanation of the proposed methodology in this part is somewhat convoluted and would benefit from clearer exposition to improve readability and understanding.

6. Missing no-buffer baseline for ProNC: Table 1 does not report ProNC’s performance without a replay buffer (which is reported in the text only). Also inserting it in the main table would increase clarity and allow a fair comparison w.r.t. replay-free methods.

### Questions
While initializing the ETF at the end of the first task close to the optimal solution of the optimization problem is an interesting idea, for subsequent tasks the ETFs are initialized orthogonally to the previous ones but otherwise randomly, without any guiding heuristic. Do you have any thoughts on how this initialization process could be improved for later tasks?

### Soundness
3

### Presentation
2

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
This paper proposes Progressive Neural Collapse (ProNC), a continual learning framework inspired by the Neural Collapse phenomenon.
ProNC progressively constructs and expands an Equiangular Tight Frame (ETF) to align class features across tasks.
After each task, it estimates the “closest ETF” from class means and expands the basis for new classes using Gram–Schmidt orthogonalization, keeping all class representations approximately equiangular.
The model jointly optimizes cross-entropy, feature-alignment, and distillation losses.
Experiments on Seq-CIFAR-10/100 and Seq-TinyImageNet under both Class-IL and Task-IL show consistent gains over strong baselines (DER++, NCT, Co2L, STAR) with good efficiency and generalization.

### Strengths
1. Grounded in Neural Collapse geometry, offering an interpretable view of feature alignment in continual learning. Achieves strong results without complex contrastive or generative modules.
2. Works as a plug-in regularizer across different CL frameworks (e.g., ER, iCaRL, DER++).

### Weaknesses
1.	The method assumes clear task segmentation (task-aware setting); its applicability to task-free or online CL remains untested.
2.	As the ETF expands over many tasks, orthogonality may gradually degrade; this possible effect is not analyzed experimentally.
3.	Gram–Schmidt expansion could become unstable when the number of classes approaches the embedding dimension; only small-scale datasets and ResNet-18 (d ≤ 512) were tested.

### Questions
When the number of tasks grows large, does ETF orthogonality noticeably degrade? Would periodic re-fitting help?
Can ProNC remain stable with higher-dimensional embeddings (e.g., ViT features) or larger datasets such as ImageNet-100?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper extends the Neural Collapse Terminus (NCT; Yang et al., 2023b, arXiv) and the ICLR 2023 work by Yang et al. (2023a), aiming to address their limitations when applying the neural collapse (NC) phenomenon to continual learning (CL). In NC-based CL, the target ETF (simplex equiangular tight frame) is typically predefined, which requires knowledge of the total number of classes and may degrade discriminability when the class number becomes large. To overcome this, the paper proposes Progressive Neural Collapse (ProNC), a method that dynamically adjusts and expands the target ETF throughout the CL process, building on the NCT framework. Experimental results show that ProNC achieves consistent and significant improvements over existing CL baselines, particularly compared with NCT. Moreover, the proposed regularization approach proves beneficial even when combined with other CL frameworks, as confirmed through ablation studies.

### Strengths
1. The paper provides a clear exposition of the background NC theory and explains its own contributions in a well-structured manner.
2. The proposed method is well-motivated by the identified limitations of prior NC-based CL works, and the use of Theorem 1 introduces a moderately novel and theoretically grounded component.
3. Comprehensive experiments demonstrate consistent and noticeable gains over a range of baselines, supporting the empirical validity of the approach.

### Weaknesses
1. The technical novelty remains limited compared with the preliminary works (Yang et al., 2023a,b). The paper reads largely as a continuation of this prior line of research, where NC-based CL formulations have already been thoroughly explored.
2. The second main contribution—the ProNC-based CL framework—largely mirrors the loss formulation of NCT (Yang et al., 2023b). While Section 3.1 introduces a genuinely new idea, Section 3.2 appears nearly identical to the corresponding part in NCT.
3. Some reproduced baselines yield noticeably lower accuracies than those reported in their original papers. For example, ER on Task-IL with Seq-CIFAR-100 using ResNet-18 has been reported above 70% (e.g., GPM, ICLR 2021), yet only 60.19% here, raising concerns about the faithfulness of baseline reproduction.

### Questions
1. Could the authors elaborate more explicitly on how the proposed ProNC framework differs technically from NCT (Yang et al., 2023b)?
Beyond the dynamic ETF expansion, are there any additional algorithmic or theoretical components that are genuinely new rather than adapted from NCT?
2. Some reproduced results (e.g., ER on Seq-CIFAR-100 Task-IL) are considerably lower than in prior works such as GPM (ICLR 2021).
Could the authors detail the reproduction settings (e.g., data augmentation, optimizer, training epochs) and justify whether these differences could account for the gap?

### Soundness
3

### Presentation
3

### Contribution
2
