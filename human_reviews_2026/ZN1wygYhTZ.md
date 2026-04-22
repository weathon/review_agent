# CRRC: Residual Cross-view Learning for Deep Multi-view Clustering

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Deep multi-view clustering aims to integrate complementary information from multiple heterogeneous views to improve clustering performance.  However, existing fusion strategies often struggle to balance shared semantics and view-specific heterogeneity, as they typically rely on direct concatenation or rigid alignment, which obscures subtle cross-view patterns and assumes equal contribution from all views.  To address these limitations, we propose CRRC, a novel framework that leverages residual connections to recalibrate view-specific features by adaptively incorporating complementary information from other views.  Specifically, CRRC introduces a dynamic gating fusion module to control residual flow based on view characteristics, and an attention-based weighting mechanism to emphasize semantically relevant cross-view signals.  These components work collaboratively to enhance feature discriminability and consistency.  Extensive experiments on benchmarks datasets demonstrate that CRRC outperforms state-of-the-art methods in accuracy, NMI, and purity, validating its effectiveness in achieving robust multi-view clustering.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents the CRRC framework, which effectively addresses the balance between shared semantics and view-specific heterogeneity in multi-view clustering by incorporating cross-view residual recalibration, dynamic gating fusion, and attention-based weighting mechanisms. The CRRC framework fuses cross-view information through residual connections, preserving view-specific differences, while leveraging dynamic adjustments and attention mechanisms to enhance feature discriminability and consistency. Experimental results demonstrate that CRRC outperforms existing methods on several benchmark datasets, showing superior performance in ACC, NMI, and PRI metrics.

### Strengths
1.The CRRC framework combines residual recalibration with contrastive learning, proposing a clustering method that is highly adaptive and robust in the presence of noise and heterogeneity. This method is not only original in theory but also highly significant in practical applications.

2.The introduction of the cross-view residual recalibration mechanism effectively addresses the information loss caused by direct concatenation or alignment of views, allowing for the flexible integration of complementary information from different views.

3.Through the Dynamic Gating Fusion module, the framework utilizes an attention mechanism to adaptively adjust the degree of fusion of cross-view information.

4.The paper extensively validates the CRRC framework’s superiority on multiple benchmark datasets, showing that CRRC outperforms existing methods in terms of clustering accuracy (ACC), normalized mutual information (NMI), and purity (PRI).

### Weaknesses
1.In Figure 1, the framework lacks a detailed description of the overall process flow.

2.The paper incorporates contrastive learning as part of the overall framework, but its independent contribution has not been sufficiently quantified.

3.The model employs only reconstruction and contrastive losses, which enhance feature discriminability and consistency but do not include clustering-oriented losses such as clustering loss or cluster-level contrastive loss. This may result in insufficiently targeted optimization for clustering objectives.

4.Although the Dynamic Gating Fusion (DGF) module helps regulate cross-view information flow, insufficient or overly restrictive learning of gating weights may lead to the excessive suppression of useful information.

### Questions
1.Although the contrastive loss and reconstruction loss help improve model performance, they are not the core innovation of this paper. Why were these two losses specifically chosen? Are there other loss functions that could further enhance the effectiveness of the residual recalibration and dynamic gating fusion mechanisms?

2.The paper employs contrastive learning as part of its loss function, but has the performance of the framework been evaluated without it? Do the ablation experiments demonstrate the specific contribution of contrastive learning to multi-view clustering? If the contrastive module were removed, could other components (such as DGF and RF) still effectively enhance clustering performance, or does contrastive learning play an indispensable role in the framework?

3.This paper uses a self-attention mechanism to weight cross-view information during fusion, a strategy also seen in other methods. What distinguishes the application of self-attention in this work from its use in previous approaches?

4.In the related work section, the authors mention that multi-view clustering typically follows two training paradigms: end-to-end and two-stage. Which paradigm does this paper adopt, and what are the reasons behind this choice? Compared with the alternative, does the selected approach offer advantages in optimization efficiency or training stability? Additionally, has the time complexity of the model been evaluated?

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
This paper addresses the problem of deep multi-view clustering, which aims to integrate complementary information from multiple heterogeneous views to improve clustering performance while balancing shared semantics and view-specific heterogeneity. The proposed CRRC leverages residual connections to recalibrate view-specific features through a dynamic gating fusion module and an attention-based cross-view weighting mechanism. The core contributions include a novel fusion strategy that adaptively incorporates cross-view information while preserving view-specific traits. Experimental results on benchmark datasets demonstrate improved performance.

### Strengths
1.	The dynamic gating fusion module and attention-based cross-view weighting provide a novel approach to multi-view fusion by adaptively controlling residual flow
2.	The paper is well-organized, with logical flow from problem statement to method description.
3.	The framework addresses key limitations in multi-view clustering, such as view imbalance and semantic alignment.

### Weaknesses
1.	The formulation of dynamic gating mechanisms is quite similar to existing gating architectures in neural networks, such as LSTM gates. The manuscript does not sufficiently distinguish its gating approach from these established methods and justify its fundamental novelty.
2.	The attention mechanism relies on global representation $H$ as queries, but $H$ itself is generated through simple concatenation and MLP processing. This approach may not adequately capture fine-grained cross-view semantic relationships.
3.	The framework heavily relies on existing contrastive learning. The manuscript does not clearly establish how the residual recalibration component provides fundamental advantages over prior methods.

### Questions
See weaknesses.

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
This paper proposes CRRC to address critical limitations in deep multi-view clustering. The framework integrates three key innovations: residual recalibration (RF) to capture cross-view complementary patterns via residual connections, dynamic gating fusion (DGF) to adaptively adjust residual flow strength, and attention-based cross-view weighting (ACW) to suppress redundant views through semantic correlation scoring. Extensive experiments on benchmarks datasets demonstrate that CRRC outperforms state-of-the art methods.

### Strengths
1.	The proposed framework innovatively combines residual learning and attention mechanisms to model cross-view complementarity while preserving view-specific features.
2.	This method achieves state-of-the-art performance across diverse datasets with consistent gains over 9 comparative methods.

### Weaknesses
1.	The combination of dynamic gating (DGF), and attention weighting (ACW) may introduce redundant computations. ACW and DGF both aim to suppress irrelevant cross-view signals, potentially leading to overlapping functionality.
2.	While the paper acknowledges sensitivity to the temperature parameter $\tau$, other critical hyperparameters are not systematically evaluated.
3.	The paper assumes clean and complete views, but real-world multi-view data often suffer from noise or missing views. CRRC’s performance under such scenarios is untested.

### Questions
Please refer to the Weaknesses.

### Soundness
2

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
4

### Summary
This paper proposes CRRC, a residual cross-view learning framework for deep multi-view clustering.
The method introduces three components: (1) a residual recalibration mechanism that preserves view-specific information while integrating cross-view differences, (2) a dynamic gating fusion (DGF) module that adaptively modulates residual information flow, and (3) an attention-based cross-view weighting (ACW) mechanism to emphasize informative views.
The authors combine reconstruction and contrastive losses to enhance semantic consistency.
Experiments on several standard multi-view datasets (NGs, Fashion, MNIST-USPS, Caltech-3V/4V/5V) show that CRRC achieves superior accuracy and NMI compared with prior methods such as SCMVC and AccMVC.

However, despite technical soundness, the novelty is somewhat limited: the framework mainly integrates existing paradigms—residual connection, gating, and attention—without providing new theoretical or algorithmic insights into multi-view fusion. The paper is well-organized, but empirical validation lacks diversity and ablation analysis does not strongly support the claimed advantages.

### Strengths
1.The paper is technically solid and grounded in contemporary DMVC literature. The modular design (reconstruction, residual recalibration, attention weighting) is clearly described and reproducible.

2.Empirical results demonstrate consistent improvement across multiple datasets.

3.The authors provide theoretical derivations (Lipschitz continuity, convergence) that enhance formal completeness.

4.The visualizations (t-SNE and convergence curves) support the qualitative effectiveness of the method.

### Weaknesses
1.CRRC mainly combines residual and attention mechanisms already well established in representation learning. The proposed design lacks a unique theoretical motivation or learning objective.

2.The paper does not clearly explain why residual recalibration benefits multi-view fusion beyond intuitive reasoning.

3.The experiments assume clean and complete views. The method’s claimed adaptivity is not validated under realistic noisy or missing-view conditions.

4.The contrastive formulation is standard; no new loss or contrastive structure is introduced.

5.The paper omits recent transformer-based or diffusion-based fusion frameworks that are now common in multi-view learning (e.g., multi-view diffusion contrastive models).

### Questions
1.How does the residual recalibration differ from simply using skip connections or attention-weighted feature concatenation?

2. The gating vector \( g_v \) in Eq. (6) depends on both \( R_v \) and \( R_{k \nleq v} \); does this introduce potential gradient coupling or instability in multi-view backpropagation?  

3.Equation (13) seems problematic in formulation.The loss function \( L_{total} = L_{rec} + L_{cm} \) is written without clear normalization or weighting between reconstruction and contrastive terms.  How are these two losses balanced during training? If both are directly summed, the optimization could be unstable since they are at different scales. Please clarify or correct the formulation.

4.Lack of large-scale and diverse datasets.The paper only evaluates on small to medium-scale datasets (e.g., NGs, Fashion, MNIST-USPS, Caltech).  Moreover, Caltech-2V to Caltech-5V are subsets of the same dataset, thus not providing true diversity in modality or scale.  How would CRRC perform on large-scale or real-world heterogeneous datasets?

5. Insufficient comparison with recent baselines (2025). The paper mainly compares against models up to 2024, missing several strong baselines proposed in 2025 from top venues (e.g., transformer-based or diffusion-based multi-view clustering models).  This makes it hard to assess whether CRRC truly advances the state-of-the-art.

### Soundness
2

### Presentation
3

### Contribution
2
