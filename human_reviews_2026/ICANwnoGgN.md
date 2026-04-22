# Model soups need only one ingredient

- Avg Score: 5.20
- Decision: Reject
- Scores: 6, 6, 4, 6, 4

## Abstract
Fine-tuning large pre-trained models on a target distribution often improves in-distribution (ID) accuracy, but at the cost of out-of-distribution (OOD) robustness as representations specialize to  the fine-tuning data. Weight-space ensembling methods, such as Model Soups, mitigate this effect by averaging multiple checkpoints, but they are computationally prohibitive, requiring the training and storage of dozens of fine-tuned models. In this paper, we introduce MonoSoup, a simple and data-free approach that achieves a strong ID–OOD balance using \textit{only a single} checkpoint. Our method applies Singular Value Decomposition (SVD) to each layer’s update, splitting it into high-energy directions that capture task-specific adaptation and low-energy directions that introduce noise but may still encode residual signals useful for robustness. MonoSoup then re-weights these components with adaptive, layer-wise coefficients that account for the spectral and geometric structure of the model. Experiments on CLIP models fine-tuned on ImageNet and evaluated under natural distribution shifts, as well as on Qwen language models tested on mathematical reasoning and multiple-choice benchmarks, show that this plug-and-play approach is a practical and effective alternative to multi-checkpoint methods, retaining much of their benefits without their computational overhead.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In the paper the authors introduce MonoSoup, a datafree approach that tries to obtain the same benefits of Model soups (recover ood perfomenace after finetuning) while using only a single checkpoint instead of averaging multiple checkpoints. The method applies Singular Value Decomposition (SVD) to each layer update in order to split into high and low energy directions. Monosoup then re-weights these directions with adaptive and layer-wise coeffiecients computed taking into account the spectral and geometric structure of the model. The authors provide experiments on both visual and text domain, using CLIP as reference model for vision domain and Qwen as model for text.

### Strengths
Simple and straightforward method: easy to implement and interpret.

Well-motivated and theoretically rooted in spectral theory: the work builds on the spectral and geometric understanding of model merging, offering clear insights into when and why merging succeeds.

Strong analysis and motivation: the introduction of Similarity-Filtered Greedy Soup empirically validates that spectral alignment is a strong proxy for effective merging, motivating the proposed method.

Efficiency: achieves the benefits of model soups while requiring only a single checkpoint, MonoSoup performs on par with or better than Model Soup and Model Stock while being much cheaper to apply.

Complementary with existing techniques: for example combining the method with Wise-FT further improves the ID–OOD performances.

### Weaknesses
Modest gains on strong checkpoints: While MonoSoup provides clear improvements on weaker or representation-collapsed models, it yields only modest gains on already well-performing checkpoints. In those cases, the improvements over Model Soup or Model Stock are often within the margin of variability, suggesting that the method’s primary advantage lies more in efficiency rather than in delivering substantial performance breakthroughs for strong models. 

Lack of variability reporting: The experimental tables report mean accuracies but omit measures of variability such as standard deviations. Given that several of the reported improvements are relatively small (often around 0.5–1%), including standard deviations would help assess the statistical significance and robustness of these gains. The absence of such information makes it difficult to judge whether improvements are consistent or within the expected variance of fine-tuning noise.

Limited interpretability of coefficients: while layer-wise coefficients are derived analytically, their per-layer impact on robustness is not deeply analyzed.

### Questions
For the models labeled M-x (e.g., M-14, M-31, etc.), could the authors clarify what these identifiers refer to? Do they correspond to specific fine-tuning configurations ?

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
5

### Summary
This paper proposes MonoSoup, a data-free, single-checkpoint approach that recovers the robustness benefits of model soups without needing multiple models. For each layer, the fine-tuning update is decomposed into high- and low-energy components (via SVD), and adaptive, per-layer interpolation coefficients are computed to recombine them. This reweighting preserves task-specific signal while retaining useful low-energy structure, leading to improvements in both in-distribution accuracy and out-of-distribution robustness. Results on vision and language tasks show consistent gains over strong baselines.

### Strengths
# Originality

The contribution is not a new mathematical primitive but a **creative reframing** of model-soup robustness: the authors show how to extract and reweight **within-checkpoint** structure so that a *single* fine-tuned model can capture much of the benefit previously obtained by averaging *multiple* checkpoints. This shift from multi-model soups to a **single-checkpoint soup** removes the dependency on collecting and storing many models while preserving the core geometric insight behind soups.

# Quality

The work is **methodologically sound**: the approach is clearly specified; comparisons include strong, relevant baselines; and the analysis (including ablations) is thoughtful and aligns with the stated hypotheses. Empirical results are broad enough to support the main claims and are reported with appropriate care.

# Clarity

The paper is **well written and well structured**. The narrative flows logically from diagnosis to method to evaluation, with enough technical detail to enable reproduction.

# Strengths

* **Conceptual creativity:** Derives a soup-like effect from a **single checkpoint** via principled layer-wise decomposition and reweighting.
* **Practical impact:** **Eliminates the burden** of curating and storing many checkpoints, reducing compute and storage overhead.
* **Empirical thoroughness:** **Extensive evaluations** with clear, interpretable metrics and ablations that illuminate why the method works.
* **Presentation quality:** Clear exposition, intuitive figures, and actionable implementation details.

### Weaknesses
* **“Plug-and-play” claim needs qualification.** The method’s practicality is somewhat overstated because performance depends on choosing (R) well. Without a label-free selection rule, users may not reliably reproduce the reported gains. 

* **Limited architecture diversity.** Most results are on Transformer backbones (CLIP ViTs, a small LLM). It remains unclear whether the recommended (R) range and mixing rule transfer to non-Transformer models (e.g., ConvNeXt, ResNet, MoE). 

* **Architecture- and dataset-dependence of (R).** Intuitively, the optimal (R) should vary with model family and pretraining corpus, yet the paper suggests a fairly stable mid-range (R) for ViTs across datasets without explaining why.

### Questions
see weakness,

**Hyperparameter selection for (R).**
While the paper argues that performance is fairly stable for mid-range (R), the work would benefit from a **practical, label-free procedure** to set (R). In particular, because spectral decay and alignment statistics can vary across architectures (e.g., ViT vs. ConvNets vs. LLMs) and datasets, a fixed default may not transfer. Please (i) justify when/why a mid-range (R) should generalize, and (ii) provide a simple selection rule that practitioners can use without labels.

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
5

### Summary
This paper addresses a key limitation of existing model merging methodologies, pointing out their reliance on two or more checkpoints. The authors argue this is misaligned with realistic scenarios where often only a single, best-performing model is stored. To address this gap, the paper proposes MonoSoup, a method that utilizes only a single checkpoint.
Using SVD, the method decomposes the single model's update into a high-energy component, which is associated with in-distribution (ID) specialization, and a low-energy component for out-of-distribution (OOD) robustness. By performing a principled weighted sum of these two components, MonoSoup demonstrates performance that is comparable to existing multi-checkpoint techniques.

### Strengths
1. The paper addresses a practical and realistic scenario. While many merging methods like Model Soups require access to dozens of checkpoints, real-world applications often only store a single, best-performing model. The paper's focus on improving robustness from this single-mode" setting is a valuable contribution.
2. The core idea of using SVD to decompose the fine-tuning update into high-energy (specialization) and low-energy (robustness) components is interesting.

### Weaknesses
1. The paper's method for calculating the alignment coefficient $cos~\alpha^{(l)}$ (lines 264-269) is theoretically unclear and appears to be a significant conceptual leap. The method computes the cosine similarity between two conceptually different entities: the pre-trained weights $W_0^l$ (an absolute state vector) and the low-energy update $W_{low}^l$ (a difference vector). There is no clear justification for why the alignment between a 'state' and a 'difference' is a meaningful measure of knowledge preservation. Even if $W_0^l$ is interpreted as a vector from the origin, it is not the pre-training update direction (unless initialized from zero), making the interpretation of this alignment ambiguous.
2. The paper's key motivation is undermined by its own data in Figure 3b. The text claims that truncating low-energy components harms both ID and OOD accuracy, motivating the need for re-weighting (MonoSoup). However, the graph clearly shows that moderate truncation (e.g., at a rank fraction of $\approx 0.7$) improves OOD performance (red line).
3. The performance gains over LiNeS reported in Table 2 are marginal. These small improvements might not be statistically significant and could be due to experimental variance or checkpoint selection. The claim of consistent improvement should be validated more robustly, for instance, by reporting results over multiple runs or across a wider set of fine-tuned checkpoints (beyond the three presented).
4. The method's generality appears limited. The key motivation (low-energy components are crucial for robustness) is derived only from the large-scale ImageNet benchmark (Figure 3b) and is explicitly shown to be absent in the small-scale 20-task benchmark. This suggests MonoSoup may only be effective for fine-tuning on large-scale datasets. This contradiction also weakens the general claim that low-energy components inherently encode OOD robustness, as this does not hold for the small-task setting.

### Questions
1. The paper argues that high task vector alignment is beneficial for merging models performing the same task. This contrasts with principles in related fields like multi-task learning, where orthogonality (low similarity) is often preferred to prevent task interference[1,2]. Could the authors clarify how to reconcile these two perspectives? Is the principle that "high alignment is beneficial" strictly limited to the same-task merging scenario?
2. In Figure 2, the authors state they analyze 2,409 pairwise combinations. However, $70 \choose 2$ (choosing 2 pairs from 70 models) should result in 2,415 combinations. Could the authors clarify this small discrepancy?
3. The symbol $\tau$ is used in the context of Similarity-Filtered Greedy Soup without a formal definition.
4. Table 1 introduces "MonoSoup (Pairwise Models)" as a baseline. The main text, however, focuses on applying MonoSoup to a single model. Could the authors briefly explain in the main text how MonoSoup is adapted and applied in the pairwise setting?

[1] Ilharco, Gabriel, et al. "Editing models with task arithmetic." ICLR (2023).
[2] Davari, MohammadReza, and Eugene Belilovsky. "Model breadcrumbs: Scaling multi-task model merging with sparse masks." ECCV (2024).

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
3

### Summary
This paper presents MonoSoup, a light-weight alternative to model soup's traditional weight-space based ensembling techniques leveraging multiple model checkpoints. More specifically, by leveraging the SVD to estimate a high and low energy weight estimates, MonoSoup is able to re-estimate the final model. Empirical evaluations shows that MonoSoup achieves strong OOD Generalization performance when compared with traditional model souping techniques.

### Strengths
The reviewer notes the following strengths:
- The paper presents a clear context for MonoSoup with a defined motivation for the development of the underlying methodology.
- The proposed methodology (MonoSoup) is light-weight and readily applicable to real-world settings.
- MonoSoup showcases strong empirical performance across multiple models & tasks.
- The author also provide strong intuitive background for MonoSoup through analysis, linking performance improvements to alignment between fine-tuning updates.

### Weaknesses
The reviewer notes the following weaknesses:
- The reviewer’s primary concern is that, while the paper’s motivation is clearly stated, the argument that storing only a single best-performing checkpoint necessitates the development of MonoSoup is unconvincing. In particular, the reviewer finds it unlikely that, in practice, there would be meaningful constraints on retaining multiple checkpoints during model training.
- Additional evaluations on other modalities like audio would also provide even more compelling evidence for the applicability of MonoSoup.

### Questions
As noted in the weaknesses above, the reviewer encourages the authors to consider an alternative motivation for MonoSoup, rather than relying on the assumption that most models in practice need to be stored as single checkpoints.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces MonoSoup, a post-hoc, data-free method that improves the trade-off between in-distribution (ID) accuracy and out-of-distribution (OOD) robustness using only a single fine-tuned model. Motivated by the geometric principles underlying model soups, the authors propose to analyze a single model’s fine-tuning update via Singular Value Decomposition (SVD), decomposing it into high-energy (task-specific) and low-energy (residual/robust) components. These components are adaptively reweighted using a combination of spectral decay and alignment with pretrained weights, yielding a single edited checkpoint that better balances specialization and generalization.

### Strengths
The paper tackles a practical and well-motivated challenge: retaining OOD robustness without storing or training multiple fine-tuned checkpoints. Its formulation is conceptually elegant, connecting the empirical success of model soups to the internal spectral geometry of a single model. The SVD-based decomposition provides an interpretable view of fine-tuning dynamics, distinguishing high-energy task adaptation from low-energy robustness-preserving directions. The adaptive weighting rule is simple, closed-form, and data-free, making the approach easy to implement as a lightweight post-processing step.

### Weaknesses
1. **Limited novelty beyond existing SVD-based merging.**

While the paper frames its contribution as extending model soups to a single-model setting, the actual main operation—SVD decomposition of the fine-tuning update followed by spectral weighting—closely parallels prior works (e.g., Task Singular Vectors, Model Merging with SVD). The paper’s novelty primarily lies in its interpretation rather than in a fundamentally new algorithmic principle. 

2. **Heuristic coefficient design without theoretical grounding.**

The adaptive weighting rule $\lambda_{\text{low}} = \rho + (1-\rho) \cos(\alpha)$ is intuitively motivated but lacks theoretical justification for why this functional form optimally balances ID and OOD trade-offs. There is no analysis of stability, sensitivity, or convergence properties with respect to R, ρ, or α, leaving the approach partly empirical.

3. **Interpretational overrstatement of “soup”.**

The name MonoSoup implies model ensembling, yet the method operates as a single-model spectral reweighting rather than a true weight-space average. This conceptual mismatch may overstate its relation to model soups and obscure the fact that it performs deterministic, full-rank parameter editing rather than multi-model averaging.

### Questions
1. Beyond natural ImageNet shifts, could MonoSoup be evaluated on stronger distributional or adversarial robustness benchmarks to validate its claimed generalization benefits?

2. Since MonoSoup performs spectral reweighting rather than true model ensembling, what geometric evidence supports describing it as a “soup”? Would framing it as spectral anisotropic re-scaling be more accurate?

3. Considering prior fine-tuning mechanisms such as Spectral Adapter (Zhang & Pilanci, 2024) and other SVD-based spectral methods that already decompose pretrained weights to modulate singular directions, what specific conceptual or methodological advance distinguishes MonoSoup from this existing line of spectral fine-tuning approaches?

### Soundness
2

### Presentation
2

### Contribution
2
