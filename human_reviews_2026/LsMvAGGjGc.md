# Benign Overfitting in Adversarial Training for Vision Transformers

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Despite the remarkable success of Vision Transformers (ViTs) across a wide range of vision tasks, recent studies have revealed that they remain vulnerable to adversarial examples, much like Convolutional Neural Networks (CNNs). A common empirical defense strategy is adversarial training, yet the theoretical underpinnings of its robustness in ViTs remain largely unexplored. In this work, we present the first theoretical analysis of adversarial training under simplified ViT architectures. We show that, when trained under a signal-to-noise ratio that satisfies a certain condition and within a moderate $\ell_2$ perturbation budget, adversarial training enables ViTs to achieve nearly zero robust training loss and robust generalization error under certain regimes. Remarkably, this leads to strong generalization even in the presence of overfitting, a phenomenon known as \emph{benign overfitting}, previously only observed in CNNs (with adversarial training). Experiments on both synthetic and real-world datasets further validate our theoretical findings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
A theoretical analysis of a two layer vision transformer with adversarial training is done. The adversarial threat model for this work is with respect to the l-infinity norm. Empirically the method is validated on a heavily modified version of the MNIST dataset and a synthetic dataset.

### Strengths
The paper is well written and easy to follow.

### Weaknesses
I am not an expert in theoretical machine learning and much of the paper’s strength lies in the fact that they have done a lot of mathematical derivations. Other reviewers will have to comment on whether that alone is enough to merit acceptance.  

However, from an experimental perspective I find the results entirely unconvincing for the following reasons:

1. The choice of datasets are too small and not impactful. The authors only test on a synthetic dataset and MNIST. Actually, even the MNIST dataset they use is not the true MNIST dataset. The dataset the authors test on is a binarized version of the MNIST dataset. These days CIFAR-10/100 experiments are the minimum for experimentally demonstrating a viable adversarial robustness technique (with Tiny-ImageNet and ImageNet also becoming norms). 

2. The attacks used in the paper are not SOTA. APGD would be a much better choice (instead of PGD) and an L2 version of the APGD code has been available for multiple years:
https://github.com/fra31/auto-attack

I also question WHY the authors only use the l2 metric. Why not also show what happens for l-inf, l-0 or l-1 attacks? Here are the related attack links:

L0: https://github.com/fra31/sparse-imperceivable-attacks
L1: https://arxiv.org/abs/2103.01208
L2: https://arxiv.org/pdf/2003.01690

 There has been much work that shows it may not be enough to just prevent one norm attack, so considering multi-norm attacks (even if the results are poor) is a much more interesting scope: https://proceedings.mlr.press/v119/maini20a/maini20a.pdf

As a reviewer I cannot mandate that you do any more experiments. However, I would say that it is nearly impossible for me to be an advocate for your paper because the scope of the current work is not justifiable in my opinion. If you could extend your framework and your experiments to the multi-norm case (even if the results then become less robust), that would be a much much stronger work.

### Questions
Please address the issues I mention in the weakness section of my review. Specifically: 

A. Why aren't SOTA attacks used in the experimental results?
B. Why only focus on L2 norm? Can you give any better justification for the scope of your current work? 
C. Why aren't more complex datasets used?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, with a solid theoretical analysis, the authors demonstrate that the phenomenon of benign overfitting also exists for ViTs. Experiments on a real-world dataset (MNIST) highlight the correctness of the finding.

### Strengths
1 The theoretical proof is solid.

2 This paper is well motivated.

3 The writing is good.

### Weaknesses
1 Acknowledging the theoretical contributions of this paper, the findings do not bring new insights to the community. For example, as listed in the second point in the contribution of this paper, the authors claim that:

1) Small perturbations yield trajectories close to clean training. (According to the definition of Adversarial training, if the perturbation is small enough, AT will collapse to natural training).

2) Moderate perturbations cause the attention mechanism to fail, such that the ViT collapses to a linear model; (Due to the misleading effect of the moderate adversarial samples, it will disrupt the attention mechanism.)

3) Large perturbations lead to significant generalization error beyond benign overfitting.  (In this circumstance, the robust overfitting will happen and resisting attacks crafted with a large attack budget is also challenging, increasing the generalization error.)

2 Theories with practical implications tend to be more appreciated. Unfortunately, this paper does not give the take-away tips on how to better perform AT on ViT Transformers.

3 The verified dataset is the MNIST dataset, which is the simplest dataset in image classification. Without the experiments on more complex datasets such as CIFAR-10 and ImageNet, the correctness of the theory can not be verified in the application of ViTs in real scenarios.

4 In Line 418, the signal and noise vectors is concatenated to form a new vector which is quite different from the application of AT in real-world datasets.

5 The theory makes an analysis on a simple Transformer architecture, ignoring the role of the linear projection layer and the MLP head.

### Questions
1 Why, in the verified experiment, only the samples of "0" and "1" labels are chosen to perform experiments? Can the theory be generalized to datasets with more classes?

2 Can the experiments be generalized to explain the appearance of the robust overfitting in Vision Transformer?

### Soundness
2

### Presentation
3

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
This paper provides the first theoretical analysis of benign overfitting under adversarial training for Vision Transformers (ViTs). The authors construct a simplified two-layer ViT model, derive convergence and generalization guarantees, and identify distinct regimes of adversarial perturbation magnitudes that influence learning dynamics. Empirical validations on synthetic and MNIST data confirm the theoretical predictions.

### Strengths
The paper provides a comprehensive and rigorous theoretical analysis of benign overfitting in the context of adversarial training for Vision Transformers (ViTs). Specifically, it extends the study of benign overfitting to the transformer architecture, offering new insights into how the interplay between attention mechanisms, signal-to-noise ratio, and perturbation magnitude determines both robust generalization and overfitting behavior.

### Weaknesses
The main limitation of the paper lies in its experimental evaluation. The experiments are conducted only on a subset of MNIST using shallow Vision Transformers (ViTs), which is too simplistic to convincingly verify the proposed theoretical results. MNIST lacks the complexity required to test the robustness and generalization behaviors predicted by the theory. At minimum, the authors should include experiments with a standard multi-layer ViT on a more challenging dataset such as CIFAR-10, to better demonstrate the practical relevance and validity of their theoretical findings.

### Questions
## 1. Generality beyond Vision Transformers:
I am wondering whether the current theoretical investigation could be extended beyond the simplified two-layer ViT model to encompass more general Transformer architectures—for example, models with multiple self-attention layers, residual connections, layer normalization, or feed-forward blocks. It would be valuable to understand whether the derived benign overfitting behavior and robustness–generalization relationships continue to hold under these more realistic architectural settings, and whether the theoretical scaling laws remain consistent when evaluated on larger and more complex datasets beyond MNIST.

## 2. Applicability to ViT variants:
I am also curious about how the proposed theorems and analysis apply to different variants of Vision Transformers, such as Swin Transformer, DeiT, or hierarchical ViTs that modify the attention mechanism or token structure. Do the key theoretical conditions, particularly those involving the signal-to-noise ratio and perturbation magnitude, still characterize the transition between benign and harmful overfitting in these variants? Some clarification or discussion on the generality of the theoretical framework across ViT architectures would strengthen the paper’s impact and scope.

## 3. Empirical validation of theoretical regimes:
The paper identifies three distinct regimes of adversarial perturbation (clean-like, linear-collapse, and failure). Could the authors provide more detailed empirical evidence or visualizations to confirm these transitions—perhaps by monitoring changes in attention distributions, feature alignment, or representation collapse across varying perturbation strengths? Such results would make the theoretical phase transition more tangible and strengthen the connection between theory and practice.

### Soundness
3

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
2

### Summary
The paper gives a theoretical account of robust benign overfitting in a simplified two‑layer ViT trained with ℓ₂ adversarial training. It proves that, under specific relationships between dataset size and signal‑to‑noise ratio (SNR), and for a moderate perturbation radius τ, the model can interpolate the training data (vanishing robust training loss) while maintaining small robust test error. Experiments on synthetic data and MNIST visualize phase transitions via heatmaps that align with the theory (e.g., boundary roughly 𝑁⋅SNR^2=Ω(1)).

### Strengths
++ The paper pinpoints when adversarially trained ViTs interpolate yet generalize robustly (e.g., N·SNR² = Ω(1) with τ ≲ ‖μ‖²₂ / log(dh)), and how robust error scales with d, SNR, τ, yielding explicit bounds and a practical “safe” τ range.

++ The analysis explains why moderate τ can “flatten” attention into near‑uniform weights—collapsing a ViT to a linear model—and contrasts convergence/SNR requirements with that degenerate baseline. This isolates an attention‑driven mechanism behind phase transitions.

++ The paper shows that no classifier can achieve nontrivial robust accuracy when τ ≥ ‖μ‖²₂ gives a sharp ceiling on what adversarial training can achieve, cleanly bracketing the benign region.

++ Heatmaps on synthetic and MNIST reproduce the predicted boundary N·SNR² = Ω(1) and show that robust gains appear only once both SNR and N clear the theoretical thresholds. The figures concretize the phase transition narrative.

### Weaknesses
-- Validation uses synthetic data and MNIST; no CIFAR/ImageNet‑scale tests or modern ViT training recipes, so the practical reach of the theory is not stress‑tested under real‑world pipelines, augmentations, or stronger attacks.

-- The two‑layer ViT and assumptions (e.g., multi‑patch distribution, specific τ/SNR scalings) help analysis but may not capture architectural and optimization nuances (depth, MHA heads, layernorm, schedules) that affect robustness in practice.

-- Results emphasize l2 training/attacks and do not discuss other threat models (l∞, l1, corruptions) or multi-step PGD details that affect robust outcomes; generalization across norms remains open.

### Questions
1. Do the phase boundaries or impossibility result change meaningfully for ℓ∞ or autoattack‑style suites? Any conjecture or preliminary evidence?

2. How do the conditions scale with number of heads M, head dimension, and depth? Can you extend the analysis (even heuristically) to stacked blocks or to pre‑norm residual forms common in ViTs?

3. Could you reproduce the heatmap phase boundary on CIFAR‑10/100 with small ViTs and ℓ₂ PGD to show qualitative agreement (even if not strictly in‑distribution with the theory)?

### Soundness
3

### Presentation
3

### Contribution
3
