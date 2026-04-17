# Maximizing Incremental Information Entropy for Contrastive Learning

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4, 6

## Abstract
Contrastive learning has achieved remarkable success in self-supervised representation learning, often guided by information-theoretic objectives such as mutual information maximization. Motivated by the limitations of static augmentations and rigid invariance constraints, we propose IE-CL (Incremental-Entropy Contrastive Learning), a framework that explicitly optimizes the entropy gain between augmented views while preserving semantic consistency. Our theoretical framework reframes the challenge by identifying the encoder as an information bottleneck and proposes a joint optimization of two components: a learnable transformation for entropy generation and an encoder regularizer for its preservation. Experiments on CIFAR-10/100, STL-10, and ImageNet demonstrate that IE-CL consistently improves performance under small-batch settings. Moreover, our core modules can be seamlessly integrated into existing frameworks. This work bridges theoretical principles and practice, offering a new perspective in contrastive learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work modifies the view generation process for contrastive learning by modfiying the query view generation process with a view-generating neural network which functionally acts as an augmentation generator. This view generator is trained with entropy maximization, making the negatives harder and therefore increasing the loss.  Overall, this results in stronger performance when combined with self-supervised methods.

### Strengths
* There is a mathematical intuition which, at a high level, is sensible is a novel motivatation for the method.
* Empirical evaluations show small but consistent improvements.
* It is not too costly, computationally.

### Weaknesses
* There are limited transfer learning results, which is the main application of self-supervised pre-training. In particular, classification results are missing. 

* ViTs are not evaluated. Would the method work with a Vision Transformer backbone? Vision Transformers are ubiquitous. 

* There are missing strong self-supervised baselines, such as DINO[1] style training. 

[1] Oquab, Maxime, et al. "Dinov2: Learning robust visual features without supervision." arXiv preprint arXiv:2304.07193 (2023).

### Questions
* How could this be extended to Vision Transformers, to modernize the method?

* I'm curious if the gains could be attributed largely to spectral regularization? In the ablation table, what would be the result of ONLY encoder regularization?

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
The paper introduces IE-CL, a novel information-theoretic framework for self-supervised learning. The motivation stems from two perceived limitations in existing Contrastive Learning methods: the inflexibility of static data augmentation and the representation compression caused by the deep encoder acting as an "information bottleneck."

### Strengths
1. The SAIB module is a genuinely clever algorithmic contribution. It replaces the inherent rigidity of manually designed data augmentations with a learnable, instance-specific mechanism.

2. The paper is well-grounded in information theory. By explicitly framing the deep encoder as a bottleneck, the authors move beyond the standard alignment/uniformity analysis to propose a more granular, principled objective focused on incremental information gain.

3. The empirical results show improvements upon previous SOTAs.

### Weaknesses
1. The IE-CL loss function is highly complex, requiring the delicate balancing of at least four major hyperparameter terms. There needs to be more ablations on the sensitivity.

2. While the method aims to make CL more accessible by excelling in small-batch scenarios, the introduction of the SAIB module, the complex loss terms, and the explicit regularization undoubtedly incur additional computational overhead. The paper critically omits a quantitative analysis of the increase in FLOPs, training time, or memory consumption relative to baselines. This lack of efficiency analysis weakens the overall practical value, as the potential computational cost might negate the performance gain, particularly in the resource-limited settings it targets.

3. The transferability of the learned augmentation should be discussed. As the standard augmentations used are not combined with specific datasets.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Existing self-supervised contrastive learning methods mainly rely on augmentation-based invariance constraints, which limit representation expressiveness. This paper introduces entropy as a measure to preserve semantic consistency and improve expressiveness. The proposed Sample Augmentation Incremental Block (SAIB) and Incremental Information Entropy (IncEntropy) objective capture entropy gain, reflecting the diversity of information lost after encoding. The main contribution is using the encoder’s information bottleneck (via SAIB) to extract semantic information while reducing noise.

### Strengths
- The approach is interesting, particularly in introducing the concept of entropy into self-supervised learning through SAIB and IncEntropy. In particular, SAIB presents an appealing way to apply entropy, showing the largest performance gain in the ablation study (Table 3).

- The method is also simple and can be easily integrated into existing SSL frameworks, as demonstrated in Table 5.

### Weaknesses
1. Although the effectiveness of the proposed method is demonstrated throughout the paper, most experiments (except Table 1) are conducted on relatively small-scale settings such as ResNet-18 or ImageNet-100. Since Table 3 highlights the strong effect of entropy generation through SAIB, it would be valuable to evaluate the method on larger or standard-scale benchmarks. The same applies to Table 5.

2. It would also be helpful to include an ablation study with semantic consistency only, to better isolate and verify the effectiveness of SAIB.

3. It is well known that dense prediction tasks, such as semantic segmentation, differ significantly from image classification, and standard SSL methods often struggle to generalize well to these tasks. Therefore, to more convincingly demonstrate the effectiveness and general applicability of the proposed method, evaluation on additional (dense) prediction benchmarks beyond the Pascal dataset would be beneficial.

4. As the authors also mentioned, SAIB relies on Spectral Normalization, which is designed for convolutional priors. Therefore, this approach cannot be directly applied to ViT-based backbones.

### Questions
1. Could the authors provide insights or results on the performance of the proposed method in larger-scale settings?
2. Have the authors considered whether there is any way to extend this approach to ViT-based backbones, despite the limitation that Spectral Normalization cannot be applied?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces IE-CL, which is a framework designed to overcome the limitations of static augmentations and large batch sizes in CL. So the authors show that maximizing the "incremental entropy" (the entropy gain between augmented views) while preserving semantics is equivalent to minimizing the InfoNCE loss.

To achieve this, they propose the SAIB which is a lightweight, trainable module that plugs into the query branch. To my understanding, I think SAIB learns to expand the representation space (increasing entropy), while a KL divergence regularizer ensures the new view remains semantically consistent with the original. The authors have conducted several experiments show SAIB improves linear evaluation performance, works well in small-batch settings, can enhance other non-contrastive methods, and helps with downstream tasks like segmentation.

### Strengths
I think the paper has some strengths:

First, I think the theoretical foundation is a major plus. The authors provide a proof for their core claim, linking the minimization of contrastive loss to the maximization of incremental entropy. In my opinion, this reframing of contrastive learning as a trade-off between entropy expansion and semantic alignment is a novel information-theoretic perspective.

Also, I think the SAIB module itself is a contribution. It's not just an arbitrary layer but a lightweight block designed to execute the paper's goal (inducing positive entropy increments) while being regularized to preserve semantics.

Finally, I think the experiments seem comprehensive. The authors validate their method across a wide range of datasets and tasks. I think it's particularly strong that they test not only linear probing but also show effectiveness in small-batch settings (a key weakness of many CL methods) and on downstream transfer tasks like segmentation and detection.

### Weaknesses
I think the paper's primary weakness is its narrow experiment, which doesn't fully support the broad claims of a improved "framework." My main issue is that all experiments are confined to ResNet architectures. The self-supervised learning field has largely migrated to Vision Transformers (ViTs), and their complete absence here is a glaring omission. In my opinion, I feel this makes the work somewhat dated and raises a critical question: is this incremental entropy principle a general SSL concept, or is it just a clever trick that happens to work well with the inductive biases of CNNs? The claim of generalizability is a little bit undermined when the method isn't tested on the field's dominant architecture.

My other concern is that the analysis of the method's new hyperparameters is too thin. The SAIB module adds a new layer of complexity, particularly the KL regularization weight, but the provided ablation study is basic. I think It's hard to tell how robust the method is or what the practical tuning cost would be for a new user. I think the paper misses the chance to investigate the interplay between the new module and existing crucial hyperparameters. For example, the temperature (τ) is important to tell how the model works. How does adding SAIB affect the model's sensitivity to temperature? Does the optimal τ change? I think demonstrating the analysis is needed for anyone trying to implement this method.

### Questions
See the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes IE-CL (Incremental-Entropy Contrastive Learning) for self-supervised contrastive learning that explicitly models entropy generation and preservation in the learning process. The central idea is to inject entropy at the input level via a learnable augmentation block (SAIB), then constrain the encoder to preserve this expanded entropy through spectral normalization. This is motivated by an information-theoretic decomposition of contrastive learning objectives under the Data Processing Inequality (DPI), suggesting that maximizing representational informativeness requires both entropy creation (from augmentations) and controlled propagation (through the encoder). The model comprises three main components: 1) SAIB (Sample Augmentation Incremental Block): A learnable transformation enforcing "volume-expanding" Jacobian to expand local volume in the feature manifold, thereby increasing input-space entropy. 2) KL Regularizer: Maintains semantic consistency between entropy-expanded samples. 3) Encoder Preservation (Spectral Norm): Constrains the encoder’s Lipschitz constant to prevent information collapse.

The provided experimental results on CIFAR-10/100, STL-10, and ImageNet-100 demonstrate consistent performance improvements over SimCLR, BYOL, MoCo-v2, and SimSiam. The ablation studies show the SAIB block contributes the most, while the encoder regularizer has only a marginal effect.

### Strengths
1. The proposed formulation leads to a new perspective on entropy control in SSL. Unlike prior works (e.g., InfoMax-SSL, VICReg, Matrix-IB) that maximize representation-level entropy, IE-CL proposes to inject entropy at the input level through a learnable augmentation mechanism. This shift from output-space to input-space entropy control is novel and conceptually meaningful.
2. The proposed method is established based on sound theoretical motivation. The information-theoretic derivation connecting augmentation entropy, encoder Jacobian, and DPI is mathematically correct and clearly stated. The idea of treating entropy propagation as an “incremental” process is intuitive and reasonable.
3. The empirical validation is comprehensive. The method is tested on multiple datasets and integrated into different contrastive baselines (e.g., SimCLR, BYOL, MoCo-v2, SimSiam). The experimental results show consistent gains, and training overhead is minimal.
4. The presentation includes sufficient implementation and training details. The ablation studies reveal transparent module effects and interactions.

### Weaknesses
1. The claim that encoder preservation is necessary appears overstated. The paper argues that spectral normalization of the encoder is required to prevent the loss of generated entropy (Section 3.3). However, Table 3 shows that removing this component leads to only a marginal change in performance (0.26 percent difference). This result indicates that the encoder likely already preserves entropy through existing normalization layers and the contrastive objective itself. Therefore, the Encoder Preservation module should be characterized as helpful rather than necessary, since the theoretical argument appears to overextend a sufficient condition into a claim of necessity.
2. The plug-and-play experiment in Table 5 evaluates only the addition of the SAIB module to other baseline methods, without reporting results for the combined configuration of “SAIB plus Encoder Regularizer.” This omission further reinforces the impression that the SAIB module is the sole component contributing meaningfully to performance improvements, while the Encoder Regularizer has little demonstrable effect.
3. The empirical gains in the experiments are limited. For example, ImageNet improvements are visible but not substantial (+1.3 \% over Matrix-SSL at 800 epochs). Also, multi-seed or statistical analysis is not provided to confirm the significance of the performance gains.
4. Although the principle of maximizing information entropy is well established, the novelty of IE-CL lies primarily in where the entropy is introduced, namely at the input level rather than the representation level. The paper would benefit from emphasizing this distinction more clearly and from explicitly differentiating its approach from prior InfoMax and variance-regularized methods such as VICReg, InfoMin, and EMP SSL.

### Questions
The authors are suggested to respond to those raised in **Weaknesses.*

**Additional Questions**

The proposed SAIB module is implemented as a small convolutional network. How can the IE-CL framework be extended to deal with Vision Transformer architectures?

### Soundness
3

### Presentation
3

### Contribution
3
