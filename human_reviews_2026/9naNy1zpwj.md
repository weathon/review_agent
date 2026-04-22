# Towards the Training of Deeper Predictive Coding Neural Networks

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 4, 2, 2

## Abstract
Predictive coding networks are neural models that perform inference through an iterative energy minimization process. While effective in shallow architectures, they suffer significant performance degradation beyond five to seven layers. In this work, we show that this degradation is caused by exponentially imbalanced errors between layers during weight updates, and the predictions from the previous layers not being effective in guiding updates in deeper layers. Furthermore, when training models with skip connections, the energy propagated by the residuals reaches higher layers faster than the one propagated by the main pathway, affecting test accuracy. We address the first issue by introducing a novel precision-weighted optimization of latent variables that balances error distributions during the relaxation phase, the second issue by proposing a novel weight update mechanism that reduces error accumulation in deeper layers, and the third one by using identity nodes that slow down the propagation of the energy in the residual connections.  Empirically, our methods achieve performance comparable to backpropagation on deep models such as ResNet18, opening new possibilities for predictive coding in complex tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles why predictive-coding networks (PCNs) fall apart beyond ~5–7 layers and pinpoints the cause as an exponential energy/error imbalance across layers (early layers starved of signal), with an extra timing issue in residual nets where skip-path energy outruns the main path. To fix this, the authors introduce time- and depth-dependent precision schedules—a “spiking precision” that briefly boosts precision when energy first reaches a layer, and a decaying precision that penalizes later layers—plus a forward-update (FU) weight rule that blends initial feed-forward predictions with converged activities to curb deep-layer drift, and BatchNorm Freezing (BF) to stabilize stats during iterative inference; for ResNets they add auxiliary units on residual paths to slow skip energy. Across CIFAR-10/100 and Tiny-ImageNet on VGG/ResNet families, combinations like spiking+FU(+BF) prevent the depth-related accuracy collapse and often match backprop: e.g., VGG-10 hits 93.27% on CIFAR-10 and 72.02% on CIFAR-100 with BF, and VGG-15 on Tiny-ImageNet reaches BP-level performance when spiking+FU+BF are combined. The core takeaway is that precision scheduling + FU re-balances layer energy so deep PCNs can train competitively with BP, albeit with some compute overhead and a slight hit to bio-plausibility from storing forward states.

### Strengths
1. Spiking precision is easy to implement and preserves locality. Moreover, iPC+spiking reaches BP-like accuracy on deep CNNs.
2. Identifies exponentially imbalanced layer errors as the root cause, much like vanishing gradients.

### Weaknesses
1. The experiments are on small scale datasets only, the reviewer is wondering how it would perform on large scale datasets such as ImageNet. 
2. PC/iPC epochs are scale with T×L and possibly slower in training. The reviewer is curious about the setting of T vs depth regrading computational and time cost.
3. The architecture used are all small models. The reviewer is worried about the feasibility of PC/iPC on large, complex networks such as transformer, ResNet101 etc.

### Questions
Please see weakness above.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Based on the observation that predictive coding networks suffer significant performance degradation beyond five to seven layers despite are effective in shallow architectures, this work proposes two algorithmic improvements (spiking precision and a novel weight-update mechanism) and two structural improvements (PCtailored batch normalization and auxiliary neurons for skip connections) that enable PC to achieve competitive performance with backprop on image classification benchmarks. While the motivation is clear, the related background and contribution significance should be clarified in detail.

### Strengths
This work may first reveal that the energy is orders of magnitude larger in layers closer to the output  in models trained with predictive coding (PC).

To regulate the energy imbalance and improves test accuracy in deep PC models and the case  incremental PC (iPC), this work proposes dynamical precision-weightings that depend on both time and layer depth, e.g. spiking precisions, that can achieve performance comparable to backpropagation in deep networks.

To achieve the goal of slowing down the feedback signal of the skip connections so that it reaches the higher layers at the same time as the main one, this work proposes to add extra families of neurons inside the skip connection that can reach performance comparable to these of backprop on ResNet18 with PC and iPC.

### Weaknesses
More details are needed to derive Equations (2) and (3).

Are the model and analysis suitable for transformer architecture?

If the algorithm's effect is only comparable to the backpropagation effect, what is its significance or advantage? It is necessary to first explain the relevant background or significance in detail.

The comparison of relevant algorithms still needs to be strengthened, and the contribution significance over related state-of-the-art works should be highlighted.

A pseudocode needs to be added to demonstrate how this algorithm operates and how to conduct related experiments.

### Questions
Please see the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper diagnoses the main failure mode of deep predictive-coding networks as a severe imbalance of prediction-error energy across layers, whereby upper layers accumulate large errors while lower layers receive effectively vanishing signals. To address this, it introduces time- and depth-dependent precision schedules, a forward-aware weight update, and architectural adjustments (e.g., BN freezing, auxiliary units on skip paths) that redistribute error energy and enable PC/iPC to approach backprop-level performance on deeper VGGs and ResNet-18 on Tiny-ImageNet.

### Strengths
It proposes practical, PC-compatible remedies—precision scheduling, forward-aware updates, and small architectural tweaks—that are simple to implement yet demonstrably restore performance to near–backprop levels on deeper CNNs, including ResNet-18 on Tiny-ImageNet.

### Weaknesses
**1. Framing/positioning issues**

* Even though energy-based models (EBMs) in deep learning have continuously advanced (e.g., JEM, diffusion models), the authors single out only Hopfield-style energy functions and predictive coding as EBMs and frame the work as an effort to “scale up” a very general EBM framework; this characterization is somewhat misleading.

**2. Notation / mathematical clarity**

* Notation and formatting are inconsistent. At line 146, the function $f$ is used without prior definition, and it is typeset in bold in one instance and non-bold in another. Similarly, several vectors are sometimes bolded and sometimes not; $\mathbf{y} \in \mathbb{R}^o$ (line 185) reuses $o$ that later seems to denote an input $\mathbf{o}$, creating an avoidable clash; and the expression “learning rate” is used for both inference- and learning-phase coefficients, which is imprecise. These issues make the mathematical development harder to follow than necessary.

* In the background section, “covariance of a specific neuron” is described, but covariance usually is defined over a set/list of variables, not a single unit; moreover, the text does not consistently distinguish between variance and covariance, which obscures what is actually being estimated and updated.

* Equation (3) appears to concatenate $\boldsymbol{\epsilon}$ and $f(\mathbf{x})$, both column vectors, in a way that is not well defined as written. The base model in the background section omits bias terms without explicitly stating this assumption, together with sporadic typos (e.g., “$x^0_0$” instead of “$\mathbf{x}^0_0$” at line 187, which weakens the technical polish of the paper.

**3. Figures/presentation**

* Figures are not reader-friendly. In Fig. 2, the train/val accuracy is difficult to distinguish by color, and the multiple per-layer curves seem to be intended to demonstrate energy imbalance; they are visually crowded. If layerwise imbalance is a key claim, a more quantitative or tabular display would present it more clearly. Similar color/legibility issues recur in other figures.

**4. Structure/organization**

* The organization is somewhat scattered: observations about energy imbalance appear inside the method section together with experimental results, and are then revisited in a separate section, which blurs the line between “what the method is” and “what we observed.” A more precise separation of motivation → method → experiments would improve readability.

**5. Novelty/relation to prior work (https://openreview.net/pdf?id=s3E08R4AMK)**

* Conceptually, much of the contribution overlaps with prior work that has already identified layerwise energy/error imbalance in PC. The proposed solution—time/depth precision spikes and forward updates—also looks very close to earlier versions of the work, raising questions about the level of novelty and, in particular, why a spike-shaped schedule is the right or unique choice.

**6. Theoretical grounding/justification**

* The path from the diagnosed problem to the proposed solution feels somewhat ad hoc. Beyond empirical evidence that the schedule “works,” the paper does not provide a stronger theoretical rationale (e.g., analysis of convergence or of error-energy propagation under the proposed precision) to justify why this particular scheduling and covariance treatment is appropriate.

**7. Methodological detail/completeness**

* Several key methodological elements are under-explained: it is not clearly specified *what* the covariance is taken over, *how* samples for covariance are collected during PC inference, *how* the substantial computational cost of covariance estimation is managed, and *why* temporal scheduling is necessary on top of the basic covariance idea. In addition, Eq. (2) already introduces (\alpha), but Eq. (4) seems to reuse (\alpha) inside (\Sigma); it is unclear whether this is intentional scaling or accidental duplication, and the resulting (\Sigma) no longer looks like a covariance. This also makes the method look very close to existing work.

### Questions
1. In an earlier version of this line of work, VGG-13 was included, but it is omitted here; is there a specific reason for excluding networks deeper than 10 layers, and does the proposed precision/covariance scheme still work reliably when depth exceeds 10?

2. Can the authors clarify the exact definition and estimation procedure for the covariance term (over which variables, over what time/window, and with what computational budget), and whether a lower-cost approximation was considered?

3. Given that several recent papers have already reported layerwise energy/credit imbalance in predictive coding, can the authors more precisely delineate what is new here beyond revalidating that observation—especially regarding the choice of a spike-shaped schedule?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The manuscript presents a modification to the standard training pipeline for Predictive Coding Networks (PCN). The proposed approach addresses the imbalance in energy distribution between the first and last layers that leads to a performance drop. The key contributions are spiking precision, forward updates, and structural adjustments for skip-connections and batch norm layers. The authors sequentially analyze the impact of the introduced modifications on the energy distribution and the resulting accuracy. The final experiments demonstrate that the proposed techniques preserve PCN accuracy in deeper models.

### Strengths
The main strength of the submission is the clear correspondence between the stated problem and the main results (in Table 2). The proposed modifications improve PCN accuracy for deeper models. In addition, the proposed structural update to skip connections and batch normalization improves accuracy for ResNet models. Moreover, the more layers there are, the greater the gain.

### Weaknesses
Below, I list the weaknesses observed in the presented submission:
1. The motivation for using PCN instead of the standard backpropagation (BP) remains unclear to me. Experiments demonstrate that the BP typically achieves higher test accuracy; therefore, the rationale for considering such a training scheme must be clearly explained at the outset, highlighting the advantages of PC over BP.  
2. Experiments are smoothly distributed over the sections, and many references to Figures are missing in the sections, for example, paragraphs "Results" in sections 4.2, 4.1.1, 4. Therefore, it is hard to determine which data support the results presented in the corresponding paragraphs.
3. I do not see any explanations why the training procedure presented in section 3 corresponds to the classification error minimization. The equations (1)-(3) do not align with the original task to make the best classification model.
4. The proposed modifications are not described with the equations (e.g., Forward Updates and Structural Updates) or lack a clear intuition (I do not say about theoretical derivation) like in the Spiking Precision paragraph. Therefore, fair analysis and comparison with baselines are impossible. 
5. I do not see any runtime or memory comparison between the proposed approaches and the mentioned baselines. 
6. Center nudging appears to be useless while the spiking mechanism works (Figure 3)

### Questions
1. Why can the proposed method be practically important? What use cases are the most suitable for using the presented approaches?
2. I see the comparison only on the classical architectures, while the current best results are obtained with the transformer-like models. Do you have any results for this or similar attention-based architectures?
3. Do you have any ideas about deriving theoretical guarantees on the convergence improvement observed empirically?
4. How robust are the proposed modifications to the noise on the gradient estimate? What is the recommended batch size for the datasets under consideration?
5. Why do the proposed approaches fail for the ResNet18 model and underperform compared to BP?

### Soundness
1

### Presentation
1

### Contribution
2
