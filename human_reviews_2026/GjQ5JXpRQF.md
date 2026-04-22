# OrthoRF: Exploring Orthogonality in Object-Centric Representations

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 2, 8, 8, 4

## Abstract
Neural synchrony is hypothesized to help the brain organize visual scenes into structured multi-object representations. In machine learning, synchrony-based models analogously learn object-centric representations by storing binding in the phase of complex-valued features. Rotating Features (RF) instantiate this idea with vector-valued activations, encoding object presence in magnitudes and affiliation in orientations. We propose Orthogonal Rotating Features (OrthoRF), which enforces orthogonality in RF’s orientation space via an inner-product loss and architectural modifications. This yields sharper phase alignment and more reliable grouping. In evaluations of unsupervised object discovery, including settings with overlapping objects, noise, and out-of-distribution tests, OrthoRF matches or outperforms current models while producing more interpretable representations, and it eliminates the post-hoc clustering required by many synchrony-based approaches. Unlike current models, OrthoRF also recovers occluded object parts, indicating stronger grouping under occlusion. Overall, orthogonality emerges as a simple, effective inductive bias for synchrony-based object-centric learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce OrthoRF, an extension of RotatingFeatures improving the state-of-the-art in unsupervised object discovery using synchrony-based representations. This improvement consists in enforcing orthogonality between the different latent spaces of the model, yielding internal object separation akin to Slot-based models. The authors evaluate the performance of their model on 2 synthetic datasets: 4Shapes and SEM and show that OrthoRF tends to outperform RF.

### Strengths
**Timely and important problem.** Object-centric learning is a central goal in vision; advancing synchrony-based models is particularly valuable because they promise binding without supervision or heavy architectural machinery.

**Clarity and presentation.** The paper is overall clear and reads well; the motivation, method, and experimental setup are easy to follow.

**Simple, intuitive, and promising idea.** Orthogonality serves as a practical geometric surrogate that encourages non-interfering latent spaces. It’s easy to implement,  and serves as a strong regularizer that constrains networks toward clean, object-separated representations.

**Good quantitative evaluation of learned representation.** The similarity and separability analyses (Section 4.2) give a concrete, interpretable view of internal representation geometry. These measurements constitute useful standardized evaluation practices for synchrony in future work, beyond task metrics alone.

### Weaknesses
*Major*

**Loss of flexible dynamic binding.** A core weakness of slot-based methods is their fixed capacity: a preset slot budget limits generalization when scenes contain fewer or more objects than during training. Synchrony-based models explicitly aim to avoid this by distributing object clusters in a continuous phase space, enabling dynamic, per-image allocation. By enforcing orthogonality across latent components, OrthoRF effectively funnels objects back into component-specific subspaces, re-introducing slot-like rigidity and sacrificing one of synchrony’s key advantages. 

Is it possible to add empirical comparisons with slot-based models showing that OrthoRF still performs better? For example: the authors could train OrthoRF on 4Shapes and evaluate with 1–8 (or more?) objects per image to demonstrate that OrthoRF can still outperform slot-based solutions when object count varies. Additionally, give the striking similarity between OrthoRF and slot-models, could the authors include the latter as baseline in addition to RF?

**Decoder comparability on 4Shapes.** OrthoRF does not clearly outperform RF on 4Shapes (especially considering RF’s performance in Table 5). The gains appear mainly when thresholding psi_final for OrthoRF while using k-means on z for RF, and performance is better only when including overlapping regions in the evaluation. This mixes architectural effects with decoder choices. 

Can RF be evaluated in the same way—thresholding psi—so that both models use comparable decoding strategies, on both datasets? Or could the authors propose ways to cleanly separate improvements due to OrthoRF’s architecture from those due to decoding?

**Benchmark scope and claim accuracy.** The Stanić et al., 2023 evaluation benchmark cited does not include 4Shapes (contrary to the statement at L268); their datasets are high-resolution, colored images (Tetrominoes, dSprites, and CLEVR). Later synchrony work (e.g., Gopalakrishnan, 2024) also reports strong phase separation on those more complex datasets. RF, even though includes some tests on 4Shapes, is also evaluated on colored naturalistic scenes (Pascal, FoodSeg). While simplified grayscale setups can be valuable diagnostically, most work on unsupervised object discovery with synchrony-based models since 2023 evaluates on RGB images, so returning to grayscale toy shapes can feel not fully aligned with current practice; framing this choice explicitly and adding at least one RGB benchmark would help align with field norms.

Could the authors (i) clarify/correct the claims, (ii) either evaluate OrthoRF on the standard color/RGB benchmarks or demonstrate that prior synchrony models specifically fail under the proposed overlap/noisy regimes, (iii) discuss scalability to known failure modes, especially same-color object collisions in RGB (a classic challenge for synchrony). Does orthogonality alleviate phase ambiguity when two objects share hue/texture, or does it merely enforce cleaner axes without resolving color-based binding failures?

*Minor*

**Notation consistency.** The authors define z_out in Eq. 5, then switch to z_final in the subsequent sentence and Eq. 6, and the two are used interchangeably elsewhere. Please clarify whether they are identical; if yes, standardize on one term throughout (update figures/captions accordingly). If not, define z_final precisely and maintain the distinction.

**Typo.** Table 2: “thesh” → “thresh”.

**Capacity fairness.** In Table 1, AE/CAE may not be directly comparable to RF/OrthoRF if they have fewer parameters. Please report parameter counts and, ideally, include capacity-matched baselines or parameter-controlled ablations (and note any compute/step differences) to ensure that performance differences are not confounded by model size.

### Questions
**Noise Robustness.** Gaussian blur and additive Gaussian noise may preserve the structure of the image. Have the authors tested more disruptive corruptions (e.g. salt-and-pepper)? Also, do the results still hold under various SNR?

**Figure 2 (n=5).** The figure indicates n=5, seemingly tying the number of channels to the number of slots. Why is this augmentation necessary? 
Also, in Figure 7, the panel shows one full scene plus four single-object images. Are these single-object images used only for evaluation (e.g., to compute metrics) or also during training as part of the n channels? If used in training, how are they generated without label leakage, and what supervision signal (if any) do they introduce?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
For the problem of forming multi-object representations, a previously proposed framework, "Rotating Features," can be improved by requiring the learned features to be orthogonal. This immediately improves the interpretability of the features, but surprisingly also improves performance in various tasks of synthetic scene understanding benchmarks.

### Strengths
* A straightforward method to add orthogonality constraints to a previously suggested loss.
* Improved performance (e.g., better handle occlusion), using a method which is much more scalable than other algorithms in the field (e.g., slot-based methods, synchrony-based methods).
* Improved interpretability, as the features are distinct rather than mixed (Figure 2 is impressive).

### Weaknesses
* The relative contribution of the two technical improvements ('Competitive binding in orientation space' and 'Orthogonality regularization') is not systematically analyzed, and it is unclear if both are needed. It seems the manuscript emphasizes the orthogonalization, but some of the heavy lifting is done by the improved 'gating' from the softmax.
* Inconsistent improvement over the previous method (as seen in Tables 1, 2, 3, 4).

### Questions
* Can you suggest in retrospect for which problems is the original RF method still superior (as seen in part in Tables 1, 2, 3, 4)?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Orthogonal Rotating Features (OrthoRF), an extension of the Rotating Features (RF) autoencoder framework for unsupervised object-centric learning (OCL). The original RF model, inspired by neural synchrony, uses vector-valued activations where magnitude encodes feature presence and orientation/phase encodes object affiliation, but often results in distributed representations requiring post-hoc clustering.

OrthoRF addresses this by imposing orthogonality as an inductive bias in the orientation space. It achieves this through two main architectural modifications:


1. Competitive Binding: A per-layer, centered softmax over orientation components to encourage "winner-take-most" specialization, mapping each object to a single, distinct component.


2. Orthogonality Regularization: An inner-product based $\mathcal{L}_{ortho}$ loss that penalizes the squared off-diagonal mass of the Gram matrix formed by the latent vectors, explicitly driving cross-component similarities toward zero, effectively enforcing a $90^{\circ}$ separation.



These modifications produce one-hot-like object encodings, which eliminates the need for post-hoc clustering and results in more interpretable representations. In experiments on the 4Shapes and synthetic SEM datasets, OrthoRF matches or outperforms current synchrony-based models on object discovery metrics (ARI-BG, MBO). Crucially, OrthoRF demonstrates superior performance on the shape completion task ($MBO_{i}^{OV}$), especially in overlapping/occluded regions, and shows a unique capability to recover occluded object parts in its intermediate representations—a strength not observed in prior slot-based or synchrony-based models.

### Strengths
* Originality and Significance: The core idea of enforcing orthogonality in the orientation space of RF is highly original and delivers a substantial improvement in the interpretability and performance of synchrony-based models. The result is a robust model that eliminates post-hoc clustering and yields one-hot-like encodings.





* Handling of Occlusion: OrthoRF exhibits a unique and powerful capability: the recovery of occluded object parts in its internal representations. This is demonstrated on the challenging, realistic SEM dataset and provides a crucial step toward more robust scene decomposition.





* Clarity and Efficacy: The method is conceptually clean (competitive binding + orthogonality loss) and empirically highly effective, with strong results across various conditions, noise, and out-of-distribution tests.

### Weaknesses
* Hyperparameter Sensitivity: The orthogonality loss is controlled by a weighting factor, $\lambda$. Table 1 shows that optimal performance varies across different settings of $n$ (orientation dimensionality), requiring a hyperparameter search for the specific dataset. For instance, $n=7$ prefers $\lambda=0.1$ while $n=5$ prefers $\lambda=0.8$. A more thorough sensitivity analysis or a proposed dynamic weighting strategy could improve the method's generalizability.




* Dependency on Final-Layer Output: The best-performing OrthoRF method (OrthoRF$^{\text{thresh}}$ on $\psi_{final}$) requires binarizing the intermediate map with a threshold (0.1) and no further post-processing. The dependence on a hand-tuned global threshold (0.1) for peak performance might limit its applicability compared to the fully unsupervised k-means approach used by RF and OrthoRF$^{\text{kmeans}}$. The reliance on an *intermediate* layer ($\psi_{final}$) rather than the final $z_{final}$ output for the $MBO_{i}^{OV}$ metric should be discussed as a limitation or characteristic.


* The authors claim training on noisy and testing on clean data degrades performance. I encourage authors to look at Extreme Image Transforms (EIT) [Crowder and Malik, 2022; Malik, Crowder and Mingolla, 2023, Biol Cybernetics]. They show that training on EITs helps with robustness on object detection tasks. 

* The authors do not compare how their methods compare to other non-RF methods. I would encourage the authors to also compare their findings with methods like Stable and expressive recurrent vision models [Linsley et al., 2020, NeurIPS].

### Questions
1. The OrthoRF$^{\text{thresh}}$ results, particularly the striking $MBO_{i}^{OV}$ performance, rely on using the intermediate $\psi_{final}$ with a manually set global threshold of 0.1. Have the authors explored making this threshold adaptive, for instance, by linking it to the statistical properties of the activations (e.g., a multiple of the standard deviation) to maintain the method's unsupervised nature?



2. The paper uses a synthetic SEM dataset to show robustness to noise and occlusions. Could the authors elaborate on how they would apply the OrthoRF framework to a real, complex dataset, like CLEVR or MOVi-C, and what modifications might be necessary to the architecture or training protocol?



3. Given the effectiveness of orthogonality, have the authors considered applying the orthogonality loss constraint directly to slot-based models (like Slot Attention) in conjunction with their reconstruction or binding losses? This might provide further insight into the general utility of this inductive bias in OCL.

4. In relevant works, the authors have missed a few important papers to cite/compare their method with. When talking about ROOTS and SAVi, the work on PathTracker and InT [Linsley et al., 2021, NeurIPS] also disentangles objects in space. Similarly the extension of this work, FeatureTracker [Muzellec et al., 2025, ICLR] disentangles the changes in feature and colorspace. 

5. I would highly encourage the authors to release the code publicly for further experimentation by the community.

6. The authors should also mention about the training details and hardware setup. 

7. The citation for paper Rotating features for object discovery is repeated twice.

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
2

### Summary
The paper introduces OrthoRF, a simple tweak to Rotating Features that makes each “orientation channel” specialize in a single object by adding a softmax competition and an orthogonality penalty. This reduces the usual phase-mixing and removes the need for post-hoc k-means, especially helping in overlapping/occluded regions where they can directly threshold an intermediate map to complete hidden parts. On synthetic 4Shapes and a custom SEM-style dataset, OrthoRF matches RF on standard object discovery and improves occlusion completion, with cleaner, near-orthogonal channels.

### Strengths
- The idea is intuitive: make channels compete and push them orthogonal so each object lands on its own axis. This yields cleaner, more interpretable features and avoids fragile clustering steps.
- The method is lightweight to implement (softmax + inner-product loss) and shows consistent angle statistics and tighter clusters. --- Qualitative plots match the quantitative gains, which is good to see. Table 2 shows substantial gains over RF in noisy conditions

### Weaknesses
- Most evidence is on binary synthetic images. It’s hard to judge real-world usefulness without RGB/textures or common OCL benchmarks.
- The headline occlusion gains rely on changing the decoding step. This makes it unclear how much of the win comes from the model vs. the decoding choice.
- The method still needs n set to at least the number of objects plus background. It’s not clear how it behaves when object count varies or is unknown at test time. The paper doesn't show what happens when n is too small (fewer components than objects), n is too large (many unused components) or object count changes at test time
- There’s no clear orthogonality ablation—it would help to see results when varying the orthogonality term, or comparing to alternative orthogonality formulations, to verify that this term is the main driver of the improvements.

### Questions
- Can you report both models with the same decoding strategy so we can separate model effects from decoding choices? A small table would clarify this quickly.
- Will you release code, configs, and a generator for the SEM dataset so others can reproduce the results?

### Soundness
2

### Presentation
3

### Contribution
2
