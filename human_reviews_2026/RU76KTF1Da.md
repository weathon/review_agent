# How does the optimizer implicitly bias the model merging loss landscape?

- Decision: Accept (Poster)
- Scores: 2, 6, 6

## Abstract
Model merging combines independent solutions with different capabilities into a single one while maintaining the same inference cost. 
Two popular approaches are _linear interpolation_, which simply averages multiple model weights, and _task arithmetic_, which combines task vectors obtained by the difference between finetuned and base models. 
While useful in practice, what properties make merging effective are poorly understood. This paper explores how the optimization dynamics affect the loss landscape geometry and its impact on merging success.
We show that a single quantity -- the _effective noise scale_ -- unifies the impact of different optimizer components on model merging. Across architectures and datasets, merging success is a non-monotonic function of the effective noise scale, with a distinct optimum. Decomposing this quantity, we find that larger learning rates, stronger weight decay, smaller batch sizes, and data augmentation all independently modulate the effective noise scale and exhibit the same qualitative trend.
Unlike prior work connecting optimizer noise to the flatness or generalization of _individual_ minima, we show that it also affects the _global_ loss landscape, predicting when independently trained solutions can be successfully merged. Our findings broaden the understanding of how optimization shapes the loss landscape geometry and its consequences for model merging, suggesting that training dynamics could be further manipulated to improve model merging.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates how optimization dynamics affect model merging success by introducing the effective noise scale as a unifying factor. The authors demonstrate that optimizer components (learning rate, weight decay, batch size, data augmentation) collectively modulate this noise scale, which exhibits a non-monotonic relationship with merging effectiveness. Through extensive experiments across multiple architectures (MLP, ResNet, DenseNet, Transformer, GPT) and datasets (SVHN, CIFAR, TinyImageNet, WILDS, TinyStories), the paper shows that intermediate noise levels enable optimal merging, while too little or too much noise degrades compatibility. The findings apply to both linear interpolation and task arithmetic merging approaches.

### Strengths
- Unifying framework: The effective noise scale provides an elegant unification of multiple optimizer hyperparameters, offering practitioners a single lens through which to understand merging compatibility.
- Comprehensive experimental validation: The paper presents thorough empirical evidence across diverse architectures, datasets, and modalities (vision and language), demonstrating the generality of findings.
- Systematic ablations: Each optimizer component is carefully isolated and studied with proper controls ensuring convergence.
- Practical relevance: The work addresses the important practical problem of model merging, with direct implications for model soup and task arithmetic methods. While a lot of effort has been dedicated in recent years to designing better model merging methods, little to no attention has been given to how to training procedure itself affects mergeability despite clearly playing an important role.
- Clarity: The paper is very well written and easy to follow. The claims are clear and the experiments are simple yet well explained and in direct support of the claims.

### Weaknesses
- Lack of theoretical analysis: The paper is primarily empirical without theoretical guarantees explaining why effective noise controls merging or why the relationship is non-monotonic. The connection to loss landscape geometry could be formalized more rigorously.
- Limited scale: The authors acknowledge computational constraints prevented large-scale experiments. The largest models tested are relatively small by modern standards, raising questions about scalability. This is problematic since model merging techniques are often used to combine relatively larger models.
- Proxy measurement: The experiments use the normalized proxy Ŝ = η/(B(1-μ)²) rather than directly measuring the full effective noise scale including Σ_A(θ_t) from Equation 3. This approximation's validity isn't thoroughly validated.
- Non-actionable insights: The non-monotonic relationship with a "sweet spot" makes it difficult to provide clear prescriptive guidance. Practitioners still need to search for optimal hyperparameters rather than having a principled selection method. A more detailed analysis of *how* to find this sweet spot would be very useful.
- Limited scope of experiments: While the experiments are extensive in that they adequately support the paper's claims regarding learning rate, weight decay, etc. they have a relatively narrow scope. There are so many interesting parameters to play with here which are also important training choices such as the actual **optimizer** used (the authors used AdamW for almost all experiments it would be valuable to compare against Adam or SGD in the same set-up); the effect of **momentum**, how does momentum affect the effective noise scale? stronger momentum minimizes the effect of each individual batch, potentially lowering gradients variance; and last but not least, the **learning rate scheduler** is also a very important factor which can have a huge effect on training and performance.
- Limited merging settings: On the merging side, there have been many recent methods proposed (e.g. TIES, DARE, etc.), it would be a very valuable contribution to look at how the merging method interacts with the effective noise scale. Can certain merging methods counteract the effects of sub-optimal optimizer hyper-parameters? For example, is the learning rate less crucial if the merging method uses some sort of pruning mechanism? Also, the settings used are a bit simple by model merging standard, i.e. only two models are merged at the time and the models were trained either on the same, or similar tasks. In practice more than two models trained on tasks that can be very different are merged.

In summary, I like the use of the effective noise scale as a unifying framework to analyze the effect of the optimization procedure on mergeability, I think the effect of training on merging has been understudied. I also think the paper is very well written and clear and that the experiments accurately support the claims. However, there is limited theoretical analysis of this framework and the experiments are somewhat limited in scope (see Weaknesses). Also, while the unifying framework is interesting and is justified by the experiments, the insights are somewhat non-actionable. I like the paper but I think it would need either (1) a more in-depth theoretical analysis, (2) more experiments to help with practical impact either of other training factors (effect of optimizer, momentum, learning rate schedule) or (3) an analysis of the interplay between effective noise and merging method in more realistic merging scenarios, OR (4) more clear actionnable insights as to how to find the sweet spot without just running a large hyper-parameter sweep which practitioners already do on new tasks. I'd be happy to increase my score if at least one or two of these 4 points are addressed.

### Questions
- Very important question and reason I gave a 2 instead of a 4: Section 3.6 starts with "In the previous sections, we analyzed settings where models are trained from scratch." Indicating that for all experiments in earlier sections the models were trained from scratch (from the same or different random initializations?). This is very surprising to me since the accuracy gain is positive in multiple experiments from these earlier sections which contradicts a large body of work on LMC and merging for models trained from scratch. [1] established that even networks trained on the same tasks aren't stable to SGD noise at initialization (i.e. no LMC for models trained from scratch) and there are multiple papers, such as [2, 3, 4], trying to permute / align the neurons of models trained from scratch to be able to merge them without a catastrophic drop in accuracy. Based on these works, successful merging of models trained from scratch shouldn't be possible, therefore I suspect there might be something wrong with the experimental framework. Perhaps the models weren't trained from a random initialization but from a pre-trained checkpoint, this would be fine, I don't think it removes much from the paper's value. **If this question is properly addressed and a valid justification is given I will gladly raise my score from a 2 to a 4 (and even further if more experiments are provided, see weaknesses).** The text also needs to me modified accordingly.
- Typos: line 296 end "+1?%"; line 357 "Figure 6 on the right" but the subplots are one over the other
- How does the effective noise scale interact with model capacity and dataset complexity? Are the optimal noise levels dataset/architecture-dependent?
- Have you measured the actual gradient-noise covariance Σ_A to validate the proxy Ŝ?
- Can you provide guidelines for practitioners to identify the "sweet spot" without extensive hyperparameter search?

**References**
- [1] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, Michael Carbin. Linear Mode Connectivity and the Lottery Ticket Hypothesis
- [2] Samuel K. Ainsworth, Jonathan Hayase, Siddhartha Srinivasa. Git Re-Basin: Merging Models modulo Permutation Symmetries
- [3] Sidak Pal Singh, Martin Jaggi. Model Fusion via Optimal Transport
- [4] Stefan Horoi, Albert Manuel Orozco Camacho, Eugene Belilovsky, Guy Wolf. Harmony in Diversity: Merging Neural Networks with Canonical Correlation Analysis

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors show that the effective noise scale, which aggregates noise across data covariance, learning rate, and momentum, has an inverted U relationship to model mergeability; too little or too much noise makes model merging difficult.

### Strengths
- The inverted U relationship to effective noise and mergeability is interesting and intuitive; too little noise gives weird solutions, too much noise moves models too far apart to merge
- The authors show the relationship with the effective noise holds for both vision and language models.
- The authors try a few common (and simple) model merging techniques

### Weaknesses
- The biggest weakness is that the contribution seems quite incremental; it's already known that solutions located on relatively flat areas of the loss landscape have better generalization behavior and that the effective noise during training helps find these flatter solutions (the authors acknowledge this). It seems reasonable to assume that it's easier to merge models on a flat part of the loss landscape. But, someone should probably check that this idea holds up and the inverted U behavior of the effective noise is interesting, so that's why I recommend a weak accept.
- I assume that the flatness of the loss landscape/effective noise perspective probably holds for other model merging techniques. It would be nice to know if that's true for some more complicated techniques as well (e.g. Fisher-weighted averaging). There's always more techniques that can be tried, so I don't think this is a major issue.

Typos/Grammar:
- Line 250: should be "ubiquitously"
- Line 252: should be "demonstrating"
- Line 268: Could use a rephrase: "Lastly, similarly as the learning rate, Appendix Figure 16 shows that a too large weight decay leads to failure in model performance and model merging."
- Line 408: Grammar: "Furthermore, as in linear interpolation merging, a “too large” becomes unstable."
- Caption for Figure 8: Could use a rephrase: "Merging pairs of similar and larger learning rates has the best performance."

### Questions
- I didn't understand what you actually did here (line 395; this should probably also be rephrased for clarity): Models w/o pretraining weight from Section 3.2 (i.e. pretraining dataset is the same as the finetuning dataset). Task arithmetic is applied to a base model and two task vectors. For each checkpoint, the base model θbase is the checkpoint itself, and the task vectors are obtained from the endpoint models τA = θA − θbase and τB = θB − θbase.
- I'm a bit unsure about exactly what's going on in the figures (e.g. Figure 2). Does each point on the figure correspond to a checkpoint during training? And, if I'm understanding correctly, two checkpoints are merged — do both checkpoints show up on the graph as separate points?

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
4

### Summary
This paper investigates how optimizer choices shape the mergeability of independently trained models. The central claim is that a single quantity—the effective noise scale $\tilde{S} \approx \eta / B(1-\mu)^2$ (augmented by data-augmentation–induced covariance)—organizes the effect of learning rate, batch size, momentum, weight decay, and augmentation on merging success. Empirically, mergeability (via linear interpolation and task-arithmetic) is non-monotonic in $\tilde{S}$: there is a “sweet spot” between too little and too much noise. The study spans CNN/MLP/Transformer/GPT, from-scratch, transfer learning (CLIP ViT-B/16), and small language modeling (TinyStories). Key reported regularities: (i) larger LR or WD (on scale-invariant nets) tends to produce more mergeable solutions; (ii) smaller batches help; (iii) augmentation helps; (iv) for task arithmetic, benefits of larger LR depend strongly on having a pretrained initialization.

### Strengths
- **Unifying lens**: Framing optimizer and data choices through a single effective noise control for cross-solution compatibility (not just single-solution generalization) is a crisp, novel angle.
- **Systematic sweeps**: Clear ablations across LR/WD/B/augmentation, with consistent procedures and multiple architectures/datasets.
- **Cross-domain evidence**: Vision from scratch, transfer learning, and (albeit small) language modeling make the phenomenon less likely to be an artifact of one setup.
- The story flows well from preliminaries → unified factor → component ablations → task arithmetic → transfer.
- **Bridges two communities**: Brings SGD-as-stochastic-process insights into model-merging practice, with implications for soups, adapters, and FL-style workflows.

### Weaknesses
- Much of the study is effectively in shared-trajectory (same init + branched checkpoints). Mergeability here may reflect trajectory alignment more than global landscape bias. Also, permutation symmetries are not addressed.
- Flatness alone doesn’t explain mergeability; representation overlap matters. One way to address this could be reporting centered-kernel alignment between penultimate-layer features of the two models; correlate CKA with merge gain at fixed $\tilde{S}$.
- The connection to SWA / soups / rebasin / SANE / KnOTS / PMA is noted but not dissected. It would be interesting tocCompare “tune noise” against (i) SWA style training aimed at wider valleys, (ii) permutation-aware merging, and (iii) subspace alignment (e.g., SVD/CCA-merge). Show complementarity or superiority.

### Questions
1. Can you disentangle selection vs creation of mergeable minima? For example, if you freeze the early trajectory (same prefix) and only vary 
$\tilde{S}$ after bifurcation, do you retain the same non-monotonic curve?
2. What breaks at larger scales (if anything)? Are there plans of extending the scale of the experiments or are there already preliminary adapter-level or LoRA-head results on larger backbones?
3. Given a target mergeability level, how would practitioners choose LR/WD/B/augmentation?
4. How does “tune noise” differ from, or complement, SWA/model-soups’ push toward wide valleys?

### Soundness
3

### Presentation
3

### Contribution
3
