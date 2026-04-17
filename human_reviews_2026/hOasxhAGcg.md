# Preventing Latent Reharsal Decay in Online Continual SSL with SOLAR

- Decision: Reject
- Scores: 4, 6, 6

## Abstract
Continual learning methods enable models to learn from non-stationary data without forgetting. We study Online Continual Self-Supervised Learning (OCSSL), in which models learn from a continuous stream of unlabeled data. We find that OCSSL exhibits surprising learning dynamics, favoring plasticity over stability, with a simple FIFO buffer outperforming Reservoir sampling. We explain this result with the Latent Rehearsal Decay hypothesis, which attributes it to latent space degradation under excessive stability of replay. To quantify this effect, we introduce two metrics (Overlap and Deviation) and show their correlation with declines in probing accuracy. Building on these insights, we propose SOLAR, which leverages efficient online proxies of Deviation to guide buffer management and incorporates an explicit Overlap loss. Experiments demonstrate that SOLAR achieves state-of-the-art performance on OCSSL vision benchmarks, highlighting its effectiveness in balancing convergence speed and final performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the long-term performance degradation issue in Online Continual Self-Supervised Learning (OCSSL), which the authors term "Latent Rehearsal Decay." They propose a replay strategy called SOLAR, which includes a loss-based buffer sampling method (the Deviation-Aware Buffer) and an additional contrastive loss term (Overlap Loss).

### Strengths
+ The paper identifies a compelling and counter-intuitive phenomenon in OCSSL: the simple FIFO strategy outperforming the theoretically more stable Reservoir strategy over long training schedules.
+ The experimental validation is thorough, covering multiple variables and validating the method's effectiveness against several baselines.

### Weaknesses
1. **Problem Reframing:** The proposed concept of "Latent Rehearsal Decay," while well-analyzed, can be seen as a well-known issue in continual learning—overfitting on the replay buffer. When the buffer's contents become static, the model repeatedly sees the same data, which naturally leads to a decline in generalization. It could be argued that the paper provides a new name for an existing problem rather than discovering a completely new phenomenon.
2. **Incremental Contribution:** The core ideas behind the SOLAR method are not entirely new. Its essence is to prioritize "hard samples" (i.e., those with high loss), a concept with well-established precedents in Experience Replay, notably "Prioritized Experience Replay" (PER). Applying this idea to a self-supervised context, using the SSL loss as a proxy for difficulty, is a relatively direct application and lacks fundamental novelty.
3. **Lack of Deeper Intuition on the Proposed Metrics**: The paper demonstrates that its proposed metrics (Deviation and Overlap) can detect the failure mode while metrics (SVD, uniformity) cannot. However, it **fails to provide a deep, intuitive explanation** for _why_ this is the case. The analysis does not go much beyond showing that "they work," leaving a gap in our fundamental understanding of the phenomenon.
4. **Compatibility with other Self-Supervised Algorithms:** The validation is exclusively on SimSiam (a non-contrastive method). It is unclear how SOLAR's "Overlap Loss" would interact with contrastive methods like SimCLR or MoCo, where it might become redundant or even conflict with the native loss function.
5. **Lack of Quantitative Computational Analysis:** Although the authors considered computational cost in their design, the paper lacks a quantitative experiment comparing the computational overhead (e.g., training time, FLOPs) against other methods.
6. **Incomplete Experimental Analysis**: The analysis of the SVD experiment in the appendix feels incomplete. Unlike the uniformity analysis, it is only performed on the final test set. A more comprehensive analysis across past, present, and future data splits could have provided crucial insights into the dynamics of the feature space, representing a **missed opportunity** to strengthen the paper's core claim.  
7. Although I know that Online Continual SSL is not proposed for the first time in this paper, I still have some major doubts about this setting. I feel that pre-training and data streaming are somewhat contradictory concepts: pre-training usually requires sufficient training time and a large amount of data to obtain general representations applicable to most data, whereas data streaming emphasizes fast adaptation and avoiding forgetting. I hope the authors can provide some real-world examples where this setting is applicable.

**Conclusion:** Given the high bar for conceptual and methodological innovation at ICLR, I am inclined to recommend rejection.

### Questions
See the weakness

### Soundness
2

### Presentation
2

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
This paper investigates the learning dynamics of online continual self-supervised learning (OCSSL) and identifies it to favor plasticity over stability. Through empirical analysis, the authors make the observation that FIFO buffer management outperforms reservoir sampling in OCSSL settings. They attribute this to a phenomenon they term "latent rehearsal decay," which arises from prolonged training on static subsets of data during replay. To address this, the paper introduces two main contributions: (1) two metrics to quantify latent rehearsal decay, and (2) a solution framework consisting of SOLAR, a deviation-aware replay buffer management strategy that prioritizes samples with higher loss, and an overlap loss for regularization. The method is evaluated using Simple Siamese networks as the SSL backbone and demonstrates improved accuracy over long training horizons.

### Strengths
- The identification of latent rehearsal decay as a specific challenge in OCSSL is novel, and the counterintuitive finding that FIFO outperforms reservoir sampling in this setting provides fresh perspective on the plasticity-stability trade-off.
- The core technical insight is sound—that samples with higher loss indicate higher feature deviation and should be prioritized for replay. The proposed metrics for quantifying latent rehearsal decay provide a useful analytical tool for the community.
- The provided solutions show practical improvements in accuracy over long training horizons. The deviation-aware buffer management strategy is a valuable contribution that could be adapted to various continual learning scenarios.

### Weaknesses
- The evidence that plasticity is favored rather than stability rests primarily on FIFO vs. reservoir comparison, which may not be sufficiently comprehensive, given that this comparison fails in the case of CLEAR dataset. This comparison seems to work for Class incremental online ssl but not in domain incremental online ssl. 

- Figure 2(a) has a confusing x-axis. The explanation of what is meant by mini batch passes is unclear as a measure of training schedule. The plot shows in Imagenet100  reservoir achieves accuracy of >46 after one minibatch pass but this is contradictory with Figure 3 where it acheives >46 accuracy after several passes

- Figure 2(b) does not clarify which dataset the results are based on. 

- Motivation for why OCSSL is an important problem area could be better articulated.

### Questions
-  Why is EMA necessary for updating buffer statistics rather than direct replacement using the current values ? 
- What is the specific contribution of extraction counts, why can't we just select based on loss ? 
- Can you clarify the apparent inconsistency between Figure 2(a) showing >46% accuracy after 1 mini-batch pass and Figure 3 showing this accuracy is reached only after 50k steps on ImageNet?

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
The paper studies identifies a novel challenge in  Online Continual Self-Supervised Learning (OCSSL), Latent Rehearsal Decay, where replay strategies that converge to a static subset (*e.g.*, Reservoir) degrade latent representations over long online training streams, degrading downstream accuracy. To quantify this phenomenon, the authors introduce two latent-space metrics, Deviation (intra-sample variability in augmented features) and Overlap (inter-sample feature intersection), and show that these metrics correlate with drops in accuracy. Building on this analysis, they propose SOLAR (Self-supervised Online Latent-Aware Replay), which *i)* uses a Deviation-aware buffer that prioritizes high-loss samples and *ii)* an Overlap loss that penalizes positive overlap between the features from the current minibatch and the ones from the buffer. Experiments on Split CIFAR-100, ImageNet-100, and CLEAR100 show that SOLAR achieves strong average and final accuracy after linear probing, mitigating the reported Latent Rehearsal Decay.

### Strengths
- Latent Rehearsal Decay represents a meaningful and interesting challenge in Online Continual Self-Supervised Learning.

- The proposed method demonstrates competitive performance across diverse benchmarks (CIFAR-100, ImageNet-100, CLEAR100), buffer configurations, and training lengths.

- The paper is well-written and easy to follow in its intuition and formulation.

### Weaknesses
1. The Deviation-aware buffer closely resembles prior loss- and gradient-based storage and retrieval strategies. The paper omits important references such as Gradient-based Sample Selection (GSS) (Aljundi et al., NeurIPS 2019, gradient-aware storage), Rethinking Experience Replay (LARS) (Buzzega et al., ICPR 2020, loss-aware storage), and Maximally Interfered Retrieval (MIR) (Aljundi et al., NeurIPS 2019, interference-aware retrieval). These works define related mechanisms for sample storage and retrieval; a theoretical and empirical comparison would clarify the novelty and advantages of the proposed approach.

2. Figures are poorly integrated into the text, as they are referenced only after extended discussion, and multi-panel figures have their subcomponents explained in separate sections. Captions are often too long and mix description with interpretation, which fragments the narrative and reduces readability.

3. The computation of Deviation and Overlap requires generating multiple augmentations per sample. The paper does not quantify the online computational cost or analyze the sensitivity of SOLAR to its hyperparameters. A complexity and sensitivity analysis would improve the understanding of SOLAR’s applicability robustness.

4. Reference works for continual learning, online continual learning, and self-supervised learning are mis-cited: when first mentioning these subjects, the manuscript relies primarily on recent secondary papers instead of referring to foundational works.

### Questions
1. Are the Deviation and Overlap curves in Figure 3 computed from the training stream, from the buffer only, or current minibatch + buffer? 

2. When the Overlap loss is applied on top of Reservoir or FIFO, the paper mentions inconsistent (and sometimes negative) gains. What's your intuition on the reasons behind this?

### Soundness
3

### Presentation
2

### Contribution
2
