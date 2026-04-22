# Memorization Through the Lens of Sample Gradients

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Deep neural networks are known to often memorize underrepresented, hard examples, with implications for generalization and privacy.  Feldman & Zhang (2020) defined a rigorous notion of memorization. 
However it is prohibitively expensive to compute at scale because it requires training models both with and without the data point of interest in order to calculate the memorization score.
We observe that samples that are less memorized tend to be learned earlier in training, whereas highly memorized samples are learned later. 
Motivated by this observation, we introduce Cumulative Sample Gradient (CSG), a computationally efficient proxy for memorization. CSG is the gradient of the loss with respect to input samples, accumulated over the course of training.
The advantage of using input gradients is that per-sample gradients can be obtained with negligible overhead during training. The accumulation over training also reduces per-epoch variance and enables a formal link to memorization. Theoretically, we show that CSG is bounded by memorization and by learning time.
Tracking these gradients during training reveals a characteristic rise–peak–decline trajectory whose timing is mirrored by the model’s weight norm. This yields an early-stopping criterion that does not require a validation set: stop at the peak of the weight norm. This early stopping also enables our memorization proxy, CSG, to be up to five orders of magnitude more efficient than the memorization score from  Feldman & Zhang (2020).  It is also approximately 140 $\times$ and 10$\times$ faster than the prior state-of-the-art memorization proxies, input curvature and cumulative sample loss, while still aligning closely with the memorization score, exhibiting high correlation. Further, we develop Sample Gradient Assisted Loss (SGAL), a proxy that further improves alignment with memorization and is highly efficient to compute. Finally, we show that CSG attains state-of-the-art performance on practical dataset diagnostics, such as mislabeled-sample detection and enables bias discovery, providing a  theoretically grounded toolbox for studying memorization in deep networks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a computationally efficient way to approximate the degree of memorization in deep neural nets. Based on the observation that memorized samples tend to have longer training time, this work proposes cumulative sample gradient (CSG) as a proxy for memorization. Theoretical results show the relation between CSG and learning time & memorization. Empirical evaluations corroborates the findings and shows superior computational performance over previous state-of-the-art.

### Strengths
Overall, this work is well-written and organized. It motivates the problem settings and draws fairly clear connection with previous work. The contribution is pertinent to the current challenge of ML. By providing a more efficient probe for the phenomenon of memorization, this work can accelerate future research in this field. The theoretical formulations are solid and with clear purpose: they do shed light into the construction of the practical proxy. The empirical improvements are encouraging. No major flaw with experiment design.

### Weaknesses
The work doesn't have apparent weaknesses that might lead to clear reject. I do have a few questions about the assumption and the source of computation edge over previous work. Having them clarified can better help the reader understand the contribution and use the tool with confidence.

There are a few grammatic glitches here and there. For example, "is plays" at Line 310 and "it's roots" at Line 258. Can be fixed by proof reading.

### Questions
1) What makes the computation of CSG faster than CSL? Seems that both metrics are cumulative and the computation of loss/gradient are not too different in general. Could you tell us more about the source of speedup?

2) The theoretical results show that CSG is upper bounded by learning time and memorization. Does that mean high CSG -> high memorization? What about the opposite direction, does low CSG -> low memorization?

3)  The theories are formulated against pure SGD. Does CSG reliably detect memorization/mislabeled samples for different optimizers? If time allows, could you show some evidence of CSG's success for other optimizer? 

4) Are previous SOTA's performance dependent on the choice of optimizer?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes CSG, a fast, theoretically grounded proxy for measuring memorization in deep networks by accumulating input loss gradients during training. CSG correlates strongly with true memorization scores while being up to five orders of magnitude more efficient, enabling validation-free early stopping, mislabeled-sample detection, and bias discovery.

### Strengths
CSG offers a theoretically grounded and computationally efficient way to estimate memorization, achieving near-perfect correlation with true scores at a fraction of the cost.

It enables validation-free early stopping and state-of-the-art mislabeled data detection, making it both practical and interpretable for large-scale deep learning.

### Weaknesses
Novelty:
My primary concern lies in the novelty of the work. The authors’ main observation that memorization tends to occur in the later stages of training is well established and has been extensively documented in prior studies [1,2]. Likewise, leveraging gradients to approximate or track memorization has been explored before [3,4], making the core idea appear incremental rather than groundbreaking. Can the authors show how their work performs in relation to [1,3,4].

Computational Cost:
While the proposed approach claims efficiency, computing cumulative sample gradients still requires forward and backward passes for each sample at every epoch, which can be prohibitive for large models and datasets. Prior work [3] already proposes strategies to reduce this overhead.

Limited Optimizer Evaluation:
The experiments rely solely on the Adam optimizer. To demonstrate broader applicability, results should be validated across multiple optimizers such as SGD, RMSProp, and AdamW. Can authors provide more info on how their method would behave across optimizers?


[1]. Agiollo, Andrea, Young In Kim, and Rajiv Khanna. "Approximating Memorization Using Loss Surface Geometry for Dataset Pruning and Summarization." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024.
[2] https://aclanthology.org/2024.blackboxnlp-1.4
[3] https://arxiv.org/abs/2008.11600
[4] https://arxiv.org/pdf/2002.08484

### Questions
Above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Cumulative Sample Gradient (CSG)—the loss gradient w.r.t. the input, accumulated over training—as a computationally cheap proxy for stability-based memorization (Feldman & Zhang, 2020). The authors provide theory that (i) expected CSG is upper-bounded by learning time (Theorem 4.2) and (ii) linearly bounded by memorization (Theorem 4.3). Empirically, they observe a characteristic rise–peak–decline trajectory for average per-sample input gradients that aligns with a peak in weight norm and the first minimum in validation loss (double-descent boundary), enabling validation-free early stopping. They also introduce SGAL (a loss accumulated only until the gradient-based stopping point) for further efficiency. Across CIFAR-100/ImageNet, CSG/SGAL correlates well with memorization scores, is substantially faster than curvature and CSL proxies, supports mislabeled-sample detection, and helps surface dataset biases.

### Strengths
•	Originality & clarity: Using input gradients accumulated over training as a memorization proxy is elegant; the rise–peak–decline alignment with the weight norm and double-descent boundary is compelling and clearly presented. 
•	Quality (theory): Theorems relating CSG to learning time and memorization provide formal grounding absent in many proxies; assumptions and proof sketches are transparent. 
•	Quality (empirics): Consistent binned linear trends; strong correlation with F&Z scores; broad comparisons (CSL, curvature, forgetting events, loss sensitivity) on CIFAR-100/ImageNet; MIA and adversarial-distance analyses support the privacy link. 
•	Significance & practicality: Large speedups (0.1–0.3× of standard training vs. 3.6–14.3× for curvature; orders of magnitude vs. F&Z) lower the barrier to dataset diagnostics at scale; mislabeled-sample AUROCs are SOTA or competitive at all noise levels.

### Weaknesses
•	Assumption sensitivity and constant opacity. The theoretical bounds rely on β-stability, Lipschitz continuity, L-bounded losses, and learning-rate conditions, and exclude first-layer skip connections. Constants involving the pseudo-inverse of batch matrices (κ terms) may be large/ill-conditioned, making the bounds hard to interpret quantitatively. 
•	Calibration claim is mixed. Table 2 shows lower ECE for the last epoch (0.1017) than for gradient-based stopping (0.1382), contradicting the blanket statement that early-stopped checkpoints have lower calibration errors; other metrics (MCE/MSCE/UCE) favor earlier stopping, so the narrative should be nuanced. 
•	Scope of datasets/models. Results are primarily on CIFAR-100/ImageNet with ResNet/Inception. It would strengthen generality to include modern architectures (e.g., ViT) and tasks beyond image classification, since input-gradient behavior and training dynamics may differ. (The Adam experiment is a useful first step.) 
•	Comparative coverage. While CSL/curvature/forgetting/loss-sensitivity are included, some adjacent proxies (e.g., EL2N, GraNd / importance-sampling-style difficulty measures) and influence-based approximations (e.g., TracIn) are not compared; these could provide a more complete picture of trade-offs.
•	Training-access requirement. Like many proxies, CSG needs access to per-sample gradients during training; this limits pure post-hoc auditing scenarios (the limitation is acknowledged). 
•	Qualitative bias analysis. The bias discovery examples are informative but largely qualitative; quantitative fairness metrics (e.g., subgroup error rates) would make the case stronger.

### Questions
1.	How sensitive is the weight-norm peak rule to weight decay, label smoothing, data augmentation strength, and optimizer hyperparameters? A small ablation across these knobs would clarify robustness.  
2.	Practically, do you compute input gradients every iteration for all samples, or on a schedule/subset? Please quantify wall-clock overhead vs. vanilla training across model sizes.  
3.	Have you tested CSG/SGAL on ViTs or transformers for NLP? If not, what obstacles (e.g., tokenization, augmentation) do you anticipate?
4.	Can you empirically estimate the constants in Lemma 4.1 (e.g., behavior of κ through training) to illustrate why the linear trends emerge despite potential ill-conditioning?  
5.	Given Table 2 shows mixed results across ECE/MCE/MSCE/UCE, can you reconcile the claim “lower calibration errors than last epoch” and specify which metrics you prioritize and why?  
6.	Since you use precomputed F&Z scores, how sensitive are your correlations to training recipe variations (architectures different from F&Z)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Cumulative Sample Gradient (CSG) as a theoretically motivated and computationally efficient proxy for memorization in deep neural networks. The authors define CSG as the gradient of the loss with respect to input samples, accumulated across training, and claim it correlates strongly with Feldman & Zhang’s formal memorization score while being orders of magnitude cheaper to compute. They theoretically show that CSG is bounded by both learning time and memorization, then empirically validate this relation across CIFAR-100 and ImageNet. Moreover, they propose that the peak of the model’s weight norm corresponds to the optimal early-stopping point, eliminating the need for a validation set. The paper also introduces Sample Gradient Assisted Loss (SGAL) as an efficiency improvement, and reports strong performance on tasks such as mislabeled sample detection and dataset bias discovery

### Strengths
1. The work links input-space gradients to memorization and learning dynamics through formal theorems, extending prior work that primarily focused on weight gradients or loss-based proxies.

2. The idea of accumulating input gradients incurs minimal additional computation during training and is potentially useful for large-scale data auditing, noisy-label detection, and privacy diagnostics.

3. The observed “rise–peak–decline” trajectory in sample gradients and weight norms provides an intuitive link between optimization dynamics and generalization behavior.

### Weaknesses
1. The main claim that “Cumulative Sample Gradient” represents a gradient of loss with respect to the input, accumulated over training, is conceptually questionable. The proposed CSG is essentially an aggregated gradient norm trajectory rather than a true differentiable functional of the loss. Treating it as a gradient object conflates sensitivity analysis (∇ₓℓ) with memorization, which lacks theoretical grounding in generalization theory. The derivations (Theorems 4.2–4.3) merely establish loose proportionality bounds without proving causality or sufficiency.

2. The assertion that the peak of weight norm universally coincides with the minimum validation loss is overstated. This correspondence may depend on architecture, optimizer, and regularization strength, and may fail under strong augmentation or non-stationary data.

3. While the authors claim that CSG generalizes across tasks, they only test standard supervised image classification. The theoretical link assumes uniform β-stability of SGD and bounded loss, which rarely holds in modern deep nets. It remains unclear whether CSG maintains predictive utility in other regimes such as self-supervised, generative, or multi-label settings.

### Questions
1. Since CSG is defined as the accumulated input gradient norm, not the derivative of a loss functional over training trajectories, do we have a rigorous reason to treat it as a “gradient of loss with respect to input samples, accumulated over training”? How does this differ from simply tracking cumulative sensitivity?

2. The paper asserts that stopping at the peak weight norm matches the minimum validation loss. Can this be proven under general conditions? How robust is this correspondence across architectures, datasets, or optimizers (e.g., Adam, Adagrad, adaptive schedulers)?

3. Do you think the CSG–memorization relationship could generalize to regression, contrastive, or generative models, where accuracy or label definitions differ? You don’t need to perform new experiments—rather, please share your intuition on whether and why such generalization might hold.

### Soundness
2

### Presentation
3

### Contribution
3
