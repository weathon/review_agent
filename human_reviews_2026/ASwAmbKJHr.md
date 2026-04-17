# Enabling Fine-Tuning of Direct Feedback Alignment via Feedback-Weight Matching

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Although Direct Feedback Alignment (DFA) has demonstrated potential by enabling efficient and parallel updates of weight parameters through direct propagation of the network's output error, its usage has been primarily restricted to training networks from scratch. In this paper, we introduce feedback-weight matching, a first method that enables reliable fine-tuning of fully connected neural networks using DFA. We provide an analysis showing that existing standard DFA struggles to fine-tune networks that are pre-trained via back-propagation. Through a thorough analysis of weight alignment (WA) and gradient alignment (GA), we demonstrate that the proposed feedback-weight matching enhances DFA's ability and stability in fine-tuning, which provides useful insights into DFA's behavior and characteristics when applied to fine-tuning. In addition, we prove that feedback-weight matching, when combined with weight decay, not only mitigates over-fitting but also further reduces the network output error, leading to improved learning performance during DFA-based fine-tuning. Experimental results show that feedback-weight matching, for the first time, enables reliable fine-tuning across various fine-tuning tasks, compared to existing standard DFA, e.g., achieving 7.97% accuracy improvement on image classification tasks (82.67% vs. 74.70%) and 0.66 higher correlation score on NLP tasks (0.76 vs. 0.10). The code is available on an anonymous GitHub.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies why Direct Feedback Alignment (DFA) struggles to fine-tune networks that were pre-trained with back-propagation (BP), and proposes a simple two-phase remedy called feedback-weight matching (FWM). The authors analyze DFA through Weight Alignment (WA) and Gradient Alignment (GA), arguing that randomly chosen feedback matrices prevent strong WA/GA when starting from BP weights. FWM first reconstructs feedback matrices from the BP weights ("feedback matching"), then re-initializes the weights to match those feedback matrices ("weight matching"), so that fine-tuning with standard DFA naturally induces strong WA/GA. They further prove that, together with weight decay, FWM reduces output error for two-layer settings and conjecture an extension to deeper nets. Experiments show consistent gains across FCNs, BERT (Tiny/Small), and ViT (Tiny/Small) models.

### Strengths
- **Clear problem framing & analysis.** The paper isolates a concrete failure mode of DFA fine-tuning from BP starts via WA/GA and proves why random feedback breaks alignment, then shows how FWM induces strong WA/GA. The link from theory (Defs./Props.) to the empirical WA/GA plots is tight and instructive for the learning-dynamics community.

- **Simple, general mechanism.** FWM is architecturally light: decompose BP weights to build feedback, then re-initialize weights to match feedback; standard DFA then applies unchanged. This “match-then-tune” recipe is easy to implement and does not require modifying optimizers or loss functions, making it appealing for practitioners exploring BP-free training.

- **Broad empirical scope.** Results span FCNs, BERT Tiny/Small, ViT Tiny/Small over CIFAR-10/SVHN/STL-10/ImageNette and GLUE tasks, consistently beating standard DFA fine-tuning and demonstrating stability improvements (WA/GA trends).

### Weaknesses
(1) **Limited transformer alignment coverage.** For transformers, alignment is only applied to dense layers; attention projections (Q/K/V) remain unaligned, which may cap attainable GA/WA. The paper mentions this but lacks systematic analysis of how much each component limits end-to-end gains, especially on larger ViTs or LLMs. Scaling experiments or per-block ablations would clarify.

(2) **Baselines and practical metrics.** The main comparator is standard DFA fine-tuning (and DFA-scratch). Stronger practice-oriented baselines (e.g., BP fine-tuning, BP with adapters/LoRA, partial-BP variants) and wall-clock, memory, and parallelism measurements are missing; these are crucial to justify DFA-based fine-tuning in real systems beyond accuracy alone

### Questions
Could the authors consider adding head-to-head comparisons against BP fine-tuning, LoRA/adapters, and partial-BP under matched budgets (same epochs/tokens, batch size, optimizer, augmentations). Report accuracy, calibration (ECE), robustness (seed/prompt sensitivity if applicable), and costs: wall-clock to target accuracy, tokens/sec or images/sec, peak memory/VRAM, and energy/FLOPs. Include at least one mid-scale vision task (e.g., ImageNet-1k fine-tuning) and NLP task (GLUE).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes feedback-weight matching (FWM) to enable reliable fine-tuning with Direct Feedback Alignment (DFA). The method reconstructs feedback matrices from pre-trained weights and then re-initializes weights to match them, yielding stronger weight/gradient alignment and improved performance; the authors also analyze how weight decay synergizes with FWM. Empirically, FWM outperforms standard DFA across FC nets, BERT-Tiny/Small, and ViT-Tiny/Small (e.g., +7.97% accuracy on SVHN and +0.66 Pearson on STS-B).

### Strengths
1. Simple, principled mechanism with theory: FWM is conceptually clean (feedback reconstruction + weight matching) and is supported by propositions showing it induces strong WA→GA; the paper further proves a GA advantage in a two-layer setting.

2. Consistent empirical gains across modalities/models: Improvements appear on FC image classifiers, GLUE with BERT-Tiny/Small, and ViT variants, often turning DFA fine-tuning from unreliable to usable (e.g., STS-B 0.76 vs 0.10).

3. Thorough ablations and interaction with weight decay: Ablations isolate feedback matching, weight matching, and weight decay, showing their contributions and a clear synergy between FWM and weight decay.

### Weaknesses
1. Scope and scale: Experiments focus on FC nets and compact Transformers/ViTs; CNNs are acknowledged as challenging for DFA, and there are no results on larger modern LMs, limiting claims of broad applicability.

2. Efficiency evidence is missing: While DFA’s appeal is “backprop-free” and parallelizable, the paper reports no wall-clock time, memory, or throughput comparisons vs BP-based fine-tuning, so the practical benefit remains speculative.

3. Theory primarily for simplified settings: Key guarantees (e.g., GA improvement) are proved for two-layer/linear cases, and extension to deep non-linear networks is left as a conjecture, weakening theoretical support for the full experimental regime.

### Questions
refer to weaknesses

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
2

### Summary
In this work, the authors analyzed the limitations of the current Direct Feedback Alignment (DFA) methods for fine-tuning pretrained fully connected neural networks via weight alignment (WA) and gradient alignment (GA). They further demonstrated that the proposed feedback-weight matching method, when combined with weight decay, can not only effectively mitigate over-fitting but also further reduce the network output error.

### Strengths
Originality: It has mitigated the limitation that DFA could only train networks from scratch, by originally proposing the feedback-weight matching method to enhance the standard DFA,,thereby enabling fine-tuning using this method.
Significance:Currently, fine-tuning still plays a role as a critical training stage for maintaining strong performance in various large models. Introducing DFA ,a method enabling efficient parallel parameter updates—into the fine-tuning process to replace back-propagation holds significance.

### Weaknesses
Quality:while Proposition 4.2 and Conjecture 4.3 attempt to link FWM with weight decay for error reduction, the proof relies on strong assumptions (e.g., two-layer networks, specific activation properties). The conjecture for L-layer networks remains ​​unproven​​, and the theoretical analysis does not fully account for the interplay between feedback matching and weight decay in deep networks. This lack of rigor weakens the theoretical contribution.

### Questions
Q1:Since the proposed feedback-weight matching method is intended to replace backpropagation,have the authors ever considered doing experiments to directly compare between the computing time of the same pretrained models but fine-tuned based on the proposed method and back-propagation?
Q2:Why the standard DFA method seems a little better than the proposed method when fine-tuning the 4-layer network from TinyImageNet in image classification tasks?

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
2

### Summary
This paper investigates Direct Feedback Alignment (DFA), an alternative to back-propagation, for the task of network fine-tuning. While DFA has shown promise for training from scratch, it is known to perform unreliably in fine-tuning scenarios. The authors provide a theoretical and empirical analysis arguing that this failure stems from a fundamental mismatch between the pre-trained weights (learned via BP) and the random feedback matrices used by standard DFA, which prevents the network from achieving 'Weight Alignment' (WA). To address this, the authors propose 'Feedback-Weight Matching,' a method that first reconstructs feedback matrices from the pre-trained weights and then re-initializes the weights to align with these new matrices before applying DFA. The proposed method is evaluated on fully connected networks and Transformer models (BERT-Tiny, ViT-Tiny/Small), showing significant improvements over standard DFA fine-tuning.

### Strengths
1. The problem studied in this paper is interesting and potentially important for developing more efficient, BP-free learning paradigms.

2. The proposed method is well-supported by a theoretical foundation.

3. The experimental results are consistent and clear, which looks reasonable to me. 

4. The presentation is generally good. The paper is easy to read.

### Weaknesses
1. The authors claim that DFA is 'back-propagation free,' which implies a computational efficiency advantage. However, the paper lacks any empirical comparison of running time or computational overhead (e.g., FLOPs, wall-clock time) against standard back-propagation fine-tuning. This leaves the practical benefits of the method (beyond being a BP-free algorithm) unsubstantiated.

2. The paper's scope may be somewhat narrow. The proposed method is only compared against existing DFA and standard back-propagation. Assessing its impact in the broader field of BP-free algorithms with comparisons to other methods can be a plus.

### Questions
Please see my comments above.

### Soundness
3

### Presentation
3

### Contribution
3
