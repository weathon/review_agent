# Enhancing Learning with Noisy Labels via Rockafellian Relaxation

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 2, 6, 8

## Abstract
Labeling errors in datasets are common, arising in a variety of contexts, such as human labeling and weak labeling. Although neural networks (NNs) can tolerate modest amounts of these errors, their performance degrades substantially once the label error rate exceeds a certain threshold. We propose the Rockafellian Relaxation Method (RRM) -- an architecture-independent, loss reweighting approach to enhance the capacity of neural network methods to accommodate noisy labeled data. More precisely, it functions as a wrapper, modifying any methodology's training loss - particularly, the supervised component. Experiments indicate RRM can provide an increase to accuracy across classification tasks in computer vision and natural language processing (sentiment analysis). This observed potential for increase holds irrespective of dataset size, noise generation (synthetic/human), data domain, and adversarial perturbation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the Rockafellian relaxation method to enhance the capacity of neural networks to tolerate noisy labels.

### Strengths
- The authors make an effort to propose a general framework for improving robustness in learning under noisy labelling conditions.

- The paper attempts to provide both theoretical analysis and empirical results, showing the authors’ intention to bridge formulation and practice.

### Weaknesses
This submission contains a significant formatting violation that should be desk-rejected. Specifically, the appendix section appears before the bibliography, causing the main text to exceed the 9-page limit.

Regarding the work itself, the current version is very difficult to follow, and the overall presentation lacks clarity. The writing requires substantial improvement to enhance readability and logical flow.

In the next revision, the authors should at least provide a clear introduction to the concept of Rockafellian relaxation and explain the motivation for applying this technique to address label noise. The paper should also avoid overcomplicating simple concepts. For example, Equations (1) and (2) are described in an overly complex manner.

### Questions
NA

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes the Rockafellian Relaxation Method (RRM), a simple, architecture-agnostic loss reweighting scheme for learning with noisy labels. RRM formulates training as a min-min problem over model parameters and example weights with a TV-style penalty, yielding a linear program with a closed-form solution that prunes high-loss samples beyond a threshold cmin + γ and redistributes their mass to lowest-loss samples. The method can be auto-tuned via an estimated contamination rate C′ and can be used with or without adversarial training (A-RRM). Experiments on CIFAR-N, Clothing1M, Food-101N, CIFAR-10, MNIST, IMDb, Toxic Comments, and histopathology show improvements over baselines and, in some cases, state-of-the-art results.

### Strengths
1. Simple, general “wrapper” that can enhance diverse methods and losses without architectural changes.
2. Clear optimization view with an explicit, efficient solution; connection to optimistic DRO/Wasserstein formulations is insightful.
3. Auto-tuning strategy to control pruning via an estimate of noise level.
4. Broad experimental sweep (vision, NLP, medical imaging) with competitive or improved results; SOTA on CIFAR-100N when wrapping strong baselines.
5. Useful analysis of u’s convergence showing selective downweighting of noisy points; practical relevance when test-time perturbations differ from training.

### Weaknesses
1. Novelty is moderate: the resulting rule effectively implements trimmed/quantile ERM focused on smallest-loss samples; closely related to self-paced/trimmed risk/reweighting/DRO literature (e.g., self-paced learning, MentorNet, trimmed loss), but comparisons to these specific baselines are missing.
2. Dependence on contamination estimate C′: despite some ablations, guidance for estimating C′ without clean data is thin; sensitivity analysis is limited.
3. Experimental fairness and reporting: some baselines lack confidence intervals; training schedules and adversarial settings are not always matched; improvements are sometimes small or inconsistent (e.g., MAE).
4. The method of weighting samples is widely used in the field of noisy label learning, such as [1][2]. However, the authors did not conduct a comparison. Furthermore, the baseline used by the author is quite outdated. It would be better to incorporate some more recent related works and baselines, such as those from 2024 and 2025.
5. The author should include a comparison of the time spent. I think this is very valuable for reference.

[1] Knockoffs-SPR: Clean Sample Selection in  Learning with Noisy Labels.

[2] Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting.

### Questions
Please see Weaknesses.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose Rockafellian Relaxation (RRM), a method for learning with noisy labels that jointly optimizes model parameters $\theta$ and per-sample weights $u$ under a total-variation style constraint. For fixed $\theta$, RRM solves a linear program to obtain weights $(1/N + u_i)$ on each training example $i$, minimizing $\sum(1/N + u_i) \cdot J(\theta; x_i, y_i)$ $+ (\gamma / 2) \lvert u \rvert_{1}$ subject to $1/N + u_i \geq 0$ and $\sum u_i = 0$. The optimal $u$ places nearly zero weight on high-loss samples whose supervised loss exceeds $c_{min} + \gamma$ and reallocates their probability mass to the lowest-loss set. Training alternates between: (i) taking $\sigma$ steps of SGD on $\theta$ using these per-sample weights, and (ii) re-solving the linear program using updated losses to recompute $u$. The authors present the method as a wrapper for existing noisy-label pipelines such as DivideMix, ProMix, and CC. They report accuracy gains on CIFAR-10N/100N, Clothing1M, Food-101N, and two NLP datasets.

### Strengths
1. The optimization view of noisy-label learning is compelling. RRM reframes sample selection as an explicit joint optimization over $\theta$ and a perturbed empirical distribution, rather than as a heuristic small-loss filter. The inner reweighting problem is convex in $u$ for fixed $\theta$ and exhibits a closed-form thresholding structure: all samples above $c_{min} + \gamma$ are effectively suppressed, and their mass is reassigned to the minimum-loss set.
2. The method seems lightweight and practical. The algorithm alternates two inexpensive steps ($\sigma$ steps of weighted SGD, then one reweight solve), and the contamination prior $C'$ optionally sets $\gamma$ through a loss quantile so that pruning aggressiveness is tied directly to an estimated corruption rate.
3. The method functions as a wrapper around already strong noisy-label learners that combine supervised and semi-supervised components. The authors convincingly demonstrate consistent improvements on realistic label noise benchmarks (CIFAR-N, Clothing1M, Food-101N).

### Weaknesses
1. The integration of RRM with semi-supervised pipelines is somewhat underspecified.
    * The authors model existing noisy label methods as having a supervised term $J(\theta; x_i, y_i)$ and an additional term $r(\theta)$, then state that they replace the training loss with $L_{RRM}$ (i.e., $\theta$ updates weight each sample's gradient by $(1/N + u_i)$).
    * What is not made explicit is how $r(\theta)$ participates in those $\theta$ updates once RRM is active. Many noisy label methods (e.g., DivideMix, ProMix, CC) employ two-branch training: one branch treats some samples as labeled and optimizes a supervised loss on them, while the other uses the remaining samples in a semi-supervised objective (consistency, pseudo-labeling, auxiliary regularizers).
    * In RRM, samples whose current supervised loss $J(\theta; x_i, y_i)$ exceeds $c_{min} + \gamma$ can be assigned weight $(1/N + u_i) \approx 0$, which zeroes their gradient contribution in the weighted SGD phase. The critical question is whether these samples still contribute through the semi-supervised branch $r(\theta)$ as unlabeled/consistency signal, or whether they are effectively removed from training altogether.
    * I point this out because methods like DivideMix are designed so that even suspected-noisy samples influence training through their semi-supervised branch rather than being discarded. If RRM can fully suppress those samples in the supervised branch when RRM and the inner wrapped method disagree on the noisy/clean partition, the optimization may not be maximally using all of the available data.
2. The authors do not fully characterize the trajectory of sample weights over time.
    * The authors report that the set of pruned samples converges and stabilizes during training.
    * However, to my knowledge, the paper does not show trajectories of individual samples. Interesting behavior might arise: for example, some samples initially downweighted to $(1/N + u_i) \approx 0$ may or may not later be restored to nonzero weight. Such behavior would show whether the method can recover a borderline but ultimately learnable sample that was assigned to near-zero weight early.

### Questions
1. When a sample is assigned $(1/N + u_i) \approx 0$ because its supervised loss exceeds $c_{min} + \gamma$, do you continue to use that sample in the semi-supervised component $r(\theta)$ (for example, as unlabeled data in a consistency or pseudo-label objective), or is it excluded from training until it regains positive weight?
2. During $\theta$-updates in your wrapped DivideMix/ProMix/CC runs, are gradients from $r(\theta)$ also scaled by $(1/N + u_i)$, or are they left unweighted?
3. More broadly, what representational insight does RRM offer that is of direct interest to the ICLR community? How does this work advance our understanding of learned representations, beyond improving robustness to noisy labels?

I would be happy to further increase my score if the authors can answer these questions and revise the manuscript accordingly.

### Soundness
4

### Presentation
3

### Contribution
3
