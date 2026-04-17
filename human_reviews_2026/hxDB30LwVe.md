# Celo2: Towards Learned Optimization Free Lunch

- Decision: Accept (Poster)
- Scores: 6, 2, 4, 6

## Abstract
Learned optimizers are powerful alternatives to hand-designed update rules like Adam, yet they have seen limited practical adoption since they often fail to meta-generalize beyond their training distribution and incur high meta-training cost. For instance, prior work, VeLO, scaled meta-training to 4,000 TPU months ($\sim$10$\times$ GPT-3 compute) to meta-train a general-purpose optimizer but it failed to generalize beyond 600M parameters tasks. In this work, we present a surprising finding: by crafting a simple normalized optimizer architecture and augmenting meta-training, it becomes feasible to meta-train a performant general-purpose learned update rule on a tiny fraction of VeLO compute, 4.5 GPU hours to be precise. Our learned update rule scales stably to a billion-scale pretraining task (GPT-3 XL 1.3B) which is six orders of magnitude larger than its meta-training distribution. Furthermore, it shows strong performance across diverse out-of-distribution tasks and is compatible with modern optimization harness that includes orthogonalization, distinct update rules for input-output and hidden weights, and decoupled weight decay. In all, this work paves the way for practically applicable _learnable_ optimization algorithms, unlocking exploration of richer meta-training and data curation recipes to further improve performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a new learned optimizer that improves upon the state-of-the-art VeLO optimizer. It achieves higher performance while requiring less compute and data during training, and it scales effectively to large tasks. The optimizer generates an update direction, after which classical update components (such as orthogonalization, normalization, and step size adjustment) are applied to produce the final parameter update.

### Strengths
- The proposed approach simplifies the training process and scales effectively to larger problems.

- The usage and implementation of the optimizer (in Algorithm 1) are well described.

- The paper presents comprehensive comparisons across multiple tasks and datasets of varying sizes, demonstrating consistent improvements.

### Weaknesses
- The paper depends on manually tuned step sizes and learning rate schedules. For instance, the authors note that they “test with 7 learning rate values sampled logarithmically between 1e-3 and 1e-5, warmup fraction 0.05 with cosine decay schedule, and plot the best performing hyperparameter setting for each optimizer.” However, only the best-performing configuration is reported, without clarifying the selection criteria. This limits the understanding of tuning sensitivity and comparability—for example how easily the method can be tuned relative to AdamW.

- The training procedure is described briefly and relies on citations rather than detailed explanation. For instance, “We use Persistent Evolutionary Strategies (PES) (Vicol et al., 2021) to meta-train our optimizer with unroll length logarithmically sampled between 100 and 2000 steps.” Re-explaining the main aspects of PES and its role in this work would make the paper more accessible and reproducible.

- It is unclear how specific training hyperparameters are chosen and applied:
  - Which step size is used during meta-training for each unrolled iteration?
  - Is the cosine decay schedule also used during training?
  - Are orthogonalization and normalization applied at training time, or only at evaluation?

These details are essential for reproducibility and for understanding how much the method’s performance depends on them.

### Questions
- How sensitive is the proposed optimizer to the choice of step size and learning rate schedule compared to standard optimizers like AdamW? Could you provide results or plots showing performance across the range of tested hyperparameters?

- During meta-training, which step size or learning rate schedule is actually used? Is it fixed, or does it follow the same cosine decay schedule described for evaluation?

- Are the orthogonalization and normalization steps used during the optimizer’s training phase, or only at test time?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new learnable optimizer Celo2. It is intended to outperforms Celo and other alternatives in terms of generalization, scalability and compute efficiency. This work is built on top of prior Celo optimizer  (Moudgil et al., 2025). This paper introduces the following changes:
a. It retains a user-tunable step size, not touching the learned scheduler. This helps the scalability.
b. It learns a small MLP to serve as a drop-in replacement for the standard Adam update rule. This makes the proposed Celo more computing efficient.
c. The proposed algorithm normalizes the MLP's output, which improves the generalization.

Celo2 shows strong results on ImageNet image classification with ViT. In exp with GPT-2 and GPT-3, it also achieves consistent lower validation loss. On Atari benchmark, it is shown effective in a PPO algorithm to learn an RL policy.

### Strengths
The proposed Celo2 does seem to simpler, scalable and generalizable based on the experiments.

There are a wide variety of experiments to back up the claims, e.g. vision, language, and RL.

### Weaknesses
The proposed method does not provide sound theoretical guarantee, making it feel ad-hoc. This raises significant concerns about its practical application. In particularly, I would be very concerning on how would a practitioner use the proposed Celo2 in real-world application. For example, it is not clear how to replace the Adam with small MLP? It is unclear if this technique was found to work only on specific, "cherry-picked" problems and whether the MLP architecture must be significantly re-tuned when switching to a new problem domain

The experimental results feel incomplete. While reporting validation loss is informative as a preliminary measure, it is not sufficiently convincing on its own. The paper would be much stronger if it included results from more established, concrete benchmarks (e.g., perplexity or BLEU scores for LLMs, not just training/validation loss).

### Questions
1. Could the authors clarify how Celo2 is expected to generalize to even larger models? Large models (LLMs/VLMs) have a massive number of parameters, and their optimizer states (like Adam's moments) consume significant GPU memory. How does the proposed MLP-based approach compare in terms of GPU memory efficiency?

2. What is the architecture of the small MLP used for ImageNet classification versus the one used for the LLM experiments? It is not clear how much this optimizer component needs to be adapted when moving between different problem domains.

3. The paper claims the RMS-normalized learned update leads to better generalization, but the mechanism is unclear. Is there a more detailed ablation study or theoretical justification to support this specific design choice?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a provocative and potentially groundbreaking finding in learned optimization (LO). It directly challenges the prevailing "scaling hypothesis" (e.g., VeLO), which posits that massive meta-training compute is necessary for generalization. The authors introduce Celo2, a learned optimizer meta-trained on a "toy" distribution (8x8 image classification) for a mere 4.5 GPU hours.

The central, surprising claim is that this "cheap" LO, built on a simple normalized MLP architecture and task augmentation, generalizes six orders of magnitude beyond its training data. It successfully and stably trains billion-scale models (GPT-3 1.3B) and vision transformers (ViT), outperforming not only the strong AdamW baseline but also the exorbitantly expensive VeLO, which famously fails on such large-scale tasks.

### Strengths
1. The paper's core finding—that a 4.5 GPU-hour "toy" meta-training can produce an LO that scales to 1.3B parameter models—is a "free lunch" that could fundamentally realign research in this field. It moves LOs from a "computationally impossible" (VeLO) to a "highly practical" domain.

2. The paper rightly centers its comparison against VeLO. The results are stark: VeLO, meta-trained with 4000 TPU-months, is unstable and fails on large tasks, while Celo2, trained for 4.5 hours, is stable and superior. This is the paper's strongest point. It demonstrates robust generalization across the three most critical axes:

Model Scale: From tiny 8x8 MLPs to 1.3B GPT-3 models.

Unroll Length: From short 2k-step unrolls to 10B+ tokens (GPT-3) and 50k steps (ViT).

Task Domain: From 2D Image Classification to 1D Language Modeling and Reinforcement Learning (Atari).

3. aThe proposed LO ("Celo2-base") is a small, 8-hidden-unit MLP (Table 1a). The design is explicitly intended as a 1-line "drop-in replacement" for Adam (Sec 4.1), which is a massive win for practical adoption.

### Weaknesses
The paper makes extraordinary claims (4.5 GPU-hour meta-training, 6-orders-of-magnitude generalization) with a simple recipe. Such "too good to be true" results demand exceptional evidence. However, the paper is missing an appendix and supplementary material, providing no code, implementation details, or full hyperparameter lists beyond what is in the main text. This makes it impossible to verify the claims or assess reproducibility, which is paramount for such a shocking result.

### Questions
1. The decoupling of the learning rate (Sec 3) is a major simplification. How much of Celo2's superior performance and stability can be attributed to relying on a hand-tuned cosine schedule, which VeLO was explicitly designed to avoid?

2. Figure 4 shows that Celo2's best performance comes from hybridization (adding Orthogonalization and AdamW@1D). Does this suggest that the future of LOs is not "pure" end-to-end learning, but rather learning "plugins" that work with hand-designed rules?

3. Could you please expand on the choice of "average loss" vs. "final loss" as the meta-objective? Does Celo2's stability come from optimizing this easier, myopic objective? What happens if Celo2 is meta-trained to optimize the final loss, as VeLO does?

### Soundness
4

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
4

### Summary
This paper addresses a major limitation of existing learned optimizers (LOs), which is that they do not meta-generalize well out of distribution, particularly when the optimizee is large or the optimization is long. To mitigate this issue, the authors propose Celo2 with a simple yet effective LO architecture that adopts orthogonalization and RMS normalization, which has been shown to be successful in hand-crafted optimizers. The experiments show that Celo2 meta-trained on relatively small datasets and optimizees in a supervised learning paradigm outperforms strong baselines in three out-of-distribution settings: larger optimizee, longer optimization horizon, and application to reinforcement learning. Detailed ablations are provided to analyze each component of Celo2.

### Strengths
The method proposed is simple yet effective, indicating a great potential for generalization and further improvement.

The experimental results provide strong support for the claims.

The contribution of this paper is significant towards the practical application of LOs.

### Weaknesses
I suggest avoiding the term "free lunch" in the paper, as the no-free-lunch theorem indicates that the inductive bias found by Celo2 will likely downgrade performance on some tasks, although these tasks may be unlikely in realistic applications.

The paper states, "We would like to emphasize that our primary objective while developing this simple approach for learned optimization was stability." However, no sensitivity analysis is provided.

Algorithm 1 is not referred to in the text.

### Questions
The ablations are conducted from a generalization-first perspective in what sense? How does the ablation analysis focus on generalization?

What is the purpose of using both TPUs and GPUs in the experiments?

What is the optimizer used for meta-training?

How many steps of optimization were used for the results shown in Fig. 1?

Why do Celo2-base and Adam have identical wall clock time? Does the MLP application on line 5 of algorithm 1 take significantly higher computation?

What are the batch sizes used in the experiments?

### Soundness
3

### Presentation
2

### Contribution
4
