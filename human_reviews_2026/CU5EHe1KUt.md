# Diffusion Negative Preference Optimization Made Simple

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4

## Abstract
Classifier-Free Guidance (CFG) improves diffusion sampling by encouraging conditional generations while discouraging unconditional ones. Existing preference alignment methods, however, focus only on positive preference pairs, limiting their ability to actively suppress undesirable outputs. Diffusion Negative Preference Optimization (Diff-NPO) approaches this limitation by introducing a separate negative model trained with inverted labels, allowing it to capture signals for suppressing undesirable generations. However, this design comes with two key drawbacks. First, maintaining two distinct models throughout training and inference substantially increases computational cost, making the approach less practical. Second, at inference time, Diff-NPO relies on weight merging between the positive and negative models, a process that dilutes the learned negative alignment and undermines its effectiveness. To overcome these issues, we introduce Diff-SNPO, a single-network framework that jointly learns from both positive and negative preferences. Our method employs a bounded preference objective to prevent winner-likelihood collapse, ensuring stable optimization. Diff-SNPO delivers strong alignment performance with significantly lower computational overhead, showing that explicit negative preference modeling can be simple, stable, and efficient within a unified diffusion framework. Code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper propose a new method called Diff-SNPO for improving preference alignment in diffusion models. It builds on Diff-NPO which uses two separate models for positive and negative preferences but that has high cost and need weight merging at inference. Instead, authors integrate both into single network using the conditional and unconditional branches from CFG. They identify problem with naive approach leading to blurry images due to likelihood collapse in DPO, and fix it by adapting Bounded DPO to diffusion setting. Experiments on SD1.5 and SDXL with Pick-a-Pic v2 show better performance and efficiency compare to baselines like Diff-NPO and CHATS.

### Strengths
- The idea of unifying positive and negative preference in one model is clever and adress the practical issues of dual-model approaches, like double compute and merging trade-offs. This make it more scalable for large models. 

- They provide good analysis of why naive single-model fail (blurring from conflicting gradients and DPO's winner likelihood decrease), and the bounded objective seem to solve it effectively based on figures.

- Experiments are solid, evaluating on multiple reward models (PickScore, HPSv2 etc.) and showing gains in alignment metrics while reduce overhead. Code will be released which is plus.

- The derivation of Diff-BDPO upper bound is detailled in appendix, and overall contribute to stable preference optimization in diffusion.

### Weaknesses
- The paper claim strong performance but evaluation is only on Pick-a-Pic v2 benchmark, which is aesthetics-focused. Would be better to test on other datasets like safety or bias alignment to show generality, especialy since abstract mention safety.

- Hyperparameters like $\lambda=0.9$ and $\beta=2000/5000$ are tuned but not much ablation on them. How sensitive is method to these? Also, training time is given (12h for SD1.5) but no direct comparison of GPU hours vs baselines.

- While they discuss blurring in naive approach, the qualitative examples in Fig3 are limited to one prompt. More diverse visuals would strengthen the claim.

### Questions
- In Eq18, how do you sample Y during training? Is p fixed or annealed?

- Does Diff-SNPO work with other samplers beyond DDIM or only tested with that?

- Have you try applying this to non-image domains, like video diffusion?

### Soundness
2

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
3

### Summary
The paper proposes Diffusion Simple Negative Preference Optimization (Diff-SNPO), which trains a single diffusion model on a preference dataset using an upper bound of the Bounded Direct Preference Optimization (BDPO) objective and a modified classifier-free guidance (CFG) dropout training setup where the unconditional branch is the other example in the preference pair. The paper shows that training a single model with the Diff-DPO objective causes generated images to get blurrier over training (and positive example probabilities to go down), and that switching to their derived upper bound of the BDPO objective mitigates this failure.

The paper compares Diff-SNPO with 1. Diffusion DPO (Diff-DPO), which trains a single model on an upper bound of the DPO objective; 2. Diffusion Negative Preference Optimization (Diff-NPO), which uses the Diff-DPO objective to train two separate models, one with pairs flipped, followed by CFG during sampling using a merged weights model instead of the trained negative model directly; and 3. CHATS, which uses the Diff-DPO objective to train two separate models simultaneously, one for each reward term in the reward margin of the objective, followed by a variant of CFG using the two models and a proxy prompt for the negative model. Diff-SNPO is more efficient than #2 and #3 due to its use of a single model, and generally improves over Diff-DPO in quality.

### Strengths
1. The derivation of a bound of the BDPO loss, as well as the CFG-like training setup for training preference pairs on a single model, seem like clear contributions and are to my knowledge novel.
2. The proposed method has a clear memory/computational wins over the two-model methods, and the move from DPO to BDPO when training a single model is also nicely showcased.
3. The additional row showing Diff-BDPO (which if I understand correctly is meant to offer a direct comparison to Diff-DPO) is helpful as an ablation.

### Weaknesses
1. The quantitative results do not show report error bars or information about statistical significance. Also, there is not a clear winner across the quality metrics.
2. The paper could benefit from some revisions of the writing (e.g., in the abstract, introduction, related work, figure 1) to better explain what currently exists and distinguish its contributions. For instance, lines 78-79 make claims about Diff-NPO that I first thought were just missing references to past works but later realized are actually new findings from the paper. Moreover, the abstract and figure 1 focus on Diff-NPO when it's not clear why that prior work should be privileged over the other methods for comparison. (The paper does compare to others, but it might be easier to, for instance, have a table comparing this method to various others rather than focus on Diff-NPO in multiple places.)
3. The implicit classification results don't seem very relevant for any actual notion of performance that matters for these models. For instance, [Rafailov et al 2024](https://arxiv.org/abs/2406.02900) (as well as others) show the lack of correspondence between better rewards and better win rate. (This can be explained by that the reward looks at offline data that need mot characterize the model well.)

### Questions
1. Are the quantitative quality results significant? Could the authors include some form of error bars, significance tests, etc.?
2. Is there a reason why Diff-NPO is the main method being compared against in the story of the paper?
3. Why should we care about implicit accuracy at all (e.g., as a downside of weight merging)?

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
3

### Summary
This paper proposes Diffusion Simple Negative Preference Optimization (Diff-SNPO), a single-network alternative to dual-model negative preference methods (e.g., Diff-NPO). The key idea is to train the conditional branch on preferred samples and the unconditional/null branch on inverted (negative) preferences within the same diffusion backbone (leveraging CFG’s two branches), thereby avoiding the cost and alignment dilution of maintaining and merging two models. The authors diagnose that a naive single-model version coupled with Diff-DPO blurs images by decreasing winner likelihood; they then adapt Bounded DPO (BDPO) to diffusion and derive an upper-bound (Diff-BDPO-UB) objective that mixes the loser term with a reference policy to prevent the winner likelihood collapse. The final objective (Eq. 18) instantiates SNPO with the BDPO upper bound and a mixing parameter λ. Experiments on Pick-a-Pic v2 and HPDv2 with SD1.5/SDXL show improved preference metrics (HPSv2, PickScore, ImageReward), higher negative implicit accuracy, and 2× training/inference speedups versus dual-model baselines.

### Strengths
1. I think the approach that uses the native conditional/unconditional branches to encode positive vs. negative preferences is both elegant and practical.
2. Adapting BDPO to diffusion and deriving a tractable upper bound yields a training signal that boosts the winner without exploding loser gradients.
3. Good empirical results.
4. Higher negative implicit accuracy than Diff-NPO post-merge, addressing a known weakness of merging

### Weaknesses
1. The main experiments use large batches and multi-GPU (e.g., 8×A6000, batch 512/2048), whereas Table 3's cost is a single-GPU batch-8 microbenchmark.
2. Training negative signals into the unconditional branch might introduce risks altering the unconditional anchor used by CFG in safety contexts.

### Questions
N/A

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
This paper proposes Diff-SNPO, a unified framework for diffusion models to learn from positive and negative preference examples. Diff-SNPO improves upon Diff-NPO by only using one model, removing the potentially undesirable model merging step and making the procedure more computationally efficient. Experiments show that Diff-SNPO outperforms Diff-NPO and other baselines.

### Strengths
1. The paper has extensive empirical verification, and good results showing that Diff-SNPO outperforms other baselines.
2. The proposed method reduces training and inference cost by 2x, which can be significant when the models are large-scale. 
3. The paper is mostly clearly written.

### Weaknesses
1. Novelty and technical difficulty: Diff-SNPO seems to be a relatively straightforward combination of CFG and BDPO, which are both existing methods. Given that BDPO addresses the problem of preference learning procedures reducing the likelihood of the preferred examples, it is not very surprising that it would be effective here. Can the authors describe the technical challenges of combining these ideas?
2. It would be great if the preliminary can include a dedicated section on CFG.
3. The paper highlights that Diff-SNPO may suffer from conflicting gradients between positive and negative examples because it uses a single model. Since DPO doesn't have this problem (there are no inverted examples), how does BDPO, which only bounds the loss scale for dispreferred examples, specifically mitigate the problem of interfering gradients?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
