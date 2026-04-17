# Rethinking Preference Alignment for Diffusion Models with Classifier-Free Guidance

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Aligning large-scale text-to-image diffusion models with nuanced human preferences remains a significant challenge. Direct preference optimization (DPO), while efficient and effective, often suffers from generalization gap in large-scale finetuning. We take inspiration from test-time guidance techniques and view preference alignment as a variant of classifier-free guidance (CFG), where a finetuned preference model serves as an external control signal. This perspective yields a simple and effective method that improves alignment with human preferences. To further improve generalization, we decouple preference learning into two modules trained on positive and negative samples, whose combination at inference can yield a more effective alignment signal. We quantitatively and qualitatively validate our approach on Stable Diffusion 1.5 and Stable Diffusion XL using standard image preference datasets such as Pick-a-Pic v2 and HPDv3.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel method that formulate preference alignment problem as classifier free guidance. Specifically, it proposes to use a preference-tuned policy model to provide guidance during inference. The author proposes two variants. The first variants PGD use the difference between the policy model prediction and a reference base model prediction as guidance signal. The second variant cPGD train two models separately on positive and negative samples of a preference dataset, and use the difference to provide guidance signal. Results show improvements across multiple metrics such as PickScore and CLIP metrics, using multiple base models including SDv1.5 and SDXL.

### Strengths
The method is straightforward and well-motivated. The presentation is clear. The experiments are comprehensive in terms of the metric covered, base model used, and baselines compared. The author also provided several ablation analyses on various design choices such as weighting.

### Weaknesses
There are three main concerns in terms of contribution and technical correctness.

First, the author assumes there are disjoint sets of positive samples and negative samples from a preference dataset, which is not true. For example, in Pick-a-Pick dataset, there are multiple images generated per prompt (say A,B,C,D), and there are preferences A > B and B > C, In such case, the positive and negative samples overlap with each other. This is fine for standard DPO method because its loss contrast **each preference pair** during the training. However, in cPGD, samples like B will appear in both positive and negative training set for cPGD. The author should provide detail discussions on how are these samples handled.

Second, some details of baseline evaluation are missing. For example, I find Figure 10 unconvincing since it reports that SDXL-DPO, which is trained on pick-a-pick, has a win rate <30% against base SDXL measured by PickScore, this greatly differs from the 89.4 win rate in Table 1. The author should provide more details on the distillation experiments. Since incorporating the proposed method will double the compute overhead, it is critical that the author provide careful analysis on the feasibility of the distillation approach.

### Questions
1. Why the "inference time is doubled"(L473)? In standard inference, we also need to predict cond_logits and uncond_logits as well. The only difference here is that we are running two models instead of one model. However, the total FLOPs should be the same? Is it possible to parallelize this inference process, provide both models can fit into GPU memory at the same time?

2. In cPGD, the author trained two separate models $\theta_{+}$, $\theta_{-}$ to provide guidance to the base policy $\pi_{ref}$, does this mean we actually need to run 3 models and the cost is tripled?

3. The cPGD approach is highly relevant to conditional SFT (cSFT) discussed in Diffusion-KTO literature. It seems the only difference is that in cSFT, a special conditional token "good" "bad" is added to the model during the training so that we can learn the distribution of both positive samples and negative samples in the same model, as opposed to use two models in cPGD. In general, it is also quite common to included negative images in the SFT training, with special tags like "deformed, low quality, etc". At inference, these tags are used as negative prompts in guidance. It seems that cPGD differs from these approaches only in that they have two separate models? The author should provide more thorough discussion on these well-established techniques and clarify what is the novel contribution of cPGD.

Overall, I find PGD a meaning contribution. The major concern is 1) the novelty of cPGD 2) the compute cost and the feasibility of distillation. If authors can address these points, I'm willing to increase my score.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose a simple approach for aligning diffusion models by training two separate models on positive and negative samples, respectively. During inference, these models are combined in a manner analogous to classifier-free guidance: at each denoising step, the method adds the positive model’s noise prediction, subtracts the negative one, and adds a reference model. The idea is clear, practical, and easy to implement.

### Strengths
1. The method is conceptually clear and intuitive — explicitly separating the contributions of positive and negative distributions provides a simple yet effective perspective on alignment.

2. The approach is easy to implement and has immediate practical value for production-level diffusion models.

3. The reported results are promising and demonstrate the potential of this straightforward modification.

### Weaknesses
1. Relying on two separate models is a major limitation. It is easy to imagine adding an additional conditioning to diffusion model and fine-tuning a single model instead, which would avoid the extra inference and memory overhead introduced by maintaining two networks.

2. Building on the previous point, the paper feels somewhat incomplete. It lacks deeper analysis or discussion of why the proposed method works. For example, does DPO actually fail to reduce the winner’s loss or increase the loser’s loss as intended? Is simple SFT on positive and negative samples truly optimal, or could training a standard DPO and a reverse DPO model for the positive and negative branches yield better separation and understanding?

### Questions
1. Could you provide an example of a single model trained in this manner, rather than maintaining two separate networks?

2. Could you elaborate on why the standard DPO framework underperforms compared to your proposed approach?

3. There may be more effective ways to model positive and negative distributions. For instance, have you considered training a standard DPO model for the positive samples and a reverse DPO model for the negatives (as an illustrative example)? If so, could you share any preliminary results or insights from such experiments?

I understand that these questions require additional experiments and are difficult to address under tight rebuttal deadlines. Nevertheless, I believe that exploring these points would significantly strengthen the paper and enhance its contribution to the community, particularly given its strong practical relevance. The authors have a clear opportunity to develop this idea into a true plug-and-play alternative to DPO that could gain broad adoption, but in its current form, the paper feels more like a strong hint of what it could become.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Preference-Guided Diffusion (PGD) and Contrastive PGD (cPGD), which reformulate human preference alignment for text-to-image diffusion models as an inference-time guidance problem inspired by Classifier-Free Guidance (CFG). PGD combines the gradients of a base model and a DPO-finetuned model to achieve better preference alignment without additional training, while cPGD trains separate positive and negative branches and fuses them at inference for improved generalization. Experiments on Stable Diffusion 1.5 and SDXL show consistent Pareto improvements over Diffusion-DPO in reward, FID, and diversity, demonstrating a practical plug-and-play alignment framework.

### Strengths
1. This paper proposes two inference-time preference alignment methods inspired by the analogy to Classifier-Free Guidance (CFG), both conceptually simple yet effective.

2. PGD is a practical approach that can reuse existing pretrained and preference-finetuned weights.

3. The experimental results are extensive — across two prompt sets and multiple evaluation metrics, the effectiveness of the proposed methods is consistently demonstrated.

### Weaknesses
1. The proposed PGD formulation (Eq. 9) is merely defined by analogy to CFG, and the equation itself lacks theoretical justification.

2. The reason why overfitting is mitigated in cPGD is not experimentally verified (although a theoretical explanation is mentioned in Section 4.2).

3. Manual tuning of the guidance weight is required — it must be adjusted for each evaluation metric or dataset.

### Questions
1. Could you provide experimental evidence showing that cPGD indeed mitigates overfitting?
    For example, performance comparisons under different training data sizes could help demonstrate this effect.

2. Could you plot the influence of the guidance weight, as in Figure 6, for the other evaluation metrics as well?
    It would be helpful to see how consistent the curve shapes are across metrics.

3. The connection to NTK is a bit interesting. could you explain why PGD outperforms DPO under the case of slightly fine-tuned models?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper takes inspiration from CFG and proposes two methods, PGD and cPGD, for preference alignment. PGD treats the DPO-tuned model as the conditional model and the reference model as the unconditional model in CFG for inference. cPGD independently trains two models from positive and negative images separately and uses both models altogether in a CFG manner for inference. Experiments show that both methods improve preference-alignment performance compared with multiple baselines.

### Strengths
1. The idea of reinterpreting preference alignment in diffusion models through CFG is interesting. While CFG is typically used for conditioning on class labels or prompts, applying it to preference signals is both novel and practically useful. This reframing also enables plug-and-play alignment.
2. Presentation is clear, well-structured, and easy to follow.  The paper does a good job of walking the reader through the standard diffusion and DPO setups before introducing PGD and cPGD. Figures are informative and well-integrated with the text.
3. Extensive experiments showing that the proposed methods are effective and robust.

### Weaknesses
1. Even though the proposed PGD/cPGD are novel reinterpretations of preference alignment, the intuition is not well-explained. The paper lacks a formal verification of these methods. For example, what is the target distribution/posterior distribution that PGD/cPGD is trying to sample from? Can the authors justify that this interpolated guidance signal (between base and finetuned models) actually approximates the posterior over preferred samples?

2. The intuition behind cPGD is also underspecified. For example, why is it preferable to split the model into two, rather than learning a joint model with contrastive loss? Is there evidence that this separation improves the reward gradients or stabilizes the guidance signal?

As a result, although performance improves in practice, the mechanism behind that improvement is not fully explained. The methods appear somewhat ad hoc, and the absence of a more principled justification weakens the theoretical contribution.

### Questions
see Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
