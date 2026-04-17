# Stable-SPAM: How to Stably Train Large Language Models in 4-Bit

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
This paper comprehensively evaluates several recently proposed optimizers for 4-bit training, revealing that low-bit precision amplifies sensitivity to learning rates and often causes unstable gradient norms, leading to divergence at higher learning rates. Among these, SPAM, a recent optimizer featuring momentum reset and spike-aware gradient clipping, achieves the best performance across various bit levels, but struggles to stabilize gradient norms, requiring careful learning rate tuning. 
To address these limitations, we propose **Stable-SPAM**, which incorporates enhanced gradient normalization and clipping techniques. In particular, **Stable-SPAM** $(1)$ adaptively updates the clipping threshold for spiked gradients by tracking their historical maxima; $(2)$ normalizes the entire gradient matrix based on its historical $l_2$-norm statistics; and $(3)$ inherits momentum reset from SPAM to periodically reset the first and second moments of Adam, mitigating the accumulation of spiked gradients. Extensive experiments show that **Stable-SPAM** effectively stabilizes gradient norms in 4-bit LLM training, consistently delivering superior performance compared to Adam and SPAM across model sizes from LLaMA-130M to LLaMA-7B. Notably, our 4-bit LLaMA-1B model trained with **Stable-SPAM** outperforms Adam by up to $3.1$ perplexity. Furthermore, when both models are trained in 4-bit, **Stable-SPAM** achieves the same loss as Adam while requiring only about half the training steps. Code is submitted.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper has two main contributions.  First, it analyzes the behaviour of different optimizers when training with lower precision, testing sensitivity to different learning rates, and observing gradient norm spikes and training divergences at lower precision levels.  Second, it introduces the Stable-SPAM optimizer, an enhancement of the SPAM optimizer, but with new techniques to mitigate the gradient norm spikes that were observed with vanilla SPAM training.  Results at different model sizes show lower perplexity compared to other optimizers.

### Strengths
It is well-motivated to improve the efficiency of LLM training, both through lower precision (reduced memory) and through mitigating instabilities.

Studying sensitivity to LRs at different precision levels is interesting, and continues a line of work (e.g., Zhao et al 2024b's work, Wortsman et al's work) into an interesting area.

I found the Abstract and Introduction relatively easy to follow, and I absorbed the key points, and these led into Section 2 nicely, which provided the fuller description.

### Weaknesses
Overall, when we get into the details of the comparison between the models, it seems like the models are not always being trained to compute-efficient token budgets, and gradient clipping is only being applied in some cases (even though I believe it to be standard, if not always mentioned) and I'm confused about how many tokens different results were trained on, and whether different warmups were used in different models, and whether the models were given their best chance (e.g., just using AdamW's default weight decay setting, yet tuning all the Stable-SPAM hyperparameters on the 60M model scale).  What if larger weight decay is all we need for AdamW to shine at lower-bits?

I would have thought that gradient clipping is standard practice with LLM training (let alone low-precision training), so anywhere in the paper a result is reported without “GradClip”, I feel like this is a straw man baseline.  And that’s a lot of the figures/tables in the paper!

I’m confused about the presentation
- The abstract said gains from 130M to 7B, but then the experiments mention testing from 60M, so do the lower models not show gains?  You might mention that specifically.  I think maybe 60M was used for hyperparameter tuning?  We can make this clear when 60M is first mentioned.
- The intro doesn’t mention Lion, but it’s tested later, it’s just weird to not mention that specific one (although technically you mentioned Adam but not AdamW)
- “As shown in Table 1, the perplexity gap between BF16 (Adam) and INT4/FP4 (Adam) exceeds 1.5” – I’m confused, Table 1 doesn’t show BF16, right?
- I put some other confusions into Nitpicks below.

Soundness:
- Note it’s my job as a reviewer to make sure the evaluation is done in practically-relevant scenarios, e.g., at the point where models are trained to compute-optimal levels or higher (20 tokens-per-parameter or greater).  But I’m having a very difficult time tracing how many tokens each model was trained on.  I have further points on this below.
- So only SPAM uses LR warmup of 150 steps?  But then I see warmup for 2000 steps in Appendix B. Can you clarify?
- I am not sure about using the HPs from the original paper for Adafactor, like, that was a long time ago, compared to SPAM, tuned in 2025, and Stable-SPAM, tuned in this very paper.
- Why does Figure 4 only go to 4K steps?  Why does Figure 3 go to 5K steps?  Do these use the same batch size, LR, etc., as the other experiments?
- So we report a batch size in tokens of 512*256.  We say 350M models train for 6.6B tokens, so that would be 50K steps, right?  And 1B should be even more.  But then Figure (1) only shows results until 20K steps.  This is really undertrained, right, compared to the Chinchilla compute-optimal setting of around 20 tokens-per-parameter (TPP).  In Table 1, the 1B model trains for 7.7, but then in Table 2, they train for 11.6B.  Why?
- The 1B models are also really undertrained, only training to 7.7 TPP, which is below the level anyone would actually train an LLM to, since a smaller model could be trained to a lower loss with the same compute by training on more tokens.  So I’m not sure how practically-relevant these results are. I.e., when you say, “Notably, Stable-SPAM performs particularly well with larger models, such as LLaMA-350M and LLaMA-1B, showcasing its strong potential for large-scale training,” could we interpret this as, “Stable-SPAM performs particularly well at compute-inefficient settings that no one would actually train to in practice”?
- I want to be clear about this: as a practitioner, I would only ever train large models to compute-efficient TPP ratios.  If I read your paper, I might think I can train stably at 4-bit if I used Stable-SPAM.  But you do not test compute-efficient TPP ratios with your larger models.  You mention Fishman’s finding that as we train on more tokens, low-bit data formats may struggle, but your results don’t show this --- is it because you don’t evaluate in these regimes, or because your methods are robust to this?  The practitioner needs to know.
- Figure 5, we only have 20K update steps, how many tokens-per-parameter is this?

Nitpicks:
- Be nice if you referred to Figure (1) in the intro.
- Recent studies (Zhao et al., 2024b; Wortsman et al., 2023b; Huang et al., 2025; Takase et al., 2023; Wortsman et al., 2023b) – multiple Wortsman citations here.
- Be nice to report the token counts in Figure (1) as well so we can see if this is a practically-relevant regime
- If we could split Figure (2) left from the other ones it would help a lot, like it seems the legend below doesn’t apply to this one, it’s just confusing, and also, what’s the point since these same lines are in the three other figures?
- Figure (4), which optimizer is this?  I’m looking for spots Figure 4 is referred to in the text, and I get “The final validation loss [of Stable-SPAM] is presented in Figure 4,” but I assume this is a typo?
- Your description of gradient clipping in Section 5: is it really “rescaling” the gradient or just clipping dimensions that go above the cutoff?

### Questions
- In all the figures, does the LR decay to 10% of the peak at the very final step as shown in the figure?  E.g., Figure 1, Figure 4, Figure 3, Figure 5, Figure 6, Figure 7.

- Do all of these training runs use the same batch size?

### Soundness
1

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
4

### Summary
The paper introduces a new optimizer, Stable-SPAM, which builds on the previous SPAM variant by adding gradient normalization and adaptive clipping to stabilize gradient norm spikes during training. However, the reported empirical results are unreliable and questionable, likely due to implementation issues.

The training configurations are only partially disclosed, leaving key details unclear—such as whether FP4 quantization is applied to gradients, the precision of optimizer states and computations, and whether the BF16 baseline follows standard mixed-precision practice or forces all operations into BF16. These ambiguities raise concerns about reproducibility and interpretation.

Moreover, the paper’s title is misleading. As noted in line 265, the experiments employ “quantization-aware training strategies” derived from LLM-FP4 (2023), which explicitly states that "quantizing both weights and activations ..., in a post-training manner" rather than full 4-bit training, yet the tile suggests the method are general for true 4-bit training.

### Strengths
The proposed methods are simple and straightforward, combining adaptive gradient normalization, adaptive spike-aware clipping (with limited novelty), and the momentum reset mechanism from SPAM.

However, given the weaknesses (stated in next part) and questionable experimental reliability, it is difficult to draw firm conclusions. It remains possible that the method genuinely stabilizes training—reducing gradient spikes in a proper BF16 mixed-precision setup with master weights and high-precision optimizer states—but it is equally plausible that the observed stability arises from implicit effects such as an effectively larger learning rate introduced by normalization.

### Weaknesses
**Implementation, baseline and FP4 training setup**

- After examining the codebase, it appears that the authors perform a full BF16 training with
```python
model = model.to(dtype=torch.bfloat16)
optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
loss = model(**batch, labels=labels).loss
```
as a result, optimizer states (and likely also intermediate activations e.g., softmax outputs) are in BF16. Since starting pytorch 1.13+, it uses
```python
state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
```
which will be same precision as the parameters. However, the standard practice is BF16 mixed-precision training (e.g., AMP or Megatron-style) that closely matches FP32 performance. Pure BF16 training with BF16 optimizer states is known to be unstable, especially when $\beta_2$ is close to 1.0, as stated in [Yu et al., 2024]. 

Several observations suggest that the experiments indeed use BF16 optimizer states (for all experiments):

i) In Table 10, BF16 (Adam) fails to match FP32 (Adam).

ii) In Table 5, Stable-SPAM (FP4) even outperforms BF16 Adam.

iii) In Table 2, BF16 Stable-SPAM and BF16 Adam + GradClip outperform plain BF16 Adam by a large margin.

These patterns indicate that BF16 Adam serves as a weak and even bad baseline, and the proposed methods primarily stabilize training under poor BF16 optimizer precision—an issue that practitioners typically avoid using standard mixed precision or techniques with Kahan summation or stochastic rounding.

**FP4-E1M2 vs INT4 equivalence**

- The paper studies FP4-E1M2 and INT4, but FP4-E1M2 is effectively equivalent to INT4 when scaled by 4.0. If the implementation follows LLM-FP4 (2023), the only difference is rounding mode—there the FP4-E1M2 uses floor rounding, whereas INT4 uses round-to-nearest. This equivalence diminishes any claimed novelty of FP4 usage in the paper.

**Why not use AdamW with weight decay?**

- Weight decay is standard in large-scale training, and AdamW is the de-facto optimizer rather than vanilla Adam. The omission raises the question of whether the proposed method is incompatible with weight decay or whether the comparison is incomplete.

**Missing Baseline optimizer configurations**

- Only Stable-SPAM hyperparameters (Tables 7–8) are reported, while the baseline optimizer settings are missing. Are learning rates shared or separately tuned? What values of $\beta_1, \beta_2$ are used? These omissions make the comparisons unclear and hinder reproducibility.

**Generality of “4-bit training”**

- As stated in line 265, the experiments rely on “quantization-aware training strategies” from LLM-FP4 (2023), which explicitly perform post-training quantization of weights and activations rather than full 4-bit training. Yet, the title suggests a general 4-bit training framework. The authors should clarify exactly which components are quantized and, if it is QAT, make this explicit in the title and introduction.

**Convergence and experimental reliability**

- Figures 1, 4, 5, and 6 show incomplete convergence; curves may align with more training. Are all models initialized identically (weights and optimizer states)? Were multiple runs with different seeds performed for smaller models? At least one long-token, large-scale run (chinchilla optimal) should be provided to verify stability and scaling behavior.

Yu et al. "Collage: Light-Weight Low-Precision Strategy for LLM Training." Forty-first International Conference on Machine Learning.

### Questions
- line 300, it refers to Figure 4, and jumps to Table 4, which is actually a figure. Also, for this study, do you have any insights on why W3A3 shows the smallest gap?

- can the authors plot the equivalent lr (treating the normalization as if scaling the un-normalized gradients & updates), across iterations for stable-SPAM and compare with the original lr schedule?

### Soundness
1

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
This paper investigates the stability challenges of training large language models using 4-bit precision and proposes Stable-SPAM as a solution. The authors conduct a comprehensive evaluation of recent optimizers (Adam, Adafactor, Adam-mini, and SPAM) under 4-bit training conditions, revealing that low-bit precision significantly increases sensitivity to learning rates and causes unstable gradient norms that can lead to training divergence. While SPAM demonstrates strong performance across different bit levels, it still suffers from gradient instability requiring careful hyperparameter tuning. To address these limitations, Stable-SPAM incorporates two novel techniques: Adaptive Spike-Aware Clipping (AdaClip), which dynamically adjusts clipping thresholds based on historical gradient maxima, and Adaptive Gradient Norm (AdaGN), which normalizes gradients using historical $\ell_2$-norm statistics. The method also inherits SPAM's momentum reset mechanism. Experiments across model sizes (LLaMA-130M to LLaMA-7B) demonstrate that Stable-SPAM effectively stabilizes 4-bit training, achieving superior performance and training efficiency compared to existing approaches.

### Strengths
- The paper is well-written and clearly structured, making the technical content accessible and easy to follow.
- The comprehensive evaluation across multiple optimizers (Adam, Adafactor, Adam-mini, SPAM) and model scales (130M to 7B parameters) provides nice observations into 4-bit training dynamics.
- The proposed techniques—Adaptive Spike-Aware Clipping and Adaptive Gradient Norm—sounds reasonable.

### Weaknesses
-  Experiments are confined to small-scale settings (1B tokens, perplexity >10) using only LLaMA-2 architecture. This resembles continual pre-training rather than full-scale pre-training scenarios. The practical significance remains unclear, as the paper lacks evidence that addressing loss spikes meaningfully improves downstream task performance—evaluation perplexity alone is insufficient validation.
- The proposed methods are incremental extensions of SPAM rather than fundamentally novel contributions, which diminishes the technical significance.
- The paper provides insufficient analysis of the root causes behind 4-bit training instability. Critical questions remain unanswered: Are loss spikes primarily caused by 4-bit model parameters, activations, or their interaction? Deeper diagnostic insights would strengthen the contribution beyond empirical observations.

### Questions
- What is the actual wall-clock time speedup achieved by 4-bit training compared to BF16/FP16? A comprehensive comparison plotting training time versus performance metrics (evaluation loss and downstream task accuracy) would clarify whether 4-bit training offers practical advantages beyond memory savings. Without demonstrated time-to-accuracy benefits, the motivation for 4-bit training remains weak.
- Does increasing batch size mitigate the loss spike issue? The observed instability appears partially attributable to stochastic gradient noise from small batches. If larger batch sizes naturally stabilize training, this would suggest a simpler solution than algorithmic modifications. Investigating this relationship could reveal whether the proposed techniques are necessary or if standard practices suffice for stable 4-bit training.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Stable-SPAM, an optimizer designed to stabilize gradient norms in 4-bit LLM training through adaptive clipping, gradient normalization, and momentum reset. It consistently outperforms Adam and SPAM across model scales, achieving equal or better performance with up to half the training steps.

### Strengths
This work develop a way to stabilize fp4 training upon SPAM, under low precision 4 bit and presented through experiment study with reasonable hyperparameter study on gamma 1 2 3 (of their algroithm)

### Weaknesses
* mxfp4 training missing, mxfp8 missing, stochasitc rounding missing, e.g., https://arxiv.org/abs/2502.20586 given mxfp8/4 are real fast , more stable and usable data-type supported from hardware for training.
* lack of SOTA mixed precision missing (or please clarify) is SR applied to stablize, e.g., https://arxiv.org/abs/2502.20566 ?
* baseline missing or distorted, what if applying technique even to bf16? not fair comparision, 
* hyperparameter tuning is not clear, clearer explanation how to adopt, like hyperparemter transfer from smaller scale, e.g., muP, was it tuned so that best to best comparison?
* not enough token training, model size,
* weight decay is critical, especially for large LLM, not considered? 
* what about recent Muon optimizer
* any theoretical analysis?

### Questions
See weakness about questions.

### Soundness
2

### Presentation
3

### Contribution
2
