# UltraViCo: Breaking Extrapolation Limits in Video Diffusion Transformers

- Decision: Accept (Poster)
- Scores: 4, 6, 2, 8

## Abstract
Despite advances, video diffusion transformers still struggle to generalize beyond their training length, a challenge we term video length extrapolation. We identify two failure modes: model-specific periodic content repetition and a universal quality degradation.
Prior works attempt to solve repetition via positional encodings, overlooking quality degradation and achieving only limited extrapolation. In this paper, we revisit this challenge from a more fundamental view—attention maps, which directly govern how context influences outputs. We identify that both failure modes arise from a unified cause: attention dispersion, where tokens beyond the training window dilute learned attention patterns. This leads to quality degradation and repetition emerges as a special case when this dispersion becomes structured into periodic attention patterns, induced by harmonic properties of positional encodings. Building on this insight, we propose UltraViCo, a training-free, plug-and-play method that suppresses attention for tokens beyond the training window via a constant decay factor. By jointly addressing both failure modes, we outperform a broad set of baselines largely across models and extrapolation ratios, pushing the extrapolation limit from $~2\times$ to $4\times$. Remarkably, it improves Dynamic Degree and Imaging Quality by 233\% and 40.5\% over the previous best method at $4\times$ extrapolation. Furthermore, our method generalizes seamlessly to downstream tasks such as controllable video synthesis and editing.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Video Length Extrapolation adapts pretrained video diffusion transformers to generate sequences longer than training in a single pass, but models typically loop content and lose quality beyond trained length 𝐿. The paper attributes both failures to attention dispersion. UltraViCo, a training-free method that decays attention to tokens beyond the training window, restores focus, extending extrapolation from 2× to 4×. This outperforms baselines across models and extrapolation ratios.

### Strengths
- Training-free, plug-and-play: Requires no additional data or fine-tuning.

- Repetition mitigation via RoPE insight: Identifies the correlation between attention periodicity observed across models and RoPE-induced harmonics, effectively reducing repetition during video length extrapolation.

- Quality preservation through attention concentration: Alleviates attention dispersion beyond the training window, mitigating quality degradation in long-horizon generation across diverse video generation models.

### Weaknesses
1. Non-uniform evaluation prompts: Prompts are not standardized across models (e.g., UltraViCo uses 100 prompts, HunYuan uses 25), complicating fairness and direct comparability.

2. Limited analysis of Eq. (6): The paper provides insufficient sensitivity/ablation on how changes in 𝛼 and 𝛽 affect outcomes, leaving robustness and trade-offs unclear.

3. Hyperparameter dependence: 𝛼 and 𝛽 must be tuned per model and per extrapolation ratio, reducing generality and increasing deployment friction—hindering active, broad use despite the training-free claim.

4. Minor suggestions: The notation in Eq. (6) makes it unclear whether the constant 1 or 𝛽 has precedence (i.e., which term is intended to dominate or be applied first). It would be clearer to revise the expression to make the precedence explicit.

### Questions
1. Given existing approaches for long video generation that rely on inference-time modifications (e.g., FIFO), what is the specific motivation and unique significance of studying video length extrapolation as a separate paradigm? 

2. In Figure 3, when visualizing Eq. (4), how many video samples were used? Was the visualization averaged over multiple videos, or based on a single example? Because Eq. (4) involves both queries and keys, it seems necessary to report statistics over multiple videos to confirm consistency of the observations.

3. For Eq. (6), is there an analysis of how varying 𝛼 and 𝛽 affects the results (e.g., sensitivity curves or robustness across ranges)?

4. Regarding the decay factors introduced in Eq. (6), could you include an ablation with quantitative metrics in Table 1 to show their individual contributions?

### Soundness
3

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
This paper tackles video length extrapolation for generating videos longer than the training sequence length in diffusion-based video transformers. State-of-the-art text-to-video diffusion models are trained on fixed-duration clips (e.g. 5 seconds) and struggle beyond that length. The paper highlights the two failure modes that emerge as sequence length grows: (i) periodic content repetition, and (ii) quality degradation affecting all models. These issues worsen at longer extrapolations, limiting practical use. Prior work such as RIFLEx addressed repetition via positional encoding tweaks but did not solve the widespread quality drop. The author(s) of this paper identify a unified cause behind both failures, i.e, attention dispersion. When generating beyond the trained window, new frames dilute the model's learned attention focus, causing it to spread attention too broadly. In extreme cases, Rotary Positional Embedding (RoPE) harmonics can organize this dispersion into a periodic pattern, explaining the looping behavior. More generally, dispersion leads to the model losing focus on fine details and mixing unrelated temporal contexts. Building on this insight, the paper introduces UltraViCo. It is a training-free, plug-and-play method to restore the model’s focus during long video generation. The author(s) claim impressive gains in video quality and temporal dynamics on multiple text-to-video diffusion models (HunyuanVideo 1.3B, CogVideoX 5B, and Wan 2.1) and various extrapolation lengths.

### Strengths
1. The paper provides a unification of two major failure modes under the single concept of attention dispersion. To me, this is a good contribution contending the earlier positional encoding-centric explanations.

2. The approach presented in the paper requires no additional training or model modifications, making it a plug-and-play inference-time fix. Despite its simplicity, it tackles both identified issues simultaneously.

3. The paper gives importance to practical implementation. Integrating the attention decay into a custom FlashAttention kernel shows strong engineering effort so that the method works on large-scale models without running out of memory.

### Weaknesses
1. UltraViCo down-weights the influence of very distant frames. A possible side effect is that the model might lose some long-term context or consistency. The paper asserts that important content is preserved while removing irrelevant far-context influence but it does not rigorously quantify scene consistency over extremely long videos. In one ablation, using too small an $\alpha$ caused a car tire to disappear in later frames, indicating that overly concentrating attention can indeed harm persistent content.

2.  The approach requires a massive computational cost for generating long videos. The authors report needing ~8 A800 GPU-hours per video for 4x extrapolation on the largest model, which forced them to use a reduced number of samples for evaluation.

3. The paper focuses comparisons on inference-time extrapolation techniques (RoPE/PE variants and RIFLEx), which is appropriate, yet omits direct comparison to some alternative approaches for long video generation. For example, sequential generation methods like FIFO-Diffusion or diffusion re-initialization schemes (NUWA-XL, Flexible Diffusion Modeling for long videos Neurips 2022, interactive video generation via domain adaptation, and others) could also produce long videos. The authors cite these works and justifiably note they are different.

4. The theoretical analysis stops short of fully justifying the chosen solution. The idea of suppressing out-of-window attention is intuitively motivated and empirically validated, but there isn’t a formal framework proving it as an optimal or necessary fix. For example, the harmonic analysis (Proposition 1) explains when periodic repetition arises, but there’s no analogous theory for why a constant decay factor is the best way to concentrate attention.

### Questions
1. Does suppressing distant-frame attention risk any long-term coherence issues? In particular, how does UltraViCo ensure that important elements present in the first few frames like background setting, main character identity, etc. are still represented in later frames, given those early frames’ influence is decayed?

2. The paper positions as complementary to other long-video generation techniques. Have the author(s) tested or considered combining UltraViCo with such methods? For instance, could one apply UltraViCo in a sliding-window generation to further improve consistency at the window boundaries, or use it alongside a method like FreeNoise to additionally boost temporal stability?

3. The implementation mentions setting the first frame decay factor to a negative value to counteract blurring. Could the authors clarify why the first frame needed special treatment?

4. How should one choose the decay parameters $(\alpha, \beta, \gamma)$ for a new model or a different extrapolation ratio? The results show different optimal values for different scenarios (for e.g.  stronger decay for 5x on HunyuanVideo, $\alpha = 0.85$ for CogVideoX at 4x).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a length-extension method for video generation models, aiming to generate longer videos without additional training. The authors achieve this by analyzing and modifying the attention mechanism. In the experiments, they compare their method with other training-free video extension approaches and demonstrate its effectiveness.

### Strengths
* This study follows a clear methodology: observation, analysis, improvement, and validation, with rigorous logic and well-structured presentation.
* Experiments demonstrate the effectiveness of the proposed method on Wan and HunyuanVideo, outperforming baseline approaches.

### Weaknesses
* The authors’ modifications to the computation process may introduce potential performance issues. Their improvements focus on the attention map, but in efficient attention implementations (e.g., FlashAttention-3), this matrix is not fully materialized. Consequently, the proposed changes could negatively impact computational efficiency.
* Generating long videos is no longer a major challenge, as several publicly available models already support segment-wise long-video generation—such as Wan2.2-S2V, Wan-Animate, and LongCat-Video. The authors should consider including these models in their baseline comparisons.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper tackles video length extrapolation in diffusion transformer–based text‑to‑video models, which suffer from repetition and quality loss beyond training length. It identifies attention dispersion as the root cause and introduces UltraViCo, a simple training‑free decay on out‑of‑window attention logits. This method extends extrapolation from 2x to 4x and significantly improves motion and visual quality across multiple models.

### Strengths
1) The paper is very well written; the figures are very helpful to understand the motivation behind the proposed method.
2) This work provides a compelling analysis that connects two failure modes (repetition and quality degradation) under the concept of attention dispersion, offering a deeper understanding of underpinnings of video diffusion transformer models.
3) The proposed solution UltraViCo is training-free, simple to implement, and integrates seamlessly with existing popular SOTA video DiTs.
4) Very good empirical results and comprehensive evaluation of the method and baselines.
5) The approach demonstrates applicability to downstream tasks like controllable video generation and video editing without retraining.

### Weaknesses
1) UltraViCo relies on manually chosen decay factors (alpha, beta) and harmonic band width (gamma), which differ across models. The paper does not provide a fully automated procedure for selecting these values, raising concerns about generalization to unseen architectures.

### Questions
1) How can alpha, beta, gamma be selected without manual tuning? Could you propose a lightweight heuristic or adaptive scheme that works across unseen models?
2) Why stop at 4x only? How does model perform beyond this threshold? If not, why do you think the method is not suitable for extreme long video generation task?
3) What do you think about other works/directions that are trying to tackle long video generation (e.g., Diffusion Forcing, Frame Pack)? Do you think augmenting/optimizing attention maps (the approach taken in this paper) is a more promising direction as opposed to others?

### Soundness
3

### Presentation
3

### Contribution
3
