# MIRO: MultI-Reward cOnditioned pretraining improves T2I quality and efficiency

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Current text-to-image generative models are trained on large uncurated datasets to enable diverse generation capabilities. However, this does not align well with user preferences. Recently, reward models have been specifically designed to perform post-hoc selection of generated images and align them to a reward, typically user preference. 
This discarding of informative data together with the optimizing for a single reward tend to harm diversity, semantic fidelity and efficiency. Instead of this post-processing, we propose to condition the model on multiple reward models during training to let the model learn user preferences directly. We show that this not only dramatically improves the visual quality of the generated images but it also significantly speeds up the training. Our proposed method, called MIRO, achieves state-of-the-art performances on the GenEval compositional benchmark and user-preference scores (PickAScore, ImageReward, HPsV2).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces MIRO, a novel pretraining framework that align with user preferences directly into the text-to-image generation process, avoiding common post-hoc fine-tuning stages. The core idea is to condition the generative model (based on flow matching) on a vector of scores from multiple, diverse reward models, such as those for aesthetics, composition, and text-image correspondence. By learning from the entire quality spectrum of the data, the method significantly accelerates training convergence (up to 19x faster) and mitigates reward hacking by balancing competing objectives. Empirically, MIRO achieves state-of-the-art results on the GenEval benchmark and user preference scores, outperforming larger models while being more computationally efficient and offering explicit control over quality attributes at inference time.

### Strengths
paper presents an interesting and practical approach to integrating alignment into the pretraining phase of text-to-image models. The main strengths are as follows:

1. The core idea of MIRO is conceptually simple and direct. It proposes to condition the model on multiple rewards. Despite its simplicity, the method is shown to be highly effective, leading to significant improvements in both generation quality and alignment.
2. A key practical contribution is the dramatic acceleration in model convergence. The paper convincingly demonstrates (e.g., in Figure 3) that MIRO converges much faster on multiple preference scores (up to 19x faster on AestheticScore). This substantial gain in sample efficiency makes the approach computationally appealing and highlights the benefit of providing dense, multi-faceted reward signals during the initial training phase.
3. The authors have conducted a thorough evaluation across a wide array of metrics. The experiments are not limited to a few aesthetic scores but also include established user preference benchmarks (PickAScore, ImageReward, HPSv2) and a compositional reasoning benchmark (GenEval).
4. The paper provides valuable insights by exploring how MIRO interacts with other state-of-the-art techniques. The analyses on improving text-image alignment, the synergy with synthetic captions, and the benefits for test-time scaling are particularly strong.

### Weaknesses
While the paper presents compelling empirical results, there are several weaknesses that limit its overall contribution and impact.

1.  The primary weakness is the limited novelty of the core method. The idea of conditioning a generative model on external signals is well-established (e.g., class-conditional generation, classifier guidance). The work of Dufour et al. (2024), which the authors cite, has already explored conditioning on a single reward (CLIP score). The main contribution of MIRO is extending this concept from a single reward to a vector of multiple rewards. While the engineering and empirical results are valuable, this extension feels more like an incremental step rather than a fundamental conceptual breakthrough.

2.  The paper highlights gains in training and inference efficiency but completely omits the significant computational cost of the data preparation stage. To implement MIRO, one must run *seven* different reward models over the entire 16M image-text pair dataset. This is a massive, non-trivial upfront computational expenditure.

3.  The paper presents the multi-reward conditioning as a universally positive approach but lacks a critical discussion of its potential downsides or complexities. It is highly probable that different rewards are not always complementary and may be in direct conflict (e.g., maximizing an aesthetic score might penalize compositional correctness or text fidelity). The paper does not explore this problem.

4. The legibility of several key figures is poor, which hinders the reader's ability to fully interpret the results. Specifically, the numerical labels and legends in the radar plots of Figure 2 and Figure 5 are too small to be read comfortably without significant zooming. 

5. This paper lacks REPRODUCIBILITY STATEMENT, ETHICS STATEMENT, and THE USE OF LLMS

### Questions
1.  The core idea builds upon prior work like Dufour et al. (2024), which conditioned on a single reward. The main contribution here appears to be the extension to a vector of multiple rewards. Could you please elaborate on the key technical or conceptual challenges that arise specifically from this multi-reward extension?

2.  The paper rightly emphasizes the impressive gains in training and inference efficiency. However, a significant computational cost is incurred upfront by annotating the 16M-sample dataset with scores from seven separate reward models. To provide a complete picture of the method's overall efficiency, could you please:
    *   Provide an estimate of this reward annotation cost (e.g., in total GPU hours)?
    *   How does this cost compare to the computational savings from faster convergence? For example, how many GPU hours were saved by the 19x faster convergence on AestheticScore?
    *   How does this annotation cost compare to the cost of a standard post-hoc alignment stage like RLHF on a similar scale?
    This information is critical for readers to perform a fair cost-benefit analysis of the MIRO framework.

3.  The current framing suggests that combining multiple rewards is always beneficial. However, it is plausible that rewards can conflict, and your own results hint at this (e.g., SciScore vs. aesthetics). I would be very interested in seeing a deeper analysis of this aspect.
    *   Could you provide a small-scale ablation study on the impact of different reward combinations? For example, showing the performance when trained with only (1) user preference rewards (HPSv2, PickScore), (2) text-alignment rewards (CLIP, VQA), or (3) a deliberately conflicting pair. This would provide invaluable insight into which rewards are most crucial and how the model handles trade-offs.
    *   What happens at inference time if a user requests conflicting high scores (e.g., maximum `AestheticScore` and maximum `SciScore`)? Could you show or describe the model's output in such a scenario?

4.  The uniform binning strategy is presented as a straightforward solution for normalization. However, reward distributions are often highly skewed, with most of the data occupying a narrow score range. Does equal-population binning lead to situations where perceptually similar scores are pushed into different bins, while very different scores (at the tails of the distribution) are grouped into the same bin? Have you experimented with other binning strategies (e.g., uniform score range binning) and, if so, how did they perform?

5.  As a final, minor suggestion, I would kindly request that you increase the font size of the axis labels, numbers, and legends in the radar plots (Figures 2 and 5) for the final version of the paper. They are currently very difficult to read and make it challenging to fully appreciate the plotted results.

### Soundness
2

### Presentation
3

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
This paper addresses the text-to-image pretraining problem. MIRO is proposed, which conditions model training on multiple reward models to directly learn user preferences. This method improves visual quality, accelerates training, and achieves state-of-the-art results on GenEval benchmarks and preference metrics (PickAScore, ImageReward, HPsV2).

### Strengths
1. The proposed approach demonstrates accelerated convergence during training, as evidenced by Figure 3, which illustrates MIRO’s significantly faster optimization compared to baseline methods.
2. MIRO consistently outperforms the baseline across all evaluated benchmarks, with Table 1 highlighting superior performance on GenEval and PartiPrompts metrics.
3. The integration of multiple reward models during pretraining represents an innovative strategy in text-to-image (T2I) generation, addressing limitations of prior single-reward optimization frameworks.

### Weaknesses
1. Experimental Limitations: The experimental design raises critical concerns regarding scalability and generalizability. The study focuses solely on a 0.36B parameter model—a relatively small architecture in T2I research—and trains it on only 16M image-text pairs. These constraints undermine confidence in the method’s ability to scale to industry-standard large models (e.g., 10B+ parameters) or real-world datasets. Additionally, the conclusions drawn from such limited experiments lack sufficient statistical rigor.
2. Theoretical Justification: The paper fails to provide a compelling theoretical motivation for integrating reward models into the pretraining phase. The authors do not adequately explain why this approach is inherently superior to conventional fine-tuning or reinforcement learning (RL) paradigms, leaving its necessity unproven.
3. Computational Burden: The proposed framework exhibits prohibitive resource requirements. Continuous reward annotation for all training samples becomes infeasible at scale, particularly when pretraining on billion-scale datasets. A more practical alternative would be to apply MIRO during the supervised fine-tuning (SFT) phase, where high-quality curated data could mitigate annotation costs.
4. Overstated Claims: Several assertions in the paper lack empirical validation:
* As claimed in Line 201, the statement that "MIRO eliminates the need for separate fine-tuning or RL stages" is unsubstantiated. Existing evidence demonstrates that domain-specific fine-tuning with curated datasets significantly enhances performance, while RL remains critical for optimizing text rendering. No experiments in this work refute these dependencies.
* The assertion in Line 206 that "MIRO leverages the entire quality spectrum" contradicts the authors’ own methodology, as they explicitly filter training data to CC12M and LAION Aesthetics 6+, effectively discarding low-quality samples.
5. Evaluation Scope: The empirical validation relies on an overly narrow benchmark set (GenEval and PartiPrompts). To strengthen the conclusions, the authors should evaluate MIRO on additional  benchmarks to ensure robustness across diverse modalities and use cases.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces MIRO (Multi‑Reward cOnditioned Pretraining), a framework that integrates multiple reward models directly into the pretraining of text‑to‑image (T2I) models. Instead of performing post‑hoc alignment or fine‑tuning with a single reward, MIRO conditions the model on multiple reward signals spanning aesthetics, preference, semantic correspondence, and compositional reasoning. By jointly learning to map text and multiple reward targets to images, MIRO achieves controllable generation, faster convergence, and stronger generalization. Experiments on GenEval and PartiPrompts show that MIRO matches or surpasses much larger models while being 100–300× more compute‑efficient, particularly in compositional and preference alignment tasks.

### Strengths
* The idea of embedding multiple reward signals into pretraining is conceptually simple yet powerful, unifying data quality, efficiency, and controllability within one framework.
* Empirical results are strong and consistent: MIRO outperforms single‑reward and baseline models across aesthetic and alignment benchmarks, reaching state‑of‑the‑art GenEval and user preference scores with much lower compute.
* The method yields clear interpretability and controllability at inference time, enabling explicit adjustment of reward trade‑offs and providing an elegant alternative to complex post‑hoc RLHF pipelines.

### Weaknesses
* The paper lacks ablation studies analyzing sensitivity to the number and choice of reward models; it is unclear how redundant or correlated rewards affect performance or training stability.
* Although MIRO shows significant efficiency improvements, the presentation of computational cost may be incomplete. Details on hardware, batch size, and training duration are sparse, making comparisons to larger models somewhat uneven.

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
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MIRO, a framework that integrates alignment directly into the pretraining phase of text-to-image models. By conditioning the model on a vector of reward scores, MIRO learns to map desired quality levels to visual characteristics, eliminating the need for post-hoc alignment stages that can harm diversity. This method preserves the full data spectrum and enables controllable inference. Empirically, MIRO converges faster on some metrics and achieves state-of-the-art results on the GenEval benchmark, outperforming the much larger FLUX-dev model. It also demonstrates greater compute efficiency at inference than FLUX-dev.

### Strengths
- Converging up to 19.1x faster on AestheticScore is a massive training speedup. And the inference efficiency achieves SOTA quality with 370x less compute than FLUX-dev. It also makes Best-of-N sampling way, way cheaper.

- The paper shows (in Fig 2) that single-reward models totally overfit and tank other metrics. By training on 7 different rewards , MIRO is forced to find a healthy balance, and it ends up doing great on all of them.

- It's especially good at tough compositional tasks like Position and Color Attribution

- This method doesn't just throw away "low-quality" data, which always felt wasteful. It learns from those samples by seeing their low reward scores

### Weaknesses
- The whole framework is now completely dependent on the quality of your N reward models. If those models are biased or flawed, MIRO will just learn to be biased and flawed

- The paper mentions augmenting 16M images. You have to run seven different reward models over all 16M images before you can even start your faster training. That's a huge, non-trivial compute cost that has to be paid first.

### Questions
- How did you land on these specific 7 rewards ? Did you try a minimal set, like just one for aesthetics and one for alignment? I'm curious what the minimum viable "MIRO" looks like.

- You binned the scores using "equal population". Why that? Did you consider other ways, like having more bins for the really high-quality scores to get more fine-grained control at the top end?

### Soundness
3

### Presentation
3

### Contribution
3
