# Fixing Model-Fitting: Compressing Guidance for Better Sampling

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 4, 8

## Abstract
Model-fitting, a phenomenon where generated samples are overly adjusted to the model used for guidance rather than the intended conditions, is a key drawback and often leads to suboptimal outcomes. The root cause of this problem is the consecutiveness of guidance timesteps throughout the diffusion sampling process. In this work, We quantify this effect and show that breaking the consecutiveness of standard guidance alleviates the problem. Based on this insight, our method, Compress Guidance, distributes a small number of guidance steps across the full sampling process, yielding substantial improvements in image quality and diversity while cutting guidance cost by over 80\%. Experiments on both label-conditional and text-to-image generation, across multiple datasets and models, confirm that Compress Guidance consistently surpasses baselines in image quality with significantly lower computational overhead.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper analyses the phenomenon of overfitting in classifier-based guidance of diffusion models (including implicit classifiers in classifier-free guidance). One of the main findings is that guiding the generation with a classifier reduces the classifier's loss during the generation; however, this loss reduction does not generalize to comparable classifiers with different parameters. It indicates an overfitting-like behavior, which the paper calls model fitting. As a solution, Compress Guidance is proposed, which reduces the number of denoising steps with guidance and distributes more guiding steps to the early denoising steps. Evaluation of the method is conducted on traditional ImageNet diffusion models (ADM and EDM2), together with more advanced models like Stable Diffusion.

### Strengths
-	The paper tackles an interesting issue: classifier-based/CFG-based generations incorporate additional computational costs. By reducing the number of guidance steps, the computational burden can be reduced. Moreover, as the paper demonstrates, this strategy also improves image quality and prevents overfitting of the guidance model.
-	Qualitative and quantitative experiments support the motivation of the paper well, and sufficient mathematical foundations are provided.
-	Quantitative evaluations indicate improved image quality compared to vanilla guidance strategies.

### Weaknesses
-	The paper writing should be improved, making it sometimes hard to follow. Here are some examples:
  - The abstract directly starts with the term of model-fitting, never mentioning the setting of image generation with diffusion models.
  - L173: “With ˜x0 is the prediction of x0 at timestep t, we can…”
  - References to equations are sometimes abbreviated, e.g., Eq. 8 (L189), and sometimes not, e.g., equation 8 (L177)
  - L195: “where the xt is the parameter of the model at the timestep t,”-> why is x the parameter of the diffusion model here?
  - Figure 4 is referenced long before Figure 3. Also, there seems to be a wrong reference in L325 (should be Fig. 3 instead of 4, I assume)
  - L342: “To avoid calculating too much gradient”
  - Font sizes in tables are comparably small, e.g., Table 5. The same goes for font sizes in figures, such as Figs. 3 and 4.
-	Most experiments focus on highly outdated and simple ImageNet models (ADM from 2021). Whereas there are some qualitative experiments on Stable Diffusion in the Appendix and some numerical results in Table 6, the focus on outdated models limits the method’s strength. Classifier-guided generations are rarely used today, and CFG stands out as the go-to method for diffusion models. 
-	Comparison to related guidance techniques is limited to Table 8. What about other settings, e.g., with Stable Diffusion/text-to-image domains?

Small remarks:

-	Metrics like FID should be cited.

### Questions
- Would it make sense to combine the method with some momentum or gradient decay, i.e., reducing the influence of gradients from the previous generation steps?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper diagnoses a previously under-appreciated pathology in guided diffusion sampling dubbed “model-fitting”: when the guidance gradient is computed at every consecutive timestep the trajectory gradually over-tunes the image to the idiosyncrasies of the guiding network rather than to the true target distribution, producing brittle, low-generalization samples and wasting compute. Through careful on-/off-sampling loss probing the authors show that (i) the majority of guidance information is injected in the early, high-noise regime, (ii) later steps contribute negligible new signal yet still incur heavy gradient cost, and (iii) an independent classifier of equal accuracy consistently disagrees with the guided outputs, confirming overfitting. Building on these observations they propose Compress Guidance (CompG), a training-free, model-agnostic strategy that selectively applies guidance at only a handful of wisely chosen steps, reuses the latest gradient for the intermediate intervals, and slightly boosts the guidance scale to preserve continuity; a simple power-law schedule skews the chosen steps toward the early denoising phase where guidance matters most. Across ImageNet resolutions from 64 to 512 px and MS-COCO text-to-image generation, both classifier and classifier-free pipelines equipped with CompG reduce gradient evaluations by 5–40× and runtime by roughly 40 % while simultaneously delivering better FID, sFID, Inception and CLIP scores than vanilla dense guidance, early stopping or uniform skipping baselines, demonstrating that compressing guidance not only saves compute but also yields visibly cleaner, more diverse and better-aligned images.

### Strengths
The paper reframes the long-observed “guidance artifacts” in diffusion models as an over-fitting phenomenon that occurs inside the sampling trajectory rather than during training. By introducing the on-/off-sampling loss diagnostic and explicitly analogizing to train/test gaps in classical learning, the authors give the community a conceptually new lens—model-fitting—that unifies disparate heuristics such as early stopping, interval guidance, and gradient reuse. The Compress Guidance schedule itself is elegantly simple, yet no prior work has derived it from a principled study of gradient redundancy and continuity requirements; the power-law timestep distribution and the gradient-caching trick are creative, synergistic additions that remove the need for extra distillation or retraining.

Diffusion sampling is the computational bottleneck in text-to-image, video, and 3-D generative systems; any training-free speed-up that also boosts quality is instantly deployable. By showing that guidance cost can be slashed 5–40× with nothing more than a changed schedule, the work impacts both research (greener experimentation) and production (lower serving cost). Equally important, the model-fitting diagnostic offers a reusable tool for future guidance research and may influence how other generative families—consistency models, flow-matching, or autoregressive transformers—think about conditional feedback.

The empirical program is unusually thorough. Experiments span four ImageNet resolutions, MS-COCO text-to-image, both classifier and classifier-free pipelines, and multiple metrics (FID, sFID, IS, CLIP, Precision/Recall, runtime). Controls include matched diversity (Recall) to ensure fair accuracy comparisons, ablations on the exponent *k*, compact-rate sweeps, and head-to-head baselines (BigGAN, VQ-VAE-2, Big-DCT, Inter-valGuidance, etc.). The 40 % wall-clock reduction with simultaneous quality gains is reproducible across settings, and the authors release full hyper-parameters and code—an indicator of solid engineering hygiene.

### Weaknesses
Compress Guidance’s core manoeuvre—applying guidance only on a sparse, early-heavy subset of steps—overlaps heavily with IntervalGuidance ( Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models ). The authors rebut that these works “lack an explicit mechanism” or treat sparsity as a by-product, yet the conceptual step (most gradients are wasted) is the same. A stronger related-work comparison should quantitatively pit CompG against IntervalGuidance under identical diffusion backbones rather than relegating it to a single line in Table 8; if CompG still wins, novelty is clearer. Explicitly citing and discussing the scheduling heuristics in Progressive Distillation (Progressive Distillation for Fast Sampling of Diffusion Models) and Phased Consistency (Phased Consistency Models) would further clarify the boundary.

### Questions
The most valuable scenario for guidance-compression techniques is **training-free conditional generation**, where an external control signal is injected into a pre-trained diffusion model without retraining its backbone.  
If the method can drastically cut the cost of guidance, the total compute of such training-free pipelines should drop to **bare diffusion-model levels**.  
The paper, however, presents only limited experiments in this setting; the evaluation ought to be **substantially expanded** to verify this claim.

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
4

### Summary
The paper has two contributions for guidance in diffusion sampling. First, model fitting, where the authors shed light into the fact that there is a difference in performance between the classifier used in classifier guidance and an external classifier. Second, they propose the Compress Guidance method, that computes guidance only for some timesteps otherwise it reuses the same from the previous step (hence offering speedups).

### Strengths
- The paper is well-written and easy to follow. 

- The model fitting analysis is very interesting and the community could benefit from this.

### Weaknesses
While I enjoyed reading the paper, there are several issues and open questions with the second part, i.e. Compress Guidance. 

1/ Scheduler G 

This seems to be the most crucial point of the proposed Compressed Guidance (this is when guidance from previous steps is being reused). G depends on K and s. 


1a/ K. 

It is unclear how the authors choose the values of K. Is there is hyperparameter search? 
If yes, it makes it may diminish the value of the method. 
If not, perhaps there is a heuristic way of selecting it?  
This seems to be very crucial as the whole Compress Guidance depends on it. 


1b/ K values in Table 11. 

Most K values are around 1, which points to uniform sampling! To my understanding, this means that in most cases Compress Guidance is not used. If this is the case, it would be beneficial for the authors to clarify this; if not, can the authors please explain? 


1c/ Guidance scale s. 

In Classifier Guidance (and Classifier-Free Guidance) the choice of s impacts the final performance heavily. How is s selected? Is there a hyperparameter search? Showing only some values in Table 11 makes it even less clear. Figures 8 and 9 show FID and IS performances with three s values, which makes it even harder to get as the performance is peaked when s=0; is this correct? I would have liked more explanation of this as it seems crucial to understand the behaviour of the proposed Compress Guidance.  


2/ Missing explanation. 

Lines 329-333 state that two prior works have both showed that delaying guidance improves performance, which is intuitive and explained in the respective papers (“where applying guidance too early causes conflicts between guidance and conditional information of diffusion model.”) . Figure 3b shows that delaying guidance hurts performance. This is not intuitive and I find the explanation offered by the authors insufficient (lines 333-334). It would be useful if the authors could expand on this. 

3/ Method clarity. 

Unlike model fitting which is very clear, understanding Compress Guidance is less clear. The algorithm really helps though. 


4/ Several choices are outdated. 


4a/ Diffusion vs Flow Matching.

A large part of the community seems to have moved from diffusion to Flow Matching and most modern frameworks (such as FLUX) use FM. It would have been useful to have the analysis presented from the FM perspective so that we can understand how we can benefit from model fitting and Compress Guidance with today’s methods. 


4b/ Classifier Guidance vs Classifier-Free-Guidance. 

To my general knowledge, CG is outdated and instead CFG is the default solution. It is unclear why the work relies so much on CG; it would have been useful to have the whole work from the CFG perspective (as opposed to the short 4.4) so that we can understand how we can transfer the findings from this paper to modern works. 

5/ Text-to-image generation.

5a/ Models

Tables 6 and 12 reports results using SD and GLIDE, which are outdated. It would have been useful if the authors had also reported results with more modern models so we can understand if their findings carry to modern models. 

5b/ Metrics

Text-to-image generation is typically evaluated with more modern metrics, such as GenEval and aesthetic scores. This is minor but I would suggest that the authors incorporate such metrics. 

5c/ Missing experiments. 

Minor: Although the paper has plenty of experiments, it seems that they are a sparse collection of experiments instead of a systematic study showing something. 
Guidance is usually associated with diversity and fidelity and FID vs CLIP Score curves could also help. 

6/ Misc.

6a/ It is unclear how Compress Guidance performs across random seeds or different classifier architectures. Given that the whole motivation of the work is based on that, I think it would have been an interesting experiment to have. 

6b/ As I mentioned above, the work has many experiments. For future reference, it would be useful to also have some failure cases or trade-offs when guidance compression is too strong. 

6c/ It is unclear if reusing gradients in Equations 11 and 12 accumulates bias or leads to drift in the sample trajectory. I would have loved to see something about this.

Minor:

1/ Table 11 is very important; yet, we cannot easily understand to what is pointing as the pointers to other tables are wrong. 

2/ Tables 12 and 13 are not mentioned and it’s hard to associate them with their respective text in the main paper. 

3/ Figure 9 is hard to read; usually we display FID (y axis) vs IS (x axis). 

4/ Are 250 steps too large?

### Questions
It would be useful if the authors address several of the points outlined in the weaknesses above. Specifically: 

Q1 (W1). Can the authors show clearly the correlation between k and s and performance? 

Q2 (W2) Explain clearly the intuition behind the “unconditional setup”.

Q3 (W2) Can the authors perhaps combine Compress Guidance with the schedulers from the related work (eg exp, sin, or other adaptive ones)?

Q4 (W4) Can the authors explain (or give some justification/intuition behind) the relevance of CF and outdated models with modern methods?  

Q5 Perhaps not relevant but an alternative could be to compare Compress Guidance with other methods with the same compute conditions (as opposed to equal steps). 

Q6 (minor?) Are 250 steps too many?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper identifies an overfitting phenomenon during the sampling process of classifier-guided diffusion models, analogous to overfitting in standard model training. In such cases, generated samples become overly aligned with the guiding classifier rather than faithfully adhering to the intended conditioning signal.

To address this, the authors propose Compress Guidance (CompG), which applies guidance only at a small number of strategically selected timesteps throughout the diffusion process, instead of at every step.

Experiments across diverse models and datasets demonstrate that CompG consistently outperforms standard classifier and classifier-free guidance. It improves image quality and diversity while significantly reducing computational cost.

### Strengths
1. Clear and well written presentation.
2. Novel perspective interpreting the sampling process as an optimization over noise.
3. Proposes a simple yet effective solution to mitigate model fitting during the sampling process.
4. Provides a thorough analysis of model-fitting across unconditional guidance (UG), classifier guidance (CG), and classifier-free guidance (CFG).
5. Demonstrates strong empirical results, with significant improvements across multiple datasets and diffusion models.

### Weaknesses
1. Unclear relation with current CFG progress (changing the guidance scale, shifting the sampling time, etc.). 

2. no discussion of SDS which is also an optimization based sampling.

### Questions
How does this approach relate to other timestep selection or adaptive guidance techniques used in text-to-image diffusion models? In particular, could this method provide insight into the effectiveness of tuned or learned CFG schedules, such as those proposed in “Navigating with Annealing Guidance Scale in Diffusion Space” (Yehezkel et al.)?

Since Score Distillation Sampling (SDS) is itself an optimization-driven sampling process and shares similarities with diffusion guidance, does the proposed method have implications for SDS-based pipelines? Can the authors clarify whether Compress Guidance improves, aligns with, or conflicts with SDS?

### Soundness
4

### Presentation
4

### Contribution
3
