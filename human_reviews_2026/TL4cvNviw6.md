# DeRaDiff: Denoising Time Realignment of Diffusion Models

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
Recent advances align diffusion models with human preferences to increase aesthetic appeal and mitigate artifacts and biases. Such methods aim to maximize a conditional output distribution aligned with higher rewards whilst not drifting far from a pretrained prior. This is commonly enforced by KL (Kullback–Leibler) regularization. As such, a central issue still remains: how does one choose the right regularization strength? Too high of a strength leads to limited alignment and too low of a strength leads to "reward hacking". This renders the task of choosing the correct regularization strength highly non-trivial. Existing approaches sweep over this hyperparameter by aligning a pretrained model at multiple regularization strengths and then choose the best strength. Unfortunately, this is prohibitively expensive. We introduce _DeRaDiff_, a _denoising-time realignment_ procedure that, after aligning a pretrained model once, modulates the regularization strength _during sampling_ to emulate models trained at other regularization strengths—_without any additional training or fine-tuning_. Extending decoding-time realignment from language to diffusion models, DeRaDiff operates over iterative predictions of continuous latents by replacing the reverse-step reference distribution by a geometric mixture of an aligned and reference posterior, thus giving rise to a closed-form update under common schedulers and a single tunable parameter, $\lambda$, for on-the-fly control. Our experiments show that across multiple text–image alignment and image-quality metrics, our method consistently provides a strong approximation for models aligned entirely from scratch at different regularization strengths. Thus, by enabling very precise inference-time control of the regularization strength, our method yields an efficient way to search for the optimal strength, eliminating the need for expensive alignment sweeps and thereby substantially reducing computational costs. The official implementation is available at https://github.com/itsShahain/DeRaDiff.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
DeRaDiff introduces a denoising-time realignment procedure for diffusion models that allows practitioners to modulate KL regularization strength on the fly during sampling without any retraining or fine-tuning, by interpolating between a pretrained reference model and a single aligned anchor model through a closed-form Gaussian mixture applied at each denoising step, controlled by a scalar parameter lambda between 0 and 1. The method extends decoding-time realignment from language models to continuous latent diffusion processes, providing an efficient alternative to the expensive practice of training separate models for each regularization strength. Experiments on SDXL and Stable Diffusion 1.5 show that DeRaDiff accurately approximates models aligned from scratch across a wide range of regularization strengths, with mean absolute errors below 0.5% on metrics like PickScore, HPS v2, and CLIP, while reducing computational costs by up to 90% when exploring multiple regularization strengths. Additionally, DeRaDiff can undo reward hacking artifacts by increasing lambda to simulate stronger regularization, offering a practical and scalable solution for hyperparameter exploration in alignment of text-to-image diffusion models.

### Strengths
While the idea of decoding-time realignment was recently introduced for language models, the paper non-trivially lifts it to the continuous, iterative denoising process of diffusion: it derives a closed-form Gaussian mixture per timestep, handles scheduler-specific posteriors, and exposes a single scalar λ that maps to any effective KL strength β/λ. This extension is not straightforward—marginalizing over latent trajectories is intractable—so the authors provide a principled step-wise approximation backed by a new theoretical result. DeRaDiff is therefore the first inference-only knob for alignment strength in diffusion, removing the need for expensive sweeps.

Alignment cost is a practical bottleneck for large diffusion models; DeRaDiff removes it by turning a multi-training sweep into a single pass plus sampling-time tuning. This lowers the barrier for researchers and practitioners to explore fine-grained alignment, potentially accelerating RLHF-style workflows, multi-reward composition, and personalised generation. The technique is architecture-agnostic (any scheduler admitting Gaussian posteriors) and complementary to existing alignment objectives (DPO, DDPO, etc.). By demonstrating that careful posterior interpolation can mimic full retraining, the work also hints at broader implications for efficient model merging and test-time adaptation beyond vision.


The submission is technically sound. The derivation (appendix) carefully justifies the Gaussian form, states regularity conditions (λ ∈ [0,1], positive variances), and warns against extrapolation (λ > 1). Empirical coverage is unusually broad: two model families (SDXL, SD-1.5), three human-centric metrics (PickScore, HPS v2, CLIP), 500 prompts spanning two datasets, and six regularization strengths. Error magnitudes are small (<0.5 % of metric means), confidence intervals are supplied, and ablations show stability inside the convex regime. Compute savings are measured in actual GPU-hours and EFLOPs rather than vague “speed-ups”. Minor gaps—no user study on λ control, limited diversity in prompts—do not weaken the overall thoroughness.

### Weaknesses
The paper trains anchors only at β ∈ {500, 1000, 2000, 5000, 8000, 10 000}.  
   - Fig. 5 shows that PickScore saturates after β ≈ 2000; the interesting “knee” region where human appeal rises fastest (β ≈ 100–1500) is sampled very coarsely.  
   - Because DeRaDiff can **interpolate** (λ < 1) but can only **weakly extrapolate** (λ > 1) before instability, a user who wants to explore β < 500 cannot do so with the supplied β = 500 anchor.  
   


   The authors never run a β-sweep **guided** by DeRaDiff. The experiment pipeline (§5.1) still assumes the user already knows which β-values to test.  
   

   Fig. 3 and Table 6 show visible degradation once λ ≳ 2.5. Many users will nevertheless try λ > 1 to obtain stronger alignment; at present the paper offers no guard-rail except a verbal warning.  
   

   PickScore, HPS v2 and CLIP give nearly identical curves (Fig. 5). The experiment therefore does **not** show that DeRaDiff can trade off **aesthetics vs. prompt fidelity**, a key reason one tunes β in practice.  
   

   All results use Euler-A, 50 steps. Many deployment pipelines use DPM-Solver-12 or 20 steps for speed. The closed-form update assumes the scheduler is **linear in the Gaussian sufficient statistics**; this is only approximately true for some solvers.  
  
   A simpler baseline is to take the aligned β = 500 and β = 2000 checkpoints and linearly interpolate their **weights** (θ-ref + λ(θ-2000 − θ-ref)). This costs zero extra inference time and is already used in “model soups” work.

### Questions
The ReAlignment problem is a notion that has only recently surfaced in the LLM literature.  
As the first work that imports this term into the generative-Art domain, the current manuscript cannot assume that most readers are already familiar with it.  

To discover what “ReAlignment” actually means, one is forced to consult *Decoding-time ReAlignment of Language Models*.  
Therefore, I believe the paper should be revised to:  

- Give a concise, self-contained definition of ReAlignment up front,  
- Explain how the generative-Art setting changes the problem.

### Soundness
2

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
3

### Summary
The authors propose an inference-time realignment of diffusion models, to be able to emulate models trained under other regularization scenarios, without extra training or fine-tuning. Inspired by DeRa (from the realm of LLMs), the authors derive a closed-form approximate posterior sampling. The authors claim and briefly discuss the reduced computational overhead.

### Strengths
- The paper is reasonably well-written with a coherent narrative. 
- The idea of extending DeRa (from LLMs) to diffusion models is an interesting angle. 
- The results are rather promising, and align with the core claims.

### Weaknesses
- I believe only Fig. 7 for comparison across base, aligned and realigned models is too little, also not discussed in necessary level of detail. In my eyes this should be established with more qualitative results and further elaboration across the images and models.  
- The paper would benefit from a thorough proof-read. Few typos, and styling inconsistencies can be seem across the document. 
- Maybe (pareto-front) reward vs divergence plots can help establish the core message from a different angle, that just looking CLIP or HPS.

### Questions
- Fig 6 (b) is hard to read and interpret. Can't this be done differently? or at least elaborated better?

### Soundness
3

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
4

### Summary
The author presents a classifier‑free guidance–like formulation designed for reward alignment in diffusion models (DDPM).
The experimental results support the formulation's validation.

### Strengths
Overall, the paper makes a worthwhile contribution with a clear presentation and credible theoretical 
support.

## Presentation: ~95th percentile

This paper presents coherence, and most of the idea is clearly addressed. I would like to thank you for saving me a lot of time reviewing your work.

## Soundness: ~75th percentile

Theorem 1 underpins the soundness of the paper. Although I have not examined every minute detail, the derivation appears to be correct.

## Contribution: 40th~70th percentile
This method seems novel to me, although I’m not sure if something similar already exists in the literature. It builds a scalable way to tune between the anchor model and the aligned model.

## Note
I hope the AC is aware that the rating is calibrated using percentiles to reduce evaluation noise effectively.

### Weaknesses
## Soundness

I would have preferred to see additional comparisons between your method and other approaches applied to similar problems. Nonetheless, the absence of such comparisons does not undermine the validity of your 
claim.

## Presentation

1. Presenting the denominator in Equations (3)–(6) as a partition function has both advantages and 
disadvantages. While it clarifies the interpretation, the repeated form of the 
equations feels redundant. If the repetition is intentional, please justify it explicitly; otherwise, 
consider consolidating the expressions to avoid unnecessary duplication.
2. Algorithm 1 appears to be a verbatim transcription of your Python implementation. For readers who 
are more comfortable with Python than with pseudocode, it would be clearer to relocate the algorithm to the appendix and present the actual Python source code there. This approach preserves the practical 
relevance of the code while keeping the main manuscript concise.

## Contribution
Regarding a [similar work](https://arxiv.org/abs/2505.18547) working on score-based SDE in the existing literature, it would be helpful to acknowledge it and clarify how your paper differs.

It is known that DDPM, DDIM, and any score‑based SDE can be reformulated as Karra’s SDE [1] in a 
bidirectional manner. Consequently, these paradigms are theoretically equivalent, although the conversion 
is not trivial. Thus, even if your work is considered concurrent or subsequent, there remains room for a 
meaningful contribution. Besides, many contemporary studies were developed without awareness of this 
equivalence.

[1] Karras, Tero, et al. "Elucidating the design space of diffusion-based generative models." Advances in neural information processing systems 35 (2022): 26565-26577.

### Questions
1. It seems odd to see the thermodynamic variable $\beta$ placed in the denominator, since $\beta$ 
is usually regarded as the inverse temperature $\tau = 1/\beta$.
2. I was confused in Line 145 that you cited Song et al. (2020) without any SDE formulation in your paper.

### Soundness
4

### Presentation
4

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
The paper proposes a novel method called DeRaDiff, which performs per-step denoising realignment. Aligning models to human preferences from scratch is often very time-consuming and computationally expensive. DeRaDiff, on the other hand performs the alignment on the fly during inference by modulating the alignment strength by a parameter $\beta$. The experimental evaluation shows that the method has comparable results to models aligned from scratch.

### Strengths
- The paper shows that DeRaDiff has a closed-form solution.
- The training-free inference-time alignment method saves computational costs
- The method is able to undo reward hacking, eliminating the need for realignment

### Weaknesses
- When evaluating human alignment a real-world human study would have been nice
- Since the pre-trained reference model is mixed with the aligned model, this could lead to biases reappearing in the output of DeRaDiff.

### Questions
Q1: Did you observe whether biases are propagated from the reference model to the output of DeRaDiff?  
Q2: In the experiments the reward-hacked model had a $\beta$ of 500. Did you test it with even more reward-hacked models that have even a lower $\beta$?  
Q3: Does the method still work if the reference model and the aligned model have different architectures (e.g. DiT and U-Net models)?

### Soundness
4

### Presentation
4

### Contribution
4
