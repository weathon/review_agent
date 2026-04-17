# EDIT: Early Diffusion Inference Termination for dLLMs Based on Dynamics of Training Gradients

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Diffusion-based large language models (dLLMs) generate tokens through iterative denoising, but answers often stabilize before all denoising steps are completed.
We introduce EDIT (Early Diffusion Inference Termination), an inference-time method that adaptively stops the denoising process once reasoning stability relative to training behavior is detected.
EDIT is built on training-gradient dynamics, typically otherwise discarded after training, where, during fine-tuning, AdamW-aggregated LoRA updates encode parameter importance signals.
We retain this information as compact reasoning maps.
During inference, EDIT measures alignment between token activations and these maps, detecting convergence when KL divergence across consecutive steps on unmasked (visible) tokens falls below a threshold. 
On reasoning benchmarks, EDIT reduces diffusion steps by 11.8–68.3\% while preserving or improving accuracy in most cases, with negligible storage overhead ($\sim$0.02\%, about 1.5–2 MB for all QKV modules in a 32-block, 8 GByte model).
These results establish a principled mechanism for transforming knowledge about training-gradient dynamics into practical test-time benefits such as reducing reasoning time.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides a mechanism to early-stop the diffusion process in a dLLM.  Basically, we collect information during LoRA fine-tuning about which dimensions of the query output projection (in the last layer of the Transformer) get the most consistent gradients, as per updates to the LoRA B matrix.  Then, during inference, we look at the dimensions of unmasked tokens in the representations after the same query projection, and see if they are aligning with the FT gradients.  If they are, we say that "reasoning stability" has been achieved, and we stop the diffusion process.  Some experiments assess whether this can really save diffusion steps on downstream tasks.

### Strengths
It's well-motivated to try to make use of the gradient information that was collected during training (or fine-tuning) and re-use that during inference.  Diffusion seems like a great place to start.

I think even the specific idea to use gradient momentum info from reasoning-based SFT to see if generation has arrived at a "reasoning"-like solution is not necessarily a bad idea.

Studying diffusion LLMs are also a very wide-open space so testing dLLMs like LLaDA-8B on downstream tasks, and finding out what the weak points are, is good to do.

Good job to have a discussion of limitations (in the conclusion), however brief.

### Weaknesses
Overall, the approach here just feels very premature, like, as a practitioner, I am really unsure whether simpler things would work, or whether other choices could unlock much bigger gains.  There's not a lot of theory or empirical rigour behind the findings, e.g., what if we used an Oracle and stopped at the optimal number of steps in each case, what are the maximum speedups that we could get, or the best improvements in quality?  It's just not well-scoped in the current submission.  This lack of rigor and testing of alternative ideas, plus a lack of comparison to baselines, and some questionable experimental decisions, makes me think this paper is not yet valuable to the community.

From my perspective, the scope of other things that COULD be tried is very large, and it's not clear how important the choices made in the paper actually are:
- What about just using the parameters themselves, or momentum itself, or just the evolution tensor, rather than the evolution tensor after the reduction?
- Why just the Query projection?  Why just the last layer?
- Why Cosine versus other measures?  Why KL divergence versus other measures?
- What if we didn’t ensure stability with Ω consecutive scores below δ, but did something else?

The current method introduces new task-specific (!) hyperparameters:
- δ, Ω, τblk
- Fundamentally, I don’t think we should be tuning hyperparameters on specific downstream tasks, as this confounds model comparison – how can we compare to models that weren’t fine-tuned, e.g., on the “Countdown” task?  Update: actually, this paper does make this mistake.
- Did we tune the number of diffusion steps on each task as well for the baselines?  It seems like you just compare at different max denoising steps (set depending on sequence length), but the EDIT count is below the smallest that you tested, so it makes me wonder if fewer steps would be better for these other ones.

Soundness:
- The fact we didn’t compare to any other methods of early stopping for denoising… this really surprised me.  Like, accepting and freezing unmasked tokens when their probability crosses a threshold?  You mentioned “output stability or confidence” and "entropy" so why not test those?  What's the downside?
- I mean, the fact we get different (and often better) accuracy numbers is surprising to me, and seems to reflect some kind of bias in the trained models that can be alleviated through hyperparameter tuning, which was only done with EDIT.  Fundamentally, I would not expect more steps to impair a diffusion model, although I believe there is prior work showing accuracy does not increase monotonically in number of steps.  Perhaps you should have plotted accuracy versus steps for the baselines for Countdown and then if EDIT is below this curve, that would alleviate some of my concerns.

Nitpicks:
- I think since the fundamental idea of this paper is that SFT reveals reasoning patterns in the activations, you should really explain the SFT dataset in more detail, right?
- Oh man, Tables 1 and 2, those are quite tiny fonts!  Is there no limit to how small we’d make them???
- Prior work specifically using momentum to identify important params: Dettmers - Sparse Networks from Scratch - Faster Training without Losing Performance - 1907.04840v2
- You know, diffusion itself does allow “guidance” in the form of gradients to be applied during the generative process.  I think maybe you could link your pseudo-gradients to this theory a bit better.

### Questions
- When we convert alignment scores into a probability distribution, what is it a distribution over?   There are many probabilistic arguments made about this distribution, but I don’t really understand, e.g., what domain the “support” of this distribution lives in.
- Is full-parameter SFT or CPT or even RL a limitation of your method, or just a limitation of your experimental evaluation?  What about other optimizers, e.g., Muon?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes **EDIT (Early Diffusion Inference Termination)**, an adaptive early inference termination rule for diffusion LLMs (dLLMs). During SFT with LoRA on Q/K/V, the method aggregates AdamW update statistics into a compact, feature-aligned “AdamW evolution” vector ($u$) that encodes which parameters consistently carried the learning signal. At inference, EDIT computes per-step cosine similarities between token activations and $u$, and halts denoising for tokens once a stability condition is met. The authors provide extensive experiments supporting the claim that training dynamics can guide safe early termination.

### Strengths
* **Insightful diagnostics.** Gradient-based “pseudo-gradient” alignment with SFT gradients; domain-wise breakdowns (e.g., GPQA subdomains) and LoRA-A vs. LoRA-B sparsity analyses justify design choices.
* **Practical gains with tiny overhead.** Reported step reductions up to ~68% and a ~1.5 MB metadata footprint for a 32-block model are compelling for deployment.

### Weaknesses
* **Metadata extraction.** EDIT relies on training-time metadata extraction. Many released checkpoints does not expose training recipe, does the selection of training recipe (dataset, hyperparam) affect extraction.
* **Scope of validation.** Experiments center on a single dLLM family (LLaDA-8B) and five reasoning tasks on one hardware stack. It’s hard to assess robustness across models, sizes, datasets. 
* **Task-tuned thresholds.** ($\delta,\Omega$) are selected per task via validation for an accuracy/steps trade-off. This introduces tuning burden and potential brittleness under distribution shift. 
* **Poor presentation** Figures and tables are too small comparing to captions, such as Figure 3/4/5, Table 1/2. Certain figures in the paper have strange frames, such as Figure 2/4/8. The theory seems to be ad-hoc.

### Questions
* **Baselines & fairness.** How does EDIT compare to strong adaptive early inference termination rule for dLLMs under identical compute budgets—and to autoregressive early-exit baselines on similar tasks? **The paper only compare EDIT with plain baseline, However, a lot of dLLM acceleration methods already exists**, such as [1][2].
* **Unclear Intuition** What is the intuition of $u$? what makes similarity between each token’s activation and the AdamW evolution vector important? The paper does not provide a clear intuition.
* **Metadata extraction.** Does SFT training recipe affects the performance of EDIT? How does performance change under domain shift from the SFT distribution?
* **Adaptive calibration.** Can $(\delta,\Omega)$ be set online *without* task-level validation?

[1] Wu C, Zhang H, Xue S, Liu Z, Diao S, Zhu L, Luo P, Han S, Xie E. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv:2505.22618. \
[2] Li P, Zhou Y, Muhtar D, Yin L, Yan S, Shen L, Liang Y, Vosoughi S, Liu S. Diffusion language models know the answer before decoding. arXiv:2508.19982. \
[3] Ben-Hamu H, Gat I, Severo D, Nolte N, Karrer B. Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking, NeurIPS 2025.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces EDIT, an inference-time early termination rule for diffusion LLMs that reuses training-time optimizer information. During SFT with LoRA, the method aggregates AdamW moments into a compact “AdamW evolution” vector $u$ that encodes parameter-importance patterns; at inference it measures cosine alignment between token activations and $u$, converts these to per-step distributions on visible tokens, and stops when matched-support KL divergence stays below a threshold for $\\Omega$ consecutive steps. The theory shows a multi-step control bound $TV \\le \\Omega\\sqrt{\\delta/2}$ and a margin condition that preserves the argmax, yielding PAC-style certificates for chosen $(\\delta,\\Omega)$. Empirically, on LLaDA-8B across Countdown, Sudoku, MATH500, GSM8K, and GPQA, EDIT reduces denoising steps by 11.8% to 68.3% with comparable accuracy on most tasks; storage overhead is about 0.02% of an 8 GB model.

### Strengths
* Clear definition of the AdamW evolution signal and why LoRA-B is preferred, with sparsity metrics and visualizations.
* Matched-support KL and multi-step TV bounds yield simple certificates and a PAC-style calibration rule.
* Minimal storage and implementation overhead, with complexity lower than attention and integration as a wrapper at inference.
* Certified early stop rates reported across tasks, indicating practical realizations of the theory.

### Weaknesses
* Hyperparameter selection uses per-task validation sets and a grid over $(\\delta,\\Omega)$; robustness to mis-tuning or cross-task portability is not thoroughly analyzed.
* GSM8K at length 512 shows a noticeable accuracy drop with EDIT, which deserves a short diagnostic beyond the brief discussion. Minor.

### Questions
* Please report end-to-end system metrics for one representative task and length (e.g., Countdown@256): wall-clock per instance, tokens per second, and peak GPU memory, alongside the step reductions already shown in Table 2. This is a light logging change.
* For GSM8K at length 512, please add a short diagnostic: either (i) a histogram of early-stop step vs. correctness, or (ii) a 2×2 grid over $(\\delta,\\Omega)\\in\\{0.05,0.1\\}\\times\\{6,12\\}$ reporting accuracy and mean denoising steps. One figure or a small table is sufficient.

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
3

### Summary
This paper proposes EDIT to speed up diffusion-style large language models. The key idea is to reuse training-time optimizer dynamics that are normally discarded: aggregated AdamW statistics on LoRA-B are compressed into a compact per-block “pathway vector” u. During inference, EDIT measures how well current visible-token activations align with u, constructs a distribution over the intersection of visible tokens across consecutive steps, and tracks a matched-support KL divergence. If the KL stays below a threshold δ for Ω consecutive steps, the method early-stops the diffusion process.
It provides theoretical support by bounding total-variation distance from KL and using a margin condition to argue that, with suitable (δ, Ω), early stopping will not change the final prediction. Empirically, EDIT reduces denoising steps by 11.8–68.3% on multiple reasoning benchmarks with little to no loss in accuracy. The approach requires only lightweight metadata extracted at training time, no architectural changes at inference, and minimal runtime overhead.

### Strengths
1.The method is clear, novel and well-supported by theory.

2.Without changing the model's reasoning structure, it enables the addition during training and no modification during inference, which differentiates it from previous efficiency-enhancing methods that require modifying the decoding or architecture.

3.The paper is easy to read, motivations are well connected.

### Weaknesses
1.The reliance on training metadata requires access to the optimization trajectory during the training phase, which is insufficient for scenarios involving closed-source models or those with only final checkpoint files.

2.Validation focuses on LoRA-SFT, it remains unclear how EDIT performs with full-parameter finetuning, other adapters, or different optimizers 

3.Small accuracy dips appear in longer sequences, and the paper does not deeply analyze when EDIT stabilizes too early.

### Questions
1.Have you tried this method on other optimizers and adapters?

2.Under the influence of prompt form changes, noise disturbances, or adversarial interventions, is it more likely to prematurely stop? Have you ever encountered the situation of prematurely stopping?

### Soundness
3

### Presentation
3

### Contribution
3
