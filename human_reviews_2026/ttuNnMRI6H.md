# Any-Order Flexible Length Masked Diffusion

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Masked diffusion models (MDMs) have recently emerged as a promising alternative to autoregressive models over discrete domains. MDMs generate sequences in an any-order, parallel fashion, enabling fast inference and strong performance on non-causal tasks. However, a crucial limitation is that they do not support token insertions and are thus limited to *fixed-length* generations. To this end, we introduce **Flex**ible **M**asked **D**iffusion **M**odels (FlexMDMs), a discrete diffusion paradigm that simultaneously can model sequences of flexible length while provably retaining MDMs' flexibility of any-order inference. Grounded in an extension of the stochastic interpolant framework, FlexMDMs generate sequences by inserting mask tokens and unmasking them. Empirically, we show that FlexMDMs match MDMs in perplexity while modeling length statistics with much higher fidelity. On a synthetic maze planning task, they achieve $\approx$ 60\% higher success rate than MDM baselines. Finally, we show pretrained MDMs can easily be *retrofitted* into FlexMDMs: on 16 H100s, it takes only three days to fine-tune LLaDA-8B into a FlexMDM, achieving superior performance on math (GSM8K, 58\%$\to$67\%) and code infilling performance (52\%$\to$65\%).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes FlexMDM, a masked diffusion model that jointly models the unmasking posterior and the insertion expectation. Unlike traditional Masked Diffusion Models (MDMs), which begin with a fixed number of masked tokens and progressively unmask them, FlexMDM introduces an additional insertion operation that dynamically inserts new mask tokens during the diffusion process. This design enables variable-length generation while preserving any-order generation capabilities.

FlexMDM can be adapted from an existing MDM with minimal changes—specifically, by adding a scalar output head at each position to predict the insertion expectation.

The authors validate their method across multiple language and planning benchmarks, demonstrating that FlexMDM:

1. Effectively learns the true length distribution of the training data,
2. Achieves comparable generative perplexity to standard MDMs,
3. Outperforms MDMs on planning and reasoning tasks, and
4. Can be efficiently scaled to larger models.

### Strengths
1. Writing is very clear and easy to follow
2. The framework is mathematically sound and theoretically grounded and solid
3. The idea is novel and the connection to traditional MDMs are very clear

### Weaknesses
I find the experiments relatively weak compared to the strong theoretical foundations of the paper, and several aspects require clarification:

Model comparison: It is unclear what specific model is referred to as “MDM” in the comparisons. For example, is it equivalent to D3PM or another variant?

Figure 4c: The figure (which should include an overall label) shows an interesting trend in perplexity as the number of sampling steps increases. Since neither curve appears to converge by 4096 steps, it would be informative to see what happens beyond this point.

Maze task: While it is reasonable that FlexMDM outperforms traditional MDMs in the maze task, this setup alone seems insufficient to substantiate the claim that it “supports FlexMDM as a principled approach for subgoal-based planning.”

Scaling up experiment: The statement “Surprisingly, we observe rapid transfer: within three days on 16 H100 GPUs, the model generates variable-length sentences” requires clarification. MDMs can already generate variable-length sequences to some extent using padding tokens, as the authors note. It would be useful to clarify in what sense this transfer is “rapid.” Also, while it's an 8B model, the authors only trains a 400M LoRA adapter and claim FlexMDM is "scalable". Can the authors clarify on this?

Ablation study: Including an ablation study on modeling the insertion process would strengthen the empirical evidence. Alternatively, can the authors confirm if the comparison between MDM and FlexMDM already serves this purpose.

### Questions
Please see weakness.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Current masked diffusion models are constrained to fixed-length sample generation, often resorting to tricks (e.g. padding) to deal with variable-length inputs. This paper addresses this issue by introducing a flexible-length masked diffusion model based on token deletion/insertion in addition to masking/unmasking operations. This behavior is achieved by learning to predict an “insertion expectation” (expected number of tokens yet to be inserted after a given position) in addition to the canonical unmasking posterior. The insertion expectation is parameterized directly as an additional scalar prediction head.

The resulting method is able to accurately match the ground-truth sequence length distribution, more so than the fixed-length baseline utilizing padding tokens. It also outperforms the fixed-length baseline on a subgoal-conditioned maze solving task, and can be cheaply retrofitted onto a 8B masked diffusion model (MDM), improving math and coding performance for longer sampling horizons.

### Strengths
### S1. Solves an important problem of MDMs
Variable-length generation has been a major limitation of MDMs, and the proposed solution is a valuable addition to both theory and practice of MDMs.

### S2. Theoretically grounded approach
The proposed approach is grounded in theory and provides some guarantees on the likelihood of the trained model.

### S3. Applicability to existing MDMs
The proposed approach can be retrofitted to existing pre-trained MDMs with minimal training (as little as 13B tokens), enabling flexible-length generation for both improved efficiency and accuracy.

### Weaknesses
(ordered by descending severity)

### W1. Meager experimental results
While the set of considered tasks is well-rounded, the selective reporting and omission of results as well as non-standard choices that skew the picture in favor of the proposed method need to be rectified:
1. Generative PPL should always be reported together with sequence entropy to control for diversity collapse. Additionally, filtering out short sequences (L1575) is non-standard and skews the picture in favor of FlexMDM. This is an inherent limitation of using the gen. PPL metric and should be controlled for via sequence entropy.
2. Validation loss, and perhaps also training curves, should be reported even if the losses aren’t directly comparable (L416). If the FlexMDM objective indeed is a likelihood bound, then both losses are meaningful and, to some extent, comparable.
3. Accuracy on text benchmarks should also be reported for the sake of consistency and comparison with the literature. These numbers are comparable even if the losses aren’t.
4. Numbers on training/inference speed of FlexMDM vs. MDM are missing. In principle, this should be favorable for FlexMDM due to its smaller average sequence length.
5. The required training time to convert an existing MDM to FlexMDM (L450) should (also) be reported in terms of training tokens, as the wallclock time highly depends on the level of optimization of the given codebase.
6. The reported performance of LLaDA on GSM8k (Fig. 5) is considerably lower than what’s reported in the original paper. The original paper (Nie et al., 2025) reports 70.3% for the base model and 69.4% for the IFT model. It is unclear how this number could get worse by directly training on GSM8k.

### W2. Reproducibility
For the sake of reproducibility, I strongly urge the authors to release all artifacts required to reproduce the results presented in this paper, including trained model checkpoints as well as training and inference code.

### W3. Some fundamental limitations of MDMs remain
While the proposed method enables flexible insertion of tokens during the generation process, some other fundamental limitations remain: Namely, the resulting model is unable to delete or revise existing tokens once filled in. Therefore, the model still risks accumulating errors throughout sampling, just like traditional MDMs.

### Conclusion
Despite the limited experimental results (W1) and concerns regarding reproducibility (W2), the theoretical contributions are strong on their own (S1, S2). Therefore, reasons to accept currently outweigh reasons to reject. The submission can be made significantly stronger still by addressing W1 and W2, and I will be happy to update my final score accordingly.

### Questions
- Q1. What are the generative PPLs of MDM and FlexMDM (175M) without filtering by sequence length? (also see W1.1)
- Q2. What is the sequence entropy (as in Zheng et al., 2024) of MDM and FlexMDM (175M) for different numbers of denoising steps? (also see W1.1)
- Q3. What is the training/validation loss (throughout training/final) of MDM and FlexMDM (175M)? (also see W1.2)
- Q4. What is the performance of MDM and FlexMDM (175M) on some relevant text benchmarks (e.g. HellaSwag, ARC-E, ARC-C, WinoGrande, PIQA, etc.)? (also see W1.3)
- Q5. What is the training and inference speed of MDM vs. FlexMDM (both 175M and 8B)? (also see W1.4)
- Q6. How many tokens were used for training LLaDA on FlexMDM? (also see W1.5)
- Q7. Why are the reported numbers on GSM8k (Fig. 5) much lower compared to LLaDA-base (Nie et al., 2025) despite explicitly training on the task? (also see W1.6)
- Q8. Will the model weights and/or training code be released? (also see W2)

Additional questions:
- Q9. Does the FlexMDM training objective constitute a likelihood bound (i.e. ELBO)? If so, this does not appear to be immediately obvious based on L278.
- Q10. Were any alternative ways to parameterize the insertion expectation considered? For example, one may consider modeling the number of insertions as a categorical distribution over the number of to-be-inserted tokens and calculating the expectation thereof.
- Q11. For large-scale pretraining, it is a requirement to have statically-shaped batches for the sake of model compilation and good FLOP utilization. Therefore, in order to apply FlexMDM at scale it will be crucial to have a recipe to do fixed-length training, e.g. through sequence packing. Can FlexMDM support such a scenario?
- Q12. Is it possible to skip the distinct masked case and directly go from empty to unmasked? This could avoid the apparent overhead of first inserting masked tokens and only unmasking them later.
- Q13. Why is a poisson distribution the correct choice for inserting mask tokens? As far as I can tell, this is not explained in the paper.

Nits (not considered for final score):
- Some equations are unnumbered (e.g. L184, L230, L259, L278). For the sake of referenceability, it is good practice to number all equations (and not just the ones that are referenced in the text itself).
- Time notation breaks the convention of t=0 being noise-free and t=1 being complete noise. This is confusing for people familiar with the diffusion literature, especially for those unfamiliar with flow-matching. For the sake of consistency with the literature, I recommend keeping the diffusion convention. Alternatively, a note on this break of convention may help prevent unnecessary confusion.
- L294: Missing parentheses around equation number. Alternatively, it is good practice to reference equations as such, i.e. “Eq. 5” or “Equation 5” or “Eq. (5)”.
- L317: Seemingly missing some words in “Since unmasking indices in an adaptive [inference setting] no longer trace [...]”
- L375: The footnote mark makes the noise schedule appear to be quadratic. Swapping the position of the period and footnote may help, i.e. `$\alpha_t = \beta_t = t$.\footnote{...}`
- L452: Missing period at the end of the line.

---

### References
- Nie et al. (2025): https://arxiv.org/abs/2502.09992
- Zheng et al. (2024): https://arxiv.org/abs/2409.02908

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes **Any-Order Flexible-Length Masked Diffusion (FlexMDM)**, a generalization of masked diffusion models (MDMs) that supports variable-length generation through a continuous-time Markov chain formulation. The authors introduce an *insertion process* parameterized by a learned insertion expectation (g_\theta) in addition to the usual denoising posterior (f_\theta). This enables length-adaptive, any-order generation while preserving theoretical guarantees and compatibility with pretrained MDM weights. Experiments demonstrate strong results on text, maze planning, and code/math reasoning tasks, including scaling to an 8B-parameter model.

### Strengths
1. **Clear theoretical formulation.** The continuous-time treatment and derivation of the rate matrix and training objective are elegant and rigorous. The paper provides the right amount of mathematical depth while maintaining readability.
2. **Minimal yet effective extension.** Introducing a scalar insertion expectation per gap is an elegant way to achieve variable-length modeling without compromising the core MDM formulation.
3. **Scalability and empirical validation.** The ability to retrofit large pretrained MDMs and obtain consistent gains across domains (text, code, math) is a strong practical signal.
4. **Well-designed sampler.** The adaptive unmasking and τ-leaping integration are both theoretically justified and empirically efficient.

### Weaknesses
0. **Missing support for addition operation**.
Compared to editflow, it supports addition while this work doesn't. 

1. **Missing discussion of Seed Diffusion.**
   The paper should reference and compare to *Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference* (arXiv:2508.02193). Both aim to enable flexible-length or insertion-based generation, but take different approaches—Seed Diffusion uses a single-model canvas growth mechanism, whereas FlexMDM introduces a learned insertion intensity. A brief conceptual comparison would clarify relative advantages (FlexMDM’s theoretical rigor vs. Seed Diffusion’s simplicity).

2. **Apparent need for two models.**
   The description suggests two learned functions—one for denoising and one for insertion—which might appear “not elegant.” It would help to emphasize that these are *two heads of the same model* rather than two independent models. The authors could also discuss whether a unified parameterization (e.g., sharing a latent event-rate representation) is feasible, as done in Seed Diffusion’s single-network formulation.

3. **Limited evaluation in real-world settings.**
   While the presented results are promising, additional experiments on **broader math and coding benchmarks** would strengthen the claims.

   * Include *MATH-500/5000* or *AIME24/25* for compositional reasoning.
   * Extend code evaluation to *HumanEval*, *HumanEval+*, or *MBPP*.
   * Report **pass@10** and **pass@k vs. compute-time** to better capture stochastic sampling performance.

4. **Minor clarity issues.**

   * An architecture figure showing the shared backbone and dual heads would avoid the “two-model” confusion.
   * A brief qualitative example illustrating insertion dynamics would improve intuition.

### Questions
See weakenesses.

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
This paper proposes Flexible Masked Diffusion Models (FlexMDM), a discrete diffusion framework that adds token insertion to standard masked diffusion models while provably preserving any-order decoding. It introduces a joint interpolant over token values and indices. The proposed model learns both the usual unmasking posterior and a new insertion expectation, and derives a CTMC-based training objective with guarantees. Empirically, FlexMDM matches MDM perplexity while modeling length distributions far more faithfully. Initializing from a pretrained MDM model (LLaDA-8B), fine-tuned FlexMDM exhibits significant performance improvements on GSM8K and HumanEval-infill as the number of sampling steps increases.

### Strengths
- This paper proposes a novel and principled approach to addressing variable-length generation, which is a known limitation of existing masked diffusion models.
- The proposed approach is theoretically grounded: a joint interpolant is defined over both token values and the number of insertions, then a neat training objective is derived based on CTMC, as a natural extension to the standard MDM training loss. 
- This paper also proves the compatibility of insertion prediction with adaptive inference in MDM
- The empirical evidence is compelling. FlexMDM models the length distribution much more faithfully than MDM without sacrificing perplexity, and demonstrates better scaling performance on GSM8K and HumanEval-infill

### Weaknesses
- It would help practitioners more if the paper could provide more descriptions of the implementation details. E.g., after inserting/deleting tokens, will the padding be dynamically adjusted?
- Some additional questions 
    - Figure 3 left: I guess $S_{\tau}$ shoud be $[0, 3, 4]$ right?
    - In algorithm 1, positions are unmasked first before insertion prediction, but I suppose the newly unmasked tokens should not be visible to the insertion module $g_{\theta}$ right?
    - Given the flexibility of combining any-order unmasking with insertion prediction, I wonder whether it's possible to simply reuse the unmasking posterior from a pretrained MDM and only separately learn the insertion expectation.

### Questions
Please see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
