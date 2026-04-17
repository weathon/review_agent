# Planner Aware Path Learning in Diffusion Language Models Training

- Decision: Accept (Oral)
- Scores: 6, 4, 8, 4

## Abstract
Diffusion language models have emerged as a powerful alternative to autoregressive models, enabling fast inference through more flexible and parallel generation paths. This flexibility of sampling is unlocked by new engineered sampling strategies, or *planners*, that select more favorable generation paths by iteratively planning---versus uniformly at random---where to denoise along the sequence. However, by modifying the reverse paths via planning, planners create an irrevocable mismatch between the uniformly random denoising paths during training and planning-based inference. In this paper, we systematically investigate the mismatch of discrete diffusion training and inference under planning and theoretically prove that the standard discrete diffusion training evidence lower bound (ELBO) does not accurately describe a denoiser that uses a non-uniform planner. To address this gap, we derive a new planned evidence lower bound (P-ELBO) that incorporates planner-based reverse dynamics directly into the training objective.
Using the P-ELBO, we introduce *Planner Aware Path Learning* (PAPL), a novel training scheme that aligns training and inference under a planned denoiser.
PAPL is implemented as a simple yet effective modification to the standard masked discrete diffusion loss, making it widely applicable and easy to adopt.
Empirically, we show PAPL delivers consistent gains across domains, including a 40\% relative improvement in protein sequences, improved text generation with up to a $4\times$ relative MAUVE gain, and 23\% relative improvement in code generation HumanEval pass@10.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the training-inference mismatch in masked diffusion language models (DLMs). While DLMs are trained with uniform random masking/unmasking, they are deployed with planners that bias the denoising order (e.g., greedy, path planning). The authors prove that this mismatch causes the standard DLM ELBO to be violated at inference (Proposition 3.1). They derive a Planner-Aware ELBO (P-ELBO) that incorporates planner dynamics into the training objective (Proposition 3.2), then propose PAPL—a practical algorithm that simplifies P-ELBO into a planner-weighted cross-entropy requiring only a one-line code change. Experiments demonstrate consistent improvements: ~40% relative gain in protein foldability, up to 4× MAUVE improvement in text generation, and ~23% relative gain in code generation pass@10, all under identical model architectures and inference planners.

### Strengths
1. The core contribution is original and well-motivated. Proposition 3.1 formally proves that greedy planner-based sampling can violate the standard ELBO inequality (log p_θ < ELBO), meaning the training objective no longer lower-bounds the inference distribution. This is a concrete theoretical finding that justifies rethinking the training procedure, not just an empirical observation.
2. The theoretical framework is rigorous and general. The P-ELBO formulation (Proposition 3.2) unifies multiple existing planning strategies: uniform (standard DLM), greedy (MaskGIT), and soft-greedy, as special cases. 
3. The implementation is practical and low-friction. PAPL reduces to standard MDLM loss augmented with planner weights and requires no architectural changes or additional forward passes during training. Algorithm 1 shows this is essentially a one-line modification, making adoption straightforward.
4. Empirical validation is comprehensive across diverse domains. The paper demonstrates improvements in protein generation (Table 1: higher pLDDT, pTM, lower pAE, +40% foldability while maintaining diversity), text generation (Table 2: consistent MAUVE and perplexity improvements across sampling budgets), and code generation (Tables 3-4: gains in both completion and infilling). The ablations (Figure 2, Figure 4) support design choices and show robustness.
5. The presentation is clear and well-structured. The method section logically progresses from dynamics formulation to ELBO violation to the new objective. Experimental setups, metrics, and baselines are clearly described with sufficient detail for reproduction.

### Weaknesses
1. The gap between P-ELBO and implemented objective is not quantified. Section 3.4 drops the correction term by detaching gradients through the planner, converting the full P-ELBO to a simpler weighted cross-entropy. The paper provides no analysis of the induced approximation error, theoretical bounds, or empirical measurements of this gap. 
2. Although the framework is generally applicable for any planner, the derived training objective and results are only for soft greedy decoding. For non-differentiable or hard planners, the correction term becomes non-trivial, and the paper does not provide a full derivation or empirical evidence beyond soft greedy-like planners. 
3. Text evaluation is limited to unconditional generation. While MAUVE and generative perplexity on OpenWebText are standard, they don't test whether planner-aware training improves conditional generation capability.

### Questions
1. Can you measure the correction empirically at different training stages across domains? Some figures may be helpful.
2. How should practitioners choose the two hyperparameters?
3. How tight is the P-ELBO bound in practice?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes planner-aware path learning for masked diffusion language models. The authors learn a model  $G_\phi(z, x_k)$, where $z$ is a sample from the denoiser probabilities and $x_k$ is the k-th state, which outputs the next index to unmask. 

The authors derive an evidence lower bound to lower bound the denoiser + planner log-likelihood, however as the objective requires $L^2$ function calls for sequences of length $L$, the authors propose a different tractable objective. The tractable objective re-weights the denoiser log-likelihood with an extra planner term.

### Strengths
See questions

### Weaknesses
See questions

### Questions
1. The proof for the counter-example in proposition 3.1 requires that the denoisers are inconsistent. However, uniform un-masking with inconsistent denoisers can lead to inconsistent sampling as well. 

It would be beneficial to a reader to see a re-written proposition 3.1, as the authors construct a counter-example rather than ***prove*** that greedy decoding does not sample from a unique learned joint distribution. Under the constructed counter-example, even uniform sampling would not sample from a single joint distribution as the denoisers are inconsistent.

1. If the denoiser is consistent, then would greedy sampling work? As MDLMs learn an any-order autoregressive model, would any decoding order at inference would produce valid samples? 
2. If the denoisers are consistent, then is the following line correct:
    1. *The key takeaway is that the ELBO in equation 1 is only valid for the uniform unmasking process (line 247)*. 
3. what does the planner learn at optimality

Experiments

1. Can the authors compare PAPL to greedy sampling in the experiments? 
2. In several experiments, the gap between the MDLM+planned and MDLM+baseline with  experiments are not substantial. Can the authors provide a confidence interval?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposed PAPL, a lightweight modification to the classic MDLM training objective that brings a performance boost. The authors first indicate a problem in the domain of diffusion language models, the mismatch between training and evaluation. Based on this insight, the authors derived an analytical form of the planner-aware ELBO (P-ELBO) and show that P-ELBO is theoretically general. Given that P-ELBO is computationally expensive, the authors proposed a cheap approximation, PAPL, and show that across protein modeling, language modeling, and coding benchmarks, adding PAPL to the MDLM loss can boost performance without extra computational expense.

### Strengths
* The storyline of the paper is clear. The paper aims to address a significant problem in the field of diffusion language models, and its solution is convincing.
* The mathematical foundation is abundant. I especially appreciate that the authors proved the P-ELBO in an analytical form. The theory part is valuable and useful for the community.
* The experimental analysis is solid, and the empirical results are strong. I am particularly surprised by the increase in pass@10 in the coding part.
* The method is simple and practical. With some minimal changes to the training objective, PAPL can improve downstream performance.

### Weaknesses
* As a person with proficiency in discrete diffusion, I have to say the math is a bit confusing. I hope the authors can add more explanation, at least in the appendix, to enhance comprehension (see Question No. 1 for details).
* The transition from Proposition 3.2 to Equation 7 is not apparent, and more explanation might be needed.

### Questions
* Is the CTMC perspective necessary for the derivation of the ELBO, i.e., Proposition 3.2? If not, can the authors add the derivation from the discrete Markov chain perspective in the Appendix? 
* Can the authors add more explanation on the transition from Proposition 3.2 to Equation 7, at least in the Appendix?
* The authors claim the new ELBO is planner-aware, so it should be better than the MDLM ELBO. But the final training objective is also a fixed objective; are there any justifications for why this objective is better for various commonly used planners? In other words, is this objective only better for the greedy-sampling planner, or is this objective better for all the commonly used samplers for masked diffusion language models?
* I would like the authors to clarify the planners used in their experiments. Are the planners just greedy sampling? If so, can the authors provide results with other planners, such as block decoding topk confidence from Nie et al. [1], confidence thresholding from Wu et al. [2], adaptive MDM inference from Kim et al. [3], and remasking samplers from Wang et al. [4]?
* From a theoretical perspective, the authors claim that P-ELBO is a unified framework for all possible planners. Can the authors provide an instantiation for the four planners mentioned above?
* In the right subplot of Figure 4, the downstream performance monotonically grows with $\alpha$; what if $\alpha$ goes beyond 5? I am assuming that at some point the downstream performance will go down, otherwise we should not interpolate between the MDLM loss and PAPL loss, but use PAPL loss alone instead.

---
**References**

[1] Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.R. and Li, C., 2025. Large language diffusion models. arXiv preprint arXiv:2502.09992.

[2] Wu, C., Zhang, H., Xue, S., Liu, Z., Diao, S., Zhu, L., ... & Xie, E. (2025). Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618.

[3] Kim, J., Shah, K., Kontonis, V., Kakade, S., & Chen, S. (2025). Train for the worst, plan for the best: Understanding token ordering in masked diffusions. arXiv preprint arXiv:2502.06768.

[4] Wang, G., Schiff, Y., Sahoo, S. S., & Kuleshov, V. (2025). Remasking discrete diffusion models with inference-time scaling. arXiv preprint arXiv:2503.00307.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper shows the mismatch of diffusion objective between training and testing under planning and address it by revising the ELBO to incorporate planner basics in training. The new ELBO allows to effectively exploit any planning strategies at test time and it is simple to implement. The method show performance improvements on various tasks, such as protein sequences, text generation, code generation.

### Strengths
1. The paper tackle the relevant and important mismatch issue of training and testing of diffusion model under planning.
2. The method is theoretically grounded though it is quite dense.

### Weaknesses
1. Using confidence as a heuristic indicator could be unreliable for training. This is exactly shown in Figure 5 in the appendix. Though I see the authors using the vanilla DLM loss as a rescue, I think this is not an efficient way since at earlier training iterations, the confidence from model is not reliable, inducing noise into training (this is bad). One alternative is to do annealing where earlier step, use uniform weight as a sole vanilla DLM and it is annealed gradually to planner-weighted DLM.
2. Look at the Table 1 and Table 2, it seems that the method negatively affect the data entropy since both are worse than vanilla model. Why?
3. What is limitation? It is worth to mention.

### Questions
1. Is the model sensitive to the temperature and the weight alpha? Need more ablation on text generation and coding. 

2. What exactly P2 sampling is  used? Since the P2 sampling has different variants. 

3. Apart from P2 sampling, have the authors tried other planning strategies like greedy decoding like MaskGIT, top-k, top probability margin (from Kim et al, https://arxiv.org/abs/2502.06768).

### Soundness
3

### Presentation
3

### Contribution
3
