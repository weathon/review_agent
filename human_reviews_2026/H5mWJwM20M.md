# Towards a Unified Theory of Quantization and Sparsity

- Avg Score: 2.67
- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Quantization and sparsification are two model compression strategies that are traditionally treated as orthogonal in the literature. Building on recent work, we show that jointly considering these techniques can meaningfully affect compression performance. First, we extend prior tensor-level analyses and prove that for any $L_p$ norm, applying sparsification before quantization ($\mathbf{S} \to \mathbf{Q}$) always yields lower errors than the reverse. However, we demonstrate that tensor-level analysis is insufficient to predict model performance, motivating the need for model-level evaluation. As such, we provide the first model-level analysis showing that $\mathbf{S} \to \mathbf{Q}$ obtains better loss in certain settings when we choose quantization and sparsification algorithms independently. Yet, this preference does have its limits. When fully relaxing model assumptions, we find it difficult to prove the superiority of $\mathbf{S} \to \mathbf{Q}$, casting doubt on the preference in the general case. To that end, we introduce Quantization-Aware Sparsification (QAS), a novel compression framework that sparsifies accounting for prior quantization, as a simple counterexample. Using this framework, we provide a simple counterexample in which $\mathbf{Q} \to \mathbf{S}$ using QAS performs comparably to $\mathbf{S} \to \mathbf{Q},$ illustrating that careful co-design between model compression steps can greatly influence performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The work advances the theoretical understanding of combining sparsity and quantisation for compressing deep learning models. First, the authors generalise previous results on the advantage (w.r.t. parameter reconstruction error) of running sparsification-before-quantisation vs quantisation-before-sparsification. Then, the authors consider the model-level with a diagonal Hessian, equivalent to a $\mathrm{diag}(H)$-weighted squared reconstruction error, showing that (under some conditions) an advantage for sparsification-first. However this is shown not to hold true in the more general case. A short empirical investigation follows, showing that sparsification-first can be better in practice, but that if the sparsification mask is taken from the original weights, then quantisation-first can also work as well.

### Strengths
The work is clearly presented and makes a reasonably robust case for the main claims. Particular strengths:

 - Notation is clear and sufficiently well-explained
 - Helpful intuition is given for some proofs, e.g. in Section 4.1
 - Results for QAS are interesting and somewhat surprising to me
 - The idea in section 6.3 of using a short re-training phase to disambiguate weights that are quantised to the same level is interesting

### Weaknesses
**Main concern:** Although I appreciate the rigorous derivation, the results in Theorems 1, 2 and 4 seem trivial and unsurprising. T1 since (as explained in 4.1), weights that are quantised to the same level cannot be disambiguated, T2 is easily reduced to T1, as stated and the claim associated with T4 is weak, having dropped the constraints that made it possible to show T3. I found T3 the most interesting, but I don't completely follow the constraints (question below). Combined with the short and small-scale empirical investigation, I do not take the results as sufficiently informative to the community to recommend acceptance.

**Specific concerns and questions:**

1. The scale bound $\delta \leq \epsilon$ isn't entirely clear to me - do we know if this is likely to be satisfied in practice, if scale is derived from a tensor-absmax?
1. Is the definition of $\epsilon_S$ in Theorem 3 (body) correct? I think as stated the following assumption would reduce to $\epsilon_S = 0$, and seems inconsistent with that of the proof in L1134, which would expect $\epsilon_S \coloneqq \Delta w_{Q \rightarrow S} - \Delta w_Q - \Delta w_S$).
1. Block (group) quantisation is common (and indeed seems to be used by default with AWQ in your supplementary scripts), while as I understand it, many of the theoretical results only apply to weights within a single block.
1. Although it is acceptable to have a limited empirical example given the theoretical focus, I think there are some obvious gaps. Although the theoretical results concern (Q, S) = (Naive Max-Scaled, Magnitude) or (Naive Max-Scaled, OBD), the results do not include these settings, always using AWQ for quantisation and not including OBD for sparsity. It would also greatly help to see sparsity-only and quantisation-only results for comparison.
1. In Table 1, despite the observation that for INT4 with 10% Magnitude pruning, the INT4 quantisation step already sets enough weights to zero to make sparsity a no-op, I can't understand why the more accurate INT8 format should under-perform INT4. (Anything that sparsity flushes to zero in the INT8 case would already have been flushed to zero in INT4.)
1. The model scale, sparsity and quantisation settings considered in the experiments are far from the state of the art (e.g. Tseng et al., 2024, Liu et al., 2024, Frantar et al. 2023), making it hard to be confident that the theoretical results can be observed in practice.

**Minor concerns:**

 - The "key insight" of L168, stating "$|w_i - \bar{Q}(\bar{S}(w))_i| = 0$ if $w_i$ is pruned" doesn't seem right, wouldn't it be $= |w_i|$ in this case?
 - While I accept Theorem 1, the implication stated in L185 concerning $L_{\infty}$ seems potentially misleading - isn't equality very highly likely in this case (assuming tensor-wise quantisation scaling), since the same-quantisation-level collision would have to occur for the extreme value?
 - Body section 4.4 references Theorem 5 from the appendix.
 - In Tables 1 and 2, I most wish to compare Sparsity Method & Order for given Precision & Sparsity levels, which would be much easier if the major grouping was on Precision & Sparsity levels and minor grouping on Sparsity Method & Order.

---

_Tseng, A., Sun, Q., Hou, D. and De Sa, C.M., 2024. Qtip: Quantization with trellises and incoherence processing. Advances in Neural Information Processing Systems, 37, pp.59597-59620._

_Liu Z, Zhao C, Fedorov I, Soran B, Choudhary D, Krishnamoorthi R, Chandra V, Tian Y, Blankevoort T. Spinquant: Llm quantization with learned rotations. arXiv preprint arXiv:2405.16406. 2024 May 26._

_Frantar, E. and Alistarh, D., 2023, July. Sparsegpt: Massive language models can be accurately pruned in one-shot. In International conference on machine learning (pp. 10323-10337). PMLR._

### Questions
I would appreciate any clarifications/corrections/counter-arguments, especially regarding my main concern and specific concerns above.

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
The paper provides a stronger theoretical result that $S \rightarrow Q$ (sparsity before quantization) is optimal over $Q \rightarrow S$ on a tensor level, and the same result on the model level under some assumptions. To mitigate error in $Q \rightarrow S$ setup, the authors propose using the unquantized weights for calculating sparsity mask, which they name Quantisation-Aware Sparsification (QAS). Then they demonstrate on OPT-125M model that $Q \rightarrow S$ can perform on par with  $S \rightarrow Q$ with QAS.

### Strengths
1. The paper tackles an open practical question of interaction between quantization and sparsity and highlights the necessity of codesigning the hybrid compression formats. 
2. This work proves on a tensor level for any $L_p$ norm that sparsity before quantization minimizes the loss.

### Weaknesses
1. Limited novelty. The work closely follows Harma et al. (2025), in particular their claim on $S \rightarrow Q$ being optimal over $Q \rightarrow S$ on a tensor level and empirical validation on a model level. Harma et al. proves the statement of Theorem 1 for $p=1$, and the authors of this work only extend it to arbitrary $p \geq 1$ without gaining new insight.
2. Too strong assumptions: Theorems 2 and 3 assume Hessian to be an identity matrix, and a diagonal matrix, respectively. The condition $\Delta w_{Q|S} = \Delta w_Q$ is particularly restrictive, it basically means the sparsity only affects the weights that would be quantized to the same value as 0, $Q(S(x)) = Q(0) = Q(x)$. This practically renders sparsity unnecessary in this setup.
3. The paper lacks contribution and insight. Proposed Quantisation-Aware Sparsification effectively makes element-wise quantization into quantization applied after sparsification. The paper offers no explanation on how would the training dynamics with QAS change for more complex quantization / sparsity schemes.
4. Limited validation: the authors conduct experiments using a single model instance, and the results in Table 1 mainly reproduce Harma et al. (2025). Sparsity rates only include 10 and 25%.

### Questions
1. What is the interplay between quantization and sparsity for other compression schemes, like SparseGPT, GPTQ, structured N:M sparsity?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper analyses the effects of quantization and sparsification of a model and the order in which they are applied. Through tensor-level and model-level analysis, they show that sparsification then quantization is more preferred than quantization then sparsification, especially when the methods for quantization and sparsification are chosen at random. They also propose a new Quantization Aware Sparsification through which they show that quantization, then sparsification, can perform better than the other way around. They show empirical evidence of their claims using the OPT 125M parameter model.

### Strengths
1. They show that S -> Q is better than Q -> S through tensor-level analysis
2. They present the novel model-level analysis of the interaction between quantization and sparsification
3. They propose a new Quantization Aware Sparsification method that prunes quantized models
4. They validate the claims made through theoretical analysis with empirical evidence.

### Weaknesses
1. The specific scenarios where Q -> S is strictly better than S -> Q are not specific theoretically; only the possibility is shown.
2. No comparison to other quantization or sparsification methods.
3. The choice of AWQ for quantization seems arbitrary, and no justification is provided
4. No ablation study of how the proposed method scales with model sizes, sparsity levels, and bit-widths is shown

### Questions
Please see the weaknesses and 
1. It would be better to explicitly mention the metric (perplexity; lower is better) used to evaluate the models.
2. In line 168, if $w_i$ is pruned, shouldn't the error be $|w_i|$?

### Soundness
3

### Presentation
3

### Contribution
2
