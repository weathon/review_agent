# Catch-22: Pareto Frontier for Detectability and Robustness in LLM Watermarking

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 0, 6

## Abstract
Large Language Models (LLMs) generate text through probabilistic token sampling, a mechanism increasingly leveraged for inference-time watermarking to verify AI-generated content. As watermarking schemes proliferate, assessing their robustness-detectability trade-off becomes essential to determine whether watermarks can survive output editing while remaining invisible to adversaries. Current evaluation relies on empirical tests lacking provable guarantees. In this work, we present the first information-theoretic framework that rigorously characterizes this fundamental trade-off. We first establish a hierarchy of sampling-time watermark detectability, ranging from undetectable (distribution-preserving) to highly detectable (biased sampling) schemes. Second, we demonstrate an inverse relationship: watermarks robust to text modifications are inherently more detectable by adversaries, creating an irreducible trilemma: no scheme simultaneously achieves high robustness, low detectability, and reliable verification. Motivated by these theoretical constraints, we propose a hybrid watermarking system that adaptively switches sampling strategies based on LLM output edit levels, achieving Pareto-optimal trade-offs. We show that distribution-preserving schemes provide perfect undetectability; however, they are only robust to near-zero adversarial edits. On the other hand, bias-free and biased sampling offer high robustness guarantees at 15-20\% output editing, but with detectable output statistics. At high output editing rates, no watermarking provides robustness guarantees. Lastly, we empirically validate our theoretical trade-off claims with Llama 2 7B and Mistral 7B models under paraphrasing attacks, thereby confirming that Pareto-optimality is only achieved by a hybrid watermarking scheme. Overall, our framework provides watermark evaluation beyond empirical testing via principled design, revealing information-theoretic limits for sampling-based watermarking and how computational hardness shapes which regimes are algorithmically achievable.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the pareto frontier for robustness and susceptibility to detection without access to the secret key of a watermark in LLMs.  A watermark is a statistical signal hidden inside text that can be detected by anyone with access to a secret key but is intended to not distort the quality of text, thus remaining undetectable to observers without access to a secret key. Many schemes have been instantiated for autoregressive models by modifying the sampling procedure of models, with a prominent such scheme being the green list approach, where a hash function looks at the recent context and returns a pseudorandom subset of the vocabulary to upweight in generation.  This paper investigates the extent to which a watermark can be both undetectable to an observer without access to a secret key and be robust to edit-level attacks and finds that there is a fundamental tradeoff.  The authors consider three approaches to watermarking and bound the total variation distance between the watermarked and unwatermarked distributions, which controls the detectability of the watermark.  They then investigate the extent to which the considered approaches are robust to distortion and identify a tradeoff before empirically evaluating their findings.

### Strengths
This paper investigates an important tradeoff between detectability and robustness of watermarks.  The existence of a pareto frontier that enforces this tradeoff is an important point and the authors do well to prove this.

### Weaknesses
First, I am confused why the authors believe that the vocabulary size in Moitra & Golowich has to be exponential; exponential in what? That paper clearly states that the vocabulary size only has to be polynomial in the security parameter.

Second, I am somewhat skeptical of the framing of the notion of detectability. While I agree that the information theoretic notion of detectability is a strong bound on the extent to which an adversary ignorant of secret keys is capable of detecting the watermark, it seems very pessimistic for at least two reasons.  First, the approach studied in Moitra & Golowich is with respect to computational indistinguishability, which allows for TV to be large as long as witnessing this gap is computationally hard.  Second, in order to take advantage of the TV gap, the adversary would need paired watermarked and unwatermarked generations from the same prompt, which seems unlikely.  Even zooming out and allowing watermarked and unwatermarked generations from different prompts, the adversary would likely not have access to unwatermarked generations from the same model.

Third, I am confused by the result in Theorem 1 for both biased and bias-free bounds.  The right hand sides of these equations seem like they are random in taht both g_t and p_t depend on the (random) history up until that point.  The left hand sides are not random.  Can the authors explain what is going on here?

Fourth, while I appreciate the difficulty of making rigorous the notion of robustness beyond edit-distance, I think realistic attacks consist of paraphrasing, not token-level approaches and it would be nice for the authors to comment on this.

Fifth, I wonder if the authors can comment on watermarks beyond the sampling approaches, such as those that imbed the watermark directly into the model weights; an example of such is *GaussMark: A Practical Approach for Structural Watermarking of Language Models*; again I am concerned that the two point hypothesis testing framework does not adequately describe the notion of detectability required.

Finally, the empirical results in Figure 2(a) suggest that the theoretical results do not even qualitatively describe the empirical realities; the bounds in Theorem 1 are all concave in $T$ but the empirical trend appears to be convex in the same.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper considers trade-offs between LLM watermark detectability and robustness in the setting where all parties have unbounded computation.

### Strengths
Unfortunately I don't think I have anything to put here. Maybe I misunderstood something and the rebuttal will change my mind.

### Weaknesses
The first result is "We first prove that detectability is determined solely by the sampling strategy, not the model architecture."
As far as I understand it, this is less of a "result" and more of an obvious consequence of the way they've set things up: Of course if you embed the watermark by biasing distribution D, then the detectability is determined solely by D...
I guess this result is formalized in Theorem 1, where they also appear to argue that any watermark which can be detected with the secret key can also be detected without. This is true information-theoretically, in the same way that encryption is impossible information-theoretically.
It appears to be a big misunderstanding: The whole point of using computational assumptions, as in the work of Aaronson, Christ et al., Zamir, Golowich & Moitra, etc. is that you can evade this trade-off. Appendix C.4 appears to be saying "if you don't change the distribution then you can't detect," which is not even relevant.

This issue then translates to their second main result, Theorem 2, where they state a "stealth vs robustness" trade-off. Again, these kinds of arguments appear to be based on a fundamental misunderstanding about computational assumptions.
And for the "detectability vs robustness" part, they're basically trying to show limits on the capacity of the edit channel. This problem has been studied before, and MUCH more is known about it than what is proven in this paper.
For instance, it is known how to construct error correcting codes that tolerate eps edits with rate 1 - O(eps): See https://arxiv.org/pdf/1710.09795.

Their bound says that the information rate should be at most (1-eps)^2, but it is already known that the information rate can be at most (1-H(eps)) / (1-eps), where H is the binary entropy function: https://arxiv.org/pdf/2107.01785v3. This is already a much better bound for eps < 1/2, and for larger eps it is not possible to do error correction at all.

### Questions
The first result is "We first prove that detectability is determined solely by the sampling strategy, not the model architecture."
But what they mean appears to be just that, if you embed a watermark by biasing the distribution at sampling time, then the only thing that matters is the distribution at sampling time. How is that not completely obvious?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a rigorous information-theoretic framework to characterize the "Catch-22" in LLM watermarking, a fundamental **trilemma** between **robustness**, **low detectability**, and **reliable verification**.

The authors establish a hierarchy of **detectability** (quantified by Total Variation distance) based *solely* on the sampling transformation (Theorem 1), proving scaling bounds for all four analyzed strategies: Greedy ($O(1)$), Biased ($O(|\delta|\sqrt{T})$), Bias-free ($O(\sqrt{T})$), and Distribution-preserving (0).

**Robustness** is then characterized using information capacity (Theorem 2). The core mechanism revealed is that the available information budget $C(\epsilon)$ **contracts quadratically** (specifically, $C(\epsilon) \approx T(1-\epsilon)^2 D_0$) with the edit rate $\epsilon$. This mathematical mechanism proves the inverse relationship: achieving high robustness (requiring a high initial information budget $D_0$) inherently necessitates high detectability (which, per Theorem 1, also scales with the parameters that increase $D_0$).

Based on these constraints, the authors derive a **Pareto-optimal hybrid watermarking scheme** (Theorem 3). This is not a simple heuristic switch, but an optimal construction derived by minimizing a **composite loss function ($\mathfrak{L}$)** that jointly optimizes for target detection power ($1-\beta$), stealth constraints ($\tau, M$), and parameter amplitude.

Experimental validation on Llama 2 7B and Mistral 7B using paraphrasing attacks confirms these theoretical bounds and, crucially, demonstrates that the proposed hybrid scheme **uniquely traces the Pareto-optimal frontier across all noise regimes**, outperforming any fixed scheme.

### Strengths
1. **Rigorous theoretical foundation**: The information-theoretic framework with formal proofs (Theorems 1-3, Lemmas 1-5) provides principled understanding beyond empirical observations.

2. **Comprehensive experimental validation**: Table 1 systematically evaluates multiple watermarking families (biased, bias-free, distribution-preserving) across two models and multiple attack scenarios, confirming theoretical predictions.

3. **Breadth of analysis**: Coverage spans greedy, biased, bias-free, and distribution-preserving sampling with unified treatment.

4. **Novel impossibility result**: Corollary 1 establishes fundamental limits showing the trilemma cannot be circumvented by clever engineering.

### Weaknesses
1. **Limited attack diversity**: Only paraphrasing attacks (DIPPER, OPT-2.7B) are evaluated. Missing: synonym substitution, back-translation, model-based attacks, and adversarial prompting attacks from Liu et al. (2025) cited in the paper.

2. **Independence assumption not validated**: The edit channel assumes i.i.d. token substitution (Eq. 43), but real paraphrasing introduces semantic dependencies. No empirical validation that this approximation holds or analysis of when it breaks down.

3. **Incomplete hybrid scheme specification**: Theorem 3 provides allocation rules but lacks concrete algorithm for runtime edit rate estimation $\hat{\epsilon}$, which is critical for deployment. The paper acknowledges this (Impact Statement) but doesn't address it.

### Questions
1. **Edit channel validity**: Can you provide empirical validation that real paraphrasing attacks approximately satisfy the i.i.d. edit assumption? What is the distribution of actual edit patterns in DIPPER outputs?

3. **Multi-key scenarios**: Theorem 1 analyzes fixed-key bias-free schemes. How does detectability change if the adversary observes outputs from multiple different keys?

4. **Comparison with Moitra & Golowich**: You mention their scheme requires exponential vocabulary (Section 2). Can you clarify the precise relationship between your impossibility results and theirs?

### Soundness
3

### Presentation
3

### Contribution
3
