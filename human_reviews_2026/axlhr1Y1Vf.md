# In-Context Learning Without Copying

- Avg Score: 3.00
- Decision: Reject
- Scores: 0, 2, 8, 2

## Abstract
Induction heads are attention heads that perform inductive copying by matching patterns from earlier context and copying their continuations verbatim. As models develop induction heads, they often experience a sharp drop in training loss, a phenomenon cited as evidence that induction heads may serve as a prerequisite for more complex in-context learning (ICL) capabilities. In this work, we ask whether transformers can still acquire ICL capabilities when inductive copying is suppressed. We propose Hapax, a setting where we omit the loss contribution of any token that can be correctly predicted by induction heads. Despite a significant reduction in inductive copying, performance on abstractive ICL tasks (i.e., where the answer is not contained in the input context) remains comparable and achieves higher accuracy than the vanilla model on 12 out of 18 statistically significant tasks, even though 31.7\% of tokens are omitted from the loss. Furthermore, our model achieves lower loss values on token positions that cannot be predicted correctly by induction heads. Mechanistic analysis further shows that models trained with Hapax develop fewer and weaker induction heads but still preserve ICL capabilities. Taken together, our findings indicate that inductive copying is not essential for learning abstractive ICL mechanisms.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The authors study a scheme to discourage induction head formation while training a Transformer model. The authors assert that training with this scheme may benefit (or at least, not harm) a model's ability to perform well on "abstractive" ICL tasks.

### Strengths
The authors study an interesting topic, and forward a mildly provocative proposal that induction heads are unnecessary for ("abstractive") ICL.

### Weaknesses
It's unclear to what extent the authors' experiments support their assertions. Their statistical methodology is unusual to me, and their mechanistic claims are vague. In its present state, I am skeptical of their results. Please see Questions below for more details.

Even supposing their results are sound, I'm unsure how to judge the significance of their work. While it's a curious fact that suppressing copying during training can nonetheless leave some ICL capabilities intact, it appears that the large majority of ICL tests do suffer (looking at tables 1, A1, and A2). Will applying this loss scheme boost downstream performance in language modeling, logical reasoning, agentic applications, etc.? Is there a broader application for these results? If not, do these results shed more light on how Transformers operate? Your mechanistic analysis seems to focus on detecting whether there are copying heads remaining in your Hapax Transformer, rather than investigating how the Hapax Transformer can perform ICL tasks at all. If the primary goal is to suggest that copying is inessential to ICL, do you have a better understanding of how ICL can be implemented in the absence of copying?

### Questions
- Simultaneous hypothesis testing typically requires stricter differences / broader confidence intervals to judge statistical significance (see e.g. https://en.wikipedia.org/wiki/Multiple_comparisons_problem). For example, if you run 20 simultaneous tests, chances are you will have 1 statistically significant result at p = 0.05 even if the underlying differences are not significant. Did you take into account simultaneous testing when summarizing your results?
-  Assuming all measures of statistical significance are genuine, Table 1 seems to indicate that only 9 out of 23 "abstractive" ICL tests are significantly better when using Hapax (and 6 are significantly worse). I'm unsure if this supports your assertion that Hapax is better for abstractive ICL?
- Over how many random seeds are your CI's generated from?
- I'm unfamiliar with McNemar's test, but a quick Google search seems to suggest that McNemar's test is used for paired 2 x 2 contingency tables (https://en.wikipedia.org/wiki/McNemar%27s_test). Are your data interpretable as paired 2 x 2 contingency tables? Why not use a traditional t-test?
- Section 4.3: I'm unsure what the takeaway is here. You appear to be claiming that ICL performance remains strong in the Hapax model using this token loss metric. However, you also acknowledge on Line 308 that this metric does not correlate well with ICL performance. So why use it at all?
- Section 5: also unsure what the takeaway is here. Why not study mechanistically why Hapax performs well on the abstractive subset of ICL tasks? It's quite plausible that your loss formulation suppresses copying (Figure 3 is quite convincing already), so I'm unsure why you need a mechanistic analysis of whether Hapax is forming induction heads?
- Figure 8: it's unclear to me whether these induction head scores are above chance. You seem to suggest on Line 431 that 0.10 thresholds the top 5 prefix matching heads from the vanilla model in Figure 6a. On inspection, however, the top 5 in Fig. 6a seem to be thresholded by ~0.9. A score of 0.10 seems to be close to the background chance level, based on your color scale? The color scale on Figure 8 seems to exaggerate the strength of the prefix scores in your random model?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors train models with a modified next-token loss, which aims to mask repetitions in order to investigate ICL in models that are disincentivized to form induction heads.

### Strengths
- The authors introduce a novel framework to investigate ICL in a setting where induction head formation is discouraged
- I believe the question of what transformers can learn when they are disallowed from forming induction heads is interesting and may lead to the discovery of important transformer circuits 
- The Hapax scheme the authors propose is an interesting lens into the formation of induction heads and what signals are required for such circuits to form, I encourage the authors to continue with this work

### Weaknesses
- It is not clear to me that input embedding similarity is the right way to resolve repetitions in text that are tokenized differently. 
    - Tokens may have similar embeddings despite representing distinct strings
    - It is not clear to me (even after reading appendix B) that the threshold chosen by the authors ($\tau = 0.3$) effectively suppresses repetitions in natural text. 
        - I cannot find experiments testing the effect of $\tau\not=0.3$, or experiments that show that embedding cosine similarity is an effective measure of token string similarity (For one thing, Figure A11 shows only the *average* edit distance, and not the full distribution, nor any notion of concentration around the mean)
- I am not sure I agree with the claim that Hapax performs better on abstractive ICL tasks. 
    - In table 1, the vanilla model still performs better on a significant fraction of the tasks, and although the authors do perform a statistical analysis that analyzes the accuracy distribution of each model trained, the results are mixed enough to call into question whether a different vanilla model trained with a different random seed might perform better than Hapax on many of the tasks here. 
        - I.e. the authors characterize statistical fluctuations only over the performance of one model, not over the distribution of models -- while I recognize that characterizing the full distribution is probably computationally prohibitive, the results for a single model are inconclusive enough that I do not feel comfortable with the statement "Hapax improves abstractive ICL"
    - The authors also note that thresholded Hapax performs worse than the vanilla model: this could be because of an improperly chosen threshold (see my earlier point), or because the distribution over models trained via Hapax is sufficiently noisy as to produce inconsistent results. In either case, I believe further investigation is necessary.

### Questions
- Can the authors provide more concrete evidence that $\tau=0.3$ is a good choice of threshold? Both in the sense of downstream abstractive ICL performance, and with respect to masking duplicate tokens without masking non-duplicate tokens that may have significant embedding similarity?
    - The fact that thresholded Hapax with $\tau=0.3$ shows poor performance on abstractive ICL tasks calls into question whether the threshold was correctly chosen. 
- Can the authors make precise their claim that Hapax performs better on abstractive ICL tasks? Currently the results are noisy enough that I am not sure that this claim would hold if one were to retrain Hapax models with different seeds.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper revisits a central assumption in mechanistic interpretability: that induction heads, which perform inductive copying of token sequences, are the foundation of in-context learning (ICL) in transformers. The authors introduce HAPAX, a novel training regime that removes the loss contribution of any token that could be predicted by induction heads—that is, any repeated n-gram within a context window. This loss masking prevents the model from receiving gradient signals for repeated sequences, thereby suppressing the incentive to learn inductive copying. Despite a 31.7% reduction in effective training tokens, HAPAX models maintain and often improve performance on abstractive ICL tasks, while performing worse on purely extractive, copy-based tasks. Mechanistic analysis shows that HAPAX models develop fewer and weaker induction heads but preserve contextual reasoning and even generate more fluent text. These results demonstrate that inductive copying is not necessary for abstractive in-context learning.

### Strengths
The paper’s originality lies in directly challenging one of the most established causal hypotheses about ICL. Rather than building another interpretability tool, the authors use a simple yet powerful experimental manipulation to test whether transformers can still learn ICL without explicit copying. This approach transforms a long-standing correlational observation into a causal experiment. The outcome is both surprising and illuminating: models deprived of copying signals still learn and sometimes excel at abstract reasoning tasks. Conceptually, this is an important reframing of how context learning emerges in large language models.

The technical and methodological quality is high. The HAPAX regime is clearly defined, with precise mathematical formalization of the masked loss and a thoughtful extension to similarity-thresholded masking. The results are carefully interpreted: the authors separate extractive versus abstractive tasks, use token-loss difference metrics to analyze contextual dependence, and provide attention-level evidence that induction heads weaken under HAPAX training. The work is also clear and well-presented, with figures and tables that clearly communicate both behavior-level and mechanistic results. Its significance is substantial, as it undermines the simplistic “ICL = copying” view and encourages a richer understanding of transformer learning dynamics.

### Weaknesses
The scope of empirical validation is somewhat limited. Experiments use 1B-parameter GPT-NeoX models trained on The Pile, which is appropriate for controlled analysis but smaller than models where ICL phenomena are most pronounced. It remains unclear whether the same findings generalize to multi-billion-parameter transformers or to multi-layer SAEs and circuit configurations used in interpretability research. The masking strategy targets repeated n-grams, which suppresses literal copying but not necessarily semantic or structural repetition, meaning that some forms of inductive behavior may persist.

### Questions
1. To what extent would the results scale to larger models such as 7B or 13B transformers? Does suppressing inductive copying remain beneficial for abstractive ICL at higher capacities?
2. Since the paper argues that abstractive ICL arises independently of induction heads, have the authors identified any alternative head types or intermediate representations that correlate with abstractive task success?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates whether **inductive copying**, often attributed to induction heads, is necessary for in-context learning (ICL).  
The authors propose **HAPAX**, a training regime that masks the loss contribution of repeated n-grams within each context.  
By excluding gradient signals from positions predictable through repetition, HAPAX aims to prevent the model from learning copy-based induction circuits.  
Empirically, HAPAX models maintain—and sometimes improve—their performance on abstractive ICL tasks, while showing clear degradation on extractive or copy-heavy ones.  
Mechanistic analysis indicates fewer and weaker prefix-matching heads, suggesting that inductive copying is not essential for abstractive ICL.

### Strengths
**Interesting empirical perspective.**  
  The paper introduces a creative intervention—masking losses on repeated n-grams—to examine how removing copy-related learning signals affects in-context learning.  
  This approach provides a valuable window into how specific training signals shape internal circuits, offering an original empirical angle rather than a purely conceptual contribution.

- **Clear experimental setup.**  
  The masking mechanism and evaluation procedure are defined precisely and are easy to replicate.

- **Mechanistic interpretability link.**  
  The analysis connects observed behavioral changes to head-level attention patterns, providing interpretable signals about circuit adaptation.

### Weaknesses
### 1. Conceptual framing may mislead rather than innovate
The central claim—that *inductive copying is not essential for ICL*—could be interpreted as overturning a previously dominant view that induction heads are the sole foundation of ICL.  
In practice, the field already recognizes that induction heads support certain ICL behaviors but are not the only mechanism.  
Thus, the framing could **unintentionally give the impression** that the paper refutes a consensus that did not exist.  
The genuine novelty lies in the **empirical intervention itself**—examining how the removal of repetition-related gradients alters model behavior and circuit formation.  
Reframing the work around this insight, rather than as a conceptual correction, would make the contribution clearer and more accurate.

### 2. Induction heads are reduced, not removed
The analysis shows that prefix-matching heads become fewer and weaker but not absent.  
The framing could more explicitly emphasize this partial suppression to avoid readers inferring complete removal.

### 3. Ambiguity in how the masking rule affects learning
HAPAX masks all token positions where a bi-gram has appeared earlier in the same context.  
This means only the first occurrence contributes to the loss.  
While this design reduces gradients from repetition, it still exposes the model to repeated patterns and may unintentionally **encourage single-exposure learning**, where the model must internalize a pattern from its first appearance.  
The results demonstrate that HAPAX changes how repeated patterns contribute to learning, but it remains unclear whether this constitutes a genuine reduction in memorization or simply a shift in its form.  
Clarifying this distinction—between preventing repetition-based learning and altering how memorization occurs—would make the causal argument more precise.

### 4. The link between “copy suppression” and “induction suppression” is theoretically underspecified
Induction heads are not mere copy circuits—they implement a retrieval mechanism that associates similar contexts \(X_q\) and \(X\) to predict the continuation \(Y\).  
Masking repeated n-grams blocks gradients for exact duplication but does not necessarily disrupt this broader retrieval mechanism.  
Thus, while the intervention reduces surface-level repetition, it does not clearly isolate the mechanisms that drive generalization or memory formation.  
A more detailed discussion of what aspects of induction behavior the masking targets (e.g., exact copying vs. contextual retrieval) would clarify interpretation.

### Questions
1. Does the mask exclude every *second and later* occurrence of a bi-gram globally, or only within a local window?  
   Could this setup unintentionally promote single-exposure learning instead of preventing memorization?  
2. How do you interpret the reduction in prefix-matching heads—does it reflect fewer copy circuits or simply less training on repeated n-grams?  
3. Have you checked whether the model continues to exhibit *semantic* or *fuzzy* induction behaviors despite masking?  
4. Would removing repeated n-grams entirely from the input, rather than from the loss, yield a clearer causal result?  
5. In what sense do you consider HAPAX “without copying,” given that repetition remains visible to the model?

### Soundness
2

### Presentation
2

### Contribution
1
