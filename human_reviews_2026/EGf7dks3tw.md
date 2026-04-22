# Beyond the Rosetta Stone: Unification Forces in Generalization Dynamics

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Large language models  (LLMs) struggle with cross-lingual knowledge transfer: they hallucinate when asked in one language about facts expressed in a different language during training.
This work introduces a controlled setting to study the causes and training dynamics of this phenomenon by training small Transformer models from scratch on synthetic multilingual datasets.
We identify a learning phase wherein a model develops either separate or unified representations of the same facts across languages, and show that unification is essential for cross-lingual transfer.
We also show that the degree of unification depends on how strongly a fact is associated with a particular language, and on how easy it is to identify the language.  
Based on these insights, we develop methods to modulate the level of cross-lingual transfer by manipulating data distribution and tokenization, and we introduce metrics and visualizations to characterize their effects on unification.
Finally, we show that our measures of representational unification correlate with cross-lingual factual accuracy in LLMs, such as Gemma.
Our work shows how controlled settings can shed light on pre-training dynamics and suggests new directions for improving cross-lingual transfer in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies why LLMs hallucinate facts when queried across languages, using a synthetic “Petri dish” to train small Transformers from scratch. It finds that models either unify or separate representations of the same fact across languages during an early training phase, and this unification—predicted by a simple similarity-based score—determines cross-lingual generalization. 
Crucially, separation is driven by how informative and easily extractable the language identity is; 
manipulating monolingual data (balancing attributes, obfuscating language cues) improves transfer without more parallel data. The score also predicts accuracy in Gemma-2B, offering a practical tool for model selection.

### Strengths
1.I think the motivation of this paper is solid: why models hallucinate facts in cross-lingual settings, even when they know the fact in another language. This is especially relevant for low-resource languages.
2. The paper is well-written and easy to understand, the experimental design is logically presented, and the figures effectively illustrate key concepts like the unification score and checker-boarding.

### Weaknesses
1. I think the baseline is a little weak, only on Gemma-2-2B models, which is not a strong LLM, and without enough justification.
Adding experiments on stronger LLMs such as Qwen, llama, could be justify the performance.
2. While the paper shows that parallel data helps, it does not disentangle whether this is due to increased exposure to facts or reduced language informativeness. A more fine-grained ablation (e.g., parallel data with balanced vs. imbalanced attributes) would clarify this.
3.The paper does not compare its findings to existing multilingual alignment methods (e.g., shared subword vocabularies, alignment losses, code-switching). I would like to see the comparison and analysis against these methods.

### Questions
see weakness part.

### Soundness
3

### Presentation
3

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
The authors present a very interesting study into generalisation in LLMs based on internal representations. Specifically, the authors focus on cross-lingual representations and cross-lingual generalisation. Their study focusses on how specific dataset statistics impact that generalisation. They develop a method of predicting this generalisation based on their developed "unification" score.

### Strengths
1. Interesting and detailed analysis into pre-training mixtures and effect on generalisation (specifically in the context of cross-language generalisation)
2. Development of an automatic metric that correlates strongly with task generalisation ("Unification Metric")
3. Development of a synthetic language for such analysis.

### Weaknesses
1. Section 3 is quite hard to follow and the plots are very small and hard to make proper use of. (although the general picture of checkboard vs. not checkerboard gets conveyed.
2. Section 6 is very biased to a single model "Gemma" (that is perhaps from the authors themselves [btw, we are not from a "competing" lab, rather this is a scientific assessment]). Secondly, Section 6 is impossible to reproduce from the very short description - limiting it's meaningfulness for the paper as well as a scientific contribution. Section 6 however, is an important contribution to the argument of the paper (as without it synthetic languages form a major limitation).
3. The "Unifcation" metric is probably not sufficiently described to properly reproduce the results.
4. Overall - reproducibility of the paper is limited and it would be hard to verify the results (or their impact).
5. The (presented) results in Section 6 (even though vague) are much lower than on the synthetic data (65% correlation)

### Questions
1. Could you describe the Unification metric in more detail. How exactly do you calculate it.
2. Could you describe the KG + Method / Code for producing the synthetic datasets in more detail. Specifically, can you share statistics of your KG and the resulting datasets.
3. Could you expand upon section 6. What are the exact training datasets (+ statistics)? How exactly do you evaluate the model "with LLM judge"? (A follow on would be, how accurate is the method?). etc.
4. Why have you not tried other models (incl. those that are fully open source OLMO, Merlin, GPT-like architectures).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the generalization of knowledge across languages. The authors suggest a Petri dish setup, where synthetic knowledge graphs are used to sample bilingual datasets, to control shared information across languages, and then train a tiny transformer to observe the generalization across languages from multiple checkpoints during training. To facilitate the observation, the authors consider Unification Score, a metric to evaluate the difference between representations in two languages. Finally, the authors leverage Unification Score to examine Gemma-2B on the ECLeKTic dataset (Goldman et al., 2025), attempting to verify all findings for a  real LLM.

### Strengths
Understanding and interpreting the emergence of cross-lingual generalization is an interesting question as it helps understand how to effectively train an LLM to support multiple languages, esp. low-resource languages.

### Weaknesses
1. The presentation of this paper is not very good, making this paper vague, unclear, and verbose. For example, in line 258, the authors state “Concretely, the unification score captures the similarity between semantically equivalent cross-lingual datapoints against a baseline of similarity between semantically distinct same-language datapoints.” , but in the following equation, they define “sim(x, y) against sim (x, x)”. Also, I believe including statistics of the synthetic datasets and more experimental setups (e.g., training steps, batch size, and learning curves) could strengthen the paper.

2. The main finding of this paper is that explosibility or frequency of appearance is the key cross-lingual transfer. However, there is a body of studies focusing on this idea. For example, [1] set up a similar experiment by controlling parallel datasets.

3. The last part, 6 LARGE LM EXPERIMENTS, is not convincing and seems disconnected from other findings. There is an important confounding factor that the frequency of appearance is not the same across languages. For example, a word X appears in English 10k times, but 1 time in other languages, thus it will increase the Jaccard similarity but not improve cross-lingual transfer or predict the cross-lingual transfer, according to other experiments in this paper.

[1] Cross-Lingual Transfer of Cultural Knowledge, ACL 2025

### Questions
Refer to Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
