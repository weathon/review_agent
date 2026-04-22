# Improving Consistency in Retrieval Augmented Systems with Group Similarity Reward

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
RAG systems are increasingly deployed in high-stakes domains where users expect outputs to be consistent across semantically equivalent queries. However, existing systems often exhibit significant inconsistencies due to variability in both the retriever and generator (LLM), undermining trust and reliability. In this work, we focus on \emph{information consistency}—the requirement that outputs convey the same core content and information across semantically equivalent inputs. We introduce a principled evaluation framework that decomposes RAG consistency into retriever-level, generator-level, and end-to-end components, helping identify inconsistency sources. To improve consistency, we propose  \textbf{P}araphrased \textbf{S}et Group Relative Policy Optimization (PS-GRPO), an RL approach that leverages multiple rollouts across paraphrased set to assign \emph{group similarity rewards}.   We leverage PS-GRPO to achieve Information \textbf{Con}sistent \textbf{RAG} (Con-RAG), training the generator to produce consistent outputs across paraphrased queries and remain robust to retrieval-induced variability. Because exact reward computation over paraphrase sets is computationally expensive, we also introduce a scalable approximation method that retains effectiveness while enabling efficient, large-scale training. Empirical evaluations across  short-form, multi-hop, and long-form QA benchmarks demonstrate that Con-RAG significantly improves both consistency and accuracy over strong baselines, even in the absence of explicit ground-truth supervision. Our work provides practical solutions for evaluating and building reliable RAG systems for safety-critical deployments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the issue of information inconsistency in Retrieval-Augmented Generation (RAG) systems, where semantically equivalent queries (paraphrases) often yield inconsistent answers due to variations in both retrieved evidence and generator sensitivity. To tackle this, the authors define and measure consistency at three levels: retriever, generator, and end-to-end. They propose PS-GRPO (Paraphrased Set Group Relative Policy Optimization), a reinforcement learning method that encourages consistent outputs across paraphrases of the same query using a group similarity reward, with scalable sampling-based approximations. The resulting model, Con-RAG, is evaluated on short, multi-hop, and long-form QA datasets (TriviaQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, ELI5), showing significant improvements in both generator- and end-to-end consistency while maintaining or improving factual accuracy. Even without explicit ground-truth supervision, Con-RAG enhances factual alignment as judged by LLM-based evaluators, highlighting its robustness for reliable RAG applications.

### Strengths
• I like that the paper disentangles the contributions of the retriever, generator, and overall pipeline to consistency — it really helps clarify where inconsistency comes from. The analysis also shows that paraphrasing and retrieval shifts have only a limited effect on final answer correctness on average, which is an important and somewhat reassuring finding.

• A particularly interesting result is on long-form QA (ELI5): even without explicit ground-truth supervision, Con-RAG significantly improves both end-to-end consistency and LLM-judged factual accuracy. That’s quite impressive and suggests the optimization signal generalizes well beyond short-answer settings.

• I appreciate that the paper treats consistency as more than just lexical overlap. By centering on information consistency and combining lexical and LLM-judge metrics, the authors offer a much more robust perspective. The fact that optimizing for consistency can jointly improve factuality (as reflected in LLM-Acc) is a strong and nuanced contribution.

### Weaknesses
• The experimental setup feels somewhat limited in terms of model diversity and scale. It’s unclear whether the observed gains would hold — or even become more pronounced — with larger or more capable base models. Including results on stronger LLMs would make the conclusions more convincing.

• I’m not fully convinced about how novel PS-GRPO is. The main innovation seems to lie in computing the n×g similarity matrix, but conceptually this could be simplified to a scalar 0/1 reward—treating paraphrases closer to the ground truth as 1 and others as 0—then applying standard GRPO training. It would be helpful to clarify what new learning dynamics PS-GRPO introduces beyond this simplification.

### Questions
• The paper shows that even without explicit ground-truth supervision, the method improves end-to-end consistency and LLM-judged factual accuracy, outperforming SFT — but it’s not entirely clear whether this effect is specific to ELI5 or generalizable to other long-form QA tasks. Can the same conclusion extend to short-form or multi-hop settings as well? Some discussion or evidence on generalizability would strengthen the claim.

• While the paper disentangles the retriever and generator effects, it still remains unclear how retriever shifts and LLM sensitivity concretely influence end-to-end consistency. What are their relative contributions? Moreover, the relationship between this sensitivity and final factual accuracy is not clearly analyzed.

• From Table 4, it seems that higher consistency correlates with better accuracy, but it’s unclear how incorrect consistency (i.e., consistently wrong answers) compares to correct consistency. Clarifying this distinction would help interpret whether consistency itself is always desirable or potentially misleading.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Authors study information consistency in RAG, decomposing it into retriever, generator, and end-to-end components, and propose Con-RAG trained with Paraphrased-Set GRPO (PS-GRPO): sample multiple rollouts across paraphrases of a canonical query and reward outputs that are similar within the paraphrase group. They also give a scalable reward approximation (subsampling paraphrases/rollouts) to keep training tractable.

### Strengths
1. PS-GRPO is clearly specified: group-normalized advantages, clipped objective, and an explicit group-similarity reward over paraphrases, with optional accuracy reward when ground truth exists.

2. A useful consistency framework that separates retriever Jaccard, generator consistency under fixed evidence, and end-to-end behavior; this makes ablations actionable.

3. Sensible evaluation choices like deterministic decoding to remove sampling noise and a reward-metric ablation that compares BLEU-n, ROUGE-L, EM as similarity functions.

### Weaknesses
1. Paraphrase generation includes the gold answer in the prompt, which risks target leakage and may inflate consistency by narrowing the paraphrase space.

2. The core reward is lexical (BLEU); optimizing surface overlap is not the same as optimizing information-level agreement

3. The retriever remains the principal bottleneck, yet the method fine-tunes only the generator; no retriever co-training or simple stabilizers are evaluated.

### Questions
1. Section A.2 (L108–121): if paraphrases are generated without gold answers, how much do consistency and accuracy drop? Please quantify across tasks and share examples where answer-priming mattered. 

2. Section 2.2 (L25–41): how sensitive are gains to κ and s in the subsampled reward—what’s the best compute-quality frontier for practitioners with fixed budgets? 

3. Section 2.1 (L182–191): when retriever Jaccard is particularly low, do generator-only gains still lift end-to-end consistency, or do they saturate? Any observed failure modes?

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
This paper addresses the issue of inconsistent outputs in RAG systems when handling semantically equivalent queries. It proposes PS-GRPO, a reinforcement learning method that uses group similarity rewards across paraphrased query sets to encourage consistent generation. The approach is applied to build Con-RAG, a consistency-optimized RAG model.

### Strengths
1.	The proposed relaxed group similarity reward is interesting and effectively reduces computational cost.
2.	The decomposition into retriever-level, generator-level, and end-to-end consistency is elegant and interpretable.

### Weaknesses
1.	There is no detailed analysis provided on why Con-RAG outperforms SFT in short-form QA tasks (Table 2), leaving the underlying factors behind the performance improvement insufficiently explained.
2.	The paper does not explicitly address how to distinguish between diverse-but-correct and inconsistent answers in open-ended QA tasks such as ELI5. The proposed group similarity reward optimizes for surface-level similarity (e.g., BLEU) and lacks mechanisms to capture semantic equivalence, making it unclear whether the observed “consistency” reflects genuine informational alignment or simply reduced diversity. Similarly, the evaluation metrics—largely based on lexical overlap and sparse LLM judgments—are insufficient to rigorously assess semantic consistency in tasks with multiple valid answers. These limitations should be explicitly discussed and empirically analyzed.
3.	The paper does not report comparative information on training time and GPU resources.

### Questions
My questions are listed in "Weaknesses". I am looking forward to clear explanations from the authors. 
I will revise my rating according to the author's feedback and the reviewer's discussion.

### Soundness
2

### Presentation
3

### Contribution
2
