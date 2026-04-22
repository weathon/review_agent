# Learning from Synthetic Data Improves Multi-hop Reasoning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Reinforcement Learning (RL) has been shown to significantly boost reasoning capabilities of large language models (LLMs) in math, coding, and multi-hop reasoning tasks.
However, RL fine-tuning requires abundant high-quality verifiable data, often sourced from human annotations, generated from frontier LLMs, or scored by LLM-based verifiers.
All three have considerable limitations: human-annotated datasets are small and expensive to curate, LLM-generated data is hallucination-prone and costly, and LLM-based verifiers are inaccurate and slow.
In this work, we investigate a cheaper alternative: RL fine-tuning on _rule-generated synthetic data_ for multi-hop reasoning tasks.
We discover that LLMs fine-tuned on synthetic data perform significantly better on popular real-world question-answering benchmarks, despite the synthetic data containing only fictional knowledge.
On stratifying performance by question difficulty, we find that synthetic data teaches LLMs to _compose knowledge_---a fundamental and generalizable reasoning skill.
Our work highlights rule-generated synthetic reasoning data as a free and scalable resource to improve LLM reasoning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper investigates an important and timely question: whether fine-tuning Large Language Models (LLMs) on synthetic data through reinforcement learning, in the absence of real-world knowledge, can improve their multi-hop reasoning capabilities on real-world question-answering (QA) tasks. The authors use two synthetic datasets: GSM-∞ (math word problems) and PhantomWiki (fictional knowledge-base QA), to fine-tune several open-source models of varying scales (Qwen series, Phi-4-mini-reasoning). The experimental results show that despite the synthetic data having no factual overlap with the real-world evaluation benchmarks (such as HotpotQA, 2WikiMultihopQA, and MuSiQue), the fine-tuned models achieve significant performance improvements on these real-world tasks. The authors argue that this improvement stems from the model learning a transferable meta-skill—"knowledge composition," the ability to chain multiple logical inference steps. The study also finds that this performance gain is consistent across different model families and sizes, and does not suffer from overfitting as training data increases.

### Strengths
1. As high-quality human-annotated data becomes increasingly scarce, exploring the value of synthetic data is a frontier direction in the LLM field. This paper explores the fundamental question of whether reasoning abilities can be learned independently of factual knowledge, which has significant implications for the training strategies of Large Language Models (LLMs).
2. The use of completely disjoint synthetic and real-world datasets effectively controls for memorization, providing clearer evidence for skill transfer.
3. The paper not only reports final results but also provides an in-depth analysis of the model's learning process by examining performance changes during training and stratifying performance by question difficulty. In particular, Figure 3 and Figure 5 clearly demonstrate how the model's progress on more difficult synthetic questions translates to performance improvements on real-world tasks.
4. The paper is well-structured and flows logically from introduction to conclusion. The research motivation, methodology, and results are all clearly articulated. The figures (especially Figure 2 and Figure 5) intuitively present the core findings and are easy to understand.

### Weaknesses
1. The training is conducted exclusively on synthetic data. Although this shows improvement in real-world scenarios, the paper lacks a comparison with a baseline trained on real-world data and tested on real-world data. There is no analysis of the potential performance gap between training on synthetic data versus real-world data.
2. The experimental models are relatively small, with the largest being 4B. It would be beneficial to see experiments on models of at least 7B parameters. Based on Figures 2 and 3, the performance improvement for Qwen3-1.7B is notably smaller than that for Qwen3-0.6B. Could this be because the 0.6B model has weaker baseline reasoning abilities, and thus training naturally improves its generalizable reasoning performance, while the larger 1.7B model shows diminished gains? What would happen with an even larger model, such as a Qwen3-8B? Would the improvement be minimal?
3. The paper states that 3 (for GSM-∞) or 11 (for PhantomWiki) CoT examples are used during RL training. Is it necessary to include these CoT examples in the prompt during inference as well? Furthermore, it would be desirable to see experiments on the training and generalization effects in a Zero-Shot setting, i.e., without including any CoT examples.
4. In Figure 2, the training results on GSM-∞ are consistently worse than on PhantomWiki for nearly all models. Is there a deeper explanation for this? Is it because GSM-∞ focuses more on mathematical reasoning rather than the type of multi-hop reasoning found in the evaluation tasks? If so, does this raise concerns about the generalizability of the reasoning skills being learned?

### Questions
See weaknesses

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates whether RL fine-tuning on purely synthetic data constructed to require multi-hop reasoning but to contain no real-world facts can teach a general skill of knowledge composition that then transfers to established, real benchmarks. Concretely, the authors fine-tune small LLMs  with GRPO on two synthetic sources. Experiments show that RL on synthetic data improves performance on all three downstream datasets and across model families and sizes.

### Strengths
* I appreciate that the paper presents a tight experimental design around a single, interpretable hypothesis: i.e., training on universes that are explicitly non-overlapping with real-world knowledge, the study isolates whether multi-hop structure learned in synthetic settings can carry over.

* The paper also tackles an important topic in reasoning/RL, namely the effort to disentangle answer formatting from reasoning.

* I also like how the paper shows curves over checkpoints and stratifying performance by question difficulty (number of hops in PhantomWiki and number of operations in GSM infinite), which provides a richer picture than just showing single endpoint scores.

### Weaknesses
* Although the empirical story is neatly organized, in my view, the novelty is modest given the rapidly expanding literature on synthetic data and RL for reasoning.
  * If I remember correctly, PhantomWiki itself was introduced as an on-demand synthetic universe generator to test reasoning and retrieval while sidestepping data leakage; it feels more like this paper leverages that dataset rather than advancing the generation framework. Likewise, GSM-infinite was created to probe reasoning under controllable arithmetic complexity and long contexts, and here it serves as a training curriculum rather than as a novel contribution.
  * Similarly from the methodology side, the paper’s RL component employs GRPO but it seems like without much methodological innovation; recent works (a la Deepseek) have shown to a degree that RL alone can elicit sophisticated reasoning behaviors without human step-by-step traces and RLVR can better incentivize process-correctness. The present study feels like a combination and transfer evaluation rather than as a new algorithm etc. The finding of using synthetic data to help with reasoning doesn't seem particular novel to me either.

* If I understand correctly, one of the paper's claims is that knowledge composition is the specific causal skill that helps with performance; in terms of evidence, the paper infers this composition skill mainly from higher F1 values on 2-4 hop datasets and from difficulty-stratified curves; however, I don't see where the paper verifies whether the intermediate steps followed by the model are logically correct and path-faithful?

* While the focus is on RL, I don't see any other baselines (e.g., SFT) on the same synthetic datasets: how do those compare? Without some of these comparisons, it's unclear to me if gains are from RL itself or via additional supervised exposure to reasoning-style data (or something else). The models used seem to also all be sub 4B; I'm not sure how transferable these findings are in generality (e.g., to even 7/8B models or eve 3-4B modes outside of phi-mini) especially since results show that different models show different degrees "malleability" to RL (but the paper defers analysis to future work). 

* I see that the evaluations are only on 3 datasets (HotpotQA, 2Wiki, MuSiQue); furthermore, results are done on sub-samples ($n=500$ with two seeds if I understand correctly; there is no re-sampling across multiple draws or paired testing. Combined with RL and small models (with potentially high variance), this seems quite small. I wonder how robust the gains shown here are to actual larger samples, increased repeated rates of sampling, etc.

### Questions
See weaknesses.

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
3

### Summary
This paper examines whether large language models can acquire general reasoning abilities solely from synthetic data, without relying on real-world knowledge.
Using reinforcement learning on fully artificial datasets such as PhantomWiki and GSM, the authors show significant performance improvements on real-world multi-hop QA benchmarks.
The results suggest that reasoning skills, such as knowledge composition, can transfer across domains, highlighting synthetic data as a scalable alternative to human-annotated reasoning datasets.

### Strengths
The study demonstrates that large language models can acquire generalizable reasoning skills purely from synthetic, knowledge-free data. It provides empirical evidence that these synthetic reasoning abilities transfer to real-world multi-hop QA tasks, achieving substantial performance gains. The approach offers a scalable and cost-effective framework for improving reasoning through verifiable, automatically generated training data.

### Weaknesses
- The experiments are limited to multi-hop QA. Even though the training data come from a synthetic world, the fact that performance improves on other multi-hop QA benchmarks is not particularly surprising.
- The applicability of the approach to grammatically or semantically complex real-world texts remains unknown.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper shows that RL fine-tuning (GRPO) on purely synthetic multi-hop datasets (GSM-infinity, PhantomWiki) improves LLM performance on real-world QA benchmarks (HotpotQA, 2Wiki, MuSiQue) by teaching a transferable “knowledge composition”.

### Strengths
- Provides clear empirical evidence that RL fine-tuning on synthetic datasets (PhantomWiki, GSM-infinity) improves LLM multi-hop reasoning on real-world QA benchmarks.

- Addresses a practical problem, the scarcity and cost of high-quality human annotated, which the paper suggests can be supplemented or replaced by synthetic reasoning data.

- Demonstrates consistent performance gains across multiple model families and parameter scales, indicating robustness and generalizability.

- The experimental details and use of open-source technologies (models and codebase) makes the setup reproducible.

### Weaknesses
- The paper’s novelty is limited as prior works have already shown that synthetic data and SFT/RLVR for reasoning works quite well. The contribution is primarily about a different reasoning setup of multi-hopping. 

- The domain of synthetic data is narrow, focusing only on arithmetic and relational reasoning, which limits claims of general reasoning transfer.

- The evaluation datasets lack diversity. HotpotQA, 2WikiMultihopQA, and MuSiQue are all two-hop or near two-hop QA tasks, reducing the strength of the generalization claim.

- The paper is unnecessarily verbose at places. The related work section covers too much ground that's not relevant. Stating GRPO equations in Section 3.2 was not really necessary.

Minor nit:
- reinforcement -> Reinforcement
- L128: “complicated, RL-based framework” -> why complicated?

### Questions
-  In the abstract, there's this phrase "“high scoring latency” which is not clear to me. Can you please explain?

- Is there any reason to prefer RL over SFT in this setup?

### Soundness
3

### Presentation
2

### Contribution
2
