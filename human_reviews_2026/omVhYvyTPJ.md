# LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 2, 4, 8

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced the reasoning capabilities of Large Language Models (LLMs) by optimizing them against factual outcomes. However, this paradigm falters in long-context scenarios, as its reliance on internal parametric knowledge is ill-suited for tasks requiring contextual grounding—the ability to find and reason over externally provided information. We identify a key reason for this failure: a reward based solely on the final answer is too sparse to effectively guide the model for identifying relevant evidence. We formally prove that the outcome-only reward leads to significant vanishing gradients for the context grounding process, rendering learning intractable. To overcome this bottleneck, we introduce LongRLVR to augment the sparse answer reward with a dense and verifiable context reward. This auxiliary signal directly incentivizes the model for selecting the correct grounding information, providing a robust learning gradient that solves the underlying optimization challenge. We validate our method on challenging long-context benchmarks using Qwen and LLaMA models. LongRLVR consistently and significantly outperforms the standard RLVR across all models and benchmarks, e.g., boosting a 14B model's scores on RULER-QA from 73.17 to 88.90 and on LongBench v2 from 39.8 to 46.5. Our work demonstrates that explicitly rewarding the grounding process is a critical and effective strategy for unlocking the full reasoning potential of LLMs in long-context applications. Our code is available at https://github.com/real-absolute-AI/LongRLVR.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on improving long-context reasoning through Rule-based Rewards. They propose a new pipeline to generate data, thereby enabling the use of rule-based rewards to evaluate the evidence used in the LLM output.  Experiments show consistent gains over answer-only baselines, supporting the method’s effectiveness.

### Strengths
The problem is significant, as RLVR may stimulate hallucinations and render the training process unstable, while its sparse rewards make effective exploration challenging in practice. 

The paper addresses the issue of vanishing gradient in RLVR under sparse outcome-reward settings, examining its causes and implications.

The choice of the F1 score as a reward makes sense to me, since it balances precision and recall rather than encourages the model to cover the evidence as much as possible.

The experiments appear to support the authors’ claims and show consistent improvements.

### Weaknesses
However, my concerns arose from the data generation pipeline and the usage of the verifier LLM. 

1. It seems that the method is only applicable for the Grounded QA, where evidence can be cleanly chunked. However, in such a case, performing rule-based rewards for the evidence suggestion should be straightforward. The usage of the F1-score is also straightforward to me, since recall encourages the policy to cover as many chunks as possible.
2. A separate verifier LLM is used, which helps identify the evidence and check its alignments with the evidence library; however, it makes the comparison with RLVR unfair. Moreover, an additional LLM can do more (e.g., directly judge whether the final answer matches the reasoning path). Why not use semantic similarity or other similar metrics?
3. The computational and human costs are nontrivial, both from data collection and the additional LLM verifier. Therefore, I am wondering: do the gains adequately justify the substantial supervision and computation?

### Questions
See weakness

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
4

### Summary
This paper addresses a fundamental limitation of RLVR when applied to long-context reasoning tasks. The authors identify that outcome-only rewards suffer from vanishing gradients for the contextual grounding process. They formally prove this vanishing gradient problem and propose LongRLVR, which augments sparse answer rewards with dense, verifiable context rewards that explicitly supervise evidence selection. The method is validated on RULER-QA, LongBench v2, and LongReason benchmarks, showing consistent improvement against vanilla GRPO and SFT. The approach requires ground-truth evidence annotations, and the authors also propose a data generation pipeline using clustering and rejection sampling.

### Strengths
1. The formal analysis of why the outcome-only reward is insufficient for the long-context retrieval-based task provides some transferable insights.
2. The modulated F-score reward, combining unconditional grounding reward and synergistic success reward, is thoughtfully designed.
3. The paper provides extensive analysis on both synthetic and real-world long-context tasks, and the paper includes thorough ablations examining reward components, data quality, hyperparameters, and chunk number robustness, making the claims more sound.

### Weaknesses
1. The comparison is a bit weak, which hinders the overall soundness of the work. Interleaving reasoning and retrieval is now becoming more popular. I would suggest comparing with some RAG baselines (which do not require RLVR but fit the same scenario), as well as some recent works like [1].
2. Assumption 1 seems too strong for the analysis. In reality, the reward for retrieved evidence, if applied, should be more continuous than the 0 or 1 sparse reward. Also, the independence assumption for chunk selection might not be true since the evidence should be highly related in a multi-hop QA scenario like the ones in LongBench.
3. A human evaluation of the validity of the generated data or some examples provided would be very beneficial.
4. There is a potentially biased evaluation regarding the training data. The vanilla RLVR and SFT baselines are trained on the same generated data with evidence, but they don't have the corresponding training signal, which may introduce bias towards the proposed method.

[1] Wang et al. 2025. Improving Context Fidelity via Native Retrieval-Augmented Reasoning. arXiv:2509.13683.

### Questions
1. Would it be helpful if some existing QA datasets with ground-truth evidence (like HotpotQA) is used partially as the training data or as a seed dataset for the data generation process?
2. Are the chunk identifiers ([CHUNK_N]) added as new special tokens?
3. The useful chunks are generated after the thinking process. Does the model actually use or refer to the chunks during the training process?
4. How does performance degrade with noisy evidence annotations?
5. Can you evaluate on datasets with evidence annotations (e.g., HotpotQA) for the retrieval accuracy?

### Soundness
3

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
2

### Summary
This paper proposes LongRLVR, a reinforcement learning framework for long-context LLMs that introduces verifiable context rewards to overcome vanishing gradients in grounding. Instead of rewarding only final answers, LongRLVR adds dense rewards for correctly selecting evidence chunks, ensuring effective credit assignment across lengthy inputs. It decomposes the policy into grounding and answering heads, uses F-score–based context rewards, and achieves large gains on long-context QA benchmarks, outperforming outcome-only RL baselines and enabling smaller models to surpass larger ones.

### Strengths
1. **Timely and impactful problem.**
The paper tackles a highly relevant and increasingly important issue — how to perform reinforcement learning effectively in long-context settings. As large-context reasoning becomes central to emerging LLM-based agents and search systems, addressing the credit-assignment and gradient-vanishing challenges identified here is both timely and of broad significance.
2. **Strong motivation, clear formulation, and well-executed methodology.**
The study is well motivated and rigorously executed. It formally defines the reward-vanishing problem in long-context RL, provides theoretical analysis to explain why outcome-only rewards fail, and introduces a principled solution through verifiable context rewards. The inclusion of a synthetic yet well-controlled dataset allows precise testing, and the resulting performance gains over baselines are substantial and convincing.
3. **Clear structure and presentation.**
The paper is clearly organized and well written, with intuitive explanations and consistent notation. The conceptual flow, from identifying the issue to formal analysis, method design, and empirical validation, is easy to follow, making the technical contributions accessible and well supported.

### Weaknesses
1. **Strong theoretical assumptions.**
The analysis relies on several simplifying assumptions that may not fully hold in practice. In particular, it adopts an all-or-nothing reward assumption, where the answer reward increases only when the entire evidence set G is selected. In reality, LLMs often produce correct answers from partial or alternative evidence, making this assumption less realistic. Similarly, the independent Bernoulli selection assumption overlooks dependencies between evidence chunks—real policies typically select evidence in a correlated or sequential manner, which could alter the theoretical gradient behavior. It would strengthen the paper to discuss under what scenarios these assumptions are likely to hold (e.g., explicit fact-retrieval tasks) and where they may fail (e.g., open-domain reasoning). Such clarification would help readers understand the practical scope of the theoretical results.
2. **Connection to broader long-context RL not explored.**
This is more like a suggestion. The paper could better relate its formulation and method to other long-context settings, such as agent RL, where the context includes environment state and action history (like adding a discussion section to appendix). Discussing which aspects of LongRLVR’s framework may transfer and which may not would improve generality and impact.

### Questions
Some other gradient vanish work in LLM RL could be discussed in related work. e.g, "Vanishing Gradients in Reinforcement Finetuning of Language Models"

### Soundness
4

### Presentation
4

### Contribution
3
