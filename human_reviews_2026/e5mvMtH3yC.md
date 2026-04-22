# Route Experts by Sequence, Not by Token

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Mixture-of-Experts (MoE) architectures scale large language models (LLMs) by activating only a subset of experts per token, but the standard *TopK* routing assigns the same fixed number of experts to all tokens, ignoring their varying complexity. Prior adaptive routing methods introduce additional modules and hyperparameters, often requiring costly retraining from scratch. We propose **Sequence-level TopK (SeqTopK)**, a minimal modification that shifts the expert budget from the token level to the sequence level. By selecting the top $T \cdot K$ experts across all $T$ tokens, SeqTopK enables end-to-end learned dynamic allocation -- assigning more experts to difficult tokens and fewer to easy ones -- while preserving the same overall budget. SeqTopK requires only a few lines of code, adds less than 1\% overhead, and remains fully compatible with pretrained MoE models. Experiments across math, coding, law, and writing show consistent improvements over TopK and prior parameter-free adaptive methods, with gains that become substantially larger under higher sparsity (up to 16.9\%). These results highlight SeqTopK as a simple, efficient, and scalable routing strategy, particularly well-suited for the extreme sparsity regimes of next-generation LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an adaptive routing strategy for sparse mixture-of-experts (MoE) models called Sequence-level Top K (SeqTopK), which replaces the conventional per-token TopK based expert activation budget with a per-sequence budget. Instead of selecting a fixed number of experts for each token, SeqTopK computes a total budget of T×K experts (where T is the number of tokens) and allocates that budget dynamically across all tokens in the sequence - effectively giving more experts to “harder” tokens and fewer to “easier” ones. During training, it is very lightweight and adds only few lines of code with minimal (1%) overhead.

### Strengths
- Clear and well-motivated problem formulation: Identifies that token-level Top-K routing misallocates compute - over-serving easy tokens and under-serving difficult ones.
- Innovative sequence-level budgeting: SeqTopK enables context-aware expert allocation by considering each token’s relative importance within the sequence, not just its standalone score. But its working during inference is not accurate.

- Practical compute bound: Introduces a simple yet effective constraint of minimum and maximum number of experts being selected for each token [1, K+2] to maintain control over compute for all tokens.

### Weaknesses
Online SeqTopK:

- I see a major flaw with the approach proposed for autoregressive decoding during inference. The method is sound during training when we have access to all tokens expert's scores in a sequence. But the methodology in section 3.2 for online SeqTopK is not correct in providing flexible K for different tokens. 

- The method proposes to limit expert selection budget to $m.K$ for every sequence index $m$ where $1<m<T$. But we can not change the past expert selection at any point in time of generation. 
- Lets say we start at seq index $m=1$ then we will select $1.K=K$ experts for current token. Next, we move to generate second token $m=2$, then our budget is to select total $2.K$ experts till $m=2$ sequence index. But since we have already selected $K$ experts for the previous token, we are left to select $K$ experts for the current $m=2$ token as well. And in the same fashion, the method continues. 
- The proposed online SeqTopK effectively collapses to the standard token-level TopK routing as the expert selection budget at each step $m$ cannot retroactively adjust past allocations, each new token still receives exactly $K$ experts. Thus, in an autoregressive setting, the online variant provides no real deviation from per-token TopK behavior.
- There is no experimental results showing otherwise for the online set up. There is one section C in appendix including results of online SeqTopK and it shows similar results as regular TopK.

Other parts:
 - Dependence on sequence length: Unclear how SeqTopK scales to models with different or longer context lengths, especially when training and inference sequence lengths differ.
- Expert allocation bound ([1, K+2]): The rationale for this specific range is unclear—should the lower bound also depend on K, and does it always guarantee exactly T×K experts per sequence?
- Lack of expert score analysis: No detailed study on how sequence-level constraints reshape router distributions or whether fine-tuning affects only the router or all parameters; unclear how tokens with >K experts adjust their distributions.

### Questions
I recommend the authors develop a practical online inference strategy so SeqTopK can adapt expert allocation during autoregressive generation rather than degenerating to per-token TopK.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a context-level MoE routing strategy that assigns experts based on sequences rather than individual tokens. The authors argue that token-level routing, which assigns a fixed number of experts per token regardless of its informativeness, is suboptimal, and that aggregating routing signals across a context window can lead to more coherent and efficient expert utilization. To implement this, the paper introduces the SeqTopK mechanism, including both an offline variant and an online variant designed for non-causal settings. The authors conduct experiments across multiple tasks and benchmarks to demonstrate the performance benefits of their approach, and provide visualizations to analyze expert usage patterns.

### Strengths
The paper presents a clear motivation and introduces the idea of context-level expert routing in a straightforward manner. The writing is concise and easy to follow, and the overall method is described with sufficient clarity. Experiments are relatively comprehensive, covering multiple tasks and including both quantitative results and visualizations to support the analysis.

### Weaknesses
* **Sequence Segmentation Strategy.**: The paper promotes context-level expert routing as a key improvement over token-level MoE, yet it fails to specify how the input sequence is segmented for routing purposes. It remains unclear whether fixed-size chunks, syntactic boundaries, or dynamic criteria are used to define each sequence. In long documents, inappropriate or static segmentation may lead to semantic fragmentation or boundary misalignment, ultimately impairing the coherence and effectiveness of expert selection. Ironically, this may negate the intended advantages over token-level granularity, especially considering that token representations already encode contextual information.

* **Inconsistency Between Motivation and Visualization**. The paper argues that assigning each token a fixed number of experts (K), regardless of its informativeness, is suboptimal—a point with which I strongly agree. As noted by the authors in Lines 54–58, *“some tokens are trivial and require little capacity (e.g., 'the')”*. However, this intuition appears to be contradicted by the empirical evidence in Figure 3(b), which shows that "the" is among the tokens associated with the highest number of experts. To reconcile this discrepancy and substantiate the paper’s motivation, more thorough and quantitative analyses are needed.

* **Limitations of Non-Causal Design**. While I appreciate the motivation behind the Online SeqTopK mechanism and agree that it offers a reasonable solution for non-causal settings, there are two potential concerns that warrant further discussion.
  * (1) The constraint that, at step m, the total number of activated experts must not exceed m·K may significantly limit the flexibility of the routing mechanism—especially in long sequences where semantic dynamics can vary drastically. This rigid upper bound could lead to under-utilization of expert capacity in later tokens, particularly when earlier tokens require fewer experts.
  * (2) It remains unclear how well SeqTopK integrates with standard decoding techniques used in autoregressive language models, such as speculative decoding or beam search. Given the increasingly widespread use of such strategies for efficient inference, a discussion (or empirical validation) of SeqTopK’s compatibility with these paradigms would strengthen the practical relevance of the proposed method.

### Questions
Please refer to Weaknesses.

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
2

### Summary
This paper proposes SeqTopK, a simple modification to standard TopK routing in MoE models that shifts expert budget allocation from the token level to the sequence level. Instead of assigning a fixed number of K experts to each token, SeqTopK selects the top T·K experts across all T tokens in a sequence, enabling context-aware dynamic allocation where harder tokens receive more experts and easier tokens receive fewer, while maintaining the same overall computational budget. The method requires minimal code changes, introduces no additional parameters or hyperparameters, and can be directly applied to pretrained MoE models through fine-tuning. Experimental results across math, coding, legal, and summarization tasks demonstrate consistent improvements over standard TopK routing, with particularly substantial gains (up to 16.9%) under higher sparsity regimes, suggesting SeqTopK is well-suited for next-generation ultra-sparse MoE architectures.

### Strengths
- The paper presents a creative and elegant solution by reframing the routing problem from token-level to sequence-level competition, allowing tokens to adaptively share expert budgets based on their relative difficulty, which appears to be a novel perspective in the MoE literature.

- The experimental evaluation demonstrates reasonable comprehensiveness across multiple base models (OLMoE, Qwen1.5), diverse tasks (math, coding, legal, summarization), and various sparsity levels, with consistent improvements that become more pronounced under higher sparsity conditions, suggesting robust empirical validation.

- The presentation is generally clear and well-structured, with effective visualizations (especially Figure 1's code comparison and Figure 2's token heterogeneity analysis) that help readers quickly grasp the core intuition and the minimal implementation changes required.

### Weaknesses
- The experimental evaluation excludes several recent adaptive MoE methods (e.g., MoE++, DynMoE mentioned but not compared), and the dismissal of these baselines based on architectural differences weakens the claim that SeqTopK represents the best parameter-free adaptive routing approach.

- The paper does not discuss scenarios where SeqTopK might underperform or fail, such as tasks requiring uniform token processing or sequences with atypical length distributions, limiting understanding of the method's applicability boundaries.

- The paper lacks systematic ablation experiments on critical design choices such as the impact of different token-level bound values, the necessity of lower/upper bounds, and how performance varies with different sequence length distributions during training versus inference.

- While Section 3.3 dismisses BatchTopK for language modeling, the experimental comparison in Tables 1-2 only reports "best performance" without showing the full sensitivity analysis or explaining under what conditions BatchTopK might still be competitive or preferable.

### Questions
- Could the authors provide theoretical analysis or formal guarantees explaining why sequence-level routing should outperform token-level routing? Additionally, can you identify and discuss specific scenarios or task characteristics where SeqTopK might underperform compared to standard TopK (e.g., sequences requiring uniform computation across all tokens, or tasks with highly variable sequence lengths)?

- he paper enforces per-token bounds (at least 1 expert, at most Ktok+2 experts) to prevent degenerate allocations. Could you provide systematic ablation studies showing: (a) performance with different bound values, (b) what happens without these bounds across different tasks, and (c) how sensitive SeqTopK is to sequence length variations during training versus inference?

- While the paper mentions MoE++ and DynMoE but excludes them from comparison due to architectural differences (additional zero/gate experts), could you clarify whether SeqTopK's gains would persist when compared against these methods? Additionally, for the BatchTopK comparison, could you provide full sensitivity analysis across different batch sizes rather than only reporting "best performance," to better understand when BatchTopK might be competitive?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SeqTopK, a context-level routing strategy for Mixture-of-Experts (MoE) models that allocates experts across sequences rather than individual tokens. The authors argue that traditional token-level TopK routing is suboptimal due to its fixed expert budget per token, and propose SeqTopK to enable dynamic, context-aware expert allocation while maintaining the same overall computational budget. The method includes both offline and online variants, the latter designed for autoregressive decoding via an "Expert Cache."

### Strengths
The idea of shifting expert allocation from token-level to sequence-level is both intuitive and innovative. SeqTopK is implemented with minimal code changes, introduces no new parameters, and maintains full compatibility with pre-trained MoE models, making it practical. The paper provides comprehensive experiments across multiple domains and model architectures, consistently showing performance gains over TopK and prior adaptive methods.

### Weaknesses
While the paper presents a simple yet effective idea and demonstrates convincing performance improvements in its experiments, the core contribution—context-aware routing—lacks sufficient explanation and validation of its underlying mechanisms and implications. The paper excels at explaining the "what" (the method design and results) but falls short in deeply analyzing the "why" (the intrinsic workings) and the "under what conditions" (the method's boundaries and limitations).

### Questions
1.The paper mentions enforcing per-token bounds to prevent "degenerate allocations." This suggests that the original, unconstrained SeqTopK might suffer from training instability or severe load imbalance. These constraints are inherently heuristic. Could the authors clarify:
a)How were these specific bounds determined? Do they introduce hyperparameters that require tuning?

b) Compared to standard TopK, does SeqTopK (even with constraints) exhibit greater expert load variance or higher auxiliary balancing loss during training? Is this a hidden cost of its dynamic allocation capability?

2.A core idea of MoE is expert specialization. Fixed-budget Token-Level TopK provides each expert with relatively stable "exposure" from all tokens. In contrast, SeqTopK's dynamic allocation might lead to "easy" tokens consistently activating only a fixed, small subset of experts, while "hard" tokens competitively activate more experts. Could this exacerbate a "Matthew effect" among experts, causing some experts to become functionally degenerate due to consistently servicing easy patterns, while others suffer in generalization due to over-specialization on hard patterns? Have the authors analyzed the distribution of expert activation under SeqTopK and whether this impacts the degree of expert specialization?

3.The paper compares SeqTopK well against other MoE routing methods like MRL-TopK and BatchTopK. However, the idea of adaptive computation has been extensively explored in broader areas like Early Exiting and Token Pruning. SeqTopK essentially performs a horizontal computation allocation within the MoE layer. Have the authors considered conceptual or experimental comparisons with these vertical or cross-module adaptive methods (e.g., allowing difficult tokens to pass through more model layers)? Discussing SeqTopK's unique position and limitations within the "adaptive computation spectrum" would help provide a more comprehensive assessment of its contribution.

### Soundness
3

### Presentation
3

### Contribution
2
