# Inference Scaling of LLM Ensembling: Bridging Token Spaces with Token Translation

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Large language models (LLMs) exhibit diverse strengths and weaknesses across tasks, motivating recent efforts to ensemble multiple models to harness their complementary capabilities, boosting test-time performance. While model diversity and capability are known to influence ensemble effectiveness, a persistent challenge in LLM ensembling arises from mismatched tokenizer vocabularies. Existing alignment strategies typically rely on token-level embeddings or string-level heuristics of tokens, overlooking the tokenizer priors embedded during LLM pretraining. Specifically, tokenizers such as Byte-Pair Encoding (BPE) and Unigram are constructed by statistically analyzing large pretraining corpora to identify frequent subword units, and they tokenize text using greedy or probabilistic algorithms that reflect these learned subword distributions. In this work, we propose a novel and remarkably simple Token Translation} (ToT) method that explicitly leverages these tokenizer priors to bridge heterogeneous token spaces. Our method is lightweight, requiring only a few lines of code, pre-computable, and highly efficient at inference. To further enhance robustness, we incorporate token-level model uncertainty to dynamically reweight each model’s contribution during decoding. Extensive evaluations across diverse model combinations and tasks demonstrate that our method consistently outperforms existing ensembling baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new cross-tokenizer ensemble approach. The method constructs a mapping matrix that aligns one tokenizer’s vocabulary to another (base) vocabulary using prefix relations. In addition, the authors propose an entropy-based weighting scheme to combine ensemble outputs adaptively. Experimental results demonstrate consistent improvements over previous methods.

### Strengths
While the method is largely heuristic, its reported performance appears reasonable and may be of some practical interest. The use of tokenizer priors is a sensible design choice and appears more effective in practice compared to earlier methods such as Unite and DeepEN.

The experimental evaluation is sufficiently thorough to support the authors’ main claims.

The proposed method is more efficient than prior works such as DeepEN.

### Weaknesses
The token-translation method is mainly heuristics and is not theoretically grounded compared to other approach, for example the character-ensemble baseline converts the every model to the same base character alphabet with statistical guarantees.

For cross-tokenizer problem, it is necessary to include the amount of overlapping tokens between the vocabularies. For example, Llama3 and Qwen2.5 shares a large amount of similar tokens.

The idea of using entropy for model combination is not new (see [1], for example). It would be important to include a comparison against other methods with entropy-weighted ensembles. In particular, Table 2 could include a separate row showing the proposed approach without entropy weighting (i.e., “ToT without entropy weighting”) to clarify its contribution.


[1] Ruan, Yangjun, et al. "Weighted Ensemble Self-Supervised Learning." The Eleventh International Conference on Learning Representations.

### Questions
Line 122-123, what do you mean by "grounded semantics correspondence"? This was never defined nor rigorously stated in the paper.

How does the selection of the target tokenizer for mapping (in an ensemble setup) influence overall performance?    

Also see weakness.

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
This paper proposes a simple, lightweight, and training-free model ensembling method called Token Translation (ToT). Instead of relying on computationally expensive or semantically weak alignment methods (like embedding similarity or string heuristics), ToT leverages the "tokenizer priors", i.e., the statistical segmentation behavior learned by each model's tokenizer. Extensive experiments demonstrates the effects of this method.

### Strengths
S1: The authors propose a simple yet highly effective training-free ensembling method that aligns the vocabularies of different models using only tokenization rules, presenting substantial practical potential.

S2: The authors conduct comprehensive experimental evaluations, demonstrating the effectiveness and broad applicability (and ease of use) of the proposed method.

### Weaknesses
1. The authors propose a simple, direct, and effective training-free model ensembling method. However, because the method relies only on the token priors and lacks additional supervision, it is susceptible to potential biases. The paper would benefit from more extensive ablation studies and analyses to demonstrate the method’s generality and robustness.

2. The proposed ToT method is asymmetric. The authors should therefore include a series of experiments that swap the roles of the main model and the assistant model, demonstrating that the method's effectiveness is not contingent on a particular model assignment.

3. The authors should expand empirical validation along several axes, for example: accuracy w.r.t. the number of ensemble members (extending beyond Figure 2); cross-family evaluations (e.g., Llama, Gemma, Qwen, GLM, GPT-OSS, etc.); different model architectures (dense models versus MoE models); and different model scales (e.g., 32B dense models or larger MoE models).

4. Given that many open-source models now share similar tokenizers, the authors should consider evaluating ensembles that include models with more divergent tokenizers to verify that the proposed technique holds when tokenizer differences are larger.

5. Because the method requires only token logits or log-probabilities, it can be applied together with closed-source models. To strengthen the broad applicability of the method and provide additional insights, the authors may attempt ensembles incorporating closed-source models (e.g., using OpenAI GPT‑4/5 API). Even if such combinations do not yield large gains, reporting these results would be valuable for follow-up work.

6. The mapping matrix derived from rule-based procedures primarily encodes tokenization priors and is therefore weakly context-aware. To increase the method’s practical credibility, the authors should present a more detailed and intuitive analysis of the mapping matrix: examine the token priors it captures, investigate underlying mechanisms such as token conflicts and cooperation, and provide theoretical discussion plus focused case studies that illustrate the behavior and limitations of the mapping matrix.

### Questions
Q1: The method intuitively works better when ensembling models of similar scale. Could the authors attempt to extend the approach to ensembles that combine small and large LMs (i.e., cross-scale ensembles, beyond the scale of RQ4) to see whether it can meaningfully raise the overall performance ceiling? Demonstrating effectiveness in the small-LM vs. large-LM setting would substantially increase the practical value of the work.

Q2: Since this family of methods seems to be particularly effective for base/instruction-tuned models, the authors might also investigate its effects on reasoning models — specifically in Chain-of-Thought scenarios. Such an analysis (e.g., whether the method changes reasoning trajectories) would help clarify the mechanism and impact of the approach on multi-step reasoning.

### Soundness
2

### Presentation
3

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
This paper addresses tokenizer mismatch in test-time LLM ensembling by proposing Token Translation, which aligns heterogeneous token spaces by decoding tokens from a source tokenizer and re-encoding with a target tokenizer. The method uses bidirectional translation (prefix and superstring weighted mappings) and uncertainty-aware fusion to combine predictions from multiple models. Evaluated on 9 benchmarks using 7 models, ToT achieves +5.95 average improvement over single models and outperforms baselines while being computationally efficient.

### Strengths
The main contribution is leveraging tokenizer priors through decode-encode in the ensembling context and distinguishes it from embedding or string similarity approaches. 
The authors experiment rigorously as they use diverse benchmarks, models, ensemble sizes and baselines. The roofline analysis showing ToT tracks oracle performance is very helpful.

Problem motivation is useful with concrete examples showing why similarity-based methods fail.

The authors addresse a real practical problem in test-time ensembling with a solution that's simple to implement, computationally efficient, and consistently effective. The  improvement and strong scaling behavior make it valuable for the community. Additionally, the sparse matrix approach enabling precomputation is an important practical advantage.

### Weaknesses
No formal analysis of when/why token translation produces valid alignments. Why is the first re-encoded token the right choice? What tokenizer properties ensure semantic coherence? Under what conditions does the method fail (e.g., radically different tokenization schemes, low corpus overlap)? The decomposition in Section 3.3 provides intuition but not rigorous justification.

Table 5 shows 3 examples. A systematic quantification of alignment quality (coverage, precision, semantic coherence), analysis of what percentage have clean 1-to-1 vs. complex many-to-many mappings, frequency of prefix/superstring disagreements, failure case characterization would be very helpful.

No principled guidance for setting hyperparameters for new model pairs or explanation of whether optimal values vary by task, model combination, or language. Ablations show robustness over ranges but don't justify the chosen defaults or explain when to deviate.

Why is entropy over top-k the right measure versus alternatives (variance, disagreement, calibration)?

No systematic characterization of what types of errors get corrected and which benchmarks benefit most and why? Tables 6-7 provide anecdotal examples but lack systematic analysis.

### Questions
Can you formalize conditions under which token translation produces valid alignments? What properties of tokenizers (vocabulary overlap, training corpus similarity, segmentation strategy) determine alignment quality?

Can you provide systematic evaluation of alignment quality beyond 3 examples? What percentage of tokens have unambiguous alignments? How do you measure/validate semantic coherence of mappings?

When does token translation fail? Can you characterize failure patterns?


You show 2-4 models which is sufficient, but what happens with 10+ models? Is there diminishing returns or computational bottlenecks?

Given the minimal ablation impact, is uncertainty-aware weighting actually essential or could you achieve similar results with simpler uniform weighting plus alignment?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes  a method for LLM ensembleing. Seems to mix ensembleing with ideas of confidence boosting popular in reasoning.

I will rate the paper after the discussion.

### Strengths
Results are very impressive.

The experiments are excessive well structured.

### Weaknesses
Claim 1 is inaccurate Phan pointed out that tokenizer mismatch is a problem and gave an algorithmic solution, needs to be toned down.

How often and when did main use an assist

### Questions
Can u discuss the picture close to table 1, what is strongest, and why is 2 not improving on 1 llm?

I would like to see more true generation tasks not just voting tasks in this context

Can u show results for larger model ensembles

### Soundness
2

### Presentation
2

### Contribution
2
