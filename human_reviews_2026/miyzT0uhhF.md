# Token-Complexity based Routing Technique within Mixture of Experts Architecture for Large Language Model

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Mixture-of-Experts (MoE) architectures have emerged as a powerful technique for improving and scaling Large Language Models by conditionally activating Feed Forward subnetworks and distributing tokens through a routing system,  within the Transformer layers. However, existing MoE methods often rely on static top-k routing strategies that do not involve token-level variability in complexity, leading to suboptimal expert utilization. In this research, we propose a novel token-complexity-based routing framework that dynamically allocates tokens to either lightweight or strong feedforward networks (FFNs) based on their estimated token complexity. Our router is trained using a few-shot classification objective to distinguish between easy and complex tokens and surrogate neural network layer. The efficacy of the framework is evaluated while integrating the router with Mistral-7B and Llama-2-7B model. We evaluate our approach on several benchmarks from various fields, and our proposed MoE framework improves accuracy up to 12% compared to the state-of-the-art results using different MoE architecture, with reasonable computational cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
this paper proposes a dynamic routing framework based on token complexity.
The framework trains a router using a few-shot classification objective and a surrogate neural network layer. It dynamically assigns tokens to either lightweight FFNs or strong FFNs based on the tokens’ semantic complexity. This complexity is calculated using features such as the hidden states of the prediction layer and the hidden states of the attention layer. For training optimization, the framework uses a total loss function that integrates task loss and load balancing loss.

### Strengths
The idea of routing different tokens based on their difficulty level is interesting.

### Weaknesses
1 The paper title on the OpenReview page is inconsistent with that in the main paper.

2 Typo: There is a duplicate "load" in the figure caption of Figure 1 ("optimal load load balancing").

3 In the experiments, the number of experts is only set to 4, and there is no exploration of more experts (e.g., 8). This means the generalizability of the proposed method needs to be verified.

4 The ratio of lightweight experts to strong experts is 1:1, and there is no analysis on how different ratios would affect the results.

5 In the experiments, most comparisons are between the proposed method and the base model, with a lack of comparisons against other router schemes.

6 The threshold α=0.478–0.5 only reflects performance degradation, and there is no analysis of the relationship between the threshold and token complexity distribution (e.g., how the proportion of complex tokens in different benchmarks affects the optimal threshold).

7 In Table 6, unlike the other two models, Mixtral-8×7B shows little fluctuation in accuracy under different thresholds. However, the authors do not explain the reason for this, which deserves in-depth analysis.

### Questions
see weakness

### Soundness
3

### Presentation
1

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
In this submission, the authors introduce a routing mechanism of Mixture-of-Experts (MoE) models. The method aims to dynamically allocate tokens based on their estimated complexity: easy tokens are assigned to lightweight FFNs, and complex ones are routed to strong FFNs.  The authors state that this dynamic routing is implemented by a router trained with a “few-shot classification objective“ and a “surrogate neural network layer“. They experiment with Mistral-7B, and Llama2-7B demonstrates improvement on a series of benchmarks, including GSM8K, MMLU, and HellaSwag.

### Strengths
The motivation is quite straightforward and intuitive: tokens with different complexity should not be allocated with the same computation budget.

### Weaknesses
The paper suffers from severe flaws when describing the proposed method, the experiments, making the core method and the whole paper hard to understand and the results untrustworthy.

**Weakness 1 Severe Clarity Issue:** The paper fails to explain what its method and how it is applied.

- The paper applies its method to dense models (llama-2 and mistral) but does not explain the “MoEfication“ process. For example, what’s the new lightweight and strong FFN architecture and how their parameters initialized. This fundamental detail could help reader better understand the paper. The paper also mentions a "surrogate neural network layer" (Fig 1) and a "Router R" (Algorithm 1) but does not explain them well.
    
- The Eq. 1 and algorithm 1 is also confusing. For example, The paper names $D_{ctx}(t)$ as "contextual difficulty features," but the paper does not define what these features are or how they are computed. The paper repeatedly mentions the "few-shot classification objective" (Abstract, Sec 5). But it is unclear what this task is, where the "few-shot" examples come from (are they from the heuristic?), or how this objective is used to train the router. Furthermore, the paper does not *state its own fine-tuning settings* (dataset, total token count, optimizer, etc.).
    

**Weakness 2 Questionable experiment results:** the paper presents problematic experiment results.

- **Skeptical Baseline Results:** Table 3 reports a baseline accuracy for Llama-2-7B on GSM8K (5-shot) of **82.6%**. This number is much higher and inconsistent with established literature.
    
- **Misleading "Cost" Metric:** In Table 6, the "Cost" column is misleading. As per Section 4.3, the cost for the proposed method (`~210M`) is the **number of fine-tuned parameters** (if I understand right), while the cost for DynaMoE (`10B`) is the **number of fine-tuning tokens**. This is highly confusing.
    
- **No performance gains on several tasks:** Table 6 shows the *base* Mistral-7B model **outperforming** the proposed method on several benchmarks (e.g., 80.2% vs 75.0% on ARC-e; 96.4% vs 86.0% on SciQ).
    

**Weakness 3 poor presentation**

In addition to present the method and experiments poorly, there are other minor things. For example, the title does not match the abstract ("Efficiency" vs. "Complexity"). And what is this Mistral-2 model (line 434)?

### Questions
Here are my questions for authors about the proposed method:

1. What is the definition of token complexity discussed in this paper? The paper seems give two definitions: a heuristic rule-based definition (line 258) and a learned function (Eq.1). How are they related?
    
2. What are the "contextual difficulty features" (`D_ctx(t)`) and "token type embedding" (`T_type(t)`), and how are they computed?
    
3. What is the "few-shot classification objective"? Please detail the task and the data, the number of "shots," and how this objective is used to train the router.
    
4. What is the *exact* architectural difference between your parallel "lightweight" and "strong" experts and the "nested" experts used by DynaMoE?
    
5. Please state your full fine-tuning details: what dataset, how many total tokens, what optimizer, and fine-tuning hyper-parameters were used? This is necessary to make a fair comparison to DynaMoE's 10B token cost.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a complexity-aware routing framework designed to enable flexible and efficient Transformer inference. It introduces a neural routing module that estimates token difficulty and dynamically allocates computation by activating either lightweight or strong expert feed-forward pathways. The routing function is trained via a small neural network with few-shot learning, allowing adaptive expert selection without full model retraining. The approach is lightweight, model-agnostic, and easily integrable into existing LLM architectures. Experimental results on multiple base models, including Mistral and LLaMA, demonstrate favorable accuracy–compute trade-offs across various configurations.

### Strengths
- Well-motivated problem formulation: Clearly identifies inefficiencies in uniform token processing and motivates complexity-aware routing as a principled way to allocate compute based on token difficulty.
- Lightweight and modular design: The proposed routing layer introduces minimal overhead, requires few-shot calibration, and integrates seamlessly with existing LLM architectures without retraining.
- Dynamic compute allocation: Effectively balances efficiency and accuracy by selectively activating lightweight or strong FFN pathways per token, demonstrating adaptive computation within standard Transformer blocks in dense or MoE base model.
- Empirical validation across multiple base models: Evaluations on Mistral and LLaMA confirm consistent accuracy–compute trade-offs, highlighting the method’s generality and practical applicability.

### Weaknesses
Several introduced concepts lack precise definitions or sufficient experimental justification for their use.

- Insufficient explanation of difficulty labeling: The process for deriving token difficulty labels is not clearly described. Clarify whether these labels are generated automatically, depend on model confidence, or require annotations from the fine-tuning dataset. The quality of difficulty labels would impact the actual routing decision being learnt. So, without much validation of the ground truth labels, the proposed method would not be broadly applicable.

- Unclear expert architecture details: The paper does not clearly specify the architectural differences between the strong and weak experts and how that will be derived based on the underlying structure of the MLP layers of any new base LLM. 

- Ambiguous token difficulty thresholds:
The chosen thresholds (≤0.48 for easy tokens, >0.5 for complex ones) appear arbitrary, with a very narrow margin separating the two classes. The authors should justify this decision.

- Undefined computation of $L_{importance}$ loss: Provide its explicit mathematical definition and intuition, and clarify how it interacts with the main training objective.

- Minor presentation and formatting issues:
The paper contains inconsistent quotation formatting (e.g., “easy” and “hard” tokens) and unclear table descriptions. Improve clarity by:
expanding table captions to be self-contained, defining metrics like Token Complexity (Table 3) and iteration-level evaluation (Table 5).
Ensuring Table 4’s description is coherent and aligned with the experimental setup.

### Questions
The writing of the paper needs a revision in order to make it more polished and coherent.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a token‑complexity‑based dynamic routing mechanism for improving Mixture‑of‑Experts (MoE) architectures in large language models. Unlike conventional fixed top‑k routing, the method employs a lightweight neural router trained with few‑shot supervision to estimate each token’s complexity and dynamically assign it to either “lightweight” or “strong” feedforward expert networks (FFNs) based on a learned threshold α. Integrated with models such as Mistral‑7B, Mixtral‑8×7B, and LLaMA‑2‑7B, the framework achieves up to 12% accuracy improvements across benchmarks including GSM8K, MMLU, and HellaSwag, while maintaining stable computational cost. Overall, this work provides an efficient, scalable approach to enhance expert utilization and inference efficiency in transformer‑based large language models.

### Strengths
- The paper explores an interesting direction by integrating token‑level complexity awareness into MoE routing, which conceptually extends prior adaptive expert frameworks like DynaMoE and Flextron toward a more fine‑grained per‑token perspective.

- The idea of using a few‑shot supervised router with a tunable complexity threshold is a creative attempt to make routing both data‑efficient and flexible, showing potential for broader applicability in resource‑constrained LLM deployment.

- The authors conduct comprehensive empirical evaluations across diverse benchmarks, demonstrating consistent, though moderate, improvements that illustrate the promise of their complexity‑based routing strategy for enhancing expert utilization efficiency.

### Weaknesses
- The overall writing quality of the paper requires substantial improvement. First, the use of citations is incorrect and inconsistent — the authors do not properly distinguish among \cite, \citep, and \citet, as seen for example at lines 131 and 140. Second, the formatting of the tables does not follow the ICLR template guidelines, since the table borders should open on both sides. In addition, some figures, such as Figure 2, are not visually appealing — there is excessive spacing between subfigures, and the font size is too small. Moreover, table references (e.g., “Table. 3.” at line 357) are not hyperlinked, making them unclickable. Overall, the writing and presentation quality of the manuscript still need significant improvement.

- The proposed token-complexity-based routing concept overlaps substantially with prior adaptive MoE systems such as DynaMoE [1] and Flextron [2], which also route tokens or features based on difficulty or task semantics. The few-shot supervised router and complexity threshold provide some variation, but the paper does not sufficiently explain what theoretical gap this fills beyond existing “dynamic routing” approaches (e.g., Token-level Load Balancing). The paper should clarify how its method differs in the design of gating functions or optimization objectives, ideally supported by quantitative or theoretical evidence—such as improved load balance entropy, lower variance in expert utilization, or convergence guarantees—to substantiate novelty claims.

- The definition of token complexity relies on heuristics—for instance, labeling mathematical or programming symbols as “complex” and common words as “simple.” This rule-based labeling can cause poor generalization across domains because it ignores probabilistic uncertainty or contextual diversity. A more rigorous quantitative measure such as token entropy, surprisal, contextual dependency length, or prediction variance should replace heuristic rules to better reflect real difficulty. Moreover, verifying the correlation between estimated complexity C(t) and empirical uncertainty measures (e.g., perplexity) would strengthen the validity of the metric. If such signals were integrated into the router, the routing decisions could become more consistent across tasks.

- The experiments evaluate a wide set of benchmarks but lack the necessary depth to isolate where the performance improvements originate. The study omits comparisons to recent dynamic router baselines such as RouteLLM and RouterBench.


**Reference:**

[1] From dense to dynamic: Token-difficulty driven
moefication of pre-trained llms.

[2]  Flextron: Many-in-one flexible large language model.

### Questions
- Could the authors clarify how their token‑complexity‑based router fundamentally differs from prior adaptive MoE systems such as DynaMoE or Flextron?  
- How was the heuristic definition of token complexity validated, and does it correlate with quantitative measures like entropy or surprisal?  
- What is the sensitivity of the model’s performance to the chosen routing threshold \( \alpha \), and how was this value determined?  
- Why were recent baselines like RouteLLM and RouterBench excluded from the experimental comparisons?  
- Can the authors provide a breakdown of the additional inference cost introduced by the routing mechanism?

### Soundness
2

### Presentation
1

### Contribution
2
