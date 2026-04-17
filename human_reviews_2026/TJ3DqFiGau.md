# Parameters vs. Context: Fine-Grained Control of Knowledge Reliance in Language Models

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Retrieval-Augmented Generation (RAG) mitigates hallucinations in Large Language Models (LLMs) by integrating external knowledge. However, conflicts between parametric knowledge and retrieved context pose challenges, particularly when retrieved information is unreliable or the model's internal knowledge is outdated. In such cases, LLMs struggle to determine whether to rely more on their own parameters or the conflicted context. To address this, we propose CK-PLUG, a plug-and-play method for controlling LLMs' reliance on parametric and contextual knowledge. We introduce a novel knowledge consistency metric, Confidence Gain, which detects knowledge conflicts by measuring entropy shifts in token probability distributions after context insertion. CK-PLUG then enables fine-grained control over knowledge preference by adjusting the probability distribution of tokens with negative confidence gain through a single tuning parameter. Experiments demonstrate CK-PLUG's ability to significantly regulate knowledge reliance in counterfactual RAG scenarios while maintaining generation fluency and knowledge accuracy. For instance, on LLaMA3-8B, memory recall (MR) of RAG response can be adjusted within a broad range (9.9%-71.9%), compared to the baseline of 42.1%. Moreover, CK-PLUG supports adaptive control based on the model's confidence in both internal and external knowledge, achieving consistent performance improvements across various general RAG tasks. Our code is available at: https://anonymous.4open.science/r/CK-PLUG-Ano-8E62

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The approach detects tokens susceptible to conflicts between these two knowledge sources by measuring per-token entropy, then interpolates between their context-dependent and context-independent probability distributions. The degree of interpolation is governed by a single hyperparameter, which can be set manually or determined automatically using a heuristic based on the entropy ratio of the two variants.

### Strengths
- The paper is clearly written and easy to understand.

- The authors introduce a conceptually straightforward and well-motivated approach to regulate the model’s dependence on retrieved context.

- The proposed method is empirically solid and thoroughly evaluated, demonstrating its effectiveness in balancing contextual and parametric knowledge and enhancing question-answering accuracy.

### Weaknesses
### Methodological Evaluation

From a methodological standpoint, the proposed approach offers **limited novelty**, as it also relies on **distribution interpolation** between context-dependent and context-independent probabilities, similar to [1].  

The main differences are:  
- **Selective interpolation:** In this paper, interpolation is applied **only to tokens whose entropy increases after adding context**, assuming these tokens indicate parameter–context conflict. In contrast, [1] applies interpolation **to all tokens**.  
- **Different interpolation formula:**  
  This paper uses  
  $$
  \alpha \log p(y \mid x) + (1 - \alpha) \log \frac{p(y \mid c, x)}{p(y \mid x)} = (1 - \alpha) \log p(y \mid c, x) - (1 - 2\alpha) \log p(y \mid x)
  $$  
  whereas [1] uses  
  $$
  (1 + \alpha) \operatorname{logit}(y \mid c, x) - \alpha \operatorname{logit}(y \mid x)
  $$  

However, the **motivation for this specific interpolation formula** is largely **intuitive**, and the **procedure for identifying conflict-inducing tokens** is not rigorously justified.  

The **improvements in accuracy** over the standard RAG baseline are **modest**—and sometimes even **negative** (e.g., on **FEVER**, performance drops from 89.5 % to 89.2 % for Mistral, see Table 2)—which is disappointing given the method requires **approximately double the compute**.  

Overall, **more analysis and empirical/theoretical justification** are needed to demonstrate that the proposed method is truly worth its computational overhead and that it **outperforms [1]** in a meaningful way.  

**Reference:**  
[1] Shi, Weijia, et al. *Trusting Your Evidence: Hallucinate Less with Context-Aware Decoding.* *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers).* 2024.

### Questions
### General Questions

- What are the exact formulas used to define **ConR** and **ParR**? They are mentioned in line 309, but no details are provided on how they are computed.  
- It would be helpful to analyze how much **context vs. parametric reliance** affects performance to justify why adjusting this balance is important. I’m particularly interested in the **question-answering accuracy** corresponding to each ratio in Table 1.  
- Could you please specify the **number of forward passes** (or total compute) used by each method in Table 2 to ensure a fair comparison?

---

### Suggested Experiments

#### Justify the Interpolation Formula
I recommend comparing the current method directly with [1], including:
- **No ConD + interpolation from [1]**
- **CK-Plug** results from Table 1 and Table 2  
Additionally, please include an **ablation on the interpolation formula** in Table 3. At present, it includes (ConD + interpolation from CK-Plug) and (no ConD + interpolation from CK-Plug); it would be informative to also test (ConD + interpolation from [1]) and (no ConD + interpolation from [1]).  

These comparisons would clarify the necessity of introducing **ConD** and justify your **specific interpolation design**. If the interpolation from [1] performs robustly without ConD, then the added component may not be needed.

---

#### Explore More Challenging Context–Parameter Conflict Scenarios
It would strengthen the paper to test **CK-Plug** in settings with **stronger context–parameter conflicts**, such as those difficult even for large models like ChatGPT.  
You could evaluate performance in scenarios like §4.4.2 (where the parametric answer is inserted as a substring into the context) or use **Table 6 in [3]** as reference.  
This would help determine whether CK-Plug can effectively guide the model to prefer the **contextual** rather than **parametric** answer under such conditions.

---

#### Justify the Use of Entropy for Conflict Detection
I suggest performing **ablations on different uncertainty measures** for identifying conflict-prone tokens, beyond entropy.  
For example, try using **maximum token probability** or more recent uncertainty estimation techniques such as [2].  
This would validate whether entropy is indeed the most suitable choice for detecting parameter–context conflicts.

---

**References**

[1] Shi, Weijia, et al. *Trusting Your Evidence: Hallucinate Less with Context-Aware Decoding.*  
Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (Volume 2: Short Papers), 2024.  

[2] Ma, Huan, et al. *Estimating LLM Uncertainty with Logits.* arXiv preprint arXiv:2502.00290 (2025).  

[3] Kortukov, Evgenii, et al. *Studying Large Language Model Behaviors Under Context–Memory Conflicts With Real Documents.*  
First Conference on Language Modeling.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CK-PLUG, a plug-and-play method for controlling knowledge reliance in RAG systems when conflicts arise between LLMs' parametric knowledge and retrieved context. The approach uses a novel Confidence Gain metric based on entropy shifts to detect knowledge conflicts at the token level. CK-PLUG modulates token probability distributions through weighted fusion of parameter-aware and context-aware predictions, controlled by a single tuning parameter α. Experiments on four LLMs (LLAMA2/3, Mistral, Qwen) demonstrate wide-range controllability on counterfactual datasets while maintaining fluency. The method also offers an adaptive mode that automatically balances knowledge sources based on model confidence, achieving consistent improvements across six diverse RAG tasks without requiring parameter modifications or retraining.

### Strengths
- Novel entropy-based conflict detection that provides interpretable, theoretically-grounded identification of knowledge conflicts through Confidence Gain metric
- Flexible control via single parameter enabling smooth adjustment from full contextual to full parametric reliance with optional autonomous mode
- Practical plug-and-play design requiring no training or architecture changes while demonstrating effectiveness across multiple models and diverse RAG tasks

### Weaknesses
1. **Insufficient Baseline Comparisons** 

   The paper lacks comparisons with existing adaptive RAG methods that also address knowledge conflicts or context utilization. Notable missing baselines include:
   - Adaptive retrieval methods: FLARE, Self-RAG, DRAGIN, SeaKR
   - Context-aware generation: RQ-RAG, QC-RAG, CtrlA

   Without these comparisons, it is difficult to assess whether the performance gains are due to CK-PLUG's novel approach or simply from any form of adaptive control. The authors should include at least a subset of these methods to demonstrate the unique advantages of their entropy-based approach.


2. **Missing Critical Related Work**

   The paper overlooks several highly relevant previous or concurrent works that employ similar entropy-based or conflict-detection approaches for RAG:
   - Entropy-Based Decoding for Retrieval-Augmented Large Language Models (arXiv:2406.17519, June 2024) - uses entropy for RAG decoding
   - Discerning and Resolving Knowledge Conflicts through Adaptive Decoding with Contextual Information-Entropy Constraint (arXiv:2402.11893, Feb 2024) - directly addresses knowledge conflicts via entropy
   - SEReDeEP: Hallucination Detection in Retrieval-Augmented Models via Semantic Entropy and Context-Parameter Fusion (arXiv:2505.07528, May 2025) - combines semantic entropy with context-parameter fusion
   - FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation (arXiv:2506.08938, Jun 2025) - models fact-level conflicts

3. **Limited Applicability to Modern Agentic RAG Systems**. 

   Current RAG systems are evolving toward agentic architectures involving multi-step planning, iterative search, self-reflection, and answer verification (e.g., Search-o1, Search-R1, Reason-RAG, Web-walker, Web-sailor, etc). CK-PLUG operates at the token-level decoding stage, and it remains unclear whether:
   - The method can be integrated into multi-turn agentic workflows
   - Conflict detection works when contexts are iteratively refined
   - The approach scales to complex reasoning chains

   The authors should discuss or demonstrate CK-PLUG's compatibility with agentic RAG frameworks to ensure practical relevance.


4. **Insufficient Analysis of Computational Overhead**
   While claimed to be "lightweight," the paper provides no quantitative analysis of:
   - Latency increases during inference (requires two forward passes for parameter-aware and context-aware distributions)
   - Memory overhead from maintaining multiple probability distributions
   - Scalability with increasing context length

### Questions
Q1: Clarification on Notation (Line 143-144). There is a typographical error with double periods: "distributions.." Please correct.

Q2: Ambiguous "Baseline" Definition in Table 1. The "Baseline" row in Table 1 is unclear. Does it refer to:
- (a) Vanilla LLM without RAG (direct question answering), or
- (b) Standard RAG with both query and retrieved context, but without CK-PLUG?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Retrieval-Augmented Generation (RAG) reduces hallucinations in Large Language Models (LLMs) by incorporating external knowledge, yet it faces challenges from conflicts between the models’ parametric knowledge (internal) and retrieved context (external)—especially when the retrieved information is unreliable or the internal knowledge is outdated, leaving LLMs unable to decide which type of knowledge to prioritize. To solve this, the authors propose CK-PLUG, a plug-and-play method designed to control LLMs’ reliance on parametric and contextual knowledge. CK-PLUG introduces a new knowledge consistency metric called Confidence Gain, which detects knowledge conflicts by measuring entropy shifts in token probability distributions after context insertion; it then enables fine-grained control over knowledge preference by adjusting the probability distribution of tokens with negative Confidence Gain via a single tuning parameter, and also supports adaptive control based on the model’s confidence in both knowledge types.

### Strengths
1. The biggest advantage of this paper is proposing a "plug-and-play" inference-time method.

2. Conflict Detector: It introduces a metric called "Confidence Gain (CG)", which identifies conflicts by comparing the entropy change of token distribution between RAG input (context + query) and regular input (query only). A conflict is determined when there is an entropy increase (i.e., the model becomes more confused), and this definition is reasonably sound.

3. Knowledge Controller: This method isolates the logits purely contributed by the "context" through log subtraction, and then uses a single parameter to perform weighted fusion of parametric knowledge and contextual knowledge (Eq. 8). This is an extremely concise and theoretically grounded approach to logits manipulation.

4. Adaptive Model Construction: The paper also proposes an adaptive mode with "automatic (parameter adjustment)" (Eq. 10), whose logic is equally intuitive — the model automatically trusts the knowledge source with lower entropy (i.e., higher confidence).

### Weaknesses
1. This is the most serious and obvious flaw of the paper. To calculate [relevant parameters] and [relevant parameters], CK-PLUG must execute two complete forward propagations in parallel at each decoding step: one for [input with context + query + generated tokens] and the other for [input with query only + generated tokens]. This almost doubles the inference latency and computational cost.

2. The core assumption of the paper is that "conflicts lead to entropy increase". However, if the erroneous context itself is highly "credible" and "fluent" (e.g., "The capital of France is Lyon"), it is entirely possible to reduce the model’s perplexity, resulting in "entropy decrease".

3. The calculation of (Eq. 6) may be numerically unstable. If [parametric distribution] assigns a near-zero probability to a certain token (with [log value] approaching negative infinity) while [context-enhanced distribution] assigns a high probability to it, [resulting value] may "explode". The paper does not discuss any suggestions for handling numerical stability.

### Questions
1. Supplement implementation details regarding the calculation of [relevant parameter], and explain whether and how potential numerical instability issues have been addressed.

2. Conduct more rigorous stress tests on the "Confidence Gain (CG)" assumption—specifically construct erroneous contexts that are "highly credible and highly fluent", and illustrate the changes in [relevant indicator] under such circumstances as well as CK-PLUG’s performance metrics.

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
5

### Summary
This paper introduces CK-PLUG, a plug-and-play method that enables large language models to dynamically balance reliance on internal (parametric) knowledge and external (retrieved) context during retrieval-augmented generation. Using a novel Confidence Gain metric that detects knowledge conflicts via entropy shifts in token probabilities, CK-PLUG selectively adjusts token-level predictions with a single tuning parameter $\alpha$ (or adaptive enhancement) to favor either parameters or context. Experiments demonstrate the effectiveness of the proposed method.

### Strengths
- The paper was well-written and had very nice figures.
- The proposed method is lightweight and effective.

### Weaknesses
- My biggest concern with this paper is novelty. The use of entropy for identifying key tokens has been explored in recent works [1–2], yet these closely related studies are not cited—especially [1], which shares a similar methodology for token-level entropy analysis. Even if applied in a different context, omitting these references significantly weakens the originality of the contribution.
- The proposed CK-PLUG method may not generalize across all scenarios. For example, if the model is confidently wrong and the retrieved context reinforces the incorrect belief, the system may still fail. The authors should clarify the underlying assumptions and delineate conditions where CK-PLUG is reliable to enhance its scientific soundness. 
- The paper lacks comparisons with prior decoding-based [3–7] and intervention-based [8–9] approaches that similarly aim to regulate factuality and knowledge conflicts. Including such baselines would better demonstrate the advantages and distinct contributions of CK-PLUG. 

[1] What is Wrong with Perplexity for Long-context Language Modeling? ICLR'25

[2] Attention Entropy is a Key Factor: An Analysis of Parallel Context Encoding with Full-attention-based Pre-trained Language Models. ACL'25 

[3] Trusting Your Evidence: Hallucinate Less with Context-aware Decoding. NAACL'24

[4] Sled: Self logits evolution decoding for improving factuality in large language models. NeurIPS'24

[5] Dola: Decoding by contrasting layers improves factuality in large language models. ICLR'24

[6] Active Layer-Contrastive Decoding Reduces Hallucination in Large Language Model Generation. EMNLP'25

[7] AdaCAD: Adaptively Decoding to Balance Conflicts between Contextual and Parametric Knowledge. NACCL'25

[8] Cutting Off the Head Ends the Conflict: A Mechanism for Interpreting and Mitigating Knowledge Conflicts in Language Models. ACL'24

[9] Taming Knowledge Conflict in Language Models. ICML'25

### Questions
Aforementioned in the weakness section.

### Soundness
2

### Presentation
4

### Contribution
2
