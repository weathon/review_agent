# Prune-then-Quantize or Quantize-then-Prune? Understanding the Impact of Compression Order in Joint Model Compression

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
What happens when multiple compression methods are combined—does the order in which they are applied matter?
Joint model compression has emerged as a powerful strategy to achieve higher efficiency by combining multiple methods such as pruning and quantization.
A central but underexplored factor in joint model compression is the compression order, or the sequence of different methods within the compression pipeline.
Most prior studies have either sidestepped the issue by assuming orthogonality between techniques, while a few have examined them only in highly constrained cases.
Consequently, the broader role of compression order in shaping model performance remains poorly understood.
In this paper, we address the overlooked problem of compression order and provide both theoretical and empirical analysis.
We formulate the problem of optimizing the compression order and introduce the Progressive Intensity Hypothesis, which states that weaker perturbations should precede stronger ones.
We provide theoretical guarantees showing that the relative benefit of one order increases with the underlying performance gap.
Extensive experiments on both language and vision models validate the hypothesis, and further show its generality to broader setups such as multi-stage compression and mixed-precision quantization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the interaction between two common model compression techniques — pruning and quantization — and specifically asks whether the order of application (Prune→Quantize vs. Quantize→Prune) affects the final model performance.
The authors formalize this question by defining the compression-order advantage
$A(Q → P)=M((P ∘Q)(φ))−M((Q ∘ P)(φ))$,
where M(⋅) denotes model performance. They also introduce the Compression-Equivalent Ratio $C_P^*$ , the quantization ratio that produces equivalent compression to a given pruning ratio so that both methods can be easily compared/combined under a unified metric.
Through a series of controlled experiments across several architectures and datasets, the paper finds that the order of operations does matter if one wishes to preserve the maximum model accuracy

### Strengths
1.	Clear and focused research question.
The “prune–then–quantize or quantize–then–prune?” question is simple but practically relevant for model deployment pipelines.
2.	Well-defined theoretical framework.
The introduction of the compression-order advantage $A(Q → P)$ and compression-equivalent ratio $C_p^*$  gives a clear formal basis for comparing compression orders.
3.	Systematic experimentation and consistency in results.
The experiments are comprehensive, covering multiple pruning ratios and quantization levels, with consistent definitions across models. And $A(Q→ P)$ behavior with $\Delta C$

### Weaknesses
1.	The compression order advantage $A$ was empirically measured for a variety of scenarios and it’s trends with $C_p^* - C_Q$ was demonstrated. But the actual value of $A$, and more specifically whether it is positive or negative, cannot be determined by the mere $C_p^*$ and $C_Q$ values. So I don’t understand how can a user decide which method to apply first,
2.	I don’t fully understand the practical meaning of "well-designed pruning" in Assumption 2. “The pruning method is chosen to minimize performance degradation”.
Isn’t that conceptually the goal of any pruning methods? 
From your phrasing it seems that there is only "one" that minimize the performance degradation. Yet, THERE ARE many pruning methods (also in your experiments). So which one of these DID minimize the performance degradation?

### Questions
$Q_1$. Figures 3, 4 and 6. It is unclear at first glance why multiple points appear for a single pruning ratio or why some curves are nearly flat around zero. I figure that the different points in each curve correspond to different compression bitwidth. Please clarify it in text for a more fluent reading (e.g., the different points in each curve correspond to $C_Q=$a,b,c,d, and e) IMHO it will make it more fluent for readers

$Q_2$. $C_p^* $ is defined for a vanilla model. That is, apply the pruning, measure the model’s accuracy and find the compression ratio $C_p^* $ of the quantization that leads to the same accuracy. I understand the reasoning for this methodology. You want to have a common metric so that you can have a proper $x$ axis. 
But what happens when you reverse the order of compressions? 
Quantization, if it is applied as the second compression, still compresses by $C_Q$ (Does it?). However, I’m not sure if pruning still achieves the same analogous compression rate $C_p^*$…

$Q_3$. How does this methodology can be used for other compression methods such as joint-compression (e.g., merging similar tokens in KV-Cache, merging heads in Attention, etc.) or Low rank approximations (LoRA). 

$Q_4$. Can you give reference to the assumptions in Assumption 1. 

$Q_5$. Figure 5 and related text. Can you clarify? The rotation without quantization did not introduce any accuracy degradation. Am I right? Only the pruning on top of the rotation introduced an increased degradation. Is it because you prune additional weights that did not exist prior to rotation?

$Q_6$. Table 1. I guess the $C_Q$ stems from quantization of 16bit to 4,5,…9bit. Please add it to make the table clearer

$Q_7$. Make the result simpler to read. For example, in Finding 5 which you put (wisely) within a frame so that any reader can take-away this message. I think you mean “Quatization and then pruning achieves higher accuracy (than pruning and then quantization)”. Add such simpler phrasing, so that anyone, even one that did not read your hypothesis can take-away your finding.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper investigates how the order of compression operations (e.g., pruning and quantization) affects model performance. The authors formalize the ordering problem, propose quantitative metrics, and validate the intuitive Progressive Intensity Hypothesis—that weaker perturbations should be applied before stronger ones. The paper presents both theoretical insights and extensive experiments on language and vision models, confirming the hypothesis and providing clear practical guidance for compression practitioners.

### Strengths
1. The topic is practical and relevant to real-world model deployment.
2. Experiments are extensive, covering multiple models and compression settings.
3. The paper provides a clear and actionable guideline that practitioners can easily adopt.
4. The writing is clear and the code is open-source.

### Weaknesses
The paper mainly evaluates compression pipelines that consist of two or three stages (e.g., prune–quantize or prune–quantize–prune). Based on the proposed theory, could this framework be extended to longer or more complex multi-stage compression pipelines, and would the same hypothesis still lead to performance gains compared with existing setups/shorter pipelines?

### Questions
Please see the weakness.

### Soundness
3

### Presentation
4

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
This paper investigates the importance of compression order in joint model compression (e.g., pruning and quantization) and proposes the "Progressive Intensity Hypothesis" (PIH). The hypothesis posits that applying weaker perturbations (compressions) first, followed by stronger ones, leads to superior model performance .

### Strengths
The paper is well-organized and easy to understand.

The paper addresses a practical yet underexplored problem in model compression: the order of joint compression.

### Weaknesses
1. The paper introduces the "Compression Equivalent Ratio" (CER) to unify the "intensity" metric. However, calculating the CER for a pruning method requires running the pruning experiment independently to measure its performance (e.g., 65% accuracy), and then finding (or interpolating) the quantization ratio $\mathcal{Q}$ that yields the same performance. This implies that to apply the hypothesis, one must first run all compression methods individually to determine their "intensity" ranking. It is questionable whether the computational cost of this ranking process is substantially lower than the cost of directly testing both compression orders (e.g., $\mathcal{P} \rightarrow \mathcal{Q}$ and $\mathcal{Q} \rightarrow \mathcal{P}$).

2. The assumptions for the theoretical analysis are an oversimplification of neural networks. The paper assumes "Layer-wise independence" , an "Error-performance trade-off"  (both in Assumption 1), and "Disjoint Selectivity" (Definition 5). These assumptions largely ignore the correlations between layers and the different sensitivities of various layer types.

3. The scope of the hypothesis is mainly focused on "post-hoc" compression settings. Additionally, its core analysis revolves almost exclusively around pruning and quantization scenarios. It is unclear how well the hypothesis generalizes to combinations involving other compression techniques (like knowledge distillation or parameter sharing).

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper revisits an underexplored question in model compression: whether the order of applying pruning and quantization matter, and why? The authors formulate the joint compression order optimization problem and propose the Progressive Intensity Hypothesis (PIH), that weaker perturbations should precede stronger ones. This hypothesis  is proved under the assumptions of disjoint selectivity and well-designed compression (Theorems 1 & 2), and extend the analysis to more realistic interference cases. Empirical validation spans language models (LLaMA-2/3 7B-13B-8B) and vision models (ResNet-18, DeiT-Base), showing consistent monotonic trends where quantization-after-pruning performs worse than pruning-after-quantization. The hypothesis is further generalized to multi-stage compression and mixed-precision quantization.

### Strengths
1. A new but meaningful problem explored in this paper: the paper introduces a rigorous formalization of compression order optimization, a previously neglected but practically crucial dimension in joint model compression. The Progressive Intensity Hypothesis provides a simple, actionable rule with theoretical grounding and broad implications.

2. Clear theoretical analysis.  Theoretical results are clearly derived. Theorem 1 establishes performance ordering under disjoint selectivity, relating compression order advantage A to per-unit reconstruction errors. Theorem 2 proves monotonicity with respect to compression equivalent ratio (CER) differences. These analyses connect signal perturbation and model compression behavior.

3. The empirical validation covers language, vision, and general pipelines, including experiments on LLaMA and ResNet/ViT variants. The paper goes beyond two-method cases to multi-stage and mixed-precision scenarios, reinforcing the hypothesis’ generality.

4. There are also some useful secondary findings in this paper. For example, rotation exacerbates pruning effects (Fig. 5). Structured vs. unstructured pruning differ in interference behavior (Table 1). Vision models exhibit stronger order sensitivity than LLMs.

### Weaknesses
1. Theoretical assumptions are strong. The “well-designed” compression assumption is idealized and not always met in real pruning heuristics. The analysis could better discuss how violations (e.g., correlated layer errors, adaptive pruning schedules) affect Theorem 2. The independence assumption between layers (Assumption 1) is strong, worth empirically validating with correlation metrics. 

2. While extensions to multi-stage and MPQ are demonstrated, the theoretical discussion remains pairwise. A more general n-method ordering theorem (Appendix B.3) is mentioned but not empirically validated.

3. Connection to prior composability studies. The paper could better position itself relative to recent works like SmoothQuant (Xiao et al., 2023) or Harma et al. (2025) by emphasizing differences in analytical generality rather than empirical scope alone.

### Questions
1. How sensitive are the results to the performance metric M(·)? Would monotonicity still hold under different metrics (e.g., loss, perplexity, BLEU)?

2. I am interested whether the Progressive Intensity Hypothesis extend to compression combinations involving low-rank (e.g., Lora) or distillation techniques?

### Soundness
4

### Presentation
3

### Contribution
3
