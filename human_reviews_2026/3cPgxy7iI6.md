# Automated Creativity Evaluation for LLMs with Semantic Entropy and Efficient Multi-Agent Judging Across Open-Ended Tasks

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 2

## Abstract
Large language models (LLMs) have achieved remarkable progress in language understanding, reasoning, and generation, sparking growing interest in their creative potential. Realizing this potential requires systematic and scalable methods for evaluating creativity across diverse tasks. However, most existing creativity metrics are tightly coupled to specific tasks, embedding domain assumptions into the evaluation process and limiting scalability and generality. To address this gap, we introduce an automated, domain-agnostic framework for quantifying LLM creativity across open-ended tasks. Our approach separates the measurement apparatus from the creative task itself, enabling scalable, task-agnostic assessment. Divergent creativity is measured using semantic entropy—a reference-free, robust metric for novelty and diversity, validated against LLM-based novelty judgments and baseline diversity measures. Convergent creativity is assessed via a novel retrieval-based multi-agent judge framework that delivers context-sensitive evaluation of task fulfilment with over 60% improved efficiency. We validate our framework across two distinct domains—physical reasoning and scientific research ideation—and with a broad suite of LLMs. Empirical results show our metrics reliably capture key facets of creativity—novelty, diversity, and task fulfilment—and reveal how model properties such as size, temperature, recency, and reasoning impact creative performance. Our work establishes a reproducible, generalizable standard for automated LLM creativity evaluation, paving the way for scalable benchmarking and accelerating progress in creative AI.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a benchmark for evaluating model innovativeness and feasibility by simultaneously considering divergent and convergent creativity. Experimental results show that, compared to other baselines, the proposed method aligns more closely with human evaluations.

### Strengths
1. The paper evaluates models along two dimensions, measuring not only divergent creativity but also the feasibility of proposed solutions.  
2. Experiments demonstrate that the proposed method correlates more strongly with human judgments than existing baselines.

### Weaknesses
See Questions below

### Questions
1. I noticed that Qwen3’s “thinking” score is not particularly high in the experiments. However, during the thinking process, models often explore many candidate solutions. Does the evaluation method treat the entire thinking trace as a single output, rather than evaluating each individual attempt as a separate path?  
2. Model performance often varies across problems of different difficulty levels—for instance, on harder tasks, models typically require multiple rounds of trial and error before arriving at a correct solution. Would it be beneficial to evaluate models separately on problems of varying difficulty?  
3. Different “thinking” models often exhibit substantial performance differences. Could the paper include results from a broader set of thinking models? Additionally, is it possible to correlate these models’ creativity scores with their capabilities in domains like math or code?  
4. The paper states that divergent and convergent creativity stem from distinct mechanisms and can be optimized independently. However, does this independence depend on the problem type? For problems with very few viable solutions, achieving a high divergent creativity score may be inherently difficult—and attempting to artificially boost divergent creativity might compromise convergent performance.  
5. Could the proposed method be adapted in the future as a reward signal to enhance model diversity during training or inference?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents an LLM evaluation framework for creative tasks. Following previous cognitive literatures, this paper advocates evaluating creativity from two perspectives: divergent creativity and convergent creativity. For the divergent creativity, this paper uses semantic entropy to measure the diversity and LLM responses. For the convergent creativity, this paper employs a multi-agent judge framework with a retrieval-based method for cost reduction. Experiments include testing the proposed framework on 3 creative datasets from different domains and several LLMs, which reveals the effectiveness of the proposed framework.

### Strengths
- This article focuses on the important problem of assessing LLM creativity and, grounded in cognitive science, advocates a systematic assessment of creativity in two aspects.
- Using semantic entropy, which is used to assess LLM hallucinations, to assess divergent creativity sounds interesting, as it seems to suggest that the two are deeply connected.

### Weaknesses
- A major weakness of this paper is the lack of technical innovation. The semantic entropy metric used to evaluate divergent creativity comes from previous work [1], while the multi-agent framework used to assess convergent creativity mainly comes from previous work [2] (with some improvements to reduce token consumption).
- The experimental part only involves three creativity-related benchmarks and lacks the verification of the proposed framework on creativity tasks in a wider range of domains.

Refs:

[1] Detecting hallucinations in large language models using semantic entropy

[2] Chateval: Towards better llm-based evaluators through multi-agent debate

### Questions
- Is using semantic entropy to assess divergent creativity affected by hallucinations? For example, when LLM responses contain hallucinations, will semantic entropy overestimate divergent creativity?
- Why is the accuracy of 'our framework: GPT-4o' better than ChatEval in Table 2? It seems that the main improvement of the proposed method is to save tokens.

### Soundness
2

### Presentation
3

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
This paper focuses on evaluating the creativity of LLM-generated text, arguing that existing methods for assessing creativity are often task-specific and lack scalability. To address this issue, this paper proposes using semantic entropy to describe the innovativeness and diversity of LLMs and designs an automated, domain-agnostic framework. Experiments on three open-ended domains demonstrate that the proposed metrics capture novelty, diversity, and task fulfilment.

### Strengths
1. Addresses an important and timely problem— automated evaluation of creativity in large language models —which lacks standardized methodology.
2. A conceptually clear method for decomposing creativity is proposed, which decomposes creativity into two aspects: divergence (semantic entropy) and convergence (multi-agent judgment).

### Weaknesses
1. Limited novelty, as both proposed components largely adapts existing ideas. For example, semantic entropy was originally designed for hallucination detection. Since it primarily describes "uncertainty," its reinterpretation of "divergent creativity" lacks both theoretical foundation and empirical depth, and is only reflected in its relevance. Meanwhile, the retrieval-based multi-agent judging framework follows ChatEval with efficiency-oriented modifications, not a conceptual innovation.
2. Lack of validation with human judgment. The paper does not assess whether semantic entropy is related to human evaluation of creativity, therefore its construct validity is not demonstrated.
3. Experimental scope is limited. Only text-based tasks are tested, without exploration of multimodal or cross-domain generalization.

### Questions
1. Correlation analysis is indirect: it focuses on the relationships between metrics rather than the consistency between humans and metrics.
2. Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an automated, domain-agnostic framework for evaluating LLM creativity, which it bifurcates into two components based on cognitive science concepts:
1. Divergent Creativity: To measure novelty and diversity, the paper adapts Semantic Entropy (SE), a metric originally developed for hallucination detection, as a reference-free measure of generative variability.
2. Convergent Creativity: To measure task fulfillment, the paper introduces a retrieval-based multi-agent judge framework. This framework aims to improve the computational efficiency of existing discussion-based evaluators by using a retrieval mechanism to condense the context, claiming over 60% improved efficiency and human-level accuracy.  

The framework is validated on three distinct datasets—MacGyver (problem-solving), HypoGen (research ideation), and BookMIA (creative writing). The paper's main findings are that (1) SE is a robust metric for divergent creativity, (2) the retrieval-based judge is efficient and accurate, (3) divergent creativity (SE) does not improve with model size or recency, while convergent creativity does, and (4) this suggests divergent and convergent abilities are distinct mechanisms in LLMs.

### Strengths
1. The paper tackles a highly significant and challenging problem: the scalable, automated, and domain-general evaluation of creativity in LLMs. This is a crucial area of research for advancing generative AI.
2. The authors validate their framework across three qualitatively different creative domains (problem-solving, scientific ideation, and creative writing), which strengthens the claim of domain-generality.
3. The paper is supported by extensive appendices that provide full prompts, implementation details, and additional ablations, which aids in reproducibility.

### Weaknesses
1. The paper's primary metric for divergent creativity, semantic entropy (SE), is a metric for hallucination/uncertainty. Using it to measure "creativity" is fundamentally flawed and invalidates all conclusions drawn from it.
2. The framework's design is contradictory. It claims to measure divergent thinking by sampling 10 steps, but then uses greedy decoding to select the path forward. This inherently evaluates a convergent, high-probability path while claiming to measure the opposite.
3. The validation for semantic entropy is unconvincing. Figure 5a is tautological (SE vs. class count), and the main validation relies on circular logic (SE vs. an LLM-judge for novelty).
4. The paper's key finding, such that divergent and convergent creativity are "decoupled", is almost certainly an artifact of this flawed metric. A more plausible interpretation of the data (Fig 17, 18) is:
 - Convergent scores (task-fulfillment) improve with model size. This is known.
 - Divergent scores (SE, i.e., uncertainty/hallucination) do not improve or even decrease with model size. This is also known, as scaling and alignment reduce hallucination.
- The paper has not discovered a "decoupling of creativity." It has simply re-demonstrated that task-fulfillment and hallucination are (thankfully) not correlated.
5. The "novel" multi-agent judge provides no clear benefit. As shown in Table 2, the framework using a weak judge (GPT-4o-mini) performs worse than a simple one-shot baseline. This proves the framework's complexity is unjustified and all performance gains come from the judge's (GPT-4o) backbone.
6. "Domain-General" Claims Undermined.
 - The core SE metric relies on an entailment model, and Table 1 shows that changing this model causes accuracy to swing from 78.1% to 47.2%, indicating high instability.
 - The correlations between SE and other diversity metrics are wildly inconsistent across datasets (Tables 8, 9, 10), further suggesting the metric is not stable or domain-general.

### Questions
* Also see the weaknesses above

1. The paper's entire premise hinges on equating SE with divergent creativity. Can the authors provide a stronger justification for why high semantic uncertainty, which often leads to incorrect or nonsensical outputs, is a reliable proxy for creative (novel yet valuable) ideas?
2. How can the authors reconcile using greedy search ("most likely candidate") to build the final solution while claiming the framework measures divergent creativity? Doesn't this design ensure you are only evaluating the least creative, most probable path?
3. Given that "Our framework" with GPT-4o-mini (55.3% acc) performs significantly worse than the "Baselines" (e.g., One-shot 64.7%) in Table 2, how can you claim the framework itself provides any value? Doesn't this data prove that all the performance comes from the GPT-4o judge, and the retrieval framework itself is ineffective?
4. Please address this counter-hypothesis: The "divergent" metric (SE) measures uncertainty/hallucination. The "convergent" metric measures task-fulfillment. And the data shows that scaling improves task-fulfillment while reducing uncertainty. Is it not more accurate to conclude that this work simply re-proven that alignment reduces hallucination, rather than discovering a "decoupling of creativity"?
5. How do the authors defend the "model-agnostic" and "generalizable" claims when Table 1 shows that the choice of entailment model causes a ~30% absolute (from 47.2% to 78.1%) swing in performance?

### Soundness
1

### Presentation
2

### Contribution
1
