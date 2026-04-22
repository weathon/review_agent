# Distributed Knowledge Storage Hypothesis: Evidence from Generalization Failure in Knowledge Editing

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Large language models (LLMs) store factual knowledge at scale, and recent editing methods allow local updates of target facts. While effective in single prompt styles, prior evaluations have overlooked whether edits generalize across task formats. In this work, we present a systematic study of cross-format generalization in parametric knowledge editing. Across six curated task formats, we show that edits that succeed in the source format often fail to transfer, and high frequency in training does not guarantee a shared, format-invariant knowledge structure. Representation-level analyses reveal that edit directions cluster by task format, supporting a *distributed knowledge subspace hypothesis* in which knowledge is distributed across multiple format-specific subspaces. To mitigate these failures, we introduce multi-format supervision with iterative expansion, which improves transferability with minimal overhead. Our study reframes generalization failure not only as a practical limitation of current editing methods but also as evidence about the distributed memory mechanisms underlying factual knowledge in language models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper examines cross-format generalization in LLM knowledge editing. It finds that edits made through methods like MEMIT and IKE often fail to transfer across different task formats. This result supports the distributed knowledge storage hypothesis which states that facts are encoded in format-specific subspaces. To solve this issue the study proposes a multi-format supervised iterative approach to improve cross-format efficacy.

### Strengths
**Strength 1**

Comprehensive cross-format evaluation for LLM knowledge editing, covering six diverse task formats to capture real-world usage scenarios.

**Strength 2**

Provides robust empirical and representation-level evidence (e.g., format-clustered edit directions) that substantiates the distributed knowledge storage hypothesis.

**Strength 3**

Proposes a practical multi-format supervised iterative method that effectively improves cross-format edit robustness while maintaining specificity.

### Weaknesses
**Weakness 1. Experimental Scope:** 

The experiments are restricted to a single open-source LLM (Llama-3.2-3B-Instruct) and a single benchmark dataset (CounterFact-1k). It remains unclear whether the observed phenomena hold for larger, more diverse models (e.g., DeepSeek, Qwen) or other factual domains. Broader validation would strengthen generality claims.

**Weakness 2. Theoretical Underdevelopment of Core Hypothesis:**

The distributed knowledge storage hypothesis relies heavily on empirical observations (e.g., format-clustered edit directions) but lacks rigorous theoretical formalization. 

**Weakness 3. Incomplete Baseline and Related Work Coverage:**

Only two editing methods (MEMIT, IKE) are systematically evaluated, excluding recent relevant approaches (e.g., AlphaEdit, PMET) with advanced localization.

### Questions
I will happily raise my score if my concerns are addressed.

**Question 1**

Do the observed generalization failures persist in larger models (e.g., Llama-3-70B, GPT-type models) or for multilingual/fine-tuned LLMs? Are there any preliminary results or hypotheses on scaling effects?

**Question 2**

Robustness to Task Domain: Does the fragmentation of edits and the distributed subspace hypothesis hold for other factual relations (e.g., temporal, causal, commonsense) or less-structured domains (e.g., multi-hop facts, subjective knowledge)?

**Question 3**

Baseline Selection: Why were methods like AlphaEdit, PMET, or ROME variants omitted from benchmarking experiments? Would incorporating these alter the conclusions about universality of generalization failure?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the limitations of current knowledge editing methods, showing that edits applied in one format cannot be effectively generalized to other formats. The authors conduct experiments across six task formats, evaluating the cross-format generalization performance of knowledge editing through various metrics, and analyzing the distribution of knowledge across different formats in the representation space. They propose the distributed knowledge storage hypothesis, suggesting that factual knowledge in LLMs is encoded within multiple format-specific subspaces. Finally, the paper introduces an adversarial procedure called Multi-Format Supervision with Iterative Expansion, which enhances cross-format generalization performance.

### Strengths
1. The authors design six distinct task formats to explore the generalization capability of knowledge editing across different forms. The data construction process is relatively comprehensive, and the experimental analysis is persuasive.
2. The analysis of representation distributions through PCA projection intuitively demonstrates the clear separation of edit directions across different task formats, supporting the distributed knowledge storage hypothesis.
3. The Multi-Format Supervision with Iterative Expansion method is simple and well-motivated. By using an agent to assist in generating new task formats, it improves cross-format generalization with minimal human intervention.

### Weaknesses
1. All experiments are conducted only on the Llama-3.2-3B-Instruct model and the CounterFact-1k dataset, without validation on other models or alternative knowledge-editing datasets. This limits the generality of the conclusions, including both the distributed knowledge storage hypothesis and the proposed method.
2. Only MEMIT and IKE are evaluated as knowledge editing baselines, lacking comparison with stronger or more diverse editing approaches.
3. The analysis of agent-assisted task format generation is relatively shallow, with no quantitative evaluation of the generated formats’ quality, diversity, or their actual impact on improving generalization.

### Questions
1. Beyond the Llama-3.2-3B-Instruct and CounterFact-1k setup, have you observed similar cross-format generalization failures on other state-of-the-art models or larger-scale editing datasets?
2. Is there a principled reason multi-format supervision plateaued at $K=4$ formats (as in Figure 5), or is this dataset/model dependent? Does the order of format addition in the iterative defense loop affect robustness or efficiency?
3. Could you provide more details about the agent-assisted experimental results?

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
This paper investigates the problem of cross-format generalization of knowledge editing. The authors show that an edit that succeeds in the source format often fails in other formats and may even break previously consistent answers. From both behavioral and representation analysis (edited value vectors cluster by format), they argue that LLMs store factual knowledge in a distributed, format-dependent way and propose an iterative multi-format supervision strategy to harden edits against such failures.

### Strengths
* The problem is well-motivated and addresses a missing dimension in current benchmarks.
* The analysis is sound.
* The proposed agent-based system is simple and actionable.

### Weaknesses
* Limited experimental scope. All results are reported on a single CounterFact-1k subset and one 3B instruct model, which significantly reduces confidence in the generality of the findings.
* The contribution of the “distributed storage” concept is somewhat limited. Under the MEMIT framework, different formats naturally yield different keys, so edits are expected to occur in different locations. Under the IKE framework, the observed failures seem to stem more from the model’s format-sensitive reasoning/common-sense abilities—an area where a 3B model is clearly weak—rather than from how knowledge is actually stored.

### Questions
* What if we apply six edits simultaneously, one per format?
* Alternatively, could we collect a sufficiently large set of prompt prefixes for a subject, cluster them to identify the major key variants, and apply edits to all of them?
* Would such a multi-key / multi-format editing strategy address the observed issue?

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
4

### Summary
This paper tackles an important and under-explored problem: how knowledge edits generalize across different task formats (e.g., from completions to multiple-choice questions). The authors find that edits successful in one format often fail to transfer to others, and can even break the model's original consistency. To explain this, they propose the "distributed knowledge storage hypothesis," suggesting that facts are encoded in multiple, format-specific subspaces rather than a single location. They back this up with representational analysis and propose a multi-format supervision method as a potential fix.

### Strengths
1. **Novel and Important Problem**: The paper is the first to systematically study generalization across task formats (like completion, MCQA, T/F), moving beyond simple paraphrasing. It clearly highlights that current editing methods fail badly at this.

2. **Novel Hypothesis with Strong Evidence**: The "distributed knowledge storage hypothesis" is an intriguing explanation. It's well-supported by strong representational analysis, like the clustering of edit directions by format (which a linear probe can identify with 98.4% accuracy).

3. **Provides a Practical Solution**: The paper doesn't just identify the problem; it also proposes a practical solution with "multi-format supervision and iterative expansion." It shows this method can efficiently improve cross-format generalization while maintaining specificity.

### Weaknesses
1. **Storage vs. Access**: My main concern is the causal leap in the "distributed storage" hypothesis. The evidence (Sec 3.4.2) clearly shows that the edit directions ($v^*-v^0$) are format-specific, but does this necessarily mean the knowledge itself is stored this way? I'm not convinced an alternative isn't just as likely: the knowledge might be stored abstractly, but the access mechanisms (or computational paths) to retrieve it are highly format-dependent, influenced by surface-level prompt differences. The paper needs to grapple with this "storage vs. access" distinction more directly.

2. **Generalizability of the Solution**: The analysis includes IKE, a non-parametric method. Its failure mode might be totally different (e.g., format conflicts in the context window) from MEMIT's. Does the proposed "multi-format supervision" solution even work for IKE? If not, it suggests the fix might be specific to parametric editing, and the scope of the solution should be clarified.

3.  **Interpreting the Consistency Drop**: I'm not sure the drop in consistency is necessarily a bad thing. A high baseline consistency could just mean the model is "consistently wrong." After an edit, it becomes "inconsistently right" (correct in one format, wrong in others), causing the score to drop. This seems like an artifact of a partially successful edit. Relatedly, I don't see why erasing the frequency bonus strongly points to distributed storage. It seems just as compatible with the "distributed access" idea (i.e., high-frequency facts have more access paths, and the edit only broke one).

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
