# Disentangling Knowledge Representations for Large Language Model Editing

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Knowledge Editing has emerged as a promising solution for efficiently updating embedded knowledge in large language models (LLMs). While existing approaches demonstrate effectiveness in integrating new knowledge and preserving the original capabilities of LLMs, they fail to maintain fine-grained irrelevant knowledge, namely facts that share the same subject as edited knowledge but differ in relation and object. This challenge arises because subject representations inherently encode multiple attributes, causing the target and fine-grained irrelevant knowledge to become entangled in the representation space, and thus vulnerable to unintended alterations during editing. To address this, we propose DiKE, a novel approach that Disentangles Knowledge representations for LLM Editing (DiKE). DiKE consists of two key components: a Knowledge Representation Disentanglement (KRD) module that decomposes the subject representation into target-knowledge-related and -unrelated components, and a Disentanglementbased Knowledge Edit (DKE) module that updates only the target-related component while explicitly preserving the unrelated one. We further derive a closedform, rank-one parameter update based on matrix theory to enable efficient and minimally invasive edits. To rigorously evaluate fine-grained irrelevant knowledge preservation, we construct FINE-KED, a new benchmark comprising fine-grained irrelevant knowledge at different levels of relational similarity to the edited knowledge. Extensive experiments across multiple LLMs demonstrate that DiKE substantially improves fine-grained irrelevant knowledge preservation while maintaining competitive general editing performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper points out that current model editing methods fail to preserve fine-grained irrelevant knowledge, as subject representations inherently encode multiple attributes, causing target and fine-grained irrelevant knowledge to become entangled in the representation space and thus vulnerable to unintended alterations during editing. To address this issue, the paper proposes **DiKE**, which employs a **Knowledge Representation Disentanglement (KRD)** module to decompose the subject representation into target-knowledge-related and -unrelated components, and a **Disentanglement-based Knowledge Edit (DKE)** module that updates only the target-related component while explicitly preserving the unrelated one. To rigorously evaluate fine-grained irrelevant knowledge preservation, the paper introduces **FINE-KED**, which assesses fine-grained irrelevant knowledge at different levels.

### Strengths
* This paper highlights the problem that current model editing methods fail to preserve fine-grained irrelevant knowledge.
* DiKE’s approach of disentangling fine-grained relevant and irrelevant knowledge before performing editing is novel and interesting.
* The paper demonstrates the effectiveness of DiKE through experiments, and notably, DiKE achieves impressive results in multi-hop editing, indicating its potential for multi-hop knowledge editing.

### Weaknesses
* DiKE seems to focus too heavily on locality. However, the efficacy and generalization of editing methods are also crucial aspects.
* DiKE appears to have limited scalability.
* DiKE seems applicable only to factual knowledge that contains a subject, which may restrict its applicability.

### Questions
* What do the abbreviations in Figure 3 stand for?
* Should “MEMIT” in line 323 be replaced with “ROME”?
* How are the negative sample sets in Equation (7) constructed? Could you provide an example?
* Some components of DiKE seem to affect efficacy — how can this be explained?
* Figure 2 is a bit confusing; it is recommended to add some numbering to help readers understand it better.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
### Summary

This paper identifies and addresses the challenge of preserving "fine-grained irrelevant knowledge" during large language model editing, where facts that share the same subject as an edit but differ in relation and object. The authors posit that current methods inadvertently alter this knowledge because subject representations are entangled, encoding multiple attributes simultaneously. To address this, they propose DiKE, a framework featuring a pre-trained KRD module that separates subject representations into target-related and unrelated components. A subsequent DKE module then updates only the target-related component while explicitly preserving the unrelated one, for which the authors derive an efficient, closed-form, rank-one update. To facilitate evaluation, they also introduce FINE-KED, a benchmark specifically designed to measure the preservation of fine-grained knowledge. Experiments demonstrate that DiKE substantially improves the preservation of this knowledge type while remaining competitive on standard editing metrics.




### Advantages

* The paper provides a clear and novel framing of a specific failure mode in knowledge editing by categorizing irrelevant knowledge into fine-grained and coarse-grained types and demonstrating the unique vulnerability of the former.
* The proposed DiKE method offers a principled and intuitive solution by explicitly disentangling representations as a preparatory step, which directly targets the hypothesized root cause of entangled knowledge interference.
* The introduction of the FINE-KED benchmark is a valuable contribution, providing the research community with a standardized tool to rigorously measure a previously under-examined aspect of editing performance.





### Disadvantages and Questions

* The proposed method introduces a separate pre-training phase for the KRD module, adding a data dependency and computational step not required by other locate-then-edit methods. To better assess the generalization capabilities of the KRD module, could the authors conduct an experiment where the module is trained on data from one domain (e.g., biographical facts) and then used to perform edits on an entirely different domain (e.g., scientific or geographical facts)?

* The derivation of the closed-form solution for the parameter update relies on an approximation that omits the non-linear activation function to simplify the fine-grained preservation constraint. Would it be possible to conduct an analysis comparing the empirical performance of the derived closed-form solution against an iterative optimization of the full, non-approximated objective function (Equation 18) to quantify the impact of this simplification?



* The definition of fine-grained irrelevant knowledge is focused exclusively on facts that share the same subject, which may not encompass all forms of semantically close knowledge that could be unintentionally altered during an edit. Could the authors provide an additional experiment to evaluate whether DiKE preserves other types of closely related knowledge, such as facts that share the same relation but have a different subject (e.g., editing "The capital of France is Paris" while measuring the effect on "The capital of Germany is Berlin")?

### Strengths
Please see my comments above.

### Weaknesses
Please see my comments above.

### Questions
Please see my comments above.

### Soundness
3

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
This paper propose DiKE, a method that disentangles knowledge representations into target-related and unrelated components, updating only the relevant part while explicitly protecting unrelated knowledge. They also construct the FINE-KED benchmark to evaluate fine-grained knowledge preservation across varying relational similarity levels. Experiments demonstrate DiKE outperforms existing methods like ROME, MEMIT and AlphaEdit in fine-grained knowledge preservation while achieving competitive or superior editing efficacy on standard benchmarks.

### Strengths
- This paper is overall clearly written.
- The experiments cover a range of benchmark, baselines, and the ablation studies help understand each component of the method.
- Overall the experiment results are good, which demonstrates the effectiveness of the proposed method.

### Weaknesses
- **Disentangler Dependence**: KRD/DiKE relies on disentanglement module quality. Poor performance (e.g., out-of-domain facts) risks reduced edit efficacy or corruption. Incoporating more case study will provide a more comprehensive understanding of DiKE
- **Benchmark Gaps**: FINE-KED is restricted to explicit subject-relation-object facts with heuristic+limited human-validated similarity. It lacks evaluation of open-domain/paraphrased knowledge generalization. While low pretraining-evaluation overlap (1.39%+) is positive, granular in-distribution vs. out-of-distribution stats are missing.
- **Batch Editing Trade-offs**: DiKE excels at subject-consistent batch edits (Figure 4), but computational trade-offs (memory/runtime as KRD scales with subjects/relations) are unaddressed. Cost-benefit analysis for large-batch/streaming use cases would add value.

### Questions
see section weakness

### Soundness
3

### Presentation
3

### Contribution
3
