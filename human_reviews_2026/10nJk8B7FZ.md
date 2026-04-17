# Relation Editing for Large Language Models

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Knowledge editing is a critical technique for the routine updating and maintenance of LLMs. Existing research predominantly assumes changes only to the object within subject-relation-object triples, with minimal exploration into techniques for editing the relation. We term this task Relation Editing(distinct from the established "Object Editing" paradigm). We first construct a dedicated relation editing dataset and benchmark existing algorithms, revealing a critical flaw: even with successful edits, prominent methods suffer from the persistent retention of outdated information, with rates reaching as high as 98.20\%. Editing failures stem primarily from two sources: the persistent retention of outdated relationships and the presence of challenging editing samples. To address the first issue, we propose a novel relation editing framework called Forgetting-and-Editing (FE). We theoretically show that existing forgetting methods (i.e., model unlearning) are unsuitable for this purpose and, to this end, introduce a new target assignment strategy within our framework. To mitigate the second challenge, we introduce a self-paced learning strategy, instantiated in a new algorithm named self-paced AlphaEdit(SPaEdit). We conduct extensive experiments on both our compiled relation-editing dataset and established object-editing benchmarks. Results demonstrate that our proposed relation editing strategy achieves satisfactory performance on the relation editing task. In addition, SPaEdit outperforms existing SOTA methods on object-editing benchmarks. Our research also suggests further study is warranted in relation editing, particularly on forgetting existing relations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
I find this to be a well-written and qualified paper overall. Therefore, I would not object if other reviewers find it worthy of acceptance. This paper is the first to systematically study the task of relation editing in large language models. I appreciate the authors' research logic: they first provide experimental proof that existing methods, such as AlphaEdit, still retain old factual relations at a high rate when performing this task. The authors have constructed a new relation editing dataset, ReEditBench (which includes manual verification), and proposed a new solution that achieves SOTA performance.

### Strengths
The authors provide detailed analytical experiments and sufficient theoretical analysis regarding traditional forgetting strategies. The authors' solution corresponds directly to the identified practical and theoretical problems of existing methods, making their argument highly persuasive.

### Weaknesses
I still have some concerns:

1. The research motivation (and even the examples used in the motivation) is identical to that of RaKE [1], particularly in lines 045-053. This reduces the paper's novelty. Meanwhile, I do not understand why the authors did not conduct more detailed analytical experiments within the RaKE framework, but instead chose to build an entirely new benchmark. The paper fails to clarify the differences and specific characteristics of ReEditBench compared to RaKE.

2. The proposed method requires a dedicated validation set and appears to be highly sensitive to hyperparameters, which seem empirical (ad-hoc). I am concerned that it may require extra adaptation for different tasks or domains.

3. I believe much of the methodology section could be streamlined or moved to the appendix, as a significant portion is identical to AlphaEdit. The authors should focus on explaining in greater detail what differs from AlphaEdit. There are still many unclear points: Does the SPaEdit "easy-to-hard" curriculum also apply to the "forgetting" data pairs? If SPaEdit only ranks the "editing" data, does using the full set of "forgetting" data in every step affect optimization stability? If both are ranked, do they share the same difficulty threshold ($\lambda$)? The paper does not clearly explain how the SPaEdit curriculum learning mechanism handles these two objectives ("forgetting" and "editing") simultaneously.

4. I think the paper could be strengthened by adding experiments that explore the semantic level of "relations," similar to the interesting experiments that can be done with simple addition and subtraction of word vectors. I believe relations themselves carry rich semantics. For example, is the difficulty of editing "CEO" to "CTO" (semantically close) different from editing "CEO" to "Father" (semantically distant)? Is the degree of forgetting also different? The paper defines sample difficulty based on computational residual rather than semantic properties. I would like to see a more intuitive explanation for the source of this difficulty, supported by analytical experiments.

### Questions
See in Weakness.

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
This paper introduces a new framework for relation editing in large language models, which focuses on updating the relations in factual triples rather than the objects. The authors construct a dedicated benchmark, ReEditBench, and find that existing object-editing methods tend to retain outdated knowledge. To address this, they propose a Forgetting-and-Editing (FE) strategy that jointly removes old relations and injects new ones, and a self-paced learning algorithm (SPaEdit) that learns from easy to hard samples to improve stability and success rates. Experimental results show that the proposed methods outperform prior approaches on both relation-editing and standard object-editing benchmarks.

### Strengths
1. The proposed Forgetting-and-Editing (FE) method explicitly separates the forgetting of outdated relations from the injection of new ones, effectively mitigating knowledge conflicts that commonly arise in conventional editing approaches.

2. This paper is clearly written, well organized, and generally easy to understand.

### Weaknesses
1.  I have some questions regarding the self-paced learning component. In your method, hard samples appear to be optimized multiple times, which seems conceptually similar to the mechanism described in NSE (Section 3.3). Could you clarify the main differences between your approach and NSE in this respect? Moreover, self-paced learning is typically applied in fine-tuning scenarios, where samples within the same task are correlated and gradual learning helps enhance model capability. However, in the knowledge editing setting, each sample is independent, and there is no cross-sample learning. This raises a concern about whether the purpose of self-paced learning here is mainly to improve overall accuracy by prioritizing easy samples first. Could you provide some results showing the relationship between sample difficulty and editing accuracy, and explain whether the performance gains are truly due to the progressive learning from easy to hard samples, rather than simply the effect of repeated optimization?

2.	This paper overly simplifies the task setting.  In practical applications, for an original triple (S, R, O), when R* changes while S and O remain the same, it is often the case that the original subject–relation pair (S, R) would naturally correspond to a new object O* instead. Ignoring this realistic phenomenon when constructing the dataset is a major limitation of the paper.

3.	In the main table results, the fluency metric was not included. From my personal experience, when the general output ability of the model drops to a certain extent, the evaluation metrics for the editing aspect have little significance. The model's output is mainly repetitive or in a garbled form.

### Questions
See Weaknesses.

### Soundness
3

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
4

### Summary
This paper studies knowledge editing in large language models. The factual knowledge is modeled as triplet $(s, r, o)$. Instead of editing the object $o$, the authors propose the challenge of editing the relation from r to $r^*.$ The main contribution is a dataset ReEditBench that contains 7918 high-quality relation editing instances and an editing framework Forgetting-and-Editing (FE), which follows the MEMIT framework and interpolate the value of outdated fact towards a new state during the forgetting stage. Experiments are conducted on the ZsRE and CounterFact benchmarks, as well as on the new ReEditBench dataset.

### Strengths
1. A new dataset on relation edit is proposed, which contains 7918 high-quality instances.

2. The proposed Forgetting-and-Editing (FE) approach looks simple and can be integrated into other editing methods.

3. The results on large-scale scenarios demonstrate the superiority of the proposed method compared to existing works.

### Weaknesses
1. In my opinion, the theoretical analysis in Sec 3.1 is redundant, if not counter-effective. Why can LLMs be modeled as a simple linear regression?

2. While focusing on hard subsets (Fig. 7) strengthens evaluation rigor, it also limits real-world applicability claims. Performance on average or easy cases—which constitute most real edits—remains unreported.

3. The proposed FE mechanism is intuitive, but lacks rigorous analysis. This may not be enough to spawn a new paradigm.

### Questions
1. What exactly does the FE strategy modify in the model? Is it a weight update, an activation-based intervention, or a regularization technique? The hyperparameters are given, but the mechanism remains opaque from the excerpts.

2. Why does SPaEdit achieve such high Fluency scores? Is this due to architectural design, decoding strategy, or post-edit smoothing? Without ablation studies, it’s unclear whether fluency gains are from editing quality or unrelated generation improvements.

3. How does SPaEdit handle conflicting edits or repeated edits to the same relation?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces and formalizes the problem of Relation Editing in large language models (LLMs), where the goal is to update the relation

### Strengths
(1) The paper addresses a novel and practical problem—relation editing—that has been largely overlooked in the knowledge editing literature.
(2) The proposed FE strategy and SPaEdit algorithm are well-motivated, theoretically grounded, and empirically validated across multiple models and benchmarks.
(3) The construction of ReEditBench is rigorous and clearly described, providing a valuable resource for future research.
(4) The paper includes extensive experiments, ablation studies, and analyses (e.g., stability, robustness to adversarial attacks, and general capability preservation), which thoroughly validate the proposed approach.

### Weaknesses
(1) While the FE strategy improves performance, retention of old knowledge remains non-trivial (often around 50% in difficult cases), suggesting that complete forgetting is still an open challenge.
(2) The method relies on several hyperparameters, and though sensitivity analysis is provided, the need for tuning may limit ease of adoption.
(3) The paper does not deeply explore the scalability of SPaEdit to very large models or extremely large-scale editing scenarios beyond the tested setups.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
