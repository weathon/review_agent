# Reforming the Mechanism: Editing Reasoning Patterns in LLMs with Circuit Reshaping

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Large language models (LLMs) often exhibit flawed reasoning ability that undermines reliability. Existing approaches to improving reasoning typically treat it as a general and monolithic skill, applying broad training that is inefficient and unable to target specific reasoning errors. We introduce Reasoning Editing, a paradigm for selectively modifying specific reasoning patterns in LLMs while preserving other reasoning pathways. This task presents a fundamental trade-off between Generality, the ability of an edit to generalize across different tasks sharing the same reasoning pattern, and Locality, the ability to preserve other reasoning capabilities.
Through systematic investigation, we uncover the Circuit-Interference Law: edit interference between reasoning patterns is proportional to the overlap of their neural circuits. Guided by this principle, we propose REdit, the first framework to actively reshape neural circuits before editing, thereby modulating interference between reasoning patterns and mitigating the trade-off. REdit integrates three components: (i) Contrastive Circuit Reshaping, which directly addresses the generality-locality trade-off by disentangling overlapping circuits; (ii) Meta-Contrastive Learning, which extends transferability to novel reasoning patterns; and (iii) Dual-Level Protection, which preserves preexisting abilities by constraining reshaping update directions and regularizing task-level predictions.
Extensive experiments with Qwen-2.5-3B on propositional logic reasoning tasks across three difficulty levels demonstrate that REdit consistently achieves superior generality and locality compared to baselines, with additional validation in mathematics showing broader potential. Our code is available at https://github.com/LzyFischer/REdit.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces REdit, a framework for reasoning editing in LLMs that extends model editing from factual correction to logical reasoning. It proposes the Circuit–Interference Law, linking neural circuit overlap with edit interference, and applies contrastive circuit reshaping with meta-learning and dual-level protection. 

Overall, it is well-motivated, technically sound, and creative, addressing how to modify reasoning pathways without interference.

### Strengths
The paper presents a novel perspective showing that modifying internal circuit traces can directly influence the final reasoning behavior of LLMs, which is an intriguing and valuable insight.

The proposed Circuit–Interference Law provides a principled and empirically grounded explanation connecting neural circuit overlap with editing interference, representing a fresh and meaningful contribution to mechanistic interpretability research.

### Weaknesses
The experimental setting is limited. The authors evaluate only on a single backbone model (Qwen-2.5-3B) and a single dataset, which constrains the generality of the conclusions.

The reported improvements are modest rather than substantial, leaving some uncertainty about the practical effectiveness and scalability of the proposed approach.

### Questions
Confused on the instance-specific noise. if edge attribution values vary substantially across samples, how are these aggregated? Is the same top-$\tau$ threshold applied uniformly, or does it adapt to distributional variance across instances?

Could the authors provide specific examples of “reasoning patterns” to clarify how they are defined and represented? Additionally, how do these reasoning patterns change after applying REdit—is there any qualitative or quantitative visualization of this transformation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes reasoning editing, which targets edits on reasoning patterns using the proposed REdit method. Through experiments, the authors also uncovered the Circuit-Interference Law, which states that interference between circuits is proportional to their overlap.

### Strengths
- The paper's idea is very interesting and shifts attention from factual knowledge toward the logic applied by models, which is a source of many flaws in their performance.
- The Circuit-Interference Law is a significant novelty for the community.

### Weaknesses
- The datasets and models used are limited. While it does not seem like the conclusions would differ with larger models, the use of the well-controlled ContextHub raises questions about how things might go wrong with wild, real-world data.

### Questions
N/A

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The article presents the reasoning editing method for point-by-point modification of logical reasoning patterns in language models without losing other skills. The authors strike a balance between generalization and locality, describe the law of circuit interference, and propose the REdit system. Tests on logical and mathematical tasks show that REdit outperforms existing methods, paving the way for more precise control over model reasoning.

### Strengths
* The shift from knowledge editing to reasoning editing is original and well-motivated. The generality–locality trade-off is crisply formulated and backed by empirical evidence
* The results include confidence intervals, which make them more reliable and indicate the stability of the findings.

### Weaknesses
* The experiments focus only on propositional logic and structured math tasks, so they don’t fully reflect how reasoning works in more open-ended, real-world settings. It’s still unclear whether the method would hold up beyond these controlled, symbolic cases.
* While “circuits” are central, empirical evidence that reshaped circuits correspond to interpretable submodules is limited to correlation plots. There’s no qualitative analysis of what circuits actually represent.
* The approach is evaluated on a single model Qwen-2.5-3B, so it’s unclear whether the results would hold across different architectures or model scales.

### Questions
See weaknesses

### Soundness
4

### Presentation
4

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
The author proposed REdit, which is one of the pioneering works in reasoning editing. The work gives a perspective from the knowledge circuit on the reasoning process, which is inspiring and interesting. Specifically, the author proposed a systematic framework for reasoning editing with the emphasis on neuron circuits, and the Redit framework combines the findings together to achieve the editing.

### Strengths
- The presentation of the paper is well-organized and convincing

- The perspective of the knowledge circuit and the finding of Circuit-Interference Law is inspiring and novel.

### Weaknesses
- The work lacks sufficient experimental validation. The current experiments in Sec. 4 are conducted on only one model (Qwen-2.5-3B) and one dataset (ContextHub), which limits the generality of the conclusions.

- The choice of base model is inconsistent between the preliminary analysis in Sec. 2.2 and the main experiments in Sec. 4. In addition, the experimental setup in Sec. 3.1 is not clearly described (not sure if this is also based on Qwen-2.5-3B).

- The method appears overly complex, but the final results do not show clear improvement, leaving me unconvinced about its effectiveness and practical value.

Overall, I am very interested in the ideas and perspectives in this work. It would be helpful for enhancing this paper if the author can provide more results on more base models and datasets to show the effectiveness of Redit.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
