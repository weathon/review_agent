# ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 4

## Abstract
Despite significant strides in statement autoformalization, a critical gap remains in the development of automated evaluation metrics capable of assessing formal translation quality. Existing metrics often fail to balance semantic and structural information: string-based methods neglect semantics, whereas proof-based approaches offer no graded similarity when proofs fail. To address these issues, we introduce ASSESS (A Semantic and Structural Evaluation Framework for Statement Similarity), which captures syntactic structure by transforming formal statements into operator trees and computes a real-valued similarity score using our novel TransTED (Transformation Tree Edit Distance) Similarity metric by incorporating semantic transformations. For rigorous validation, we present EPLA (Evaluating Provability and Likeness for Autoformalization), a benchmark comprising 1,247 expert-annotated formal statement pairs derived from miniF2F and ProofNet, distinctively labeled for both semantic provability and structural likeness. Experiments on the EPLA benchmark demonstrate that TransTED Similarity surpasses existing methods, achieving state-of-the-art accuracy and Kappa score. The benchmark dataset, code, and detailed experimental results are available at https://github.com/XiaoyangLiu-sjtu/ASSESS.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a similarity score metric to assess the semantic and structural correspondence between pairs of formalized statements in autoformalization. Specifically, the proposed TransTED method leverages tree edit distance (TED) on operator trees extracted from Lean, and further augments TED through transformations based on tactics applied to equality relations between statements. To evaluate TransTED, the authors construct the EPLA dataset, which consists of autoformalized formal statements collected from MiniF2F and ProofNet. Experimental results show that TransTED achieves higher accuracy and Kappa coefficient compared to several baseline metrics on the newly constructed EPLA dataset, and demonstrates greater robustness than the BLEU score.

### Strengths
The paper is generally easy to read and well-organized. I appreciate the formalization and clear logical flow of the methodology section. The proposed method is well motivated and intuitively explained. Experimental results show that it outperforms existing baselines by a reasonable margin in accuracy and is better aligned with manual evaluations, demonstrating its effectiveness.

### Weaknesses
The proposed evaluation metric extends the traditional TED score through several augmentations; however, in practice, the implementation primarily applies a limited set of Lean tactics to establish the equality of two formal statements, and relies on a curated subset of these tactics. When the tactics fail to prove equivalence, the method simply falls back to the standard TED computation. Thus, the approach can be viewed as augmenting or relaxing proof-based equivalence checking with TED-based similarity, rather than achieving full proof-based validation. As a result, the evaluation metric does not guarantee 100% precision, unlike BEq or identity matching.

I also find some aspects of the experiments a bit unclear (see questions below). In my opinion, definitional equality should yield 100% precision—the reported misclassified examples mostly involve implicit type variables, which could reasonably be interpreted as equivalent or easily resolvable by instantiation.

Finally, I would suggest avoiding the claim that “to our knowledge, this represents the largest and most comprehensive dataset dedicated to this evaluation task.” Several recent works have autoformalized and manually verified a larger number of statements than those included in EPLA, such as:

[1] Autoformalize Mathematical Statements by Symbolic Equivalence and Semantic Consistency

[2] FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models

[3] Improving Autoformalization Using Type Checking

### Questions
How are semantic provability and structural similarity jointly used to evaluate the actual equivalence between two formal statements in the experiments? Could the authors provide a few concrete examples illustrating how these two aspects interact during evaluation?

In the example mathd_algebra_132, the labeled statements appear to perform a one-step reasoning that directly substitutes the point coordinates into the line equation. I am curious whether current LLMs also perform such autoformalization. Moreover, if an LLM conducts multiple reasoning steps during autoformalization, would it still be considered equivalent under the proposed metric?

What is the average computation time of TED and TransTED during evaluation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
ASSESS is a two-stage framework to evaluate similarity between formal statements by combining structure and semantics. It parses statements into Operator Trees, then scores them with TransTED Similarity—an enhanced tree edit distance that applies semantic-preserving transformations. The authors introduce EPLA, a 524-pair expert-annotated benchmark (from miniF2F and ProofNet) labeled for both semantic provability and structural likeness. On EPLA, TransTED achieves state-of-the-art accuracy and the highest Cohen’s Kappa, outperforming string-, syntax-, proof-, and LLM-judge baselines. It is CPU-only, reproducible, provides continuous similarity scores, and ablations show the transformation component drives the gains.

### Strengths
- The proposed TransTED metric effectively combines syntactic and semantic information, overcoming the major limitation of existing purely structural or proof-based methods.  
- Introduces a semantically enhanced Tree Edit Distance that accounts for logical transformations, providing graded similarity instead of binary judgments.  
- Builds the largest expert-annotated dataset for this task (524 pairs), allowing rigorous and reproducible evaluation.

### Weaknesses
- No source code is provided.
- Why not compare with models like Kimina-Autoformalizer-7B and Goedel-Formalizer-V2-8B?  
- Perhaps you could compare the recall accuracy with some embedding-based retrieval systems.

### Questions
Please refer to Weaknesses.

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
3

### Summary
This work introduces ASSESS, a two stage evaluation framework that first translates natural language formal statements using the Lean Language Server into Operator and then measures the distance between the translation and a ground truth via a new metric TransTED. TransTED is a core contribution of the paper, and it extends from Tree Edit Distance by applying transformations to Operator Trees that surface semantic similarity (making the metric less prone to structurally different but semantically equivalent parses of formal statements). To prove the effectiveness of ASSESS the authors introduce a new dataset in this work, EPLA. EPLA is the other core contribution of this paper, and consists of 524 annotated pairs of natural language proofs and their formal translations annotated by experts.

The authors show that on EPLA, their method (ASSESS) consistently outperforms traditional evaluation metrics for autoformalization (this is a meta-evaluation, their evaluation framework is better at evaluating the task of auto-formalization).  The authors take care to compare against several common approaches for evaluating autoformalization including string based approaches, model-based approaches (like DeepSeek majority voting), and proof-based approaches.  ASSESS not only achieves higher accuracy, but has clear benefits to the alternative methods (including lower computational budget and reproducibility).  The authors compare their method by using a simple "Tree Edit Distance" (no translations for surfacing semantic similarity) and find that it leads to significant degradations when removed, showing the efficacy of their TransTED metric.

### Strengths
- **A large contribution: a dataset benchmarking evaluators for autoformalization + a method for evaluations**, a lot of work has gone into this paper, and it shows. The dataset EPLA is a nice contribution to the field for researchers working on evaluating and measuring the effectiveness of auto-formalization methods.
- **Strong main results: the leading approach behind theirs requires large amounts of compute (deepseek)**, excluding their ablated method, in order to achieve both high precision and recall when evaluating auto-formalization, their method appears to be the best as well as the most cost-effective.
- **Well written and clear results**, the math is a bit dense, but the paper overall is quite clear and well written. The graphs and tables show the results of their experiments clearly; figures 1 and 2 also help make their contributions a bit more concrete (showcasing an overview of the ASSESS framework and an example of an Operator Tree, respectively).

I lack the expertise to critically evaluate if the math theorems and proofs introduced in the paper are accurate or well-constructed, but the prose surrounding them was clear enough for me to understand and follow. Overall, this paper seems like a good contribution.

### Weaknesses
- **A lack of analysis** The method introduced does perform the best over all other evaluation methods; however, there is no discussion on the failure modes of the proposed method.  This makes it a bit difficult to understand where improvements could be made or more research is necessary (the EPLA dataset has some room left for improvement, so this may be a nice addition to the work).  *It is possible that some of this is discussed in the Appendix, and I missed it; if so, I'd still like a callout in the main body.*
- **Lack of ablations** There is one ablation, that of removing all the translations and using "TED" as the metric in ASSESS, but I do wonder how much of the gain TransTED gets from having each individual translation.  It may be worrisome if most of the gain from TransTED came from only a few of the translations in Table 1, or conversely, it may be insightful to see that all the translations improved the performance of TransTED, indicating that maybe future researchers could focus on devising more of these in order to push performance on EPLA further.  I think what I'd like to see is more discussion on whether the translations introduced provide the most gain and other translations won't help that much, or if adding a whole library of translations could improve performance significantly.


---

Minor notes:
- There's a weird artifact on lines 293 or 294 where the top of the text is slightly cut off.

### Questions
- Why is there no mutually provable & like* label? (I may be misunderstanding the relation between mutually provable and structurally similar under transformation)

### Soundness
4

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
This paper tackles a practical problem in automated theorem proving: how to automatically evaluate whether a machine-generated formal statement correctly captures the meaning of a natural language mathematical problem. The authors propose ASSESS, a framework that tries to balance both the structural form and semantic meaning when comparing formal statements.

**The approach works in two steps.** 

First, they use the Lean Language Server to parse formal statements into operator trees (OPTs), tree structures that represent the syntactic hierarchy of the statement. They apply some preprocessing like adding placeholders and removing redundant parentheses to standardize these trees. 

Second, they compute a similarity score using their proposed TransTED metric, which extends the classic Tree Edit Distance algorithm by incorporating semantic transformations. These transformations are implemented as Lean tactics (like `apply congrArg`, `ext`, etc.) that can recognize when two syntactically different statements are semantically equivalent. The algorithm uses heuristic search with TED as the guiding heuristic to find the minimum achievable distance.

**The paper makes three main contributions.** 

- First is the TransTED Similarity metric itself, which outputs a continuous score between 0 and 1 rather than a binary decision. The authors provide theoretical justification through Theorem 1, which frames the problem in terms of pseudometric spaces. 

- Second, they introduce EPLA, a new benchmark with 524 human-annotated statement pairs drawn from miniF2F and ProofNet. Each pair gets fine-grained labels covering both semantic correctness (provable vs unprovable) and structural similarity (like vs unlike), creating five categories in total. 

- Third, they show experimental results where TransTED achieves 78.82% accuracy on miniF2F and 70.86% on ProofNet, with Kappa scores of 0.46 and 0.40, beating several baselines including BLEU, LLM-based voting, and proof-based methods.

The authors argue their method offers a sweet spot. It's more semantically aware than simple string matching, more robust than brittle proof-based approaches, and more efficient and reproducible than using LLMs as judges.

### Strengths
**Originality**

The paper makes several novel contributions. Applying operator trees to formal statement evaluation bridges mathematical information retrieval techniques to theorem proving in an unexplored way. The key innovation **augmenting Tree Edit Distance with semantic transformations via Lean tactics** elegantly addresses the challenge that semantically equivalent statements can have very different syntactic forms. The pseudometric space formulation (Theorem 1) provides theoretical grounding. The EPLA benchmark's five-category annotation scheme is notably more nuanced than existing binary evaluation datasets.

**Quality**

The technical approach is sound. Leveraging the Lean Language Server for OPT construction is pragmatic engineering. The choice of transformation tactics (Table 1) reflects good mathematical intuition. The experimental evaluation covers diverse baselines (string-based, proof-based, LLM-based). The ablation study provides clear evidence that transformations contribute meaningfully (+6.70% on miniF2F). Implementation details demonstrate attention to reproducibility.

**Clarity**

The paper is well-structured with logical progression from motivation through methodology to experiments. Mathematical notation is introduced systematically with helpful examples. Figure 3 effectively demonstrates TransTED's robustness to threshold selection compared to BLEU's sensitivity. Results are presented with appropriate metrics and clear tables.

**Significance**

The work addresses a real bottleneck where existing evaluation methods have well-documented limitations. An efficient (CPU-only), reproducible automated metric could significantly accelerate research by enabling faster iteration without expensive human evaluation. Outperforming LLM-based majority voting while being much cheaper suggests genuine practical utility. The EPLA benchmark could become a standard evaluation resource for the community.

### Weaknesses
**1. Missing Citation of Highly Related Concurrent Work [Major - Critical for Novelty Assessment]**

The paper does not cite GTED (Liu et al., "Generalized Tree Edit Distance: A Faithful Evaluation Metric for Statement Autoformalization", arXiv preprint, arXiv:2507.07399v1, July 10, 2025), despite substantial overlap in core contributions. Both works use operator trees extracted via Lean Language Server, extend Tree Edit Distance with semantic transformations, and evaluate on miniF2F and ProofNet. The timeline raises concerns: GTED appeared on arXiv on July 10, 2025, while the ICLR 2026 deadline was September 24, 2025, where a 2.5-month gap that should have been sufficient for discovery and citation, especially given the direct relevance to this work.

Without proper comparison, several critical questions remain unanswered. I cannot assess whether the pseudometric formulation (Theorem 1) is substantially different from GTED's "generalized tree transformation" framework, or whether they are merely different mathematical presentations of the same underlying idea. The transformation sets appear different quantitatively (ASSESS: 6-7 tactics vs GTED: primarily α-conversion), but it's unclear if this represents a genuine methodological advance or simply different implementation choices. Most importantly, the reported performance differences (GTED: 70.73% accuracy on miniF2F vs ASSESS: 78.82%) cannot be properly interpreted without knowing whether they stem from algorithmic improvements or merely different evaluation datasets.

If the authors were genuinely unaware of GTED during development, this represents an unfortunate collision of independent work and should be acknowledged. However, if they were aware but chose not to cite it, this falls into more serious issues of academic integrity. Either way, the current manuscript must address this relationship. I believe the contribution lacks sufficient demonstrated novelty until the authors: (1) explicitly cite GTED and clarify their awareness timeline, (2) provide detailed technical comparison showing what is genuinely new beyond GTED, and (3) ideally, evaluate both methods on identical test sets to enable fair performance comparison.

I believe this issue significantly impacts my assessment of the paper's originality and must be resolved for acceptance.

---
**2. Dataset weakness**

2.1 Selection bias

The benchmark includes samples that successfully compile after translation, creating severe selection bias. Based on GTED's reported numbers using Herald translator alone, the compilation success rates are approximately 84% for miniF2F (205/244) and only 25% for ProofNet (93/371). The ProofNet success rate is particularly concerning, the evaluation excludes 75% of the original test set, precisely the cases where translation is most challenging.

While ASSESS uses both Herald and Gemini translators (producing 373 miniF2F pairs and 151 ProofNet pairs as shown in Table 2), the underlying selection bias remains: the metric is only evaluated on cases where at least one translator produced compilable output. We have no information about how TransTED would behave on the harder cases where translation completely fails. This severely limits conclusions about the metric's general applicability.

2.2 Translator bias

The benchmark relies exclusively on two translators (Herald and Gemini 2.5 Pro). Different autoformalization systems may produce systematically different output patterns, where some favoring verbose formulations, others producing compact statements, or using different Mathlib conventions. TransTED's performance might not transfer to outputs from other translation systems. Including diverse translators would strengthen claims about the metric's general applicability. This selection bias severely limits the conclusions we can draw about the metric's general applicability. The paper should explicitly discuss this limitation and its implications.

2.3 Simple statement-only evaluation

Both miniF2F and ProofNet contain only simple, single-statement theorems , and typically one-line formulas expressing direct mathematical relationships. Real world formalization is substantially more complex, often involving custom definitions (def), structured types (structure, inductive), multi-statement developments with auxiliary lemmas, and nontrivial type hierarchies. 

For example, a predicted formalization might define a helper function that accomplishes the same mathematical purpose as the ground truth's , but with different naming and internal structure. The current OPT-based approach, which only compares the final theorem statement in isolation, might struggle with such cases. The paper provides no discussion of this limitation. It remains unclear whether TransTED can handle complex multi-component formalizations, whether OPT construction can parse and align definitions appropriately, or whether transformation tactics generalize beyond simple theorems.

### Questions
**Q1: Relationship with GTED**

As noted in the weaknesses, GTED has substantial overlap with your work. This is the most critical issue affecting our assessment:

- Can you clarify whether you were aware of GTED during development?
- Please provide detailed technical comparison showing what is genuinely new in your approach
- Would you be willing to evaluate both methods on identical test sets for fair comparison?

Your response will directly determine whether we view this as independent innovation or incremental refinement of GTED.

---

**Q2: Handling complex formalizations [Important - could mitigate scope limitation]**

The evaluation focuses on simple, single-statement theorems. We understand this reflects the datasets used, but it raises questions about practical applicability.

- Have you conducted any experiments (even informal) with formalizations containing definitions, structures, or multi-statement developments?
  
- Can you discuss theoretically how the OPT construction and transformation tactics would handle such cases?

Even preliminary evidence or principled discussion would help assess whether the scope limitation is fundamental or merely reflects current evaluation constraints.

---

**Q3: Transformation tactics design [Moderate - would strengthen contribution]**

Table 1 presents seven transformation tactics. Can you provide more details like

- How were these tactics selected? Systematic exploration or intuition?
- Have you analyzed their individual contributions (ablation study)?
- Which tactics are most frequently used or impactful?
This would help understand the design rationale and completeness of the transformation set.

---

**Q4: Statistical reliability of improvements**

For ProofNet (151 samples), the +3.31% improvement represents only about 5 additional correct predictions. Have you checked whether this difference is statistically significant? A simple test would help confirm the reliability of this improvement, though the miniF2F results (+6.70% on 373 samples) are more clearly substantial.

### Soundness
3

### Presentation
3

### Contribution
3
