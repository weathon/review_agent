# A Hybrid Feature Tree-Based Approach for Explainable LLMs in Domain-Specific Knowledge Management

- Decision: Reject
- Scores: 2, 0, 2, 2

## Abstract
The "black-box" nature of Large Language Models (LLMs) poses a significant barrier to their adoption in high-stakes, regulated domains like finance and healthcare, where verifiable explanations are mandatory. We propose a novel hybrid framework that enhances LLM explainability by generating hierarchical feature trees from individual Question-Answer (Q\&A) pairs and merging them into a unified, global "Uber Tree." This structure provides both local explanations for specific answers and a global overview of the model's knowledge landscape. Our method combines the semantic understanding of LLMs for tree generation and merging with traditional recursive algorithms for robustness, ensuring scalability. Crucially, we introduce a formal consistency verification step to validate the alignment between individual explanations and the global knowledge structure. Applied to the domain of mortgage compliance using a comprehensive dataset of 1000 Q\&A pairs, our framework demonstrates high-quality tree generation, effective merging that outperforms purely algorithmic baselines, and strong consistency (95\%). A human evaluation with domain experts confirms a significant improvement in explainability and auditability over standard Chain-of-Thought explanations. This work offers a practical pathway toward auditable and verifiable AI systems at enterprise scale.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a method for meta-analysis of Q&A systems using different tree-generation and aggregation prompts. The authors aren't completely clear which domain(s)/use cases necessitate such a metadata tree. The authors evaluate their method on a Chain of Trees baseline on a dataset of mortgage compliance Q&A.

### Strengths
1. Overall the method is well motivated, in this specific domain that attribute trees can be constructed and verified. 

2. The authors give some qualitative evaluation, though it would be greatly improved over more domains and a higher number of domain experts.

### Weaknesses
1. The paper doesn't clarify their key contributions well. The related work is in 5 areas that the authors don't give sufficient technical depth/differentiation. 

2. The authors dont give sufficient technical details on the merging step. On 3.2.1 the authors dont give a clear algorithm or prompt generation. Furthermore, the authors arent clear what a 'good' aggregated tree ought to contain. There is no measure on candidate trees. If it's left to the LLM through prompting, this is really not a suitable contribution. 

3. The human evaluation is lacking. 3 expert labelers over 30 instances is pretty light. But again, the authors haven't educated the reader on the tree quality. Please give specific examples of the human evaluation task. Demonstrate the tree differences side-by-side. 

Furthermore, the measures aren't even given(!?) in Table 3? What are the value scales for FT, CoT columns? What is the specific p-value test?  

4. The tree evaluation is quite unintuitive and needs more space. In table 2, the authors note 100% success-rate with a heuristic non-LLM fallback. OK? It's not obvious from table 2 that Hybrid tree construction is better. It's sparser, but we can always add sparsity penalties to the other methods. Since there's no(?) tree-level quality scoring, it seems any method could 'game' the statistics on this table.

### Questions
1. What is the prompt for tree merging? The specific prompt. 

2. Is there an explicit measure for tree quality?

3. Can you visually present the evaluation task? What are all the columns in Table 3 representing, and why is FT qualitatively better?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The paper introduces a method to explain LLM outputs in regulated domains, e.g. finance, and health. They do so by restructuring the outputs of LLMs in verifiable feature trees that are then merged into a single global "Uber Tree" representing the global structure of the model. The method is tested on a Q&A dataset of mortgage applications.

### Strengths
The paper addresses a timely and important challenge: improving explainability of LLMs in sensitive, high-stakes domains. Introducing structural constraints to the model’s outputs to better capture and understand its internal reasoning structure is a well-motivated and sensible approach. The proposed hybrid framework—combining LLM-based semantic abstraction with algorithmic merging—demonstrates technical creativity and practical relevance, especially for regulated areas like mortgage compliance where interpretability and auditability are essential.

### Weaknesses
While the use of structured outputs is a sensible way to probe LLM behavior, the approach falls short of what would be required in a real regulated deployment. In such settings, structural constraints should ideally be integrated into the model’s training or architecture, not imposed only through prompt engineering or in-context control.

Although the paper claims broad applicability to regulated domains, all experiments are conducted solely on mortgage compliance data. For a study submitted to ICLR, this narrow scope—limited to a single dataset and a single proprietary LLM—restricts the generalizability and scientific contribution of the work.

Finally, reliance on a closed-source, proprietary LLM undermines the paper’s stated goal of developing verifiable and auditable AI systems. Guarantees in regulated environments require transparency about the predictive model’s inner structure, which is not available here.

### Questions
Have you considered evaluating the proposed framework on datasets from other regulated domains such as healthcare or legal compliance, to test its generalizability beyond the mortgage domain?

Rather than constraining outputs only through prompting, did you explore modifying or fine-tuning LLMs to natively generate structured, hierarchical explanations? If not, what do you see as the main barriers to doing so?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a way to improve LLM explainability: generating hierarchical feature trees from individual QA pairs, and merging into a unified global "Uber Tree". They chunk the individual feature trees and prompt the LLM to recursively merge them chunk by chunk, removing duplicated information and getting globally semantically related nodes. They use the merged Uber Tree to check if invidiual trees are consistent with it and find that they are mostly very consistent, showing they are consistent in the internal knowledge being used. They validated with human experts and show that these feature trees a re more clear than cot, but not as comprehensive, and maybe similarly easy to verify.

### Strengths
1. The paper is well-written and easy to understand.
2. The paper gives a comprehensive structural and semantic evaluations of the tree-based method, demonstrating that the merged hierarchy and the individual explanations are internally consistent, having the shared consistent domain knowledge.

### Weaknesses
1. In the human evaluation, the authors only compared clarity, comprehensiveness, and ease of verification between feature trees and CoT, but this could simply be because FT is a tree structure. There are other merging methods, and they did not report human evaluation results on those.
2. The authors mainly validated whether LLMs can generate the required JSON structure (which current LLMs can already do reliably), whether the global knowledge looks coherent and abstract (which makes sense since they use the same Uber Tree), and whether the outputs have high internal consistency. However, they did not evaluate the accuracy of the results against any external ground truth, so a tree could be coherent and self-consistent but still factually incorrect.
3. It is also unclear whether the explanations themselves align with expert intuition. The human study only evaluated presentation-level properties (clarity, comprehensiveness, ease of verification), but did not assess whether the key factors, logical steps, or abstractions actually match how domain experts reason about these regulations.
4. In the abstract, they authors claim that the proposed method improved auditability, but the human evaluation in Table 3 shows no statistically significant improvement (p=0.5834) in "Ease of Verification" between feature trees and CoT explanations, weakening this claim.
5. The authors did not evaluate the faithfulness of the feature tree-based explanations. Thus, even if the explanations are clear, we don't know if they can inform the correctness of the answer, and thus we do not know if the proposed method makes the explanation more auditable.

### Questions
1. Could you do human evaluation also on the other tree-based baselines?
2. What are the end accuracy of the answers following each method's explanation?
3. Do the experts agree on the domain-knowledge being used by the LLM despite thinking the explanations are clear?
4. How faithful are the explanations and corresponding answers? How often do the answers logically follow the explanations? How logically consistent is each feature tree internally?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a framework to enhance the explainability of LLMs in high-stake domains. The method uses an LLM to generate structured, hierarchical Feature Tree from an individual Q&A pair. Then merges these local trees into Uber Tree. The paper verifies consistency where an LLM audits the alignment between Uber Tree and local Feature Trees, producing a consistency score. The framework evaluated on the mortgage compliance dataset, achieved high structural consistency and improved clarity over CoT explanation in human expert evaluation.

### Strengths
1. The paper is well-written and easy to follow.
2. Consistency verification step is a compelling idea.
3. The experiment is done comparing with human evaluation with domain expert.

### Weaknesses
1. The problem statement in the introduction is very broad. The described challenge is fundamental to almost all xai research.
2. The main text of the table needs to provide guidance on how to interpret the results in the table (e.g., what the takeaway is).
3. The proposed Feature Tree seems like a post-hoc structuring of the answer's content generated by a separate LLM call which is a bit different from explanation of LLM's internal reasoning process for generating an answer (like LIME, SHAP as mentioned as limitations in the paper)
4. The paper does not clearly specify whether for the generation of the Feature Tree from the pair, which is a post-hoc step, is the LLM given only the text of the question and answer of does it have access to the reasoning trace of the model that produced the pair.
5. The experiment is only done in mortgage compliance domain. I wonder the experiment result of other high-stake domains like medicine.
6. The experiment relies on one LLM (GPT 4.1) which lacks generalization.
7. The human evaluation only involves 3 domain experts which lacks statistical power to draw generalizable conclusions. Also need to give specific information about the domain experts.

### Questions
Look at the weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
