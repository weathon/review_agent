# Quantifying Edits Decay in Fine-tuned LLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Knowledge editing has emerged as a lightweight alternative to retraining for correcting or injecting specific facts in large language models (LLMs). Meanwhile, fine-tuning remains the default operation for adapting LLMs to new domains and tasks. Despite their widespread adoption, these two post-training interventions have been studied in isolation, leaving open a crucial question: if we fine-tune an edited model, do the edits survive? This question is motivated by two practical scenarios: removing covert or malicious edits, and preserving beneficial edits. If fine-tuning impairs edits as shown in Figure 1, current KE methods become less useful, as every fine-tuned model would require re-editing, which significantly increases the cost; if edits persist, fine-tuned models risk propagating hidden malicious edits, raising serious safety concerns. To this end, we systematically quantify edits decay after fine-tuning, investigating how fine-tuning affects knowledge editing. We evaluate two state-of-the-art editing methods (MEMIT, AlphaEdit) and three fine-tuning approaches (full-parameter, LoRA, DoRA) across five LLMs and three datasets, yielding 232 experimental configurations. Our results show that edits decay after fine-tuning, with survival varying across configurations, e.g., AlphaEdit edits decay more than MEMIT edits. Further, we propose selective-layer fine-tuning and find that fine-tuning edited layers only can effectively remove edits, though at a slight cost to downstream performance. Surprisingly, fine-tuning non-edited layers impairs more edits than full fine-tuning. Overall, our study establishes empirical baselines and actionable strategies for integrating knowledge editing with fine-tuning, and underscores that evaluating model editing requires considering the full LLM application pipeline.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the interplay between knowledge editing (KE) and fine-tuning (FT) in LLMs. The authors systematically evaluate the persistence of knowledge edits after fine-tuning, examining multiple editing methods (MEMIT, AlphaEdit) and fine-tuning approaches (full-parameter, LoRA, DoRA) across five different LLM architectures. To quantify the preservation of edits, they introduce the Edit Flip Ratio (EFR), demonstrating that knowledge edits generally decay after fine-tuning. Additionally, the paper explores how selective-layer fine-tuning strategies affect the retention or removal of edited knowledge.

### Strengths
1. This paper investigates an interesting and underexplored aspect of the LLM lifecycle: the interaction between knowledge editing and subsequent fine-tuning, which is an important factor in assessing the efficacy of knowledge editing.
2. The introduction of the Edit Flip Ratio (EFR) provides a direct and interpretable measure of edit retention.
3. The authors not only quantify edit decay but also discuss the impact of selective-layer fine-tuning strategies on edit persistence.

### Weaknesses
1. The experimental setup is limited and somewhat overstated. Regarding editing methods, only two approaches (MEMIT and AlphaEdit) from the same paradigm (locate-then-edit) are examined, while other paradigms such as fine-tuning and meta-learning are not explored. As for LLMs, although the authors claim to examine five models, the DeepSeek model is excluded due to poor editing efficacy, and results for Llama3 are not reported.
2. The authors have a limited understanding of the editing datasets used and present some incorrect information about them. Specifically, according to [1], the COUNTERFACT dataset contains 21,919 samples, not 21,890 as stated in the paper. Additionally, as noted in [2], the editing version of the ZsRE dataset consists of counterfactual statements, where the editing target for a given question is an alternative prediction rather than the ground truth, contrary to the claim in the paper that they are real-world factual statements.
3. Although this paper demonstrates that fine-tuning can remove knowledge injected through editing, this may be part of the catastrophic forgetting induced by fine-tuning, a common phenomenon in which knowledge acquired during pretraining can also be lost. Therefore, it is important to compare the retention of edited knowledge with that of pretraining knowledge. However, this comparison is not addressed in the paper.

[1] Locating and Editing Factual Associations in GPT

[2] Editing Factual Knowledge in Language Models

### Questions
Although the authors repeatedly emphasize their motivation to examine the dual goals of preserving beneficial edits and removing malicious ones, their analysis focuses solely on the overall retention of edits. Consequently, the conclusions are not directly aligned with the stated dual goals.

### Soundness
2

### Presentation
2

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
This paper investigates the under-explored interaction between two common language model update techniques: knowledge editing (KE) and fine-tuning (FT). The authors argue that these methods have been studied in isolation, leaving open the critical question of whether knowledge edits persist after a model undergoes subsequent fine-tuning. The paper systematically quantifies this "edit decay" across a wide range of configurations, involving multiple LLMs, KE methods, and FT approaches. Their findings demonstrate that fine-tuning generally impairs edits, and they explore factors influencing this decay, proposing selective-layer fine-tuning as a potential mechanism for controlling edit persistence

### Strengths
1.The problem is clearly defined, timely, and highly relevant for the real-world deployment and maintenance of LLMs.

2.The paper is easy to read.

3.The authors provide basic experiments to validate the proposed method.

4.The study is methodologically sound and impressively comprehensive

### Weaknesses
1.	All experiments were fine-tuned using only the "common sense reasoning" dataset. This severely limits the generalizability of the conclusions. The relevance of the fine-tuning data to the edited facts is a key variable in determining the degree of edit decay, which this paper does not explore at all.
2.	The paper introduces the "Edit Flip Ratio" as a new metric. However, as defined, it measures the proportion of initially successful edits that become unsuccessful after fine-tuning. This is a form of conditional probability. The paper itself notes (Section 4.1) and demonstrates (Table 3) that EFR follows similar trends to the simple decrease in Efficacy Success (△ES). This raises questions about whether EFR is a fundamentally necessary new metric or a re-framing of existing ones that offers limited additional insight.
3.	The analysis in Section 4.3 reveals a fascinating but complex result: fine-tuning non-edited layers also impairs edits, in some cases more than full fine-tuning. This finding seems to contradict the simple hypothesis that edits can be preserved by freezing the layers where they were made. The paper's conclusion could more deeply address this tension, as it has profound implications for the concept of knowledge localization that underpins many KE methods. The current framing as an "actionable strategy" is slightly undermined by this complexity.
4.	The study exclusively uses a commonsense reasoning dataset for all fine-tuning tasks. While a valid choice, the dynamics of edit decay could differ significantly under other common FT scenarios, such as instruction-tuning on diverse tasks, domain adaptation to a specialized corpus. The paper's conclusions about FT's impact on KE may not generalize to these other critical use cases.
5.	The paper found that AlphaEdit's edits decay more easily than MEMIT, but did not analyze why fine-tuning has different effects on the two methods based on their underlying mechanisms.

### Questions
1.	The example in Figure 1, where fine-tuning changes the model's output to "Elon Musk," is illustrative but feels arbitrary. A more realistic outcome of fine-tuning would be the model reverting to the original fact ("Joe Biden") or producing a more semantically related but incorrect answer, which would better reflect the mechanisms of knowledge decay discussed in the paper.

### Soundness
3

### Presentation
3

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
This paper studies how knowledge edits in large language models decay after fine-tuning. By evaluating two editing methods (MEMIT and AlphaEdit) and three fine-tuning strategies (full, LoRA, DoRA) across multiple models and datasets, the authors find that fine-tuning generally weakens or removes prior edits, with the extent of decay varying by method, model, and task. Selective fine-tuning of edited layers can erase edits more effectively but slightly reduces downstream performance, while tuning non-edited layers does not preserve them. The study concludes that knowledge editing and fine-tuning are closely intertwined processes, emphasizing the need for future editing methods that remain robust under fine-tuning.

### Strengths
1. The experiments are highly comprehensive, covering comparisons across different baselines and a wide range of dataset configurations and parameter combinations.

2. The paper tackles a relevant and timely problem and presents its analysis in a clear and organized manner, offering useful empirical observations about how knowledge editing interacts with fine-tuning.

### Weaknesses
1. Many of the conclusions in this article are not particularly novel and can be drawn without the need for experimental verification. For instance, the author mentions that larger models have greater robustness, which is obvious. As the model size increases, the orthogonality among the internal representations within the model becomes stronger, and thus the impact is naturally smaller. Another example is that the author states that full fine-tuning causes more damage to editing than LORA and DoRA. From the perspective of the low-rank structure in the LORA category, this conclusion is quite clear. Or, the conclusion that fine-tuning the non-editing layer has a greater impact on editing than fine-tuning the editing layer is also quite obvious. From the perspective that the FFN layer of large models is kv storage, fine-tuning the non-editing layer causes the k to change in the editing samples, thereby causing a change in v, and ultimately leading to a decline in editing performance. While fine-tuning the editing layer directly affects the mapping relationship between kv, which leads to a decline in editing performance. In summary, from the perspective of viewing editing as local fine-tuning, many of the conclusions summarized by the author are obvious and have been widely studied in the field of fine-tuning.

2. Some of the logical relationships mentioned by the author are also somewhat awkward to understand, as exemplified by lines 403 to 411. The author mentioned, "Taken together, these findings provide a feasible approach for removing unwanted edits by fine-tuning only the edited layers." In the previous text, the author stated that fine-tuning the unedited layers could also remove the edits. However, when fine-tuning the editing parameters, although the edits could be deleted, it would have an impact on the downstream tasks. But in the summary of "taken together", the author overlooked this drawback of editing the fine-tuned layers, and did not analyze why not to delete the unwanted edits by fine-tuning the unedited layers. The conclusion was made directly without further analysis.

3. It is suggested that the author add some simple and feasible methods. As this article is an analytical piece, it does not draw many valuable conclusions. It is recommended that the author propose some adjustments during the stage of using KV cache to construct loss constraints, etc.

### Questions
See weaknesses.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the stability of knowledge edits in large language models (LLMs) after subsequent fine-tuning, filling a notable gap in the current literature. The authors systematically study how fine-tuning (both full-parameter and parameter-efficient variants such as LoRA and DoRA) affects knowledge edited using state-of-the-art techniques (MEMIT and AlphaEdit) across multiple LLMs and datasets. Extensive experiments reveal that most knowledge edits are unstable under fine-tuning, with both practical and safety implications. The paper further explores selective layer fine-tuning as a potential strategy for controlling edit persistence and offers detailed benchmarks for the field.

### Strengths
- The paper tackles a highly practical and timely question—how robust are model knowledge edits to downstream fine-tuning—via a comprehensive empirical framework. By evaluating two prominent knowledge editing algorithms (MEMIT, AlphaEdit) and three fine-tuning strategies (full, LoRA, DoRA) on several LLMs and datasets, the paper delivers a broad and reproducible empirical baseline.
- The work convincingly argues that KE robustness to fine-tuning needs to become a standard evaluation axis, impacting both future algorithm design and LLM safety auditing.

### Weaknesses
- While the empirical results are extensive, the paper only speculates (Section 4.3, Discussion) on why edits decay so markedly under fine-tuning. There is little theoretical or mechanistic explanation of why edited knowledge appears so fragile or dispersed, limiting the generality of insights. For instance, Section 4.3 suggests that edits are not strictly localized, but this is not formalized or linked to deeper theory.
- The centrality of the Edit Flip Ratio (EFR) is well-motivated, but the description in Section 3 could be improved for precision. The mathematical definition for EFR in Formula 1 is somewhat ambiguously written, and the text should clarify if EFR is computed only over edits initially successful in $M_{ed}$, especially for the counterfactuals where success rates are lower. Furthermore, equations related to metric definitions in Section 3 and Appendix C are scattered and would benefit from being presented compactly and with more explicit notation for all variables.
- The fine-tuning dataset (Commonsense Reasoning, Section 3.1) and edits (zsRE, COUNTERFACT) are both focused on factual/toy domains. It remains unclear if the observed patterns would generalize to more complex or knowledge-rich downstream tasks.

### Questions
Same as Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
