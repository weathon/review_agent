# Dynamic Weight Grafting: Localizing Finetuned Factual Knowledge in Transformers

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
When an LLM learns a new fact during finetuning (e.g., new movie releases, newly elected pope, etc.), where does this information go? Are entities enriched with relation information immediately, or do models recall information just-in-time before a prediction? Or, are "all of the above" true, with LLMs implementing multiple redundant heuristics? Existing localization approaches (e.g., activation patching) are ill-suited for this analysis because they usually replace parts of the residual stream, thus overriding previous information.
   To fill this interpretability gap, we propose dynamic weight grafting, an analysis technique that selectively grafts subsets of weights from a finetuned model onto a pretrained model. Using this technique, we show two separate pathways for retrieving finetuned relation information: 1) "enriching" the residual stream with relation information while processing the tokens that correspond to an entity (e.g., "Zendaya" in "Zendaya co-starred with Timothée Chalamet" and 2) "recalling" this information at the final token position before generating a target fact. In some cases, models need information from both of these pathways to correctly generate finetuned facts while, in other cases, either the "enrichment" or "recall" pathway alone is sufficient. We localize the "recall" pathway to model components---finding that "recall" occurs via both task-specific attention mechanisms and an entity-specific extraction step in the feedforward networks of the final layers before prediction. By targeting model components and parameters, as opposed to just activations, we are able to understand the mechanisms by which finetuned knowledge is retrieved during generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces dynamic weight grafting, a technique to localize fine-tuned knowledge retrieval mechanisms in LLMs. Unlike activation patching (which replaces activations and destroys previous computations), dynamic weight grafting selectively swaps weights from finetuned models into pretrained models at specific layers, components, and token positions. The authors identify two retrieval pathways: (1) "enrichment" at entity tokens and (2) "recall" at the final token position. Key findings: both pathways together nearly recover full fine-tuning performance and grafting everything except these pathways yields near-zero accuracy. Recall operates through task-specific attention at entities/final token and relation-specific FFNs in final layers. Experiments use Llama3, Pythia, GPT2-XL, Gemma on synthetic relation completion datasets.

### Strengths
Addresses previous method gap: activation patching conflates which components compute vs. pass through information, weight grafting isolates true mechanisms. Conducted ablations demonstrate both pathways are sufficient along with complement grafting to back the claims. Consistent results across multiple models Llama3/Gemma (strong recall) and GPT2-XL/Pythia (weaker recall, stronger enrichment). Granularity is impressive, localizes recall to ATTN O matrices + FFNs with task-specific attention, more precise than prior feedforward/attention papers. Importantly tests training task-specific vs. relation-specific models.

### Weaknesses
Synthetic data limitation, may not reflect true complexity in other natural language prompts. Only relation completion is tested. I wonder if the findings extend to other tasks like reasoning or open-ended generation?

A few of the final conclusions were already known.

### Questions
How do results change with other types of prompts (e.g., Wikipedia text)? Are pathways applicable to paraphrasing beyond templates?
What happens with real factual data where models have partial pre-existing knowledge (compared to fully synthetic facts)?
Can the authors explicitly highlight the findings of their work that were already known and the ones that were novel?

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
This paper proposes "dynamic weight grafting," a novel interpretability method that intervenes on model parameters (weights) instead of activations, avoiding the limitations of activation patching. Using this method, the authors investigate knowledge retrieval from SFT. They find that new knowledge is retrieved via two primary, localized pathways: an "Enrichment" (E) path at the entity's tokens and a "Recall" (R) path at the final token. Experiments establish these pathways as both necessary and sufficient. The paper further uses component grafting to localize the "R" pathway to specific attention mechanisms, O-matrices, and FFNs.

### Strengths
**Novel and Sound Methodology**: The core contribution, "dynamic weight grafting," is a clever and valuable addition to the interpretability toolkit. It correctly identifies a key flaw in standard activation patching (conflating information computation with information passing) and proposes a more precise causal intervention by swapping the mechanisms (weights) themselves.

**Clear, Rigorous Findings**: The identification of the 'E' and 'R' pathways is a clear and compelling finding. The authors were rigorous in their analysis, using complement experiments ((FE+LT)^C) to establish both the sufficiency and necessity of these pathways.

**Deep Mechanistic Localization**: The paper doesn't stop at the token level. The component grafting experiments in Section 3.3 are particularly strong, using the "reversal curse" setup to dissect the "R" pathway into its constituent parts (task-specific attention vs. relation-specific FFN/O-matrix). This provides a granular, plausible mechanism for how recall functions.

### Weaknesses
## Experimental Concerns

1. **Unexplained Model-Specific Strategies**: A significant concern is the inconsistent behavior across models, which the paper notes but fails to adequately explain. In Figure 2, the performance of LT (Last Token) versus FE^C (which includes LT) shows large, unexplained disparities in models like GPT2-XL. More importantly, the paper notes that models like Gemma/Llama favor a strong 'R' path, while GPT2-XL/Pythia favor an 'E' path. For an interpretability paper, why these divergent strategies emerge is as important as the fact that they do. The paper's attempt to correlate this with architecture (e.g., RoPE) is unsatisfying, as the groupings are inconsistent (Pythia has RoPE but groups with GPT-2). A more concrete hypothesis and-ideally-a simple experiment to test it are needed. The current explanation feels arbitrary.

2. **Limited Scope (Synthetic SFT Data)**: The reliance exclusively on synthetic, templated SFT data severely limits the generality of the findings. Are these 'E' and 'R' pathways a general mechanism for knowledge retrieval, or are they an artifact of this specific, narrow, templated SFT setting? One could argue that the model is simply overfitting to the template, learning to "plug in" information at the entity (E) or "look up" the answer at the end (R). Maybe it can include even a preliminary discussion or experiment using dynamic weight grafting to explore pre-trained knowledge (e.g., on LAMA probes), to contrast it with these SFT-induced mechanisms.

3. **Weakness of Top-5 Accuracy Metric**: The choice of Top-5 accuracy as the primary metric is questionable and potentially misleading. The authors' defense of this metric (citing "uncertainty" in Appendix A.2) is directly contradicted by their own data in Appendix C.9.2. An example for a Gemma 'LT' graft shows the target "Uta" with a probability of 0.008, while the incorrect token "John" has a probability of 0.901. This is not "uncertainty"; this is high-confidence error. This metric choice masks the important nuance of how a mechanism fails (e.g., by confidently predicting the wrong thing). Using a more sensitive metric like logit difference (on the correct token) or mean target rank (as shown in C.1.2) as the primary metric in the main text would be far more rigorous and convincing.

## Writing and Clarity Concerns

1. **Undefined Terms**: The core concepts of "enrich" (E) and "recall" (R) are introduced abruptly in Section 3.1 as if they are experimental conclusions. The paper fails to provide a clear, a priori definition of what information flow processes these terms are hypothesized to represent. What does it mean for a token to be "enriched"? This makes the results section difficult to follow, as the reader is learning the definitions from the experimental results themselves.

2. **Bloated and Repetitive Result Presentation**: The presentation of results in the main text and appendix is highly repetitive and buries the key insights. The appendix is inundated with dozens of charts (e.g., Fig 6, 7, 10, 11, 12) that all show the exact same experiment on different models or datasets, reinforcing the same basic point. The true insight—that different model architectures have different E/R preferences—is lost in this flood of redundant figures.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Dynamic Weight Grafting to localize how SFT-injected relational knowledge is retrieved from LLMs at inference time. Instead of activation patching, DWG swaps parameters of a fine-tuned model into a pre-trained model selectively by token position or component during generation. The authors construct synthetic relation datasets to train on GPT-2-XL, Pythia-2.8B, Llama-3, Gemma-1.1. The authors identify two retrieval pathways: (i) enrichment at the first-entity tokens and (ii) recall at the last token before prediction. Either pathway can partially recover SFT performance; FE+LT recovers near-SFT, while grafting the complement drops to pre-trained level. Component grafting further pins recall to task-specific attention at FE/LT and relation-specific O-projection + FFN in late layers.

### Strengths
- The paper proposes an innovative way to probe how relational knowledge is retrieved from parameters without directly overriding residual stream, like activation patching.
- Datasets and experiments are tightly controlled, yielding strong, clean and strong evidence that enrichment at the first token and recall at the last token jointly recover fine-tuning performance.
- Component grafting reveals insights for how late-layer O-projection + FFN at the final token and task-specific attention at the first token are necessary for retrieval and recall.
- Overall, the writing is very clear. The experiments and ablations are well designed and easy to reproduce.

### Weaknesses
- It is not clear if similar mechanism can generalize to real-world data and if the findings still hold as model size scales.
- The study is conducted on one-hop relational task; findings for position and component grafting may not transfer to multi-hop settings. The claims should be tempered accordingly to the scope of the task.

### Questions
1. How sensitive are conclusions or findings to evaluation choices? Was any human auditing performed to check robustness?
2. In Table 1, the QA appears synthetically phrased to mirror the headline. Do the findings persist under lexically diverse or paraphrased question formulations?

### Soundness
3

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
2

### Summary
The paper introduces Dynamic Weight Grafting (DWG), a new method for studying how fine-tuned knowledge is retrieved in Transformer-based LLMs. Unlike activation patching, DWG swaps model weights rather than activations, allowing analysis of mechanisms without overwriting upstream computations. Using this method, the authors identify two distinct retrieval pathways for fine-tuned facts:

- Enrichment Pathway: Relation information is integrated while processing entity tokens.

- Recall Pathway: The model retrieves stored information at the final token before prediction.

Experiments across several LLMs (Llama3, GPT-2 XL, Gemma, Pythia) show that both pathways together nearly recover fine-tuning performance, and either can sometimes suffice independently. The recall mechanism is localized to attention and feedforward layers in later Transformer blocks.

### Strengths
- Novel methodology: DWG is an original and potentially useful interpretability technique that avoids the destructive limitations of activation patching.
- Comprehensive experiments: Multiple models and datasets are tested, showing consistency of findings.
- Clear conceptual framing: The paper offers an intuitive distinction between “enrichment” and “recall” processes in LLM memory retrieval.
- Potential for broader application: The approach could generalize to other interpretability or knowledge editing analyses.

### Weaknesses
- Synthetic and simplistic setting: The experiments rely heavily on artificial datasets (e.g., fake movies and actors), which limits external validity.
- Limited theoretical insight: The method identifies where retrieval occurs but not why or how specific mechanisms encode relations.
- Insufficient analysis of failures: Cases where grafting fails (e.g., certain models or directions) are mentioned but not deeply analyzed.

### Questions
Please see the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3
