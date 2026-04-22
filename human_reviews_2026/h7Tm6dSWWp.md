# Focus Directions Make Your Language Models Pay More Attention to Relevant Contexts

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Long-context large language models (LLMs) are prone to being distracted by irrelevant contexts. The reason for distraction remains poorly understood. In this paper, we first identify the contextual heads, a special group of attention heads that control the overall attention of the LLM to the contexts. Then, we demonstrate that distraction arises when contextual heads fail to allocate sufficient attention to relevant contexts and can be mitigated by increasing attention to these contexts.
We further identify focus directions, located at the key and query activations of these heads, which control the amount of attention activated from the attention sink to the contexts.
With a proper amount of attention activation, the contextual heads could allocate more attention to relevant contexts.
Motivated by this, we introduce an automated magnitude control method that keeps attention activation within a proper range, enabling practical use of focus directions.
We comprehensively evaluate the effect of focus direction on various long-context tasks and find that focus directions can help mitigate the poor task alignment of long-context LLMs.
We believe our findings could promote further research on long-context LLM alignment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper finds that the distraction happens when these Contextual Heads fail to pay enough attention to the relevant context. It then identifies Focus Directions, which are specific directional vectors located in the key and query activations of these heads, that directly control the attention mechanism. They also introduced an automated magnitude control method to determine the optimal strength for applying the Focus Directions.

### Strengths
1. The central idea of introducing Automated Magnitude Control to dynamically adjust the strength of the intervention (Focus Directions) is novel and addresses a practical challenge in applying direct attention-head steering, making the method more robust and less reliant on manual hyperparameter tuning.
2. The authors have provided a large volume of detailed experiments across several LLMs and context lengths. The presentation of the methodology and results is clear, making the paper relatively easy to follow

### Weaknesses
1. The experiments predominantly focus on the simple one-hop Needle-in-a-Haystack (NIAH) task. It is a major concern whether the proposed methodology can effectively scale to more challenging, real-world long-context tasks, such as multi-hop reasoning or complex question answering, where the relevant context is distributed and requires multiple retrieval steps.

2. The method's performance improvement appears to be inconsistent and sometimes non-existent across various context lengths. This lack of a stable, monotonic improvement raises serious concerns regarding the overall robustness and reliability of the proposed approach under diverse operational constraints.

### Questions
please refer to weaknesses.

### Soundness
2

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
The paper tackles long-context getting distracted by finding the few attention heads that naturally focus on the right spans and then gently steering them at inference time. The authors learn simple focus directions in those heads so the model pays less attention to sink tokens and more to likely relevant text, without finetuning or editing the input. Across multi-document QA and several long-context benchmarks, this yields consistent, interpretable gains. Overall, it’s a lightweight, diagnostics-friendly way to help LMs look in the right place.

### Strengths
1. The work gives a simple, inference-time way to nudge a few attention heads, so it can get gains without finetuning or rewriting the input.

2. On multi-document QA and several HELMET tasks, it improves accuracy under long contexts across different models and context lengths.

3. The head scoring and attention evidence are easy to understand.

### Weaknesses
1. The method requires labeled relevant span supervision (e.g., the 20-doc, 1-relevant setup) to identify contextual heads and learn focus directions, so transfer to domains without such supervision is uncertain.

2. Performance depends strongly on the number of intervened heads and the intervention magnitude, and small changes can flip gains into regressions, implying per-model and per-task tuning is necessary.

3. Deployment becomes difficult for closed-weight models because the approach needs access to internal activations to intervene at inference.

4. The learned directions likely favor semantic overlap with the query, which can reduce robustness on paraphrases or counterfactual spans where evidence is indirect.

5. Improvements are uneven across tasks and models, leaving unclear guidance on when to enable the method, which heads to pick, and what intervention strength to use.

### Questions
1. How would the method work in domains without labeled relevant spans?

2. Does the approach bias toward lexical overlap with the query, and can you report robustness on paraphrase and counterfactual setups where evidence is indirect?

3. Given the uneven gains across tasks and models, can you provide a simple decision rule or usage guideline (when to enable, which heads, what α) validated across datasets?

### Soundness
3

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
3

### Summary
This paper investigate the context distraction issue in long-context LLMs, where the relevent parts for QA burried in a large portion of irrelevent parts of the input. The authors first identify contextal heads, which are a small subsets of attention heads that are senitive in focusing on relevant spans of input. Then they introduce applying focus direction on those attention head by adding a vectors to the key and query representations, this aims to steer attention toward relevant spans. They have experiments on multi-doc QA and a long-context benchmark HELMET, they show improved performance using various LLMs.

### Strengths
1. They address a well-known problem of distraction in long-context LLMs in a mechanistic perspective, which is interesting.
2. The proposed method adpots steering vector approach which is efficient.
3. The movtivation is clear by showing that contextual heads exist.

### Weaknesses
1. The idea of steering attention activateions has been well explored in several prior work, the proposed focus directions are another instance of attention steering, which lack novelty for ICLR conference.
2. The method heavily reply on gold spans or releveant documents in a dataset or domain to identify contextual heads and train focus directions. The transferability to different domain in zero shot  is limited.
3. Experiments only consifer synthetic distractors and a subset of HELMET. How the method could be generalized to multi-hop reasoning where there are multiple relevant spans?
4. Addressing distraction for long-context LLMs is a well-explored domain, however, baselines and related approaches are missing in this paper. For example, "Never Lost in the Middle" which directly address the distraction issue without training, and "Reducing distraction in long-context language models by focused learning" directly addresses distraction with training, which is more closely related.

### Questions
See weaknesses.

### Soundness
2

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
4

### Summary
The core contribution of this paper is a thorough systematic study that seeks to better understand why LLMs get distracted by irrelevant contexts in long-context settings. Through a controlled study, the authors demonstrate that by increasing the LLMs attention weights towards the relevant contexts can mitigate the propensity for LLMs to get distracted. Using this, they propose focus directions which allows the model to allocate more attention towards the relevant contexts, thereby improving performance.

### Strengths
- The study of the cause of LLMs distractions is novel and well-motivated, allowing for a better understanding of LLMs failures. I enjoyed reading section 2.
- The results from section 3 show clear improvements when using the proposed method, further justifying the validity of the method.

### Weaknesses
- While the model shows strong gains in section 3, the improvement in section 4 (on HELMET) are minimal, barely improving over the baseline. This suggests that the approach is limited to settings in which in-domain training data is available.
- The method to obtain focus directions depends on datasets where relevant and irrelevant contexts are annotated. This limits scalability and makes it hard to apply in real-world settings where such labels are unavailable.
- The performance highly depends on the magnitude parameter ($\alpha$). Too strong or too weak interventions can break the attention distribution and lead to performance drops. Although the authors propose automated magnitude control, it is still preliminary.
- Training and applying focus directions require caching key/query activations and modifying attention weights during inference, which can't be applied to modern techniques in inference speed up like FlashAttention2.

### Questions
- How would a simple two-step approach that first filters out relevant context with a relevance classifier and feeds that to an LLM compare?
- Can you provide some qualitative or visual analysis of how contextual heads or focus directions behave, and how they differ from other known functional heads like retrieval heads?

### Soundness
3

### Presentation
3

### Contribution
3
