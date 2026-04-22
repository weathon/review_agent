# CRED: Contrastive Residual Embedding Decoding for Adaptive Concept Unlearning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Large language models (LLMs) trained on web-scale data inevitably encode outdated, private, or undesired knowledge, posing challenges for privacy, safety, and factual reliability. While existing machine unlearning methods typically rely on retraining or fine-tuning, these approaches are costly and risk catastrophic forgetting. In this work, we propose CRED, an in-context unlearning method that enables LLMs to forget specific concepts at inference time without any parameter updates. CRED formulates unlearning as a decoding-time intervention: given a query, it constructs retrieval-augmented prompts from both a retain set and a forget set, then computes a contrastive residual vector from their decoder embeddings. This residual is injected into the decoder of the original prompt, guiding generation away from forget-set content while preserving relevant knowledge. Experiments on the TOFU and MUSE benchmarks demonstrate that \modelnameshort achieves effective concept erasure with minimal quality degradation. Additional analyses confirm its stability under 8-bit and 4-bit quantization, highlighting its robustness and practicality for real-world deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CRED (Contrastive Residual Embedding for Decoding), an inference-time unlearning method that removes the influence of unwanted knowledge from large language models without modifying model parameters. Instead of retraining, CRED constructs prompts based on separate retain and forget knowledge graphs, performs retrieval, and computes a contrastive residual between the model’s responses under retain vs. forget contexts. This residual is injected during decoding—layer-wise—to steer generation away from undesired concepts while preserving useful information. Experiments on TOFU and MUSE benchmarks show that CRED achieves effective forgetting with minimal utility loss and remains robust under 8/4-bit quantization.

### Strengths
1. Introducing a knowledge graph into the unlearning process is interesting.

2. The paper is well-written and easy to follow.

3. The experimental section is comprehensive and includes robustness evaluations of the proposed unlearning method.

### Weaknesses
The idea is interesting, but I am not fully convinced that introducing a knowledge graph is necessary. If a prompt filter can be trained to accurately classify whether a query belongs to the unlearning set—which may be more efficient—would the knowledge graph still be required?

### Questions
I am willing to reconsider my score if the motivation for introducing a knowledge graph is better justified.

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
CRED applies contrastive decoding for unlearning by computing a residual between logits from a single frozen model run on retain vs forget evidence, then adding that residual to the original logits; a simple multi-layer fusion stabilizes decoding. Evidence is sourced via KG-guided retrieval to build the retain/forget prompts. Results on TOFU and MUSE-News show strong forgetting while keeping utility, with some analysis under 8/4-bit quantization.

### Strengths
1. Training-free deployment. The method requires no parameter updates and only adds a small number of extra forward passes, making integration into existing systems straightforward.
2. Clear, simple mechanism. The residual formed between retain- and forget-conditioned logits is easy to implement and reason about, and the layer-wise fusion helps preserve fluency and coherence.
3. Competitive empirical results. Across the two benchmarks, the approach achieves strong forgetting while maintaining utility, supported by ablations and analysis.
4. Quantization-aware evaluation. The paper explicitly examines behavior under 8- and 4-bit inference and proposes a simple stability measure, which is practical for real deployments.
5. RAG-friendly design. The method fits naturally with retrieval pipelines and can be updated by modifying the external knowledge split rather than retraining the model.

### Weaknesses
1. Limited conceptual novelty. The core technique is essentially a steering-style residual in logit space; the contribution lies more in the overall packaging than in a new decoding principle.
2. Retrieval and KG dependence. Performance is likely sensitive to extraction quality, hop depth, and retrieval parameters; robustness to noisy or incomplete graphs is not systematically evaluated.
3. Unclear triggering reliability. A prompt classifier gates when the method is applied, but its precision/recall and the impact of misclassification on utility and forgetting are not reported.
4. Baseline comparability. Several baseline figures are imported rather than re-run, which weakens strict apples-to-apples comparisons.
5. Residual privacy leakage. On MUSE, the privacy leakage metric remains noticeable, indicating that forgetting may not be airtight.

### Questions
1. Prompt classifier. What model and thresholds gate intervention? Please report precision/recall and quantify how false positives/negatives affect forgetting and utility.
2. Retrieval/KG robustness. How sensitive are results to extraction errors, hop depth, and top-k? Provide a controlled-noise study to show stability under imperfect graphs.
3. Residual design. Why restrict fusion to the last N layers? Did you evaluate adaptive \alpha(q) or concept-subspace projections, and how do they change the forgetting–utility trade-off?
4. Baseline parity and overhead. Which baselines were re-run in your stack versus imported, and what is the end-to-end latency/throughput (including retrieval) relative to those baselines?

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
5

### Summary
This paper proposes an in-context concept unlearning method named contrastive residual embedding for decoding (CRED). Given a query, CRED constructs retrieval augmented prompts from a retain set and a forget set, computes a contrastive residual from multi-layer decoder embeddings, and injects this residual into the decoder of the original prompt to achieve the unlearning goal. Experiments on TOFU and MUSE demonstrate the effectiveness of CRED in unlearning concepts at inference time.

### Strengths
1. In general, the proposed method makes sense

2. The paper is easy to follow

3. Experimental results demosntrate the effectivenss of the propsoed method

### Weaknesses
1. The novelty of the proposed method is limited. The idea of applying contrastive decoding to unlearning has been explored by UCD, making the novelty of the proposed method limited.

2. It is unclear how accurate the prompt classifier is. As an important component, the authors should report the accuracy of the prompt classifier.

3. The proposed method needs to keep a knowledge graph constructed from the forget set, which seems to be not reasonable. Generally, for unlearning knowledge, the model owner not only needs to unlearn the knowledge from the model, but also delete the corresponding data, as it might cause privacy and copyright issues. The authors might need to better explain why the setting is reasonable.

4. Given a prompt query related to unlearned knowledge, it is unclear what will be retrieved from the retain knowledge graph. Can the authors give some examples and a case study? If the retrieved information from the retain knowledge graph is some random things, why would it be useful? It would be better to also conduct an ablation study without using the retain knowledge graph.

5. Some important experiments, such as the ablation study, are put in the appendix, and the authors did not mention these in the main content. The authors should try to put some important experimental results in the main content and mention those that are put in the appendix.


[1] Suriyakumar, Vinith M., Ayush Sekhari, and Ashia Wilson. "UCD: Unlearning in LLMs via Contrastive Decoding." arXiv preprint arXiv:2506.12097 (2025).

### Questions
Please see above

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
In this paper, the author(s) propose(s) training-free unlearning methods that are using the designed residual. The residual is calculated based on the content conditioned on the forget knowledge graph  and the content conditioned on the retain set. This method can help LLMs keep less sensitive content rather than simply making the models refuse to answer. Experiments validate the effectiveness of the proposed method.

### Strengths
This paper has the following strengths:
- This paper is well-written and clearly motivated.
- This submission encourages LLMs to output less sensitive messages instead of directly refusing to provide any information.
- Conducted experiments validate the effectiveness of the proposed method.

### Weaknesses
This paper has the following weakness:
- Evaluated models are limited. It would be better to explore different backbones and different model sizes.
- Some experimental design is not very reasonable. For example, why the ablation study in C.1 use LLaMA3.2-1B which is different from the main experiments. It would be better to use the same backbone when conducting ablation studies.
- The model's performance heavily relies on the prompt classifier to judge whether a given prompts in related to the forget set. However, this important component is simply mentioned without any details.

### Questions
I have the following questions/suggestions:
- The first question is the question in the weakenss point 2.
- Could the authors give more details about the prompt classifier? What about its generalizability?
- The phrase “retrieval augmented prompts” in the abstract might be better written as “retrieval-augmented prompts.”
- Another point is that there are some inaccurate references. Please also list their published proceddings rather than just author names and the title. For example, The first one below should be IEEE Transactions on Dependable and Secure Computing and the second one below should be ICLR.
     - Line 590-591: Shang Wang, Tianqing Zhu, Dayong Ye, and Wanlei Zhou. When machine unlearning meets retrieval-augmented generation (rag): Keep secret or forget knowledge?, 2024.
     - Line 618-620: Zhiwei Zhang, Fali Wang, Xiaomin Li, Zongyu Wu, Xianfeng Tang, Hui Liu, Qi He, Wenpeng Yin, and Suhang Wang. CATASTROPHIC FAILURE OF LLM UNLEARNING VIA QUANTIZATION. 2025.

### Soundness
2

### Presentation
3

### Contribution
2
