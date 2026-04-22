# PostAlign: Multimodal Grounding as a Corrective Lens for MLLMs

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Multimodal Large Language Models (MLLMs) have shown remarkable performance in vision-language tasks, such as image captioning and visual question answering. However, these models often struggle with fine-grained visual understanding and are prone to hallucinations, primarily due to over-reliance on linguistic priors that distract them from leveraging actual visual information. This results in outputs that are often unanchored in the visual content, leading to errors. To address these challenges, we introduce MMGrounded-PostAlign, a post-multimodal alignment framework designed to enhance the visual understanding capabilities of MLLMs and mitigate hallucinations. In the framework, the visual grounding module identifies the referred objects in the image, while the textual grounding module generates the rationale for the final answer. This dual grounding approach ensures that outputs are firmly anchored in both visual and textual evidence. In particular, we incorporate a negative rejection mechanism within the visual grounding module to distinguish between grounded entities and non-existent objects influenced by linguistic biases. Moreover, we propose a selective reasoning mechanism within the textual grounding module to adjust the model’s reasoning strategy based on the complexity of the query. These innovations together work to resolve the issues associated with hallucinations and enhance the overall alignment between visual and textual modalities. Extensive evaluations on benchmarks such as POPE, HaloQuest, ReasonSeg, MME, and MMBench demonstrate significant improvements in fine-grained visual understanding and hallucination suppression, showcasing the effectiveness of our approach in real-world multimodal tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MMGrounded-PostAlign, a post-multimodal alignment framework designed to mitigate hallucinations and enhance the visual understanding capabilities of Multimodal Large Language Models (MLLMs). The core idea is to use multimodal grounding as a "corrective lens" to anchor the model's outputs in actual visual and textual evidence, thereby reducing its over-reliance on spurious linguistic priors.

### Strengths
- The paper presents a novel and compelling perspective. Instead of using MLLMs for grounding tasks (the common approach), it inverts the relationship by leveraging grounding to enhance the MLLM itself.
- The paper is generally well-written and well-structured. The motivation is clearly established, the framework is explained with the aid of a pipeline diagram, and the findings are presented logically.

### Weaknesses
- the `<SIMPLE>`/`<COMPLEX>` labels are assigned at the dataset level, not per sample. This is a coarse-grained heuristic that may misclassify individual queries.
- It is uncertain whether the performance gains are due to the novel grounding-as-a-lens concept or simply the introduction of any additional post-processing signal. It is possible that a simpler method like contrastive decoding could achieve similar hallucination suppression on POPE without the need for a full segmentation model.
- The framework is a multi-component system where the visual grounding, textual grounding, and final answer generation are interlinked. What happens if the visual grounding module fails (e.g., produces an incorrect or low-confidence mask for a present object, or fails to trigger <REJ> for an absent one)?

- Some figures have relatively small, which affects readability. For example, Figure 3(a) and (c).

### Questions
See weakness please

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
Addressing the hallucination problem and insufficient fine-grained visual understanding of Multimodal Large Language Models (MLLMs) caused by their over-reliance on linguistic priors, this paper proposes the MMGrounded-PostAlign post-multimodal alignment framework. The framework integrates visual grounding (incorporating a negative rejection mechanism to distinguish between real and non-existent objects) and textual grounding (incorporating a selective reasoning mechanism to adjust reasoning strategies based on query complexity). Built on the base models of LLaVA-1.5-7B/13B and ViT-H SAM, and optimized through LoRA fine-tuning and multi-loss function, the framework’s effectiveness—including suppressing hallucinations, enhancing visual understanding, and preserving the reasoning capabilities of MLLMs—has been validated on benchmarks such as HaloQuest, POPE, VQAv2, MME, MMBench, RefCOCO, and ReasonSeg.

### Strengths
- It accurately identifies two key issues of Multimodal Large Language Models (MLLMs): "hallucinations" (generating non-existent content) and "insufficient fine-grained visual understanding", which are caused by the models' over-reliance on linguistic priors. These two types of issues serve as core bottlenecks that undermine the robustness and reliability of current MLLMs in vision-language tasks, making the research direction highly practically significant and necessary.
- While enhancing visual understanding and suppressing hallucinations, it does not compromise the inherent reasoning and generalization capabilities of MLLMs (e.g., achieving performance equal to or better than the baseline on MME and MMBench). It also avoids the problem of degraded reasoning ability in some grounding methods (such as BTL-Generation) due to overfitting to visual information, demonstrating an excellent balancing effect.

### Weaknesses
- In textual grounding, the <SIMPLE>/<COMPLEX> labels are categorized at the "dataset level" (e.g., queries in the COCO dataset are classified as <SIMPLE>, while those in the ReasonSeg dataset are classified as <COMPLEX>), rather than being annotated at the "sample level". Although this approach reduces annotation costs and ensures training stability, it fails to handle scenarios where "simple queries and complex queries are mixed" within the same dataset. This may lead to inaccurate matching of reasoning strategies for some samples (e.g., complex reasoning queries expressed in a simple form are misjudged as <SIMPLE>).
- The <REJ> samples used to train the "negative rejection mechanism" are only sourced from the gRefCOCO dataset (containing 32,202 queries that "refer to non-existent objects") and do not cover more scenarios (such as negative samples of different object types, different image styles, and different linguistic expressions). When facing unseen "false query-image" combinations, the effectiveness of the negative rejection mechanism may decrease, and its robustness needs further verification.
- During training, the model is enabled to automatically judge the complexity of queries through "self-reflection prompting". However, the model's judgment of difficulty may also introduce biases and hallucinations. For instance, it is common for the model to be overconfident, which leads to the generation of hallucinations.

### Questions
See above

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
This paper introduces MMGrounded-PostAlign, a framework to reduce MLLM hallucinations by grounding outputs in evidence instead of unreliable text priors. It features a visual grounding module with a "negative rejection mechanism" to deny non-existent objects and a textual grounding module that uses "selective reasoning" to add rationales only for complex queries , thereby improving visual accuracy and suppressing hallucinations.

### Strengths
1. The paper clearly identifies and addresses a critical problem in MLLMs: the over-reliance on linguistic priors, which leads to hallucinations and a failure to ground responses in visual evidence. The proposed "post-alignment" framework is a well-motivated and logical approach to re-center the model's outputs on visual information.

2. The experimental analysis provides valuable insights. "Finding 1" (Figure 3), which empirically demonstrates how linguistic priors can override visual information in the model's later layers, offers a strong motivation for the method. Furthermore, the ablation studies (e.g., Tables 1 and 2) are thorough, providing a solid comparison of different grounding strategies (segmentation, detection, BTL vs. explicit grounding) and validating the paper's design choices.

3. The `<REJ>` token is a practical and effective mechanism for negative grounding. By giving the model an explicit option to "abstain" from grounding a non-existent object, this method directly targets object hallucination.

### Weaknesses
1. Insufficient baseline comparisons: A significant weakness is the lack of comparison against the original, unmodified baseline model, as well as other well-established MLLMs. The model is built on LLaVA-1.5, yet Tables 1-3 primarily compare variants of the proposed method against an internal baseline (the framework with modules removed), not against the original LLaVA-1.5. This makes it difficult to assess the true impact (including any potential performance trade-offs) of the added components. This is particularly concerning given "Finding 3" (retaining reasoning abilities), which cannot be fully verified without this comparison. For instance, the reported 63.9 on MMBench-EN (7B) may not be competitive with the public LLaVA-1.5-7B score (64.3).

2. Unclear architectural novelty: The novelty of the visual grounding module's architecture is not well-explained. The Method section (Section 3) describes a model (with SAM-based decoder, `<LOC>` and `<REJ>` tokens) that seems to reimplement established paradigms from prior works like LISA, GLaMM, and GSVA. While these are cited in Related Work, the Method section itself does not attribute these design choices or clearly differentiate what is adopted from prior work versus what is a new architectural innovation. The contribution appears to be more in the application of this module, but the presentation makes the architectural contribution ambiguous.

3. Dataset-level reasoning labels: A significant limitation, as acknowledged by the authors in Appendix, is that the `<SIMPLE>`/`<COMPLEX>` labels for selective reasoning are applied at the dataset level, not the sample level. This is a very coarse heuristic. A dataset labeled "complex" may contain many simple queries, and vice-versa. This design choice weakens the "selective reasoning" strategy, as the model isn't learning to distinguish query complexity on a case-by-case basis but is instead learning a bias associated with the data source prior.

4. (Minor) Clarity of "Selective Reasoning": The "selective reasoning" mechanism is not very clearly explained in the Introduction. The Introduction is vague. The reader must wait until Section 3.3 to understand the concrete implementation (i.e., the `<SIMPLE>` and `<COMPLEX>` tokens). Briefly explaining this mechanism earlier would improve the paper's readability and flow.

5. (Minor) Citation formatting: The paper does not consistently follow the ICLR template's citation guidelines (`\citep` and `\citet`). This should be corrected for the final version.

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes MMGrounded-PostAlign, a post-alignment framework that augments an MLLM with  
(1) visual grounding (segmentation + bounding box) driven by a special LOC token and a negative rejection token REJ, and  
(2) textual grounding via selective reasoning that emits rationales only for complex queries (SIMPLE/COMPLEX gate).

The method aims to reduce hallucinations caused by linguistic priors and improve fine-grained visual understanding.  
Experiments on HaloQuest, POPE, VQAv2, MMBench, MME, RefCOCO series, and ReasonSeg show consistent gains; ablations compare to BTL (boxes-as-tokens) variants.

### Strengths
- Clear motivation: Tackles language-prior-driven hallucination via explicit multimodal grounding; neat idea of "grounding as a corrective lens".  
- practical design: Simple LOC/REJ interface to a multi-task decoder; selective reasoning avoids unnecessary rationale generation.  
- Broad evaluation: Covers hallucination, general V+L, and grounding benchmarks with meaningful ablations.

### Weaknesses
- Limited generality:Only tested on LLaVA-1.5 (7B/13B) + SAM-ViT-H; cross-backbone evidence (e.g., Qwen-VL, other grounding encoders) missing.  

- The idea of labeling queries as SIMPLE vs COMPLEX is good, but doing so at the dataset level rather than per sample raises concern. Some “simple” dataset queries might still require reasoning, and vice-versa. 

-  The REJ token is an interesting idea, but the paper does not show the  cases where referent exists but system rejects.

### Questions
I note that the Related Work section references reinforcement learning (RL) approaches in the vision–language modelling domain, yet the paper does not include any empirical comparison involving RL.  

My thought is that a more general RL-based training paradigm—one in which the vision-language model learns in a sequential decision-making setting—might offer better generalization across vision-language tasks rather than designing a task-specific chain-of-thought process solely for the grounding task.  

So my question is: what are the advantages of your approach compared to a more general reinforcement learning method for MLLMs?

### Soundness
2

### Presentation
3

### Contribution
2
