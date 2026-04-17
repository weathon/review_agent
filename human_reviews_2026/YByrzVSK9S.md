# Last Layer Logits to Logic: Empowering LLMs with Logic-Consistent Structured Knowledge Reasoning

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Large Language Models (LLMs) achieve excellent performance in natural language reasoning tasks through pre-training on vast unstructured text, enabling them to understand the logic in natural language and generate logic-consistent responses. However, the representational differences between unstructured and structured knowledge make LLMs inherently struggle to maintain logic consistency, leading to *Logic Drift* challenges in structured knowledge reasoning tasks such as Knowledge Graph Question Answering (KGQA).
Existing methods address this limitation by designing complex workflows embedded in prompts to guide LLM reasoning. Nevertheless, these approaches only provide input-level guidance and fail to fundamentally address the *Logic Drift* in LLM outputs. Additionally, their inflexible reasoning workflows cannot adapt to different tasks and knowledge graphs.
To enhance LLMs' logic consistency in structured knowledge reasoning, we specifically target the logits output from the autoregressive generation process. We propose the *Logits-to-Logic* framework, which incorporates logits strengthening and logits filtering as core modules to correct logical defects in LLM outputs. Extensive experiments show that our approach significantly improves LLMs' logic consistency in structured knowledge reasoning and achieves state-of-the-art performance on multiple KGQA benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a decoding-time framework to reduce logic drift when LLMs reason over knowledge graphs. The framework includes multiple stages, like compiling legal KG paths into NFAs for ranking, and modifying the last-layer logits via "strengthening" and "filtering". Experiments report gains on several KBQA benchmarks across KGs and tasks.

### Strengths
1. Presents a clear motivation and introduces an interesting approach to mitigate logical inconsistencies in LLM reasoning over structured KG on output-side logit corrections rather than only prompt-level guidance
2. The proposed method is model-agnostic and can be plugged into any decoder without retraining
3. The use of NFAs to model KG paths and constrain decoding is a clever way to improve path validity
4. Extensive ablation studies and visualizations that help to illustrate the impact of core modules

### Weaknesses
1.The evaluation suite relies mostly on older KGQA datasets. This limits the external validity of the SOTA claim and may underrepresent contemporary KG schemas and LLM-era challenges. 
2. Several mathematical derivations need to be clarified: the overall objective in Sec. 3.2 treats Pθ,q and Pθ,G as independent without justification; Sec. 3.2.2 states “we calculate the difference between original and masked outputs” but Eq. for Dq is a convex combination, not a difference; in Sec. 3.2.3, the filtering equation sets δ as 0/1 but z are logits (not probabilities), therefore setting δ({art,award})=0 at the logit level does not forbid tokens, potentially allowing probability mass leakage. 
3. Prior work on constrained decoding with FSAs/tries and KG-constrained generation should be thoroughly discussed.
4. Compiling full KGs into NFAs (Sec. 3.2.1) could be computationally prohibitive for large-scale KGs like Wikidata, raising concerns about the approach's scalability. Complexity with respect to number of candidate paths, tokens per label, and beam size should be analyzed. 
5. The method performs SFT with 1/10 of training data from CWQ and WebQSP "to teach the model correct path output format" -- I wonder if this brings an unfair advantage over the agentic reasoning baselines that do not include SFT.

### Questions
1. The paper cites "constrained decoding" in section 3.2.2, but it should be "contrastive decoding."
2. How is the sentence-transformer chosen and tuned for scoring NFA paths - why not use the LLM itself for consistency?
3. How is MASK token handled in prompts— what exactly is masked (top-1 path only or top-K?), how long is the masked span, and how do you align time steps between original and masked runs?
4. Logits strengthening should require two forward passes (original vs masked) per decoding step; yet Table 6 lists “1 LLM call per question.” Can you explain this?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Logits-to-Logic, a framework to address Logic Drift in LLMs in the scenario of KGQA. The core innovation is targeting the logits distribution in LLMs’ autoregressive output process to align it with the logical constraints of KGs and question semantics. To achieve this goal, paths are used to filter tokens not in KG and keep consistent of the LLM logits with semantic and structure information.

### Strengths
1. This paper is the first KGQA method that considers output-level control that corrects LLM outputs by manipulating logits.
2. The results look promising in a few benchmarks. It adapts to different KGs and tasks (multi-hop QA, slot filling).
3. The form of displaying results is good, where multiple different kinds of figures are used to show results.

### Weaknesses
1. The information in Figure 1 is not clear. "current approaches" is very vague. I cannot get which methods and datasets are tested.
2. The "logic" formulated in this paper mainly depends on paths, not true logics. Along with this problem, the novelty of this paper is a concern where there are many path-based methods like RoG (Luo et. al.). The authors should have a separate subsection in related works to discuss methods lie in this type.
3. Based on the results in Figure 2, the main technique that works is $Z_f$, which serves as a filter that masks the logits of tokens that are not in the searched paths. So I wonder whether the NFA is still needed or just an approach for story telling.
4. It seems that this method is quite expensive. For LLaMA3.1-8B model, it takes two A800 GPUs to compute in parallel for inference. The computing time is also not well compared.
5. The layout of figures and tables in this paper is in chaos. Like Figure 2 is mentioned before  Figure 1. Table 4 lies in Section 4.6, which mainly discusses results in Figure 4.

Minor issues:
- The caption of Figure 3 is called "overview of our framework", while part (a) seems not the proposed method.
- The concept of "acceptable path" is not well defined.
- The range of $\omega$ is not clear.

### Questions
1. Please discuss the difference between logic drift and hallucination.
2. In Table 4, why bigger model (Qwen2-1.5B) can be much weak than Qwen2-0.5B in WebQSP?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The manuscript proposes a novel framework, Logits-to-Logic, to enhance teh logical reasoning capabilities of LLM when applied to KGs. The main problem tackled by this manuscript is that LLM struggle with Logic Drift, where their reasoning paths often do not align with the logical structure of the KG, leading to errors in structured knowledge reasoning. 

The authors propose Logits-to-Logic, a framework that directly operate on the logits output by the LLM during their autoregressive generation. Logits-to-Logic consists logic compiling, logits trenghthening, logits filtering.

### Strengths
+ The manuscript addresses the issue of logic drift by directly intervening in the logits.
+ Extensive experiments show significant improvements.
+ The framework demonstrates significant computational efficiency

### Weaknesses
- Evaluation mainly focus on LLaMa and Qwen model. Other foundation models or larger models are not validated due to resource constraints. It would benefit analyzing how the framework scales with larger models.
- Although the class-agnostic loss helps prevent overfitting to specific classes, the overall framework may stil struggle with class imbalance or biased training samples
- The framework relies heavily on the predefined hyperparameters for the loss terms. While the paper show an empirical study on these values, further dynamic or adaptive tuning methodology may benefit the flexibility of the framework.

### Questions
Please refer weakness

### Soundness
3

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
2

### Summary
This paper proposes a knowledge graph question answering method based on large language models (LLMs). When using LLMs to answer knowledge graph (KG) questions, they sometimes generate paths that do not exist in the KG or produce incorrect paths, leading to wrong answers. To address this, the authors propose aligning the logits of the LLM output with the logical structure of the knowledge graph to ensure the generated answers are faithful. Experimental results show that the proposed method outperforms existing state-of-the-art approaches.

However, the paper is somewhat difficult to understand, particularly because it lacks an introductory section or preliminary knowledge, making it challenging for readers to assess its contributions. Moreover, the contribution appears somewhat incremental.

I will try to review the paper again during the rebuttal period, so I hope the authors can provide sufficient information at that time.

### Strengths
The idea of aligning the logits of LLM outputs with knowledge graph logic to ensure that LLM outputs follow KG information is interesting. However, the idea is somewhat difficult to understand. LLM outputs are probability distributions over tokens, while in a KG, an entity name or relation may consist of multiple words. It is unclear how this mapping is performed.

The experimental results are promising, but in Table 1, only Hit@1 is reported; F1 scores are not provided. It would be better to include the F1 scores for all baseline methods for a more comprehensive comparison.

### Weaknesses
The paper lacks an introduction to preliminary knowledge, which makes it difficult to understand and to assess the true contributions. Moreover, the contribution seems somewhat incremental.

No example is provided in the paper. It would be helpful to include a complete example that illustrates the entire process, from beginning to end, to facilitate understanding.

### Questions
no

### Soundness
2

### Presentation
1

### Contribution
2
