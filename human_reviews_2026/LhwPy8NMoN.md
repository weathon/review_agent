# PLP-RC:Point–Line–Plane Fusion for Discriminative Relation Classification with LLMs

- Decision: Reject
- Scores: 8, 4, 4, 0

## Abstract
Relation classification is a fundamental NLP task that involves identifying the semantic relations between entity pairs in a given text. While pre-trained language models have advanced this area, effectively integrating local entity information with global context remains a key challenge. Large Language Models offer rich world knowledge, but their generative use often suffers from hallucinations, limiting reliability. To address these issues, we propose a Point–Line–Plane fusion framework for discriminative relation classification with LLM embeddings. Entity spans are modeled as local point representations, the end of sequence token provides a global plane representation, and an attention-based line representation aligns the two. This discriminative paradigm avoids hallucinations while fully exploiting LLM representations. Our method achieves new SOTA performance on TACRED, TACREV, and RE-TACRED benchmarks,  outperforming both discriminative and generative baselines. Ablation studies provide further evidence for the effectiveness of our design in achieving context-aware relation classification.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes PLP-RC, a framework for Relation Classification that leverages Large Language Models (LLMs) embeddings but avoids the hallucination issues common in LLM generative usages. The core innovation is a "Point-Line-Plane" geometric fusion mechanism: Point represents local entity span information, Plane encodes global context  with the [EOS] token, and Line (attention scores between entities and [EOS]) bridges the two levels of granularity. These features are integrated into fused entity representations for relation prediction. PLP-RC achieves new state-of-the-art results on the TACRED, TACREV, and RE-TACRED benchmarks, outperforming both discriminative and generative baselines.

### Strengths
* The paper proposes a discriminative approach with LLM embeddings instead of directly using the LLM generative power
* The Point-Line-Plane geometric analogy for feature fusion is interesting
* The experimental evaluation is comprehensive, including benchmarks, baselines, ablation studies, model scaling and computational costs
* The paper is well-written and clear

### Weaknesses
* The evaluation benchmarks primarily consist of single-sentence contexts. The authors may briefly discuss the proposed approach's adaptation to long-range dependencies.
* The ablation studies show that the "Line" component's contribution is relatively modest compared with others.
* It would be nice if the authors could analyze further whether specific types of relations affect the overall performance

### Questions
1. Given the most contribution from "Plane" and instruction, how does the model's performance change if the instruction changes, e.g., placed at the beginning instead of the end?
2. The "Line" feature uses attention scores from the [EOS] token to entity tokens. Did you try other attention scores like those between the subject and object entities?

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
4

### Summary
The paper presents a novel framework, Point–Line–Plane Fusion (PLPF), for relation classification using Large Language Model (LLM) embeddings. The work addresses a long-standing challenge in balancing local entity representation with global contextual understanding while mitigating hallucination issues common in generative models. The proposed geometric abstraction models entities as points, context alignment as lines, and overall semantic scope as planes, which is conceptually elegant and empirically validated.

### Strengths
1.The paper introduces an innovative geometric fusion paradigm based on the point–line–plane concept, which offers a clear and interpretable approach to integrating local and global features for relation classification.

2.The proposed method leverages Large Language Model embeddings in a discriminative framework, effectively mitigating hallucination issues that are commonly observed in generative LLM applications.

### Weaknesses
1.The core "Point-Line-Plane (PLP)" fusion mechanism lacks sufficient theoretical justification. The paper frames the mechanism as "conceptually grounded in geometric and information-theoretic principles" but provides no formal connection to these principles (e.g., how line/plane representations map to information-theoretic metrics like mutual information).

2.The methodological description lacks mathematical rigor and theoretical foundation. The "geometric perspective" remains largely metaphorical without formal mathematical formulation or theoretical guarantees about the representation properties.

3.The paper fails to address cross-sentence relation classification, a critical limitation of existing methods highlighted in the Introduction. All experiments are conducted on sentence-level datasets, yet the PLP framework is claimed to "capture long-range dependencies"—no evidence is provided for this capability, and the [EOS] token’s causal attention (autoregressive) cannot model cross-sentence context effectively.

4.The paper confuses discriminative vs. generative paradigms. It claims PLP-RC avoids hallucinations by using a "discriminative framework" but uses decoder-only LLMs (Qwen3) pretrained with generative next-token prediction.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

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
This paper proposes a Point-Line-Plant fusion framework based on LLM embeddings for entity relation classification, which leverages the representational capacity of LLMs while mitigating their hallucination issues. Experiments were conducted to validate the effectiveness of the proposed method.

### Strengths
The proposed approach utilizes a LLM as text encoder, effectively transferring its rich representational capacity to the discriminative task of relation classification. This enables the generation of semantically richer embeddings, leading to improved classification performance. Furthermore, the method requires no fine-tuning of the LLM, thereby avoiding substantial computational costs.

### Weaknesses
1. Lines 52-54 of the paper mention that LLMs have inherent deficiencies in capturing contextual content, yet there is no further explanation or citation of relevant arguments in the paper. Taking the Qwen3 model used in the paper as an example, it can support a maximum context length of 128K, which is fully capable of covering some basic tasks including Relation Classification.

2. Experiments demonstrate that the PLP-RC method proposed in this paper is effective in relation classification, significantly outperforming other approaches. However, PLP-RC uses Qwen3 as its backbone, while the comparison methods adopt GLM-10B, Mistral-7B, and LlaMA2-7B. It is important to note that Qwen3 is a new-generation model; its 4B version even outperforms Qwen2.5-7B, its predecessor. Moreover, Qwen2.5-7B itself shows better performance than models like Mistral-7B and LlaMA2-7B. Therefore, it remains unclear whether the advantage of PLP-RC over other models stems from the method itself or from Qwen3, resulting in a lack of fair comparison.

3. As a mature and fundamental task, relation classification can already achieve good results by directly using LLMs to generate answers. PLP-RC treats LLMs as encoders and transforms relation classification into a discriminative task, but the paper lacks experiments to illustrate the advantages of PLP-RC compared to direct answer generation.

4. The writing expression of the paper needs further polishing, and the presentation should be consistent throughout the text. Some parts of the article lack necessary citations, such as the reference to the strategies of previous work in lines 222-223.

### Questions
Refer to the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper introduces the Point–Line–Plane (PLP) framework for relation classification, demonstrating limited performance improvements on TACRED and related datasets. However, the work lacks substantial novelty and does not meet high standards for innovation and impact.

### Strengths
This paper would be a good negative example to warn students not to write similar papers.

### Weaknesses
The paper lacks novelty, practical application, and sufficient experimental rigor. To improve, the authors should focus on contemporary challenges, explore more innovative approaches, and validate real-world applicability.

### Questions
Suggestion：The authors should consider more research-worthy directions.

### Soundness
1

### Presentation
1

### Contribution
1
