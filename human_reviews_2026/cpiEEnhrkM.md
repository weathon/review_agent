# SynDoc: A Hybrid Discriminative-Generative Framework for Synthetic Domain-Adaptive Document Key Information Extraction

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Domain-specific Visually Rich Document Understanding (VRDU) presents significant challenges due to the complexity and sensitivity of documents in fields such as medicine, finance, and material science. Although existing Large (Multimodal) Language Models (LLMs/MLLMs) achieve promising results, they still suffer from hallucinations, limited domain adaptation, and heavy reliance on extensive fine-tuning datasets. We introduce SynDoc, a novel framework that combines discriminative and generative models to address these challenges. SynDoc features a robust synthetic data generation workflow that extracts structural information and generates domain-specific queries to produce high-quality annotations. Through adaptive instruction tuning, SynDoc improves the discriminative model's ability to extract domain-specific knowledge. In parallel, a recursive inference mechanism iteratively refines outputs from both models to achieve stable and accurate predictions. This integrated framework demonstrates scalable, efficient, and precise document understanding, bridging the gap between domain-specific adaptation and general world knowledge for key information extraction tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes SynDoc, a hybrid framework for Key Information Extraction(KIE) from complex, domain-specific, visually rich documents. SynDoc's method has two main parts. First, it uses a Synthetic Data Generation workflow, where an MLLM automatically creates verified question-answer pairs from unlabeled documents. Second, this synthetic data is used to train a discriminative "Warmer" model, turning it into a domain expert. During inference, this "Warmer" provides domain-specific hints to a generative MLLM. The two models work together in a recursive loop, iteratively refining the answer.

### Strengths
1. The paper's originality stems from its creative combination: a hybrid discriminative-generative framework coupled with a "recursive inferencing" mechanism for iterative refinement. It also includes a synthetic data workflow with a multi-step verification process to enhance data quality.
2. The paper demonstrates quality through its well-defined, multi-component SynDoc framework. This framework is logically decomposed into four modules, each fulfilling a specific purpose.
3. The paper is clearly written and well-structured. The methodology is broken down into distinct components, and the use of diagrams visualizes the model architecture and the recursive workflow.
4. This work addresses the significant problem of zero-shot key information extraction in domain-specific documents. By proposing a framework that avoids reliance on manual annotations, it offers a scalable approach for adapting models to specialized domains.

### Weaknesses
1. The framework's novelty requires clearer articulation. The proposed SynDoc architecture appears to effectively integrate several familiar techniques, such as synthetic data generation and iterative refinement. It would strengthen the paper to more clearly define the specific conceptual novelty that distinguishes this architecture from a sophisticated engineering of existing components.
2. The empirical validation of SynDoc's effectiveness could be strengthened. The results in Table 1 show performance gains that appear marginal over the baseline MLLMs. Furthermore, the comparison is limited to these off-the-shelf models. Including comparisons against other recent, state-of-the-art methods in domain-specific KIE would provide a more robust validation of the framework's advantages.
3. A more granular ablation study of the "Warmer" component is recommended. The "Warmer" is a core contribution, yet its "Semantic Adaptation" step is not independently validated in the ablation studies. While Table 2 shows the value of "Structural Adaptation," a more detailed breakdown of the semantic tuning components would be necessary to fully justify the Warmer's design.
4. The paper would benefit from an analysis of the accuracy-latency trade-off. The recursive inference mechanism inherently increases computational costs and latency with each iteration. A discussion is needed on whether the resulting accuracy improvements are significant enough to compensate for this added overhead.

### Questions
Q: The empirical gains shown in Table 1, while positive, appear quite marginal over the baseline MLLMs in several cases. Could the authors discuss the practical significance of these improvements, especially considering the added complexity of the SynDoc framework?

Q: The recursive inference mechanism inherently increases computational costs and latency. Could the authors provide a more explicit analysis of this accuracy-latency trade-off?

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
3

### Summary
The paper addresses the task of key infomation extraction in documents from a double perspective. On one hand, the paper introduces a framework to generate synthetic QA data to train models for key information extraction. On the other hand, it nntroudces a framework for key information extraction that combines a discriminative specific model trained with synthetic data with a generative generic MLLM in an iterative scheme. The discriminative model is based on the adaptation of an existing document understanding model through specific fine-tuning tasks using the generated synthetic data. Experimental results on standard benchmrks for key information extraction show how, in general, the proposed recursive framework improves over using only a MLLM.

### Strengths
- The paper proposes a specific framework to generate synthetic QA data for the task of key information extraction. Although the framework is very specific to the task of key information extraction and the generation of data is based on the simple use of LLMs, the pipeline of generation of data (based on semantic question generation + structural question generation + question validation) is interesting and, with some additional work, could be used in other domains. 
- The paper proposes new training strategies for information extraction, particularly grid matching and entity retrieval taking advantage of the generated synthetic data.

### Weaknesses
- The description of the method is not clear in several aspects:
a) In section 3.3, what is the sentence representation s? A global embedding of the sentence? What is the difference of the sentence representation with C (defined in the first line of page 5 as the encoding of the surrounding context), what is this surrounding context? And, in equation (2) why combining the aggregation of c_i with s (that is defined before as the representation of c)?
b) Which are the exact inputs to the Spanning-based QA head and to the Grid Matching Head? 
c) It would be necessary to align notation between figure 3 and the text in section 3.3. I understand that "r" in the figure corresponds to "a" in eq. (1) and "t" corresponds to "C+B" in eq. (1). Is this correct? it should be clarified.
d) It is not clear how the warmer can generate a different output for each recursive step? Are the inputs to the warmer cahnged somehow, at every iteration?
e) It would be helpful to specify how the prompt to the MLMM is built, how the otuput of the warmer is integrated and which is the actual input to the MLMM.
f) At inference time, the input and the output of the warmer is the same as in training?, or only some modules are active at inference? How are different outputs generated in the top-K configurations in order to provide several options to the MLMM?
g) In table 1, what does the row w/bbox corresponds to? What do you mean by "best configuration with bounding boxes"?
- In most of the datasets, the improvement of the proposed approach with respect to using only Gemini is very low. The same for the recursive version vs. the non-recursive version. However, with the proposed approach an overhead in computation is introduced that does not seem fully justified according to the obtained improvement in accuracy. 
- Table 1 only compares the proposed approach with generic MLLMs. I miss a comparison with more specific SoA methods for each dataset.
- Table 2 includes an analysis of the impact of structural adaptation. I miss an ablation study on the impact of the two semantic adaptation tasks.

### Questions
- In Table 1, which is the configuration used in the recursive setting (which is the maximum number of allowed iterations)?

### Soundness
2

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
The paper presents SynDoc, a unified framework for domain-specific Visually Rich Document Understanding (VRDU) that integrates discriminative and generative models to overcome hallucinations and poor domain adaptation in existing LLMs/MLLMs. SynDoc introduces a synthetic data generation pipeline that leverages document structure and domain-specific queries to create high-quality annotations, along with adaptive instruction tuning to enhance domain knowledge extraction. A recursive inference mechanism further refines model outputs for stable, accurate predictions. Experiments show that SynDoc achieves scalable and precise understanding across specialized domains such as medicine, finance, and material science.

### Strengths
- The paper introduces a scalable pipeline that automatically produces high-quality, domain-specific annotated data, reducing dependence on costly manual labeling.
- Adaptive Instruction Tuning is proposed, which enhances the discriminative model’s domain adaptation and knowledge extraction through targeted, domain-aware instruction tuning.
- The authors design a Recursive Inference Mechanism, which integrates discriminative and generative reasoning in a feedback loop, achieving more stable, accurate, and interpretable document understanding results across domains.

### Weaknesses
- The idea of integrating discriminative and generative models is good; however, the overall pipeline is somewhat engineered, which weakens the novelty of the proposed method.
- Besides the improved performance, the computation and storage overload is ignored in this manuscript, which is important for application is real-world system.

### Questions
Please refer to the weakness part.

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
3

### Summary
The authors introduce SynDoc, a novel method for improving information extraction from documents. In this approach, they have a generative MLLM and discriminative warmer model. They use synthetic data generation pipeline to train the warmer, which then provides hints to the MLLM in a recursive inference loop.

### Strengths
- One of the main strength is the  synthetic data generation pipeline, which solves the labelled data bottleneck problem in visually rich documents.
- Their approach of using a discriminator to provide hints to the MLLM is very interesting and useful.

### Weaknesses
- The authors did not discuss about latency. Given that their approach involves recursive inferencing at test time, the latency will significantly increase.

### Questions
- The authors provide zero-shot comparison, in which their approach improved Gemini performance. It would be interesting to see how this would compare to fine-tune models. Can it improve performance of fine-tuned models as well?
- The approach depends on the data generator and warmer model. This can result in bias amplification, example where the warmer learns the generators flaws and reinforces those biases.

### Soundness
4

### Presentation
4

### Contribution
4
