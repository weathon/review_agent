# Proactive Reasoning-with-Retrieval Framework for Medical Multimodal Large Language Models

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Incentivizing the reasoning ability of Multimodal Large Language Models (MLLMs) is essential for medical applications to transparently analyze medical scans and provide reliable diagnosis. However, existing medical MLLMs rely solely on internal knowledge during reasoning, leading to hallucinated reasoning and factual inaccuracies when encountering cases beyond their training scope. Although recent Agentic Retrieval-Augmented Generation (RAG) methods elicit the medical model's proactive retrieval ability during reasoning, they are confined to unimodal LLMs, neglecting the crucial visual information during reasoning and retrieval. Consequently, we propose the first **Multimodal Medical Reasoning-with-Retrieval framework, Med-RwR**, which actively retrieves external knowledge by querying observed symptoms or domain-specific medical concepts during reasoning. Specifically, we design a two-stage reinforcement learning strategy with tailored rewards that stimulate the model to leverage both visual diagnostic findings and textual clinical information for effective retrieval. Building on this foundation, we further propose a **Confidence-Driven Image Re-retrieval (CDIR)** method for test-time scaling when low prediction confidence is detected. Evaluation on various public medical benchmarks demonstrates Med-RwR's significant improvements over baseline models, proving the effectiveness of enhancing reasoning capabilities with external knowledge integration. Furthermore, Med-RwR demonstrates remarkable generalizability to unfamiliar domains, evidenced by 8.8% performance gain on our proposed EchoCardiography Benchmark (ECBench), despite the scarcity of echocardiography data in the training corpus. Our data, model, and codes will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Med-RWR, a multimodal Reasoning-with-Retrieval framework for the medical VQA task. Med-RWR first develops an environment with a curated multimodal dataset and medical knowledge base to train the medical MLLM to use external reliable sources. Then, it introduces a two-stage (text-only to multimodal) RL training strategy to guide the model to retrieve visually and textually related references (semantic reward) and gain confidence through retrieving (confidence gain reward). During inference, it introduces a Confidence-Driven Image Re-retrieval strategy if low confidence is detected.

### Strengths
1. The idea of reasoning-with-retrieving is sound, with support from reliable medical KB construction and reward designs (semantic reward and confidence gain reward).
2. The paper is overall well-written, with each component clearly explained with visualizations.
3. The experimental results show that Med-RWR effectively enhances performance across different medical VQA benchmarks.

### Weaknesses
1. Misconception on Test-Time Scaling (TTS). This paper mentions that the Confidence-Driven Image Retrieval (CDIR) method is for test-time scaling. However, this might be a misconception. Will retrieving more similar cases further improve performance? Providing a figure on this would be helpful. If the answer is yes, then it is fine to use TTS. Otherwise, it would be better to use another terminology for clarity.

2. Too many hand-crafted hyperparameters. For example, the reward weights in Eq. 4 and the confidence threshold in line 471.

3. Lack of baselines. This work incorporates multimodal RAG with reasoning to enhance medical MLLMs' capability. However, it is mainly compared to reasoning medical MLLM. How does it perform compared to multimodal RAG methods [1, 2]?


[1] MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models. ICLR 2025.

[2] RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models. EMNLP 2024.

### Questions
Please see weaknesses. I would be willing to adjust my ratings if the concerns are addressed.

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
5

### Summary
This paper introduces MED-RWR, a novel framework for medical multimodal large language models (MLLMs) designed to address the problem of factual inaccuracies and hallucinations by actively integrating external knowledge during reasoning. Unlike existing methods that rely solely on internal knowledge or unimodal retrieval, MED-RWR enables the MLLM to proactively query for information based on both visual and textual inputs. The core of the method is a two-stage RL strategy. The framework is evaluated on several public medical benchmarks and a newly curated ECBench, demonstrating significant improvements in accuracy and generalizability to scarce domains.

### Strengths
- The paper tackles a critical and high-stakes problem in medical AI: the unreliability and hallucination of MLLMs, which stems from their reliance on static internal knowledge. The proposed reasoning-with-retrieval approach is a well-motivated solution.
- The reward engineering is a key strength. The Query Semantic Reward is novel for jointly encouraging textual relevance and visual grounding. Furthermore, the Confidence Gain Reward is an nice way to optimize for the utility of the retrieved information by measuring its impact on the model's confidence in the correct answer.
- The paper includes a comprehensive set of ablation studies that validate the key design choices.

### Weaknesses
- Lack the Ethics statement and Reproducibility statement in the main text.
- The CDIR mechanism, while interesting, has questionable scalability. At inference time, it computes image similarity against a randomly selected subset of 10,000 images from the multimodal corpus. This selection seems arbitrary, and the paper does not address how this method would scale to a more realistic, much larger corpus (e.g., PubMedVision corpus).
- The framework's retrieval mechanism is effectively limited to a single turn, which is insufficient for complex, multi-hop medical reasoning. The paper's own prompt design in Table 10 is contradictory, stating both "Only one query is allowed" and "Multiple think-query-retrieve cycles may occur". In practice, all case studies provided (e.g., Figures 9-14) demonstrate only a single query-retrieval step. This single-turn design is brittle. If the initial retrieval is inaccurate (as shown in the failure case in Figure 15), the model has no mechanism to correct its course by re-querying. This lack of iterative reasoning may explain why the overall performance gains are not as substantial as one might expect from a more sophisticated multi-step agent.
- Lack a section of discussion of recent retrieval-based methods [1,2,3,4] and Reasoning-with-Retrieval methods [5,6,7].

[1] MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models. ICLR 2025.

[2] Fact-Aware Multimodal Retrieval Augmentation for Accurate Medical Radiology Report Generation. NAACL 2025.

[3] RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models. EMNLP 2024.

[4] Patho-agenticrag: Towards multimodal agentic retrieval-augmented generation for pathology vlms via reinforcement learning. arXiv preprint arXiv:2508.02258.

[5] MMSearch-R1: Incentivizing LMMs to Search. arXiv preprint arXiv:2506.20670.

[6] Webwatcher: Breaking new frontier of vision-language deep research agent. arXiv preprint arXiv:2508.05748.

[7] DeepMMSearch-R1: Empowering Multimodal LLMs in Multimodal Web Search. arXiv preprint arXiv:2510.12801.

### Questions
- See weaknesses.
- If the authors can address the limitations mentioned in the weaknesses, I will consider raising my score.

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
This paper proposes MED-RWR, a comprehensive Multimodal Medical Reasoning-with-Retrieval framework, to address the factual inaccuracies in existing medical MLLMs by encouraging MLLMs to proactively retrieve external knowledge using both visual and textual information during reasoning. The framework utilizes a two-stage reinforcement learning strategy with customized rewards to facilitate effective multimodal retrieval, and further introduces a Confidence-Driven Image Reretrieval (CDIR) mechanism to augment potentially insufficient information when the model's confidence is low.

### Strengths
1. The paper is well-written, with clear logic and is easy to understand.
2. Extensive experiments on multiple benchmarks demonstrate the superior performance of the proposed MED-RWR.
3. Comprehensive ablation analysis shows the effectiveness of each component.

### Weaknesses
1. Why does Equation 2 only compute the semantic similarity between the image and the query, instead of also considering the retrieved content as in Equation 1?
2. Section 3.1 mentions that the difficulty levels of the samples were stratified during dataset construction for progressive curriculum training, but curriculum training does not appear to be utilized subsequently in the methodology.
3. In line 309, it is mentioned that “We apply accuracy and format rewards to instill the model’s fundamental medical reasoning-with-retrieval capabilities.” How is the retrieval capability obtained in this context?
4. Based on my understanding, the last row in Table 2 should represent the complete MED-RWR model. If so, why do the numerical values not correspond to those reported in Table 1?
5. The base model that MED-RWR was fine-tuned upon is not specified.
6. What knowledge base did the Agentic Search MLLM (used in the comparison methods) utilize for external knowledge retrieval?
7. Some presentations need optimization. For example, Figure 1 is somewhat cluttered and a bit hard to interpret, while the headers in Tables 2 and 3 appear to be misaligned.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces an agentic retrieval framework with reasoning augmentation, enabling the base model to learn how to better retrieve evidence from external knowledge bases and perform reasoning for medical visual question answering. 
Different from previous text-only agentic retrieval systems that simply copy or rephrase the question to expand the search, the proposed med-rwr first observes visual clues and then generates a multimodal query. to build such a system, the authors curate a dedicated dataset including think, query, retrieve, and answer items, and train the model using a composite reward. Experiments on multiple datasets show consistent performance improvements.

### Strengths
1. the proposed method is solid and supported by comprehensive experimental settings and ablation studies.
2. the target problem it aims to address, agentic retrieval and multimodal information fusion, is important for the medical analysis domain.

### Weaknesses
Unclear hyperparameter design: the reward function is composite, but the weights assigned to each component vary widely without sufficient explanation or justification. It’s unclear how these weights were determined or whether any sensitivity analysis was performed.

Inappropriate RAG baselines: Although the authors compare their approach with a training-free RAG setup (see Figure 3), they don’t clearly specify what the base model is (is it Med-RWR, and is it based on qwen or lingshu?) In addition, they only compare against two general-domain agentic retrieval methods (Visual-ARFT and MMSearch-R1), but not against medical-domain RAG methods such as MMed-RAG (ICLR ’25) or RULE (EMNLP ’24). 
Comparing purely reasoning-based models is also not very meaningful here, since retrieval introduces additional information by design, and the general-domain retrieval/reasoning models are not trained on medical data.

Insufficient generalization validation: While the authors introduce ECBench, it appears to be private, making it hard for others to verify or fully understand the dataset details. To support the generalization claims, I recommend adding experiments on at least one public medical benchmark.

### Questions
I think the motivation behind confidence-driven test-time compute scaling is that the retrieved information isn’t always sufficient. So a straightforward fix would be to just retrieve more text chunks, basically scale up retrieval.

But the current setup instead uses another knowledge base for multimodal retrieval. I’m not totally sure I understand the reasoning here. could you help explain the logic? also, why not just use both databases directly at the same time, instead of doing it in two stages?

### Soundness
3

### Presentation
2

### Contribution
3
