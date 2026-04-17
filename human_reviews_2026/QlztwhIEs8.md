# LLM4CTI: Uncovering Security Entities and Their Interactions from Unstructured Cyber Threat Intelligence

- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Unstructured Cyber Threat Intelligence (CTI) articles are a critical source of detailed threat knowledge. Structuring this information into knowledge graphs is crucial for downstream applications, including enhancing Large Language Models (LLMs) factuality and enabling advanced predictive analysis. However, extracting security-related knowledge from these long articles poses a fundamental dilemma for LLMs: capturing relationships scattered across distant sentences requires full-article context, yet processing long inputs triggers issues like the “lost in the middle” phenomenon, while naive chunking severs these crucial long-range dependencies. To resolve this dilemma, we present LLM4CTI, an approach that employs a novel dual-context, chunk-wise processing pipeline. For each text chunk, LLM4CTI provides the LLM with both the local chunk for fine-grained analysis and the full article for global context. A formal, security-oriented framework further guides the LLM to focus on threat-relevant information. Evaluation shows that LLM4CTI achieves a state-of-the-art F1-score of 85.13%. This is driven by an average recall of 81.58%, substantially outperforming competitors like CTINexus (65.97%), CTIKG (76.73%), and GraphRAG (70.25%), while maintaining a high precision of 89.87%. Notably, the version of LLM4CTI powered by a local 32B open-source model also achieves a strong F1-score of 81.58%, surpassing baselines that rely on the much larger GPT-4o. Furthermore, LLM4CTI’s integrated GNN leverages the extracted knowledge to deliver predictive insights. It successfully forecasts relationships between threat-related entities, such as connections between malware families, with over 80% Hits@10 accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents LLM4CTI, an LLM-based framework for extracting structured cyber threat intelligence (CTI) knowledge from long, unstructured CTI articles and transforming it into a knowledge graph. The method introduces a dual-context, chunk-wise processing pipeline that gives an LLM both local and global context to mitigate issues like “lost in the middle” and hallucination. It is guided by a formal threat-centric schema inspired by STIX 2.1 and integrates a Graph Neural Network (GNN) for downstream relationship prediction among security entities. Experiments on large CTI corpora show that LLM4CTI achieves state-of-the-art performance.

### Strengths
- **Originality**: Introduces a dual-context, chunk-wise reasoning framework that balances local accuracy and global context in long CTI articles—a novel application of LLMs to cyber threat intelligence.  
- **Clarity**: Features a clean structure with step-by-step explanations, clear phase definitions, and illustrative diagrams for ease of understanding.  
- **Significance**: Delivers open-source tooling that enables scalable, structured extraction and prediction of threat intelligence from unstructured text, outperforming commercial systems using open-source models.

### Weaknesses
- **Novelty**: While the paper proposes combining local and global context in its pipeline, this concept is not new; previous work has already addressed this challenge in both training and inference phases (e.g., via adaptive chunking) [1, 2]. Moreover, the fixed segmentation of documents into 400-token chunks appears arbitrary and lacks methodological justification.  
- **Unfair Comparison**: In comparing its approach to the three baselines (CTINexus, CTIKG, and GraphRAG), the paper uses GPT-4o as the baseline model. Notably, Table 2 shows that LLM4CTI achieves only ~50% F1 when using GPT-4o, which raises questions about the efficacy and generalization of the proposed method.  
- **Lack of Analysis**: The presentation of results lacks depth as there is minimal discussion of when or why the approach succeeds, fails, or underperforms. For instance, the link-prediction outcomes reported in Table 7 are hard to interpret without a clearly defined baseline or detailed breakdown of error modes.  


[1] An, Shengnan, et al. "Make your llm fully utilize the context." Advances in Neural Information Processing Systems 37 (2024): 62160-62188.


[2] Duarte, André V., et al. "Lumberchunker: Long-form narrative document segmentation." arXiv preprint arXiv:2406.17526 (2024).

### Questions
- While the authors briefly discuss why GPT-4o performs so poorly with the proposed system, the other three baselines still perform much better with GPT-4o. Can the authors provide the corresponding baseline results using o4-mini or QWQ 32B, or explain why those baselines do not suffer from the same performance degradation as LLM4CTI?  
- For RQ3, a proper baseline is needed to gauge the difficulty of the task and the size of the improvement achieved. Please include such a baseline in the evaluation.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes LLM4CTI to extract CTI evidence from long reports via dual-context, chunk-wise prompts (local chunk + full article) and merges results into a STIX-inspired knowledge graph. It then trains a GNN to predict latent links (e.g., malware–actor). Experiments show state-of-the-art F1 with notably higher recall than baselines and strong Hits@10 for link prediction. The problem is important to both AI and cybersecurity.

### Strengths
The methodology design is clear. The paper relies on a KG structure construction and conduct its design in a sequential way: extracting entities --> parse relations from unstructured texts.

### Weaknesses
1. There is a significant concern of novelty. Overall, this paper aims to construct KGs from articles, however, existing works such as ThreatKG [1] already did the same thing. How could this work claim its novelty and significance?

2. CTI from articles are limited in its scope. Mostly, CTI is collected from public threat feeds, daily reports, system logs, or vulnerability databases. Targeting only on articles are quite constrained on the impact and comprehensiveness.

3. The design heavily relies on AI-generated annotation or identification. The author also claims that using Gemini-2.5 achieve 85% agreement with human annotators, which implies a significant noise level (15%) in the produced KG.

4. The evaluation datasets are not comprehensive. The author only considers 55+302 articles among the daily evolving cyberspace, which simply yields a tiny set of CTI and cannot provide strong evaluation insights.

[1] ThreatKG: An AI-Powered System for Automated Open-Source Cyber Threat Intelligence Gathering and Management

### Questions
Please check the comments above.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the problem of translating Cyber Threat Intelligence (CTI) reports into knowledge graphs (KGs) using large language models (LLMs). It focuses on handling long CTI reports by dividing them into smaller chunks and proposes constraint on KG structure tailored specifically for CTI data. The empirical results show improvements over existing methods, although the gains are not very substantial.

### Strengths
- The problem is timely and important.

### Weaknesses
- While the proposed approach is effective, it appears to be mainly engineering-oriented, with limited novelty in methodological design.

- The paper lacks ablation studies to examine the contribution of each component in the proposed framework. In particular, it remains unclear how the length of CTI reports impacts the model’s effectiveness. It is possible that the observed improvements are due to other factors rather than the handling of long documents.

### Questions
- Provide ablation studies to investigate the effect on the CTI's length.

### Soundness
3

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
This paper presents LLM4CTI, a system for automated CTI extraction and knowledge graph construction from long unstructured CTI articles.
The method attempts to address the challenges of extracting information from long CTI report/sources such as entities spread across the long content, irrelevant content. This is done by  splitting the content into chunks each processed by an LLM together with the full doc. as context. In addition, a GNN-based link prediction model is used to infer relationships among the entities. 
The method was evaluated using a large CTI corpus and was compared with prior approaches, CTINexus, CTIKG, and GraphRAG in entity/relation extraction task. Results show high precision/recall using both proprietary and open-source LLMs.

### Strengths
1. Problem and motivation is clearly described
2. Evaluation was performed using a large dataset
3. Evaluating with local models, which shows good results
4. Demonstrating how the GNN-based link prediction can provide actionable value for analysts.

### Weaknesses
1. Related works is not informative and missing relevant papers, for example:
[a] Entity and relation extractions for threat intelligence knowledge graphs
[b] KnowCTI: Knowledge-based cyber threat intelligence entity and relation extraction
[c] CyberEntRel: Joint extraction of cyber entities and relations using deep learning
[d] CyberKG: Constructing a Cybersecurity Knowledge Graph Based on SecureBERT_Plus for CTI Reports
[e] Ctinexus: Automatic cyber threat intelligence knowledge graph construction using large language models
and there are more...

2. Limited novelty - basically applying a common practice of splitting the input to an LLM into small sections to improve its performance.

3. An evaluation of removing important component of the method such as chunk-wise processing will help in understanding the effectiveness of the method. Also, what if removing the GNN? the dual-context...
 
4. Figure 1 is hard to read and should be improved. 

5. Table 1 - the text is too small and hard to read.

6. Need to discuss and evaluate the robustness of LLM4CTI to prompt injection attacks.

### Questions
1. How do you address images within the CTI, some are relevant and some are not?
2. You are relying on prompts and LLMs. How robust is your system? 
3. How was the dataset labeled and annotated?
4. Can you provide details on the average execution time?

### Soundness
2

### Presentation
2

### Contribution
3
