# Enhancing Conversational Recommender Systems with Tree-Structured Knowledge and Pretrained Language Models

- Decision: Reject
- Scores: 6, 3, 3, 5

## Abstract
Conversational recommender systems (CRS) have emerged as a key enhancement to traditional recommendation systems, offering interactive and explainable recommendations through natural dialogue.
Recent advancements in pretrained language models (PLMs) have significantly improved the conversational capabilities of CRS, enabling more fluent and context-aware interactions. 
However, PLMs still face challenges, including hallucinations—where the generated content can be factually inaccurate—and difficulties in providing precise, entity-specific recommendations.
To address these challenges, we propose the PCRS-TKA framework, which integrates PLMs with knowledge graphs (KGs) through prompt-based learning. By incorporating tree-structured knowledge from KGs, our framework grounds the PLM in factual information, thereby enhancing the accuracy and reliability of the recommendations. Additionally, we design a user preference extraction module to improve the personalization of recommendations and introduce an alignment module to ensure semantic consistency between dialogue text and KG data. Extensive experiments demonstrate that PCRS-TKA outperforms existing methods in both recommendation accuracy and conversational fluency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Conversational recommender systems (CRS) enhanced by pretrained language models (PLMs) face challenges. The PCRS-TKA framework, integrating PLMs with knowledge graphs via prompt-based learning, addresses these by enhancing accuracy and personalization. Experiments show it outperforms existing methods in accuracy and fluency.

### Strengths
(1) This paper presented a signifcant extension on PLM based CRS with newly proposed method and extensive experiments. 

(2) The results have shown improved performance on the task of conversational recommendation.

### Weaknesses
(1) The introduction could be improved by incorporating more movitation and rationale, e.g., how and why the proposed method would improve the performance?

(2) The proposed method seems somehow complicated. Can it be simplified to some extent? A simple yet efficient method is important in practice. 

(3) Do you try larger language models except DialogGPT?

### Questions
See my comments in the weaknees

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
In this paper, authors focus on challenges included in PLMs and propose the PCRS-TKA framework, which integrates pretrained language models (PLMs) pretrained language models (PLMs) with knowledge graphs (KGs) using prompt-based learning to enhance conversational recommender systems (CRS). 
By leveraging tree-structured knowledge from KGs, PCRS-TKA aims to improve the accuracy and reliability of recommendations while addressing challenges such as hallucinations and the need for personalized interactions. 
This framework includes a user preference extraction module and an alignment module to ensure semantic consistency. Experimental results indicate that PCRS-TKA outperforms existing methods in both recommendation accuracy and conversational fluency.

### Strengths
1. **Integration of PLMs with KGs**:
   - The framework effectively combines the strengths of PLMs and KGs, enabling richer contextual understanding and improved recommendation accuracy.

2. **Reproducibility**:
   - The availability of the source code enhances the reproducibility of the results, allowing other researchers to validate and build upon the findings.

3. **Experimental Results**:
   - Extensive experiments demonstrate significant improvements in recommendation accuracy and conversational fluency, providing empirical support for the proposed framework.

4. **Personalization**:
   - The inclusion of a user preference extraction module shows a thoughtful approach to improving the personalization of recommendations, which is critical in conversational settings.

### Weaknesses
1. **Lack of Novelty**:
   - The framework appears to be a straightforward extension of existing methods, lacking unique ideas or contributions in the proposed modules, architecture, and training objectives.

2. **Insufficient Literature Review**:
   - The paper does not adequately survey recent advancements in prompt learning, which diminishes the contextual grounding of the proposed approach within the broader research landscape.

3. **Hallucination Challenge**:
   - While the authors mention hallucinations, they do not provide a concrete solution to this issue or conduct experiments to compare models concerning this challenge, limiting the impact of their claim.

4. **Generalizability Concerns**:
   - The choice of RoBERTa and DialoGPT as PLMs raises questions about the generalizability of PCRS-TKA. The framework's effectiveness with more recent models, such as LLama-3-instruct, remains unexplored.

5. **Limited Discussion on PLM Variability**:
   - There is a lack of discussion regarding the applicability of PCRS-TKA across diverse PLMs, as well as the potential advantages and disadvantages of different PLMs in the context of the framework.

### Questions
1. **Novelty and Unique Contributions**:
   - Can you elaborate on the unique contributions of PCRS-TKA compared to existing frameworks? How does it differ fundamentally from other prompt-based learning approaches?

2. **Addressing Hallucinations**:
   - Could you clarify how your framework specifically addresses the issue of hallucinations? Are there potential strategies you envision for mitigating this problem?

3. **Generalizability Across PLMs**:
   - Have you considered testing PCRS-TKA with state-of-the-art models like LLama-3-instruct? What are your thoughts on how the framework might perform with different PLMs?

4. **Discussion on Prompt Learning**:
   - How do you plan to integrate the latest findings in prompt learning into your framework? What implications might this have for your proposed methods?

5. **Future Research Directions**:
   - What future research directions do you foresee for enhancing the PCRS-TKA framework? Are there specific areas you believe warrant further exploration?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes PCRS-TKA, a framework that enhances conversational recommendation systems by integrating pre-trained language models with knowledge graphs through knowledge-enhanced prompt learning.
The framework consists of four main components: a feature encoder module for encoding dialogue and graph information, a knowledge tree module for KG triple transformation, a user preference module for preference extraction, and a soft-prompt module for task guidance. The combined prompt template is then used by DialoGPT to generate conversational recommendations.

### Strengths
- Clear presentation of the framework and results.
- Interesting perspective on representing KG triples as a tree structure for prompt learning for CRS.

### Weaknesses
- PCRS-TKA heavily relies on established techniques and follows UniCRS[1] architecture, While the knowledge tree module introduces some innovation, other components like constructive learning and prompt-based design are mainly adaptations of existing methods from CRS [1][2].
- The choice of PLMs needs justification given LLMs have demonstrated promising performance in CRS.
- The benchmark comparison focuses on older models (up to 2022); the authors should include later models.
- The authors emphasize PLM's logical reasoning abilities and KG for hallucination reduction, there's no systematic evaluation of these. Also, the human evaluation survey from the provided link actually shows many responses contain incorrect facts or hallucinations

[1] Wang, X., Zhou, K., Wen, J.-R., Zhao, W.X., 2022. Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning, in: Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. Presented at the KDD ’22: The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, ACM, Washington DC USA, pp. 1929–1937. https://doi.org/10.1145/3534678.3539382   
[2] Zhou, Y., Zhou, K., Zhao, W.X., Wang, C., Jiang, P., Hu, H., 2022. C2-CRS: Coarse-to-Fine Contrastive Learning for Conversational Recommender System, in: Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining. Presented at the WSDM ’22: The Fifteenth ACM International Conference on Web Search and Data Mining, ACM, Virtual Event AZ USA, pp. 1488–1496. https://doi.org/10.1145/3488560.3498514.  
[3] Yang, B., Han, C., Li, Y., Zuo, L., Yu, Z., 2022. Improving Conversational Recommendation Systems’ Quality with Context-Aware Item Meta-Information, in: Carpuat, M., de Marneffe, M.-C., Meza Ruiz, I.V. (Eds.), Findings of the Association for Computational Linguistics: NAACL 2022. Association for Computational Linguistics, Seattle, United States, pp. 38–48. https://doi.org/10.18653/v1/2022.findings-naacl.4

### Questions
* In line 415, the authors attribute PCRS-TKA's higher performance on INSPIRED to "more dialogues", yet INSPIRED actually contains fewer dialogues than ReDial. Could the authors clarify this contradiction and provide a more accurate analysis of the performance gains?

2. How is the tree-structured knowledge representation significantly different from MESE’s[3] approach of using item metadata as knowledge prompts, especially when the knowledge tree is set to hop=1? And what specific advantages does this added complexity bring over simpler metadata-based methods?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper addresses two key limitations of existing Conversational Recommender Systems (CRS): (1) hallucinations and (2) difficulties in providing precise, entity-specific recommendations.
To overcome these challenges, this paper proposes the PCRS-TKA framework. This framework is designed to enhance the accuracy and reliability of CRS by effectively aligning tree-structured information from knowledge graphs with the semantic information in conversation history through prompt learning.

### Strengths
- The authors clearly and thoroughly explained the proposed method to address the limitations of Conversational Recommender Systems (CRS).
- The authors demonstrate the effectiveness of the proposed framework and its components in enhancing recommendation capability.

### Weaknesses
- **The proposed method does not fully handle the limitations.** 
The authors pointed out the issue of inaccurate generation caused by hallucinations. However, the proposed framework has different input prompts for the recommendation and response generation tasks. While the proposed module components can effectively bridge the semantic gap between the conversation prompt and the entity prompt, they cannot completely resolve the issue. Providing different input prompts to the same model can lead to additional semantic misalignment issues, as the prompts may cause the model to perform distinct tasks even if they are connected through a semantic alignment process. This could lead to the fundamental issue of semantic inconsistency between recommendations and conversations, as highlighted in previous research[1]. Therefore, additional experimental or statistical evidence is needed. For instance, it is necessary to examine how consistently the generated dialogue aligns with the predicted recommended items and the related entities from the knowledge graph to effectively address the hallucination problem.


- **The conducted dataset exhibits a "repeated item shortcut" problem, which refers to data leakage where items have appeared in previous conversation turns [2][3].**
More than 15% ground-truth items are repeated items in INSPIRED dataset. This issue suggests that the proposed structure utilizes the embedding of the ground truth (GT) item as the input prompt. Additional validation is needed to assess whether the proposed method remains effective in scenarios eliminating repeated items as ground truth.

[1] Wang, Xiaolei, et al. "Towards unified conversational recommender systems via knowledge-enhanced prompt learning." Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2022.

[2] He, Zhankui, et al. "Large language models as zero-shot conversational recommenders." Proceedings of the 32nd ACM international conference on information and knowledge management. 2023.

[3] Zhu, Lixi, Xiaowen Huang, and Jitao Sang. "How Reliable is Your Simulator? Analysis on the Limitations of Current LLM-based User Simulators for Conversational Recommendation." Companion Proceedings of the ACM on Web Conference 2024. 2024.

### Questions
- I agree that the proposed framework effectively handles the structured information of the KG and text semantic information through various fusion modules. But I believe further analysis of the two well-known problems mentioned above is necessary.
- The results of the analyses on the degree and depth of the knowledge tree in Section 4.5 are intriguing. I agree that an increase in depth can introduce noisy information and may lead to the loss of information regarding the nearest relationships. However, could it also be interpreted that the embedding model fails to adequately reflect the structural information of the proposed construction of the knowledge tree text? I wonder about the potential outcomes if the knowledge tree text is better structured or if a more powerful embedding model is used.

### Soundness
3

### Presentation
3

### Contribution
2
