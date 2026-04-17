# Counterfactual LLM-based Framework for Measuring Rhetorical Style

- Decision: Accept (Poster)
- Scores: 2, 6, 8, 4

## Abstract
The rise of AI has fueled growing concerns about "hype" in machine learning papers, yet a reliable way to quantify rhetorical style independently of substantive content has remained elusive. Because bold language can stem from either strong empirical results or mere rhetorical style, it is often difficult to distinguish between the two. To disentangle rhetorical style from substantive content, we introduce a counterfactual, LLM-based framework: multiple LLM rhetorical personas generate counterfactual writings from the same substantive content, an LLM judge compares them through pairwise evaluations, and the outcomes are aggregated using a Bradley--Terry model. Applying this method to 8,485 ICLR submissions sampled from 2017 to 2025, we generate more than 250,000 counterfactual writings and provide a large-scale quantification of rhetorical style in ML papers. We find that visionary framing significantly predicts downstream attention, including citations and media attention, even after controlling for peer-review evaluations. We also observe a sharp rise in rhetorical strength after 2023, and provide empirical evidence showing that this increase is largely driven by the adoption of LLM-based writing assistance. The reliability of our framework is validated by its robustness to the choice of personas and the high correlation between LLM judgments and human annotations. Our work demonstrates that LLMs can serve as instruments to measure and improve scientific evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work presents a counterfactual, LLM-based framework for measuring rhetorical style independently of content. It uses a calibrated panel of LLM personas to generate counterfactual texts, aggregates pairwise comparisons with a Bradley–Terry model, and infers rhetorical strength for new writings. Validated against human judgments, the method is robust to persona choice and produces fine-grained scores. Applied to ICLR submissions, it shows that rhetorical style predicts citations and media attention, revealing a notable increase in rhetorical strength since 2023.

### Strengths
1. This work introduces a counterfactual, LLM-based framework to measure rhetorical style independently of content.

2. It develops a calibrated panel of LLM personas to generate counterfactual writings, aggregates pairwise comparisons via a Bradley–Terry model, and provides a method to infer rhetorical strength for new texts. 

3. The approach is validated against human annotations, is robust to persona choice, and produces fine-grained scores. Applied to a large dataset of ICLR submissions, the framework reveals that rhetorical style predicts citations and media attention, and that rhetorical strength has increased notably since 2023.

### Weaknesses
1. It remains unclear whether the Bradley-Terry score is sufficiently representative or effective for mimicking rhetorical style, especially given its low correlation with reviewer scores. Further analysis is needed (see my questions below).

2. The experimental section feels somewhat weak. There is no clear evidence of the reliability of the Bradley-Terry score, nor a detailed discussion of how it might generalize to broader research contexts.

3. The method depends on LLM-generated counterfactual abstracts and a calibrated panel of LLM judges. This setup introduces potential biases stemming from the choice of LLM personas and their training data, which may influence the assessment of rhetorical strength.

4. It is unclear whether the observed trends generalize beyond ICLR submissions, as all findings are drawn from a single conference domain.

### Questions
1. The selection of personas will largely affect the model performance. Therefore, it's important to elaborate how these personas are selected? Whether the LLMs strictly follow the personas.

2. According to Line 308, if rhetorical style were measured across full papers rather than just abstracts, would the correlation with peer-review scores increase, and could it then meaningfully predict reviewer evaluations?

3. If abstracts (Y) are used as a proxy for rhetorical style while the full paper content (X) represents the substantive content, could the limited scope of abstracts lead to an incomplete or biased estimation of Z? In other words, does using only abstracts risk conflating substantive content with rhetorical framing, since abstracts may omit key details present in X?

4. In Line 101, the authors compare the setup with several methods such as GAN, RLHF, DPO. It remains unclear to me how the setup is connected to this method. It would be helpful if the authors could elaborate more on it.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper assumes that each paper abstract has a particular style of presenting the problem, methods and results, which they call the “rhetorical style” of the paper. The authors build a system that can predict the rhetorical score for a paper given its abstract. The method involves 30 different rewrite prompts that use LLMs to rewrite an abstract in different styles. These styles are assumed to be “counterfactuals”. The method is used for a series of ICLR paper submissions and the results show that, once controlling for review scores, the papers with a higher rhetorical score tend to be cited and discussed more.

### Strengths
I found the correlation between the rhetorical score and the “popularity” of a paper an interesting result, which may show how humans are biased by the presentation style.

### Weaknesses
I don’t see any serious weaknesses. Wondering what the practical application and implications are. Shall we all adopt a writing style that results in a high rhetorical score? :) Also not clear at all why the method is called “counterfactual”.

### Questions
It was not clear to me why the method is called “counterfactuals”. Not clear why the rewriting prompts are assumed to produce counterfactual abstracts. Could you please clarify this term choice? It seems to be inappropriate for this context. 

What are the practical implications and/or applications of this technique?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a counterfactual framework powered by LLM that automatically measures the rhetorical style of research papers independent of its substantive content. The framework models rhetorical style as a one-dimensional variable which is independent to the substantive content of the paper, which jointly derive the surface text form of the paper. To measure this rhetorical style variable, the paper proposes to 1) extract objective and descriptive content of the paper; 2) steer the LLM with a wide spectrum of persona to generate diverse abstracts with counterfactual writings; 3) use LLM-as-judge to provide Bradly-Terry based pair-wise comparison for estimating the rhetorical strength score for each persona to form a spectrum of rhetorical strength; 4) situate the actual (query) abstract in this spectrum to obtain the rhetorical strength estimation for the real abstract. With this framework, the authors estimates the rhetorical strength of paper abstracts of ICLR submissions from 2017 to 2025 and conduct statistical analysis of the correlation between the strength and review scores/downstream attention. The results showcases that 1) the rhetorical estimation by the proposed framework is more effective than baseline methods; 2) while the estimated rhetorical strength showcase minimal correlation with review scores, stronger rhetorical style does lead to larger downstream attention,

### Strengths
1. The proposed framework address the issue of entanglement of substantive content with rhetorical style, overcomes the challenges of biased measurement of rhetorical style in prior work.
2. With the multi-persona counterfactual generation & Bradley-Terry scoring approach, the proposed framework yield high quality and less biased estimation of rhetorical scores than prior approaches.
3. The analysis of the predictive power of rhetorical scores on peer-review scores and downstream impact/attentions provide insightful findings of how different rhetorical styles affect the recognition of the work by the community.

### Weaknesses
1. The single-dimension formulation of the rhetorical strength measurement might have made an over-simplified assumption. For instance, the strength of rhetorical style might be multi-faceted: a paper might argue significant generalizability of their contributions and simultaneously put less emphasis of the novelty/impact. I am thus concerned if the single-dimension rhetorical strength could capture such variability.

### Questions
Please see the weaknessess above

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
4

### Summary
This paper proposes BiasRetriever, a contrastively-trained dense retriever for detecting intersectional biases in LLMs, along with two new paragraph-level datasets (Indic-Intersect and Western-Intersect, totaling 7,404 paragraphs). The authors train a retriever using triplet loss to enable generalization to unseen intersectional bias combinations, reporting up to 10% higher Jaccard scores compared to classification-based approaches.

Novelty. The paper claims to present "the first unified framework" for intersectional bias detection. However, recent literature presents several related approaches. HInter (March 2025) provides automated intersectional bias detection using metamorphic testing across 18 LLMs [1]. Ma et al. (2023) introduced intersectional stereotype benchmarks evaluating LLMs through uncertainty analysis [2]. BiasAlert (July 2024) employs retrieval-augmented bias detection with Contriever embeddings [3]. The contribution appears to be the paragraph-level extension of existing sentence-level datasets and systematic exploration of triplet mining strategies, building on established contrastive learning techniques.

Significance. The work addresses an important problem in AI fairness. However, several considerations affect the scope of contribution. WinoIdentity (2025) provides 245,700 prompts across 50 bias patterns [4] compared to 7,404 paragraphs in this work. The datasets rely on synthetic LLM-generated data without validation on naturally-occurring biased text. Recent work has demonstrated concrete impacts through resume evaluation studies affecting over 190,000 individuals [5], while this paper focuses on benchmark evaluation without downstream task validation. The primary contribution is empirical rather than methodological.

[1] Souani, B., Soremekun, E., Papadakis, M., Yokoyama, S., Chattopadhyay, S., & Le Traon, Y. (2025). HInter: Exposing Hidden Intersectional Bias in Large Language Models. arXiv:2503.11962.

[2] Ma, W., Chiang, B., Wu, T., Wang, L., & Vosoughi, S. (2023). Intersectional Stereotypes in Large Language Models: Dataset and Analysis. In Proceedings of EMNLP 2023.

[3] Fan, Z., Chen, R., Xu, R., & Liu, Z. (2024). BiasAlert: A Plug-and-play Tool for Social Bias Detection in LLMs. In Proceedings of EMNLP 2024, pp. 14778–14790.

[4] Khan, F. A., et al. (2025). Investigating Intersectional Bias in Large Language Models using Confidence Disparities in Coreference Resolution. arXiv:2508.07111.

[5] An, J., Huang, D., Lin, C., & Tai, M. (2025). Measuring gender and racial biases in large language models: Intersectional evidence from automated resume evaluation. PNAS Nexus, 4(3), pgaf089.

[6] Sahoo, N., Kulkarni, P., Ahmad, A., Goyal, T., Asad, N., Garimella, A., & Bhattacharyya, P. (2024). IndiBias: A Benchmark Dataset to Measure Social Biases in Language Models for Indian Context. In Proceedings of NAACL 2024, pp. 8786–8806.

[7] Sap, M., Gabriel, S., Qin, L., Jurafsky, D., Smith, N. A., & Choi, Y. (2020). Social Bias Frames: Reasoning about Social and Power Implications of Language. In Proceedings of ACL 2020.

[8] Dou, Y., et al. (2023). GPTBIAS: A Comprehensive Framework for Evaluating Bias in Large Language Models. arXiv:2312.06315.

### Strengths
- Systematic triplet generation strategies: Comprehensive exploration of four different triplet curation approaches (SR-k4, SR-k10, SR-k4-UN, SR-k4+LLM) with ablation studies demonstrating their relative effectiveness
- Paragraph-level contextual data: Extension beyond sentence-level datasets provides richer narrative contexts for intersectional bias detection, potentially capturing more realistic bias manifestations
- Cross-domain transfer analysis: Demonstrates that models trained on Indic-Intersect transfer to Western-Intersect and vice versa, with label-agnostic retrieval enabling transfer where classification baselines fail
- Cultural context coverage: Addresses Indian context with caste and regional biases, complementing IndiBias (2024) [6]
- Rigorous annotation process: Strong inter-annotator agreement (Cohen's κ = 0.89-0.91) with multi-phase validation including sentence-level and paragraph-level quality checks
- Comprehensive empirical evaluation: Thorough analysis across multiple experimental conditions, retrieval depths, and generalization scenarios with clear presentation of results

### Weaknesses
- Positioning relative to existing work: The relationship to HInter (2025) [1], Ma et al. (2023) [2], and other intersectional bias detection frameworks requires clarification
- Limited comparison with BiasAlert: BiasAlert (2024) uses retrieval-based detection with Contriever encoders [3]; the specific advantages of training the retriever versus BiasAlert's RAG approach need more detailed empirical comparison
- Dataset scale: The 7,404 paragraphs represent a smaller scale compared to WinoIdentity's 245,700 prompts [4] and SBIC's 150k annotations [7]
- Data generation methodology: Reliance on LLM-generated data (GPT-4o-mini, Llama) without validation on naturally-occurring biased text from social media, news, or other authentic sources
- Evaluation scope: Mising comparisons with HInter's metamorphic testing [1] and GPTBIAS framework [8]; limted to Jaccard similarity without analyzing harm severity or bias magnitude
- Application validation: No downstream task evaluation in real-world scenarios such as hiring, content moderation, or hate speech detection to demonstrate pratical utility
- Methodological foundations: Builds on established contrastive learning and triplet loss techniques; the application context is the primary differentiator

### Questions
1) How does BiasRetriever compare directly against HInter's metamorphic testing approach and BiasAlert's RAG-based detection on the same benchmark? What are the specific performance differences?
2) What empirical evidence demonstrates the advantages of training the retriever over BiasAlert's approach of using frozen Contriever with LLM judging? Could ablation studies quantify this benefit?
3) Have you validated that detected biases reflect real-world stereotypes rather than patterns in GPT-4o-mini's generation? What would evaluation on naturally-occurring biased text from social media or news sources reveal?
4) What is the rationale for the dataset size (7,404 paragraphs) compared to concurrent work like WinoIdentity (245,700 prompts)? Does the paragraph-level format provide sufficient advantages to justify this scale?
5) Could you provide downstream task evaluation (e.g., resume screening, content moderation, hate speech detection) to demonstrate practical utility beyond synthetic benchmark performance?
6) How does performance degrade when the retrieval database lacks coverage of a bias type?
7) Given that IndiBias addresses Indian context and SBIC addresses Western context, what is the specific value proposition of creating paragraph-level versions rather than leveraging existing datasets directly?
8) How does the framework handle cases where biases are expressed implicitly or require substantial cultural context to interpret correctly?

### Soundness
3

### Presentation
2

### Contribution
2
