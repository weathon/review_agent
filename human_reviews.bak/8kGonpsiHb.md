# Lens: Rethinking Multilingual Enhancement for Large Language Models

- Decision: Reject
- Scores: 5, 6, 8, 8

## Abstract
Despite the growing global demand for large language models (LLMs) that serve users from diverse linguistic backgrounds, most cutting-edge LLMs remain predominantly English-centric. This creates a performance gap across languages, restricting access to advanced AI services for non-English speakers. Current methods to enhance multilingual capabilities largely rely on data-driven post-training techniques, such as multilingual instruction tuning or continual pre-training. However, these approaches encounter significant challenges, including the scarcity of high-quality multilingual datasets and the limited enhancement of multilingual capabilities. They often suffer from off-target issues and catastrophic forgetting of central language abilities. To this end, we propose \textsc{Lens}, a novel approach to enhance multilingual capabilities of LLMs by leveraging their internal language representation spaces. Specially, \textsc{Lens} operates by manipulating the hidden representations within the language-agnostic and language-specific subspaces from top layers of LLMs. Using the central language as a pivot, the target language is drawn closer to it within the language-agnostic subspace, allowing it to inherit well-established semantic representations. Meanwhile, in the language-specific subspace, the representations of the target and central languages are pushed apart, enabling the target language to express itself distinctly. Extensive experiments on one English-centric and two multilingual LLMs demonstrate that \textsc{Lens} effectively improves multilingual performance without sacrificing the model’s original central language capabilities, achieving superior results with much fewer computational resources compared to existing post-training approaches.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a novel method for enhancing multilingual performance in large language models by refining the model's text representation space. The approach applies singular value decomposition to analyze and separate cross-language similarities and differences within the representation space. During model training, this method aims to minimize the "language-agnostic" subspace and separate out the "language-specific" representation space, enabling better knowledge sharing with English (the central language) and improving the unique characteristics of each language. Experimental results indicate that this method effectively enhances multilingual capability, especially improving language fidelity.

### Strengths
1. The methodology is straightforward, with clear explanation and rationale.
2. Comparative results validate the effectiveness of the proposed approach.

### Weaknesses
1. The technique primarily builds on existing methods (such as https://aclanthology.org/2023.findings-emnlp.190/) and lacks significant novelty.
2. Language selection for experiments could have accounted for the proximity of each language to the central language, which would add meaningful insights.
3. Performance inconsistencies are observed, such as the inferior results for the Phi dataset compared to llama's performance in the appendix.

### Questions
1. Since the method doesn't leverage external data to augment knowledge, wouldn't languages closer to English theoretically benefit more? It raises the question: does the "alignment" operation inherently favor languages that are closer to or in the same family as English?
2. Why do the results in Figure 2 show greater improvement for Chinese and Japanese, but limited improvement for Arabic and Bengali?

### Soundness
3

### Presentation
3

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
This paper introduces a method to enhance the multilingual capabilities of LLMs by leveraging the central-language internal language representation as pivot signal. Specifically, the authors decouple the internal language representation spaces into language-agnostic and language-specific subspaces. In the language-agnostic subspace, they pull the target language representations closer to those of English to inherit its capabilities, while in the language-specific subspace, they push the target language representations away from English to ensure distinct expression.

### Strengths
1. **Resource-efficient** Compared with previous resource-intensive methods like MSFT and continual pretraining, the proposed method enhances multilingual capabilities efficiently with fewer data resources and computation costs.

2. **Competitive Performance** This method demonstrates comparable performance with open-source LLMs that conduct large-scale post-training to enhance multilingual capabilities. Moreover, it surpasses current strong baselines in multilingual enhancement by a large margin.

3. **Good Interpretability** Inspired by previous findings on LLM interpretability, the authors manipulate the internal language representations in the top layers of LLMs, applying these findings to multilingual enhancement successfully. The results of the visualization analysis underscore the interpretability advantages of the proposed method.

### Weaknesses
1. **Missing Reference** Previous work [1] has explored how to enhance multilingual abilities through aligning internal sentence representations, but there is a lack of detailed introduction to this relevant research.

2. **The results of the ablation study do not fully support the authors' claims** The authors claim that target languages inherit capabilities from English by pull the target language representations closer to those of English. However, the left part of the figure 3 demonstrate that the performance improvement does not primarily stem from this component. There is only a slight performance variance, even when the hyperparameter is set to zero.

3. **Incorrect Color in Table1** The performance of SDRRL on XCOPA outperforms original backbone. However, the authors highlight it with red color. Additionally, I believe that comparable performance should not be indicated in green. Some results that are clearly lower than the original backbone is still marked in green, which could lead to misunderstandings about performance for readers.






[1] Improving In-context Learning of Multilingual Generative Language Models with Cross-lingual Alignment (https://aclanthology.org/2024.naacl-long.445) (Li et al., NAACL 2024)

### Questions
1. **Lack of explanation** Consider adding an explanation for what the bold and underlined text indicates in the caption of Table 1.

2. **Caption Error** The caption for Figure3 (Line408) contains an error, please correct it.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper, LENS, is an innovative approach that enhances multilingual capabilities of large language models (LLMs) by manipulating their internal language representation spaces. It operates on the top layers of LLMs, aligning target languages with a central language in a shared semantic space while differentiating them in a language-specific space. This method significantly improves multilingual performance without compromising the original language abilities and does so with fewer computational resources compared to existing post-training techniques.

### Strengths
1. Novelty of Approach: LENS introduces a novel perspective on multilingual enhancement by leveraging the internal language representation spaces of LLMs, offering a fresh approach compared to traditional data-driven post-training methods.

2. Efficiency and Effectiveness: LENS demonstrates high efficiency and effectiveness by achieving superior multilingual performance with significantly less computational resources, making it scalable and practical for large-scale applications.

3. Preservation of Central Language Abilities: A key strength of LENS is its ability to enhance multilingual capabilities without sacrificing the model's original central language performance, addressing the common issue of catastrophic forgetting.

4. Comprehensive Improvement: LENS shows a comprehensive improvement across various multilingual tasks, including both comprehension and generation, which is a significant advancement over methods that focus on only one aspect of language performance.

5. Transparency and Interpretability: LENS provides transparent and interpretable solutions for multilingual enhancements, allowing for better understanding and control over how language models process and represent different languages.

### Weaknesses
1Typical Multilingual General or Unique Case Performance:
 The LENS approach, while effective in enhancing multilingual capabilities, may encounter challenges when dealing with languages that have unique grammatical structures or vocabularies significantly divergent from the central language. The method's reliance on internal representations might not fully capture the intricacies of such languages, potentially leading to suboptimal performance in tasks requiring deep linguistic understanding.
2 Alignment in Multilingual Tasks such as Machine Translation:
Although LENS improves multilingual performance by manipulating language representations, it might not fully address the complexities of tasks like machine translation, especially for low-resource languages. The scarcity of high-quality parallel corpora for these languages could hinder the model's ability to learn the fine-grained linguistic nuances necessary for accurate and fluent translations.

### Questions
1. Typical Cases Analysis:    How does the paper analyze typical cases for the language-agnostic and language-specific subspaces in Parts 4 and 5, and what implications do these analyses have for the performance of LENS in handling multilingual tasks?

2. LENS vs. xSFT-Full Performance:   In Figure 2, why does LENS not perform better than xSFT-Full for the Swahili language, and what factors might contribute to xSFT-Full's superior performance in this specific case?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a novel method, Lens, to enhance the multilingual capabilities of LLMs. Lens first explores the subspaces of Language-Specific and Language-Agnostic features, and introduces three training objectives—pull, push, and retain—to optimize the model's multilingual performance. Experiments across various understanding and generation tasks demonstrate that Lens effectively prevents catastrophic forgetting and significantly improves the performance of LLMs.

### Strengths
-	This paper is innovative, offering a fresh perspective on enhancing the multilingual capabilities of large models. It not only provides new ideas for future multilingual research but also helps the Chinese NLP community better understand large models.
-	The proposed method verifies its effectiveness on multiple datasets including NLU and NLG and avoids problems such as catastrophic forgetting.
-	This paper is well written, and the figures and tables are well drawn, making it easy to understand.

### Weaknesses
-	The experiments in the paper primarily compare Chinese and English, with additional languages including Japanese, which belongs to the same language family as Chinese, as well as low-resource languages like Bengali and Swahili, which are more distant from the representation space of English in LLMs. I am curious about the extent of improvement the method proposed in the paper can offer when the representation space of LLMs for languages within the same language family as English is closer to that of English, such as Es, Fr, and De.

-	This paper compares two SFT schemes with different data sizes, xSFT and xSFT-Full. But I still recommend that the authors compare SFT based on the LoRA version, as some experimental work [1] suggests that LoRA-based SFT can effectively prevent catastrophic forgetting, particularly when data quality is insufficient, such as when training data is sourced from automatic translation.
-	Some explanatory text needs to be added to explain the author's motivations and make it easier for readers to read. For examples, the function Span() in Equation (2)， the design of  Equation (3).
-	Typo: line 1009， “batchsize” -> “batch size”.

[1] MindMerger: Efficient Boosting LLM Reasoning in non-English Languages

### Questions
How different data sizes affect Lens, and how much improvement can be achieved using more training data.

### Soundness
3

### Presentation
3

### Contribution
3
