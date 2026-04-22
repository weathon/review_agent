# Beyond Text: LLM-Based Multimodal and Cross-Lingual Transfer Learning for Low-Resource Tigrigna Sentiment Analysis

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Sentiment analysis in low-resource languages remains underexplored, particularly for Tigrigna, where communication frequently combines text, emojis, and memes. We introduce TigXMM, a cross-lingual, multilingual, and multimodal framework for Tigrigna sentiment analysis, along with the first multimodal sentiment dataset for this language. The dataset, collected from social media, integrates text, emojis, and meme content to capture real-world communication patterns. We benchmark widely used multilingual models, including mBERT, AfriBERTa, XLM-RoBERTa, XLNet, BLOOMZ, and LLaMA, and highlight their limitations in processing multimodal signals. To address these challenges, we design an LLM-based cross-lingual transfer model with multimodal adapters for text, emoji, and meme fusion using hybrid attention and additive–hierarchical strategies. Experimental results demonstrate that our approach consistently improves sentiment classification performance: from 78.4\% accuracy on text-only inputs to 81.2\% with emojis, 86.3\% with memes, and 89.7\% when combining all modalities, achieving state-of-the-art performance for Tigrigna sentiment analysis. Beyond performance gains, this work contributes the first multimodal dataset and a reproducible framework, providing open resources to advance sentiment analysis for underrepresented African languages.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces TigXMM, an LLM-based framework for Tigrigna sentiment analysis that combines cross-lingual transfer with multimodal inputs (text, emojis, and memes). The model is built on a LLaMA backbone with LoRA adapters, plus an emoji-aware embedding layer and a meme module fusing OCR text, captions/hashtags, and visual features via multimodal adapters. This paper propose two robustness metrics—Emoji Sentiment Coverage (ESC) and Meme Sentiment Consistency (MSC)—to assess modality-specific reliability. Experimental results show that TigXMM outperforms some baselines.  As claimed in this paper, this paper propose the first multimodal sentiment dataset for Tigrigna sentiment analysis, providing crucial data for researching multimodal sentiment analysis in low-resource languages.

### Strengths
1. Low-resource African languages, especially Tigrigna, are indeed under-served by sentiment resources; focusing on multimodal social media (emojis/memes) addresses how users actually express sentiment. This paper propose the first multimodal sentiment dataset for Tigrigna sentiment analysis, providing crucial data for researching multimodal sentiment analysis in low-resource languages.

2. The framework proposed in this paper is simple, intuitive, and easy for readers to understand.

### Weaknesses
1. The framework proposed in this paper lacks technical innovation and resembles a standard fine-tuning pipeline for multimodal large language models.

2. As the core contribution of this paper, the proposed multimodal sentiment analysis dataset lacks essential statistical information, details on how it was acquired and processed, and crucial insights (e.g., why constructing this dataset is significant? Why previous work did not consider this approach?). Building this dataset resembles more of an engineering endeavor, while meaningful, it falls short of the technical innovation and in-depth analysis required for an academic paper.

### Questions
The authors need to elaborate in detail on the innovation and motivation behind the core contributions of this paper (the framework and dataset), which are currently lacking in the manuscript.

### Soundness
2

### Presentation
2

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
This paper proposes TigXMM, a cross-lingual and multimodal LLM framework aimed at Tigrigna sentiment analysis in low-resource settings. The model combines text, emojis, and memes using LoRA-based adapters and a hybrid attention mechanism. It also introduces two novel evaluation metrics  called Emoji Sentiment Coverage (ESC) and Meme Sentiment Consistency (MSC)  to quantify multimodal robustness. Experiments include text-only and emoji/meme-rich datasets, comparing TigXMM with mBERT, AfriBERTa, XLM-R, BLOOMZ, and a few vision-language baselines such as CLIP and LLaVA.

### Strengths
1 Relevant and underexplored topic. Addressing Tigrigna sentiment analysis with multimodal data fills a notable research gap in African-language NLP.

2 Introduction of ESC and MSC metrics. These metrics provide a potentially useful framework for evaluating emoji and meme understanding in low-resource contexts.

3 Reasonably clear structure and motivation. The paper is organized around identifiable stages and includes both model- and data-level contributions.

4 Effort to combine cross-lingual and multimodal transfer. The idea of leveraging Amharic and English as bridge languages is conceptually sound.

### Weaknesses
1 Limited novelty in modeling. The technical framework relies on standard, well-known techniques (LoRA, late fusion, multilingual transfer) without introducing new architectures or training objectives. Most of the improvements arise from fine-tuning and scaling rather than conceptual innovation.

2 Outdated references and baselines. The paper still frames LLaMA and LLaMA-2 as “recent,” which is inaccurate in late 2025. Similarly, the baselines omit strong multilingual and multimodal SOTA models such as Mistral-8×7B, Qwen2-VL, Gemini 1.5, or Yi-VL, making the comparisons less meaningful.

3 Weak data curation and transparency. The dataset construction process lacks examples, annotation details, and inter-annotator agreement scores. This undermines the reproducibility of the results and the credibility of the benchmark.

4 Unclear cross-lingual rationale. The use of Amharic as a source language is not sufficiently justified. The authors should explain the linguistic or structural relationship between Amharic and Tigrigna and show ablations confirming its benefit.

5 Insufficient scope for research questions. The second research question (RQ2) mixes cross-lingual and multimodal elements; it should instead focus exclusively on Tigrigna performance to match the paper’s stated aim.

6 Presentation and formatting issues. Result tables have inconsistent decimal formats, and one table is awkwardly placed at the end of the main paper. Figures and captions are minimal and do not meet ICLR presentation standards.

7 Lack evaluation in generalization. Despite claiming cross-lingual robustness, the experiments cover only English ↔ Amharic ↔ Tigrigna. Testing on additional African or Semitic languages would substantiate the generalization claim.

### Questions
What is the empirical justification for selecting Amharic as one of the bridge language?

How many samples are in the Tigrigna multimodal dataset, and how was quality ensured?

Were ESC and MSC correlated with human judgment scores to validate their reliability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses sentiment analysis for low-resource languages, focusing on Tigrigna (spoken in Eritrea and Ethiopia). It proposes TigXMM, a multimodal, cross-lingual framework leveraging Large Language Models (LLMs) and Parameter-Efficient Fine-Tuning (PEFT) to integrate text, emojis, and memes.

The work first proposes multimodal sentiment dataset for Tigrigna (text + emoji + meme) and proposes a new LLM-based multimodal architecture (TigXMM) with adapters for emoji and meme fusion.

However, the proposed method and framework make a relatively minor contribution and lack clear novelty compared to existing approaches.

### Strengths
1. The work first proposes multimodal sentiment dataset for Tigrigna (text + emoji + meme) and proposes a new LLM-based multimodal architecture (TigXMM) with adapters for emoji and meme fusion.

### Weaknesses
1. the proposed method and framework make a relatively minor contribution and lack clear novelty compared to existing approaches. The work primarily builds upon established techniques without introducing substantial methodological or theoretical advancements. As a result, its overall impact on advancing the field appears limited.

2. the manuscript format requires a major revision and improvement.

### Questions
1. the motivation of the proposed three research questions? and explain the relationship among the proposed research question?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The author constructed a cross-lingual sentiment symbol-aware emotion dataset and evaluated the performance of several models on this dataset.

### Strengths
The author is the first to propose a sentiment analysis dataset that includes emotional symbols and cross-lingual features.

### Weaknesses
1. The author proposed a dataset but did not introduce any new methods to evaluate it.

2. The experiments lack performance comparisons of state-of-the-art large language models on the dataset, such as Qwen, DeepSeek, and GPT-5.

3. Figure 1 is unclear and unreadable.

4. There is a lack of detailed analysis and description of the dataset.

### Questions
Refer to weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
