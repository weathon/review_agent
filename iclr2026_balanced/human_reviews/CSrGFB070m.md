## Human Reviewer 1

### Summary
The authors introduce an efficient tokenizer for Indic LLMs built on top of SuperBPE, achieving state-of-the-art parity scores across Indic languages, code, and English. Their proposed IndicSuperTokenizer combines the strengths of both subword and multi-word tokenization, along with script-specific pretokenization strategies that yield more linguistically aligned segmentations. When applied to train and evaluate models across English and multiple Indic languages, the tokenizer largely preserves model performance while boosting inference throughput by 44%

### Strengths
- Adapting the original SuperBPE algorithm to low-resource, non-Latin script languages like Indic is commendable and makes a meaningful contribution to the multilingual NLP community.
- The paper includes very fine-grained analysis and ablations, covering key factors in training superword tokenizers for Indic languages, such as vocabulary size, merging strategies, and pre-tokenization techniques.
- Using script-agnostic pretokenization in the first stage of tokenizer training improved token-to-word ratios by 38–40% on Indic scripts. 
- Their tokenization approach preserves model performance while boosting inference throughput.

### Weaknesses
- Section 2.4 contains comparisons with different baseline toenizers. But is fertility fairly comparable here, given the potential difference in data distributions each tokenizer baseline was trained on? At the very least, a short description of how each tokenizer was trained, if possible, would make comparisons more meaningful. 
- It would also really help to see a direct comparison between IndicSupertokenizer and a regular BPE trained on the same data with script-agnostic pretokenization. Right now, it’s hard to tell whether the improvements come from the tokenizer itself or just differences in the training data.
- For downstream evaluation, it’s not clear what results are zero-shot or from finetuning. I know that some of the Indic task-specific datasets have train/test splits. 
- More on the downstream evaluation, is there a reason why IndicBPEtokenizer (just using the first stage) wasn’t considered for these extrinsic evaluations? I believe it would help to isolate the contribution of the super words learned in the later stage. 
- In section 4.3, can you provide more details on the 200 examples used for analyzing inference efficiency? Are these parallel sentences? Also are these the same models evaluated in section 4.2? 
- Compared to the original SuperBPE paper, the performance improvements here appear smaller. Any thoughts on why that might be would be valuable.

### Questions
- Do you have any insights from your analysis of vocabulary allocation strategies? Prior work has shown that training language-family or script-specific tokenizers and then merging their vocabularies can sometimes benefit low-resource languages. Though it increases vocabulary size, it can lead to small gains in downstream performance. You focus on fertility here, but it would be interesting to see whether similar trends hold for your setup.
- Why does English dominate in your dataset? Was that to preserve English performance in LLaMA or to encourage cross-lingual transfer to Indic languages, or was it just a result of data availability?
- Finally, pointing readers to the appendix for key metrics (as in line 42) isn’t ideal. It would be better to report an aggregated number, like average fertility, directly in the main text for readability.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper considers the problem of tokenization for LLM training, focusing on the highly multilingual context of languages spoken in India. The authors use recent insights on super-word tokenization to train IndicSuperTokenizer, a SuperBPE (Liu et al., 2025) tokenizer for 22 Indian languages, English, and code. In their experiments, the authors show that IndicSuperTokenizer results in substantially better intrinsic metrics compared to baselines (e.g., higher efficiency), while exhibiting comparable extrinsic performance on downstream benchmarks. The paper also contains several post-hoc analyses, such as an experiment on glitch tokens and a comparison of SuperBPE with BoundlessBPE (Schmidt et al., 2025).

### Strengths
The experimental setup and analyses are methodologically sound. The intrinsic performance improvements are substantial. It is great to see that the authors actually pretrained an LLM using their tokenizer (even though the performance improvements are only modest at best). The writing of the paper is also clear and easy to follow.

### Weaknesses
The main weakness of the paper in my opinion is that it is highly incremental, especially for a venue like ICLR that focuses on technical advances. The authors use an existing method (specifically, the SuperBPE tokenizer) and apply it in a new setting. While some design decisions are original (e.g., the vocabulary allocation strategy), they are pretty minor, and it seems that the main improvements are due to the use of SuperBPE.

I think this paper could be published as is at a specialized venue (e.g., a workshop). To be of interest for a broader audience, the authors would need to show better how they are making technical contributions that go beyond the application of an existing method in a new setting.

### Questions
What are the novel technical contributions that your work is making?

### Soundness
4

### Presentation
4

### Contribution
2

### Rating
2

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper addresses the significant inefficiency of standard tokenizers for multilingual LLMs, particularly for morphologically rich Indic languages which suffer from high token-to-word ratios. High fertility increases training costs, inference latency, and context size usage. The authors propose IST, an optimized tokenizer for 22 Indic languages, English, and code.

contributions:
1. IST achieves a new state-of-the-art fertility score, improving 39.5% over LLaMA-4 and 18% over Sutra on average.
2. Pretraining a 1B model from scratch with IST results in a 44% improvement compared to the LLaMA-4 tokenizer in inference throughput while maintaining comparable performance.
3. The paper justifies its design choices through extensive ablations.
4. The authors show that IST can replace the tokenizer of an existing pre-trained model, achieving the same efficiency gains while preserving the original model's performance.

### Strengths
1. This paper addresses an important practical issue: `the inefficiency of standard tokenizers for morphologically rich non-English languages`.
2. The IndicSuperTokenizer demonstrates genuinely impressive and state-of-the-art results on intrinsic metrics like fertility score and NSL.
3. The 44% improvement in inference throughput is a substantial practical gain.
4. The authors performed an extensive set of ablations to justify their design choices.

### Weaknesses
1. This paper's primary weakness is its lack of algorithmic novelty. The authors state the method is `inspired from SuperBPE` and follows its `curriculum principles`. The core contribution is not a new tokenization algorithm, but rather the careful application and tuning of an existing one to a new domain, combined with other existing components. This feels more like a strong engineering effort than fundamental research suitable for ICLR.
2. This work focused on specific Indic languages. While this is valuable work for that community, its direct contribution to the general machine learning and representation learning audience at ICLR is limited. The findings are an application.
3. Table 8 is not referenced in main text.

### Questions
1. Given that the core two-stage training algorithm is from SuperBPE and the pre-tokenization regex is from LLaMA-4, what do the authors consider to be the primary novel algorithmic contribution of this work?
2. Could the authors provide the quantitative data of latency for the abandoned morphology-aware approach?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
2

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper proposes a new tokenizer for Indic languages that results in lower token fertility score compared to baselines.

### Strengths
1. The paper is comprehensive and relatively well-written.
2. The authors compared the latency and throughput for a model trained with their tokenizer and a base tokenizer.
3. The paper also looks at glitch tokens and at the possibility of fine-tuning pre-trained models with their new tokenizer.

### Weaknesses
1. Not particularly novel methodology. The pre-tokenization step is simply using the regex from Llama-4 and allowing cross-word tokenization within the sentence. Then, standard BPE is applied.
2. It is not surprising that a tokenizer designed for a specific language family would perform better than such designed for other languages or more languages. While it is interesting that it outperforms two other Indic language tokenizers, there is no explanation or discussion in the paper as to why that may be the case.
3. Bytes-per-token seems to be a strange metric as it depends on the specific encoding scheme and the length of words in characters. Unicode encodes different scripts with different numbers of bytes (one to three) and languages vary in how long (in number of characters) their words are (Chinese tends to use three times less characters than English for the same content). Therefore, this metric seems to be confounded with other aspects of language, making it not particularly suitable.
4. Looking at Table 8 and contrary to the claims in the paper, there is little if any difference in performance between the model trained with the Llama-4 and the IST tokenizers.
5. Overall reads more like a technical report than a scientific paper: it provides details on how the authors built and designed a specific instance of a tokenizer but not much scientific or transferable insight.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
1

### Rating
2

### Confidence
5