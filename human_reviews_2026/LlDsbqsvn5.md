# RedNote-Vibe: A Dataset for Capturing Temporal Dynamics of AI-Generated Text in Social Media

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
The proliferation of Large Language Models (LLMs) has led to widespread AI-Generated Text (AIGT) on social media platforms, creating new challenges for content authenticity. The identification of AIGT on social media platforms presents unique challenges due to engagement-driven content and temporal dynamics. To bridge this gap, we introduce a novel RedNote-Vibe dataset, collected from RedNote (Xiaohongshu), one of the most influential Chinese social media platforms. This dataset contains user posts and their parallel AIGT variants generated using diverse LLMs, spanning from before ChatGPT's release to the present. We further propose a detection method based on psycholinguistic principles, namely PsychoLinguistic AIGT Detection Framework (PLAD), which achieves SOTA performance compared to recent model-based methods and provides superior interpretability. Our analysis also reveals temporal trends of AI content adoption and engagement pattern differences between human and AI-generated content.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper looks at how to spot AI-generated text in dynamic social media environments. It introduces the PsychoLinguistic AIGT Detection Framework (PLAD) — an interpretable method that uses psycholinguistic features. Not only does it deliver strong performance using model-based techniques, but it also gives clear insights into the style fingerprints of different LLMs and shows how AI-generated text connects to user engagement. By digging deep into the dataset, the study uncovers trends in AI adoption over time and highlights differences in engagement patterns between human-written and AI-generated content.

### Strengths
1. This work focuses on the social media domain, capturing a large-scale real-world dataset from RedNote and constructing a parallel version with AI-generated content. The dataset is intentionally designed to support research on AI-generated discourse (AIGD).
    
2. From an interpretable perspective, the study develops a framework based on psycholinguistic features, which is used for training and predicting with AI content detectors.

### Weaknesses
1. The paper lacks sufficient details about the dataset construction process.
    

   - Line 078: Please clarify the source and location of RedNote’s official user behavior report.
    
   - Line 084: What are the filtering criteria? Is the metadata from raw crawled data, or is it already processed after filtering?
    
   - Please provide a clear description of the language distribution of the dataset.
    
   - Regarding the parallel AI-generated posts: they appear to be rewrites rather than entirely original creations, as the original content is supplied and then rewritten based on prompts. If so, please explain the rationale and methodology behind this process.
    
   - Line 117: What are the “6 providers”? Are they specific companies? Please elaborate.
    

2. As the core contribution of the paper, the description of the psycholinguistic features is overly superficial.

   - In Section 3, the “four dimensions of human language expression” are only defined in terms of what aspect they measure, without listing the 31 linguistic features in detail or explaining how each feature is extracted.
    
   - The appendix only shows six representative features for illustration, with comparative results, making it hard to understand the implementation and reproduction process.
    
   - Please describe the feature extraction methodology clearly, especially for different languages — are the extraction methods identical across languages? The paper does not address this point.
    

3. Concerns About Experimental Reliability

   - Some average sequence lengths exceed 512 tokens, yet several PLM-based models used in the paper have a maximum input length of 512. Was the input simply truncated? If so, please clarify. Considering alternatives like Longformer could be beneficial [1][2].
    
   - Previous works [1] found that using a small set of features as model input did not surpass PLM-based methods in accuracy. This paper uses 31 features, which sounds promising, but the importance and justification of this choice are not clearly discussed.
    
   - The paper does not provide enough details on dataset splits (training/validation/test).
    
   - Prior studies have reported PLM-based methods reaching >98% accuracy [1][2]; however, the paper’s reported accuracy is below 90%, raising questions about the credibility of the results.
    
   - For Subsection 4.2 regarding zero-shot experiments, prior works [1][2][3] suggest that PLM-based models are not as weak in cross-source detection as the reported results here. This discrepancy needs explanation.
    

4. Although the **Ethics Statement** briefly mentions masking personally identifiable information, the dataset construction section does not explain how privacy protection was implemented. A more comprehensive description is necessary.

5. While the paper claims to focus on AI detection in dynamic data, the dataset itself is static. The time-related analysis is only performed through the “Exploration Set.” This is similar in spirit to [3], except the domain of data differs.

6. Some models used (e.g., Claude, Gemini) are not commonly used in the Chinese internet context. This could introduce bias in the training data. Using such models to train and then evaluate predictions on real-world data may weaken the validity of conclusions.

[1] Beyond Binary: Towards Fine-Grained LLM-Generated Text Detection via Role Recognition and Involvement Measurement

[2] MAGE: Machine-generated Text Detection in the Wild

[3] Large language models penetration in scholarly writing and peer review

### Questions
See weakness.

015：Xiaohongshu? maybe rednote?

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
This paper presents a real AIGT dataset from social media spanning 5 years and develops an AIGT detector based on psycholinguistic features.

### Strengths
1. A 5-year AIGT dataset with a large amount of real data is released, and some analysis of the dataset is provided.
2. PLAD demonstrates superior detection performance.

### Weaknesses
1. I believe temporal analysis of the data is a very important contribution. Unfortunately, in Section 5.1, the authors only analyze frequency. Several important analyses are missing: How has AIGT data changed over the 5 years? How has the detection difficulty evolved?
2. The authors lack some basic statistical information about the dataset. For example, what is the yearly frequency distribution of the data?
3. An important issue concerns data bias. The authors only state that “We adopt a web crawler to collect 120,000 notes from January 2020 to July 2025,” but provide no measures to ensure the data is unbiased.
4. The authors construct data through customized LLM simulations. However, a more important issue is how to obtain genuinely published AIGT data from social media.
5. The data construction seems inconsistent with their claimed 5-year span. In line 118, they actually use only data before November 2022 (the pre-LLM period).

### Questions
I suggest the authors to include some specific data examples in the paper.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduce RedNote-Vibe, a 5-year social-media dataset from Xiaohongshu (RedNote) with post text plus rich engagement metadata (likes, comments, collections) spanning pre-LLM to July 2025, expressly to study temporal dynamics of AI-generated text (AIGT) in the wild. They also propose PLAD -- a psycholinguistics-based, interpretable detector for AIGT.

### Strengths
1. Presentation: It was very easy to read and follow. I appreciate the authors' effort for their clear presentation.
2. Longitudinal, engagement-aware dataset: I like the way the authors combined pre-LLM and post-LLM contents, and framed its usability for AI-content detection and analysis.
3. Interpretable detector (PLAD): The paper proposes a psycholinguistics-based, feature-driven approach aimed at interpretability rather than a purely black-box classifier.

### Weaknesses
1. Some subsections need more explanation/details. For example, the 'Data Collection' subsection should be clearer, the 'Feature Extraction and Classification' subsection should add more details, the 'Experiment Setup' subsection should mention how the other baselines were implemented/tweaked for specific identification, etc.

2. For the zero-shot experiment, I would recommend adding more detectors as baselines; only BERT-base is not enough.

3. For the 'Analysis' section, authors should explicitly mention how they are identifying the AI-text and AI-augmented authors. It's not clear from the paper in its current state.

4. There are prior works that covered AI-text in social media, such as [1], Reddit posts in [2], etc. The author needs to clarify how their dataset differs from these existing datasets and why PLAD adds more value to the community. 

4. There are some related works that might be worth mentioning, such as [3,4] for AI-augmented users, etc.

References:

[1] Are We in the AI-Generated Text World Already? Quantifying and Monitoring AIGT on Social Media

[2] RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors

[3] Almost AI, Almost Human: The challenge of detecting AI-polished writing

[4] LLM-as-a-coauthor: The challenges of detecting LLM-human mixcase

### Questions
1. I did not understand some of the parts in the 'Data Collection' subsection, e.g., *"We first extract the example tags provided in the report for each category, then expand them to approximately 50 representative tags per topic through manual curation."*, etc. Can you elaborate it? 
2. How were the statistics-based and model-based methods adopted for the model- and provider- identification task?
3. Is there any specific reason to include only the BERT-base model for zero-shot comparison?
4. For the 'Analysis' section, how are you deciding/filtering the AI-texts?
5. For the subsection 'Author-Level AI Usage and Engagement', how did you identify the AI-augmented authors?

### Soundness
4

### Presentation
4

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
RedNote-Vibe - the first longitudinal AIGT dataset.

### Strengths
The real novelty is Chinese Xiaohongshu with engagement and a 5-year span.

### Weaknesses
Please check the questions.

### Questions
Can you compare with the closest time-aware datasets? Like SAID and MultiSocial, which can do longitudinal analysis too. Only designed to longitudinal analysis is not a good novelty.

Psycholinguistic feature scores partly produced by a proxy LLM will decrease the interpretability.

Please add a human audit with inter-rater agreement, and quantify the bot risk.

Please release the exact prompts and compare with a control where the prompts vary widely.

Please add strong Chinese baselines trained under identical budgets.

Unlabeled the “exploration set”, then using PLAD to analyze. It is circular.

### Soundness
2

### Presentation
2

### Contribution
2
