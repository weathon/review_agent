# Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training

- Decision: Accept (Oral)
- Scores: 6, 6, 8, 8

## Abstract
Large Language Models (LLMs) are pre-trained on large data from different sources and domains. These datasets often contain trillions of tokens, including large portions of copyrighted or proprietary content, which raises questions about the legal use of such models. This underscores the need for truly open pre-training data that complies with data security regulations. In this paper, we introduce Common Corpus, the largest open dataset for LLM pre-training. The data assembled in Common Corpus are either uncopyrighted or under permissive licenses and amount to about two trillion tokens. The dataset contains a wide variety of languages, ranging from the high-resource European languages to some low-resource languages rarely represented in pre-training datasets. In addition, it includes a large amount of code data. The diversity of data sources in terms of covered domains and time periods opens up the paths for both research and entrepreneurial needs in diverse areas of knowledge. In this paper, we present the detailed provenance of data assembling and the details of dataset filtering and curation. We train two small language models on Common Corpus and find that they perform comparably to other models of their size, indicating that our dataset is suitable for multilingual pretraining. Common Corpus represents a key contribution to the ecosystem for open science research on Large Language Models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Common Corpus, a 2T-token multilingual dataset built entirely from open or permissively licensed sources (public domain, CC, open code). It aims to offer a legally clean, transparent alternative to web-scraped corpora. The authors describe the collection process, cleaning (PII removal, OCR correction, toxicity filtering), and train two small models showing comparable multilingual performance to baselines.

### Strengths
- Timely and relevant: strong contribution to open and compliant LLM research.
- Impressive scale (2T tokens) and careful documentation of provenance and licenses.
- Multilingual coverage beyond English, rare for open corpora.
- Clear adherence to emerging best-practice frameworks (e.g., dataset documentation, PII filtering).
- Demonstrates feasibility through working models and released tools.

### Weaknesses
- Empirical section is limited, only small models and a few benchmarks.
- No clear comparison to similarly “open” corpora (e.g., Dolma, KL3M) in terms of quality or coverage.
- Curation process, though detailed, lacks quantitative measures of data quality after filtering.
- Language balance is heavily skewed to English (~50%).
- Evaluation of ethical filtering (toxicity, PII accuracy) could be better substantiated.

### Questions
- How scalable is the current pipeline to truly support trillion-token multilingual expansion?
- How do the authors ensure consistent quality across OCRed historical data?
- Could releasing the filtering tools lead to reproducibility or bias-transfer risks?
- Do they plan to release validation splits or subsets for standardized benchmarking?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the Common Corpus, an open dataset for pre-training LLMs. It is constituted of texts that are either uncopyrighted or under permissible licenses, for a total of two trillion tokens across a variety of languages and tasks such as coding, as well as a diversity in terms of regions and time. 

The authors go into the data collection and curation process in great detail, and also train two LLMs of small size on the dataset that show that they perform on a similar level to other models of their size across a variety of datasets.

### Strengths
Developing an explicitly curated dataset based on data licensing is important and a crucial contribution to the AI community.

There is an emphasis on diversity in terms of languages and regions.

The code used for creating the dataset is available, allowing others to reproduce it.

The resulting dataset can be filtered based on different criteria, including license and language, which makes it useful for developers and researchers working on specific languages or historical periods.

Personally Identifiable Information is removed with the Presidio tool, which means that there is almost no risk of data leakage from the trained models.

### Weaknesses
- "Ethical data" is a very relative/hard-to-define concept -- maybe "consensual data" or "legal data" would be better alternatives?

- Figure 1 is hard to read because languages such as English, Spanish or French are spoken in multiple places, so simply putting a dot on Madrid or Paris isn't representative of where the language is from 

- "synthetically rewrite the document without the harmful language" - how is this done and verified? Doesn't introducing synthetically-generated text dilute the corpus?

### Questions
- How are the six collections defined, what are the criteria? 

- How are you sure (are you sure) that none of the data is LLM-generated?

- Are the audio transcripts AI- or human- generated?

- How has the set of Wikidata been adapted in natural language? The example provided isn't very clear

- "Segmentext should work correctly on diverse document formats" - did you do testing? In general, providing more information about the tools that you developed and how they work would help understand their limitations and applicabliity.

### Soundness
3

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
3

### Summary
This paper accompanies the release of the largest open dataset for LLM pretraining: the Common Corpus. The authors fully emphasize the “open” aspect of their data, providing full transparency around the provenenance, source, and licensing of their data. They also engage in extensive data cleaning and quality improvement steps for different segments of their data. Unlike its recent open data precusors, Common Corpus also includes substantial multilingual data.

### Strengths
The writing of this paper was very clear, and this work was done with much care, detail, and rigor. For example, the authors “involve local communities” in gathering data from diverse sources and did not machine-translate the multilingual component of their dataset. This resource is an invaluable contribution and far exceeds its predecessors in size and composition. Aside from data, this work also contributes several data cleaning tools.

### Weaknesses
I understand that the main focus of this paper is on the data and not on model training, so it is okay that the model training results aren’t groundbreaking. One minor detail is that your choice of benchmarking results to show in Section 5 is a little strange. You focus on a few multilingual benchmarks, but compare against OLMo 1B, which may have been intended to be monolingually English. 

It would have been nice to see a clear tabulation (like, a table) of how this dataset overlaps, extends, or differs from existing open datasets.

### Questions
Data may be copied and then mislicensed. Have you checked for overlap between your dataset and data that has less permissive licenses? This might be really tricky to do, so I am just curious. 

Is there a way for someone to remove data about themselves or produced by themselves from this dataset? Right to be forgotten, etc.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces Common Corpus, a ~2T-token multilingual dataset  for LLM pre-training that contains only data that are in the public domain or released under permissive open licenses.  The authors document data sources, licensing, cleaning tools (Segmentext, OCRonos, Celadon, PII filtering), and present benchmark results for small Llama-style models trained solely on this corpus.

Overall, this is a valuable contribution to open LLM research. The motivation is strong, and the authors make extensive efforts to demonstrate the usefulness of the dataset for pre-training purposes. 

However, several aspects including the very Euro-centric nature of the multilingual coverage and lack of quantitative information about ambiguous licenses need further clarification.

### Strengths
1. Timely contribution to open-data infrastructure for LLMs, especially given that many existing datasets had to be taken down due to license issues.

2. Transparent and detailed documentation of dataset provenance, content, and filtering pipeline

3. Releases and discusses useful tools for OCR correction and sentence segmentation

4. Demonstrates that open data sourced from only permissive licenses can still yield competitive model performance.

### Weaknesses
1. The claim of “multilingual diversity” is overstated as the dataset is heavily Euro-centric. The top ten languages are all European, and there is little inclusion of African or Asian languages.

2. It is not clear how trustworthy the licensing information is. The authors do not report how many documents had ambiguous, conflicting, or missing license information.

3. For certain types of data, licenses can apply inconsistently across sub-components. It is not clear how these were disambiguated.

4. The data pre-processing pipeline, including Segmentext and OCRonos are underdescribed. It is not clear if any de-duplication was done or required.

5. The evaluation lacks detail on per-language performance and broader benchmark coverage.

### Questions
1. Can the authors provide a full language inventory and token count distribution, including low-resource languages?

2. How are ambiguous or missing licenses handled, and what proportion of the dataset do they represent?

3. What steps were taken to ensure deduplication across overlapping sources?

### Soundness
3

### Presentation
2

### Contribution
3
