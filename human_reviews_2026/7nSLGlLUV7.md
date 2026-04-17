# Element2Vec: Build Chemical Element Representation from Text for Property Prediction

- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
Accurate property data for chemical elements is crucial for materials design and
manufacturing, but many of them are difficult to measure directly due to equip-
ment constraint. While traditional methods use the property of other elements, or
related properties for prediction via numerical analyses, they often fail to model
complex relationships. After all, not all characteristics can be represented as
scalars. Recent efforts has been made to explore advanced AI tools such as lan-
guage model for property estimation, but still suffer from hallucinations and a
lack of interpretability. In this paper, we investigate Element2Vec to effectively
represent chemical elements from natural languages to support research in the
natural sciences. Given the text parsed from Wikipedia pages, we use language
models to generate both a single general-purpose embedding (Global) and a set of
attribute-highlighted vectors (Local). Despite the complicated relationship across
elements, the computational challenges also exists becuase of 1) the discrepancy
in text distribution between common descriptions and specialized scientific texts,
and 2) the extremely limited data, i.e., with only 118 known elements, data for
specific properties is often highly sparse and incomplete. Thus, we also design
test-time training method based on self-attention to mitigate the prediction error
caused by Vanilla regression clearly. We hope this work could pave the way for
advancing AI-driven discovery in materials science.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a framework named Element2Vec, which aims to use LLMs to extract chemical embeddings from the Wikipedia text for material property prediction. The embeddings consist of global embeddings that summarize entire text and local embeddings specific to attribute texts (e.g., mechanical, optical, thermal). To handle the high sparsity in chemical property data, the paper also designs a test-time training strategy based on self-attention, transforming property prediction into an imputation problem. Experimental results show that local embeddings achieve better clustering by element family in t-SNE visualization for classification. In property regression task, the global embeddings combined with the 'test-time training' strategy perform best under high data missing ratios.

### Strengths
1.	This paper presents an interesting perspective, using large language models to extract embeddings from text for predicting chemical properties, which can capture richer contextual knowledge missing from traditional databases.

2.	This paper proposes a Test-Time Training strategy, which treats all elements as a whole for 'imputation' prediction. According to the experiments in the paper, this method outperforms traditional inductive training methods at all data missing ratios.

### Weaknesses
1.	The authors employ LLM to generate chemical representations from natural language text, however, there are several concerns. The paper fails to sufficiently address the issue of data leakage; since Wikipedia may already contain explicit numerical values or strongly correlated descriptions of the properties being predicted, it is unclear whether the model is learning genuine chemical relationships or merely retrieving and regurgitating memorized information. The authors need to provide a rigorous analysis to rule out this possibility.

2.	Furthermore, the justification for choosing the specific Gemini embedding model over other general-purpose large models (like GPT) or domain-specific models (like MatSciBERT or SMILES-BERT) is insufficient. The paper needs to clearly articulate the specific advantages of the chosen method relative to these alternatives.

3.	The paper's introduction to the global and local embedding generation methods lacks the necessary detail. The process of how text is segmented, summarized, and fed into the model to generate local embeddings is not clearly explained. Concurrently, the paper fails to provide a convincing scientific justification for the necessity of local embeddings. Although the authors hypothesize that local embeddings can highlight specific attributes, the empirical evidence provided appears contradictory, for instance, the results in Figure 7(b) show that the global embedding generally has a lower RMSE in property prediction than the local embeddings, which weakens the motivation for adopting the more complex local embedding method. This work resembles a simple text-processing workflow applied to the chemical domain rather than a substantial new contribution to chemistry or materials science.

4.	The paper lacks modeling of the relationships between local representations. The authors generate independent vectors for attributes (e.g. atomic, chemical, and thermal), but in chemistry, these properties are deeply correlated and interconnected. The current method appears to treat them as mutually independent.

5.	The paper does not provide sufficient detail regarding the dataset used for embedding generation and property prediction. Although Wikipedia is mentioned as the source, the authors need to clearly specify the data collection and cleaning procedures, as well as comprehensive statistics for the final corpus (e.g., average document length, vocabulary size, etc.).

6.	The paper's experimental evaluation is weak due to a lack of adequate baseline comparisons. The authors primarily compare different variations of their own method. A benchmark against non-text-based methods, such as GNNs, is required.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
To address the critical challenge of property prediction for chemical elements, this paper employs a Large Language Model (LLM) as an encoding module to extract knowledge from text in the form of embeddings. The proposed Element2Vec framework leverages an LLM to construct vector representations of chemical elements derived from unstructured text sources, specifically Wikipedia pages. The key contribution lies in generating two types of embeddings that capture information at different levels. The global embeddings utilize the entire document or page of an element as input to capture holistic information, while the local embeddings are learned from text grouped under specific attributes. Overall, this idea of using LLM to encode chemical text has been explored in several related works which limit the contribution of this paper.

### Strengths
1. The proposed Element2Vec provides an effective way for translating human-experienced, qualitative knowledge like Wikipedia into numerical representations that are machine-readable.

2. The proposed method utilizes both Global and Local (attribute-specific). The local embeddings include the information of specific characteristics (e.g., optical vs. thermal properties), which is vital for materials design and scientific analysis.  The global embeddings capture more holistic knowledge.

3. The training-free framework is another strong aspect. By relying on pre-trained LLMs as feature extractors and content classifiers, the embedding generation pipeline does not require additional training. This makes it straightforward to apply to new elements or attributes without extensive retraining, thereby facilitating faster research and experimentation.

### Weaknesses
1. The main limitation of this work is that Element2Vec relies entirely on Wikipedia pages. Consequently, the quality, depth, and neutrality of the generated embeddings depend heavily on the completeness and accuracy of this data source. If a particular element’s Wikipedia page is sparse, outdated, or biased, the resulting embedding may be inaccurate. A potential improvement would be to incorporate additional sources of domain-specific knowledge, such as scientific publications or chemical databases, to enhance representation quality.

2. Wikipedia is a general data source, and most modern LLMs have already been trained on it during pretraining. Therefore, a more appropriate baseline would be to compare this approach against recent, powerful models applied directly to property prediction tasks. Additionally, several models have been fine-tuned for chemistry-related tasks, and including such comparisons would strengthen the paper. Overall, while the application of LLMs to chemical property prediction has been investigated in previous studies, this work would benefit from a clearer demonstration of its unique technical contributions and distinctions from existing approaches.

### Questions
Please check the weaknesses.

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
2

### Summary
This paper is concerned about learning representations for chemical elements (e.g., atoms) from text data such as Wikipedia webpages. The proposed representation consists of `Element2Vec-Global` and `Element2Vec-Locals`; the former is a representation of the whole relevant text data, while the latter is a collection of representations, each of which is obtained by prompting an LLM to focus on specific (pre-defined) attribute (details are in Figure 2).
Since the number of all elements is limited, the authors also propose a test-time training approach for prediction, instead of the standard supervised learning (Section 4.3 ).

The benefit of the proposed representations has been validated from several perspectives.
Section 4.1 visually examines the validity of the proposed local representations, as compared against several different representations.
Section 4.2 quantitatively examines the benefit of the local representation.
Section 4.4 examines the effectiveness of the proposed embedding and test-time training method.

### Strengths
- Learning a representation from text data is an interesting way of utilizing LLMs.
- It is insightful that the authors have shown several ways to define local embeddings and have explained why the proposed embedding is selected among others.

### Weaknesses
One of the major concerns is the empirical validation. As far as I am aware of, a quantitative validation is done on a task to predict the van der Waals radius of an element, without any existing methods to be compared. Since I am not an expert in materials science, I failed to understand the importance of the prediction task, and thus, I thought the experiment was rather a toy task rather than a real-world problem. In addition, without performance comparison with other existing methods, it is difficult to understand whether or not the proposed representation is useful or not in real applications.

### Questions
- I would like to ask the authors to clarify how $p_k(x)$ is computed in Section 4.2.
- In Section 4,4, the authors state that "$R_{\mathrm{vdW}}$ is difficult to determine and not uniquely defined", and I'm curious about how the authors determine the ground truth labels.
- I would like the authors to clarify the relevance of the van der Waals radius prediction task to real-world problems.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Element2Vec, a framework for representing chemical elements as vectors by leveraging textual descriptions. The authors use Wikipedia pages for each of the 118 chemical elements as the data source. An LLM-based pipeline is employed to produce two types of embeddings for each element: a single Global embedding capturing the overall content of the element’s page, and multiple Local embeddings that are attribute-specific. To obtain the Local embeddings, the approach first uses an LLM to classify each sentence of an element’s page into one of eight predefined attribute categories (e.g. Atomic, Chemical, Thermal, etc.), and then aggregates the text of each category (with a brief summary of the whole page) to generate an embedding for that attribute . The goal is that these Global and Local embeddings encapsulate meaningful chemical knowledge extracted from text, which can then be used for downstream tasks such as classifying an element’s periodic-table family and predicting various material properties

### Strengths
The paper's key strengths lie in its innovative cross-domain approach that bridges NLP and materials science by leveraging large language models to extract knowledge from scientific text, moving beyond traditional hand-designed features. The introduction of attribute-aware embeddings significantly enhances interpretability by producing multiple vectors for each element corresponding to human-understandable categories (mechanical, thermal, chemical properties), which demonstrably organize the latent space in ways that respect known scientific classifications like periodic families. The proposed test-time training method effectively addresses sparse data challenges, showing substantial performance improvements over conventional baselines especially when 50-80% of data is missing, by cleverly allowing unlabeled instances to influence the model during inference. The work is supported by thorough empirical evaluation including comprehensive ablation studies (examining summary length effects, attribute contributions, and embedding dimension overlap) that reveal meaningful insights such as the model's ability to rediscover periodic families from text alone and capture real scientific relationships like the shared features between melting and boiling points. This combination of strong performance, interpretability through attribute-specific analysis, and alignment with domain knowledge makes the approach both scientifically valuable and trustworthy for deployment in materials research.

### Weaknesses
The approach suffers from fundamental limitations in its dependence on Wikipedia as a single, uneven source of truth, with sparse coverage forcing the exclusion of 22 elements from certain analyses. More critically, the heavy reliance on LLM-based sentence classification and summarization lacks any validation or accuracy assessment. The paper provides no evidence that the LLM correctly categorizes sentences into attribute buckets or avoids hallucination during summarization, despite these steps being central to the embedding process. This unverified pipeline could propagate errors throughout the embeddings, yet the authors offer no robustness analysis or manual verification of these critical automated decisions.
The experimental evaluation lacks essential baseline comparisons that would contextualize the method's performance. No comparisons are provided against simple alternatives like one-hot encodings, manual feature sets, or naive whole-text embeddings without attribute segmentation. Furthermore, the results reveal a surprising weakness: the sophisticated Local attribute embeddings actually underperform the simpler Global single-vector approach in property prediction tasks, with the authors themselves acknowledging that "global embedding generally exhibits the lowest error." This undermines a core contribution of the paper and suggests the attribute segmentation may lose important holistic information rather than enhance it.
The work's practical applicability is limited by its narrow scope (118 elements only, with no demonstration on compounds or real materials) and reproducibility concerns stemming from dependence on proprietary models like Gemini. The authors provide no plan to release computed embeddings or discuss computational costs, making it unclear how others could replicate or extend this work without access to the same commercial AI services.

### Questions
1. How do you handle elements with incomplete Wikipedia attribute coverage—do they receive fewer Local embeddings, default vectors, or some imputation method?
2. Did you compare Element2Vec against simpler baselines like linear regression on atomic features (e.g., atomic number, group, period) to quantify the advantage of text-derived embeddings?
3. How reliable was the LLM's sentence classification into attribute categories, and did misclassifications (especially for ambiguous sentences spanning multiple categories) impact embedding quality?
4. Why did concatenated Local embeddings underperform Global embeddings for regression, and did you explore learned fusion methods like attention mechanisms to weight attribute relevance for specific properties?
5. How sensitive is test-time training to hyperparameters and the ratio of known-to-unknown elements, and at what point does including too many unknowns cause overfitting or instability?
6. Is the Gemini embedding model publicly accessible, could alternative models like SBERT produce similar results, and will you release the Element2Vec embeddings for the 118 elements?

### Soundness
1

### Presentation
2

### Contribution
3
