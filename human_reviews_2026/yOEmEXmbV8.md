# Seeing Through Words: Controlling Visual Retrieval Quality with Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Text-to-image retrieval is a fundamental task in vision-language learning, yet in real-world scenarios it is often challenged by short and underspecified user queries. Such queries are typically only one or two words long, rendering them semantically ambiguous, prone to collisions across diverse visual interpretations, and lacking explicit control over the quality of retrieved images. To address these issues, we propose a new paradigm of quality-controllable retrieval, which enriches short queries with contextual details while incorporating explicit notions of image quality. Our key idea is to leverage a generative language model as a query completion function, extending underspecified queries into descriptive forms that capture fine-grained visual attributes such as pose, scene, and aesthetics. We introduce a general framework that conditions query completion on discretized quality levels, derived from relevance and aesthetic scoring models, so that query enrichment is not only semantically meaningful but also quality-aware. The resulting system provides three key advantages: 1) flexibility, it is compatible with any pretrained vision-language model (VLMs) without modification; 2) transparency, enriched queries are explicitly interpretable by users; and 3) controllability, enabling retrieval results to be steered toward user-preferred quality levels. Extensive experiments demonstrate that our proposed approach significantly improves retrieval results and provides effective quality control, bridging the gap between the expressive capacity of modern VLMs and the underspecified nature of short user queries. Our code is available at https://github.com/Jianglin954/QCQC.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the problem of Quality-Controllable Retrieval, which is an extension of the text to image retrieval problem where in the input query can have additional constraints such as high relevancy, low aesthetics, etc and the goal is to retrieve images from a gallery which satisfy the original query and the additional constraints as well. To enable existing VLMs such as CLIP to do this, they use a query augmentation pipeline where they train an LLM to generate modified queries based on certain additional constraints (relevancy, aesthetics in their case). The trained LLM is then used during inference time to change the query based on the additional constraints. Experiments on MS-COCO and a new Flickr2.4M dataset show the method can effectively steer retrieval results. Conditioning on "High" quality levels yields images with significantly better average aesthetic and relevance scores compared to "Low" conditions or baseline methods.

### Strengths
1. The paper tackles a practical and significant problem. Users frequently use short, ambiguous queries and most existing T2IR systems just return the top-k images by cosine similarity rather than controlling for additional quality metrics. The formulation of "quality-controllable retrieval" is a strong and useful contribution. 
2. Simple and elegant solution: their approach allows to convert any existing VLM into a quality controllable one by using a fine-tuned LLM which can augment the input text query to constrain the search. 
3. Good visualizations / interpretable results: The paper contains several qualitative examples which show that by fine-tuning the LLM to augment queries, they are able to get precise augmented queries where the retrievals which match the users preferences.

### Weaknesses
1. The same CLIP model and EV_A model are used to generate the training data quality labels and for measuring the average metrics during evaluation as well, this might lead to the fine-tuned LLM learning patterns which work only for a particular gallery/dataset and might overfit the query augmentations to that dataset. It would be great if the authors can either do a small human study or use other stronger SoTA multi-modal models as relative judges of the metrics. 
2. Missing per-condition re-rank baselines in table 6: the paper only compares with a retrieve -> sort by aesthetics post filtering pipeline in Table 6. This effectively is only covering the H/H sub-segment. Since the main idea behind the paper is QC^2, the table should also include the (L/M/H X L/M/H) segments like in Table 3/4 and also include stronger retrieval baselines like the following:
a) a weighted joint scorer which ranks the images based on a weighted average between rel and aes distance with the queries gt_rel and gt_aes
b) constrained retrieval: since the rel and aes values are already binned, they can first be used to filter out all images which do not belong in the gt bin and then performing search over the reduced sub-space. 
These results should help shed more light into how simple retrieval methods perform against more complex query augmentation methods 
3. Dataset dependency: As already mentioned by the authors, for each new dataset, they need to fine-tune the query augmentation LLM again so that the LLM can learn the specifics of rel and aes scores and the corresponding captions. This questions the scalability of the method if it needs to be deployed in a real scenario where web-scale retrieval is performed which has a lot of noise/distribution shift. 

4.Mode-collapse/diversity risk: QC^2 relies on an LLM to generate quality-conditioned completions, yet the paper reports no diversity or entropy analysis of those texts. Without such checks, the LLM may collapse to a few stylistic templates which led to low loss with a fixed rel and aes value, which can lead to poor diversity in the images which the users would see. It would be great if the authors can provide some metrics on the diversity of queries generated per (rel,aes) tuple. One such metric can be follows: 
for a fixed instruction (query: q, rel: r, aes: a), sample n different completions from the llm with different temperature parameters, and then check how diverse these completions are (dispersion based on text-clip distance can be a good metric).

### Questions
refer to weaknesses section for questions

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
4

### Summary
This paper proposes Quality-Controllable Retrieval (QCR), which allows users to control retrieval results through quality dimensions such as relevance and aesthetics. The proposed approach uses an LLM as a query completion function conditioned on discretized quality levels. Experiments on multiple datasets demonstrate that quality-aware query completions can enhance retrieval performance, although only two quality metrics are explored.

### Strengths
- The paper has a good motivation, addressing the challenge of underspecified queries that often lead to ambiguous retrieval results. Quality-controllable retrieval is both practically useful and of research interest.
- The paper is well-structured, and the methodology is described in sufficient detail.

### Weaknesses
- While the concept of quality-controllable retrieval is attractive, the current work only explores two quality dimensions (relevance and aesthetics). This limited scope is insufficient to demonstrate the true practical value of controllable retrieval in applications.
- Despite the claim that LLM-based completions avoid irrelevant or hallucinated content, there is no explicit mechanism or evaluation to manage or detect query artifacts that could mislead retrieval or introduce out-of-distribution details.
- The use of large language models for query rewriting or expansion is already a common and mainstream approach. The main innovation of this paper lies in introducing two quality indicators and training data to guide LLM-based query completions. My primary concern is that this level of novelty does not meet the expected standards of ICLR.

### Questions
- The training data consists of concise sentence summarizing the main content of the image with annotated aesthetic and relevance levels. Through training, the LLM learns the relation between text descriptions and the two quality metrics. However, I am concerned **whether such concise sentences are sufficient for the LLM to truly capture the connection between these quality metrics and the actual visual content of the image.**

- Although the numerical results suggest separability, they do not always match intuitive perception. For example, in Table 1 (teddy bear), the left image (aesthetic 4.788, relevance 0.359) seems more aesthetically pleasing and more representative of a teddy bear than the right image (aesthetic 5.818, relevance 0.437). This raises concerns about the reliability of the scoring process.

- The authors claim that the method can be extended to many different metrics. However, the paper only provides experiments with relevance and aesthetics.  **I wonder how the LLM behaves when more metrics (three or more) are involved. Would the model prioritize certain metrics while ignoring others? Will the LLM's completion ability degrade under multiple metrics**？
- Using LLM for query completion is a clever choice, but while the authors claim to avoid hallucinations, no explicit mechanism is provided. Have the authors  observed redundant, voague, or inaccurate queries, and were any filtering strategies considered?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new method for text-to-image retrieval called quality-conditioned query completion . The method addresses the problem of short and underspecified user queries. It uses a large language model to expand short queries with descriptive and quality-aware details. The approach does not modify pretrained vision–language models and provides interpretable, controllable retrieval behavior. Experiments on MS-COCO and Flickr2.4M show that QC² consistently improves both semantic relevance and visual quality, outperforming existing baselines.

### Strengths
1. The paper proposes a new task: quality-controllable text-to-image retrieval, which considers both semantic relevance and aesthetic quality during retrieval. This setting aligns well with real-world search scenarios.
2. The method uses a large language model to generate query expansions with quality cues, requiring no modification to existing vision–language models and remaining relatively simple to implement.

### Weaknesses
1. The paper mainly focuses on short queries but does not evaluate on datasets with long or descriptive queries. When the textual input already contains sufficient information, the effect of query completion may diminish or even introduce redundancy or semantic drift. Will this strategy still work for underspecified long user query?
2. The paper's quantitative evaluation relies on a limited set of test queries, namely 80 concrete object nouns. Consequently, the method's performance in handling more abstract, complex, or queries involving emotional atmospheres (such as "a sense of calm"，"a vintage vibe") remains entirely unvalidated.
3. This method requires fine-tuning a large language model for each specific image retrieval gallery. This means it is not a "plug-and-play" solution. On the contrary, if the gallery is replaced or updated, the dataset must be rebuilt and significant resources must be invested in retraining, which severely limits its scalability.
4. How can this be extended to image-to-text retrieval?

### Questions
1. How does the method perform on the datasets from the LoTLIP[1]?
2. How does the scalability of the QCR method which requires an LLM to be fine-tuned for each gallery, comparing to VISA [2], a "plug-and-play" method that performs re-ranking at test-time without training?

[1] LoTLIP: Improving Language-Image Pre-training for Long Text Understanding

[2] Visual Abstraction: A Plug-and-Play Approach for Text-Visual Retrieval

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores using an LLM to rewrite short text queries into more detailed, quality-aware descriptions, aiming to control the aesthetic and relevance level of retrieved images without modifying the vision-language model. The method learns to expand queries based on quality labels and shows that higher-quality prompts lead to higher-quality retrieval results.

### Strengths
1. The motivation is clear. The paper introduces aesthetic cues to explicitly control and improve retrieval quality. 
2. The method is plug-and-play and easy to apply in existing systems, requiring no modification to the visual model.

### Weaknesses
1. Experiments mainly use single-word, single-object queries（e.g., “a dog”）. Real retrieval queries are usually longer, involve multiple objects and relations. Current setup looks more like keyword/entity retrieval.
2. The paper doesn't report preprocessing cost or inference latency, so it's hard to judge the efficiency of method.
3. The approach assumes the database has many visually similar images with different aesthetic qualities (like Flickr30k/COCO). In many datasets this won’t hold (VisualNews,fashion200k).

### Questions
1. In real retrieval scenarios, queries often contain multiple objects, relations, and context. How would the method scale to complex, natural multi-entity queries?
2. How would the method perform when the image databse diversity is limited, or when aesthetic cues are less meaningful (news/fashion domain)?
3. The theory suggests increased rank leads to better discrimination. But multiple  multiple LLM rewrites could also increase rank. On theoretical, what is the concrete advantage of your controlled rewriting over simple rewriting ?

### Soundness
3

### Presentation
3

### Contribution
2
