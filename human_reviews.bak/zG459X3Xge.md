# VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Retrieval-augmented generation (RAG) is an effective technique that enables large language models (LLMs) to utilize external knowledge sources for generation.  However, current RAG systems are solely based on text, rendering it impossible to utilize vision information like layout and images that play crucial roles in real-world multi-modality documents.  In this paper, we introduce VisRAG, which tackles this issue by establishing a vision-language model (VLM)-based RAG pipeline.  In this pipeline, instead of first parsing the document to obtain text, the document is directly embedded using a VLM as an image and then retrieved to enhance the generation of a VLM. Compared to traditional text-based RAG, VisRAG maximizes the retention and utilization of the data information in the original documents, eliminating the information loss introduced during the parsing process. We collect both open-source and synthetic data to train the retriever in VisRAG and explore a variety of generation methods. Experiments demonstrate that VisRAG outperforms traditional RAG in both the retrieval and generation stages, achieving a 20–40% end-to-end performance gain over traditional text-based RAG pipeline. Further analysis reveals that VisRAG is efficient in utilizing training data and demonstrates strong generalization capability, positioning it as a promising solution for RAG on multi-modality documents. Our code and data are available at https://github.com/openbmb/visrag.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes VisRAG, a model that leverages vision-language models (VLMs) for retrieval-augmented generation in multi-modal documents. The framework comprises two components: VisRAG-Ret for retrieval and VisRAG-Gen for generation, utilizing a data construction method that combines visual question answering (VQA) datasets with synthetic data. Experiments demonstrate that VisRAG outperforms traditional RAG in both retrieval and generation tasks, exhibiting improved training data efficiency and generalization capabilities. It achieves a significant relative increase in end-to-end accuracy, highlighting its potential to replace text-based RAG for multi-modal documents.

### Strengths
1.  The paper provides a clear and detailed description of the VisRAG framework., which makes it easy to understand.
2. The experimental results on document-related VQA tasks are promising, suggesting that the proposed method could be a practical solution.

### Weaknesses
1. The rationale behind the proposed VisRAG-Ret is unclear. If no OCR information is included, how can the target image be retrieved, especially for answers that require precise text content? From my perspective, a pooling-based representation feature does not offer sufficient information.

2. The experimental setup is not sufficiently convincing, indicating that the VQA task alone cannot fully demonstrate the method's effectiveness in the retrieval task. In other words, employing such a complex pipeline to address the VQA task is akin to using a sledgehammer to crack a nut.

3. The methods discussed are insufficient. To the best of my knowledge, there are several document-specific MLLMs, such as mlug-docowl, ureader, and text monkey, among others. While these works may not be directly related to RAG, they should at least be mentioned and analysized.

4. In my view, the comparison is unfair, as both Minicpm and Siglip can not only perform VQA tasks but also handle other tasks related to document image understanding and content perception.

### Questions
See "weaknesses"

### Soundness
2

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
4

### Summary
The paper focus on the research of vision-language model (VLM)-based RAG
Pipeline and propose a novel method named VisRAG. It  broadens the external data processed by RAG from text to image. To achieve the goal, the authors introduce two vision-optimized modules: VisRAG-Ret and VisRAG-Gen. This VisRAG-Ret directly utilizes the images without extracted textual content and establish  the map between the query and the documents in the latent space. The VisRAG-Gen leverage the help of the existing LLMs and VLMs for generating the answers. The image processing technology is integrated to realize the efficient generation of multiple inputs. Moreover, a new dataset is built combining a vision question answering (VQA) dataset and synthetic data.

### Strengths
In this paper, the authors propose an image-based RAG method named VisRAG, and construct a new dataset to verify the relevant effects. This paper does a lot of comparative experiments to verify its innovation. A large number of state-of-the-art methods have been combined and proven to be effective.

### Weaknesses
The method in this paper is mainly based on the retrieval of image data, which is basically consistent with the text retrieval framework, and lacks innovation. In addition, similar to the comparative model Colpali's innovation, the VisRAG is fine-tuned with a new dataset and new VLM which is not sufficient to express its innovation in multimodality area.

### Questions
1. In the comparison process, Colpali and Vis RAG-Ret should be fine-tuned based on the same data, rather than using the pre-trained model to compare directly. Please add more comparing results.
2.The article needs to provide more comparison results with the current best models, such as GPT o1, Qwen and other leading models that can directly input image for VQA task.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces VisRAG to address the limitations of text-only RAG systems by incorporating vision-language models (VLMs). VisRAG enables the processing of documents as images. By training the VisRAG retriever with open-source and synthetic data and exploring various generation methods, experimental results show VisRAG's superiority in both retrieval and generation tasks. The authors also plan to make their code and data publicly available.

### Strengths
1. Improve text-only RAG by establishing a vision-language model (VLM)-based RAG pipeline.
2. The experiment is thorough under the designed scenario.
3. The authors have promised to make the data and code open source.

### Weaknesses
1. Lack of novelty: For retrieval, it uses the dual-encoder-style and just changes them to VLM embeddings. The position-weighted mean pooling is also off-the-shelf. For generation, they only try a few straightforward strategies.
2. Lack of design for document understanding: There is a lack of design considerations specifically for document understanding. The proposed method is not tailored for document understanding scenarios. Alternatively, why not validate the method on non-document (common image) datasets?
3. Confusion regarding the application: Why does a person always need the assistance of document images when asking general questions? For example, in Line 884, Why would a person directly ask a question like 'Where were the two magnesium containing papers made at?' without reference to any document? If he already has this document as a reference, why does he still need to do a retrieval process at first?
4. Why only experiment with text-only or image-only features, not image+text(+layout)?

### Questions
See Weaknesses.

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
4

### Summary
This paper proposes VisRAG, a multimodal retrieval-augmented generation pipeline. Different from traditional RAG methods, which rely on text parsing to retrieve information from visual content, this paper utilizes a VLM-based retriever and generator to navigate information relevant to the query and generate responses. The VisRAG design consists of two main stages: (1) Retrieval: Given a query, the VLM retrieves a set of relevant images from the dataset leveraging the cosine similarity between the query embedding and image embeddings; (2) Generation: the VLM uses a combination of retrieved images and the query to produce responses. VisRAG is compared with a series of baselines and show advantages in both retrieval and generation capabilities.

### Strengths
1. [Originality] The proposed method is promising. With the VLMs showing increasing capabilities in understanding images, encoding images with VLMs is a natural improvement from traditional text-based retrieval methods.
2. [Quality] The paper presents relatively comprehensive experiments, demonstrating advantages in both retrieval and generation capabilities of the proposed methods.
3. [Significance] The proposed method clearly outperforms various baselines.
4. [Clarity] The paper is generally well-written and easy to follow.

### Weaknesses
Major concerns:
1. In the paper, when retrieving relevant images, the authors adopt position-weighted mean pooling over the last-layer VLM hidden states. This choice seems heuristic-based. Although the paper points out that VLMs are with causal attention, potentially emphasizing weights on later tokens, after multiple VLM layers, we could expect information to propagate across most tokens. In fact, previous works often use the leading token or simply learn a linear layer from all tokens. How would the model perform with these designs? Would the proposed position-weighted mean pooling produce significantly better results?
2. When using multiple images to generate responses, one method proposed in the paper is image concatenation, which is performed horizontally. This could change the resolution of the final image. Would this affect VLM performance, particularly when handling a larger number of retrieved images (e.g., more than five)?
3. The paper does not report retrieval efficiency compared to text-based methods. Would using the VLM significantly increase retrieval time? How would the number of retrieved images affect retrieval time?

Minor comments:
1. When measuring the performance of generation, how is accuracy measured? To me, this is particularly unclear for synthetic data, where ground truth answers and queries are generated by GPT-4. Is an additional LLM used to evaluate the textual responses?

### Questions
1. Significance of position-weighted mean pooling: How would the model perform with these designs? Would the proposed position-weighted mean pooling produce significantly better results?
2. Effects of concatenated multiple images: Would this affect VLM performance, particularly when handling a larger number of retrieved images (e.g., more than five)?
3. Efficiency of VLM-based retrieval: Would using the VLM significantly increase retrieval time? How would the number of retrieved images affect retrieval time?
4. Clarity on accuracy measurement: How is accuracy measured? Is an additional LLM used to evaluate the textual responses?

### Soundness
3

### Presentation
3

### Contribution
3
