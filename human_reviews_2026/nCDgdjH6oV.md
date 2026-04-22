# Neural Catalog: Scaling Species Recognition with Catalog of Life–Augmented Generation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Open-vocabulary species recognition is a major challenge in computer vision, particularly in ornithology, where new taxa are continually discovered. While benchmarks like CUB-200-2011 and Birdsnap have advanced fine-grained recognition under closed vocabularies, they fall short of real-world conditions. We show that current systems suffer a performance drop of over 30\% in realistic open-vocabulary settings with thousands of candidate species, largely due to an increased number of visually similar and semantically ambiguous distractors. To address this, we propose Visual Re-ranking Retrieval-Augmented Generation (VR-RAG), a novel framework that links structured encyclopedic knowledge with recognition. We distill Wikipedia articles for 11,202 bird species into concise, discriminative summaries and retrieve candidates from these summaries. Unlike prior text-only approaches, VR-RAG incorporates visual information during retrieval, ensuring final predictions are both textually relevant and visually consistent with the query image. Extensive experiments across five bird classification benchmarks and two additional domains show that VR-RAG improves the average performance of the state-of-the-art Qwen2.5-VL model by 18.0%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces NeuralCatalog, a retrieval-augmented generation (RAG) framework for open-vocabulary bird species recognition. It leverages Wikipedia as an external knowledge source, retrieving concise species descriptions to complement visual features from images. By integrating retrieval, re-ranking, and multimodal reasoning within a large language model, NeuralCatalog enables accurate recognition of both seen and unseen species, achieving state-of-the-art performance in bird species recognition.

### Strengths
1. The problem of open-vocabulary species recognition is practical and of scientific interest. 
2. The improvements obtained are strong and non-trivial in nature.

### Weaknesses
I believe this paper has several weaknesses that are not adequately addressed in the current submission. My main concerns center on positioning and evaluation.

W1. While the paper’s motivation around biological species recognition is strong, the inclusion of a new Pokémon dataset feels abrupt and insufficiently justified. As presented, it appears more like an attempt to expand the technical scope rather than a well-motivated extension of the main problem.

W2. The evaluation against encoder-only models raises concerns about fairness. Since the proposed method ensembles multiple encoder-based VLMs (e.g., CLIP), comparing against these same models individually in Tables 2 and 3 seems misaligned. Moreover, the paper omits several relevant baselines from recent zero-shot prompting and retrieval works [1,2,3], which weakens the empirical positioning of the method.

W3. The retrieval pool includes over 11,000 species—substantially larger than prior benchmarks—yet it remains unclear whether species appearing in test sets (e.g., CUB, iNat) are excluded from the retrieval corpus during evaluation. If not, this introduces a fairness issue, as the model could indirectly access prior knowledge of test species during inference.

References
[1] Prompting Scientific Names for Zero-Shot Species Recognition - EMNLP 2023

[2] Visual Classification via Description from Large Language Models - ICLR 2023

[3] Learning Customized Visual Models with Retrieval-Augmented Knowledge - CVPR 2023

### Questions
Please refer to the weaknesses outlined above.

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
This paper presents VR-RAG (Visual Re-ranking Retrieval-Augmented Generation), a system for large-scale fine-grained species identification, focusing mainly on birds.
The authors argue that current closed-world benchmarks (e.g., CUB-200, NABirds) fail to reflect the real complexity of biodiversity recognition, where species are numerous, visually similar, and new species are continuously added.
VR-RAG combines three pretrained components:

Multimodal retrieval using CLIP / OpenCLIP / SigLIP to find candidate species descriptions from a curated Wikipedia-derived knowledge base (11 k+ species).

Visual re-ranking using DINOv2 to reorder retrieved candidates by pure image–image similarity, claimed to suppress environmental bias.

Reasoning with a pretrained MLLM (e.g., Qwen2.5-VL or InternVL) that receives the image and top-k candidate summaries and generates the final predicted species name.
The authors report consistent improvements in retrieval quality (mRR@k) and final classification accuracy across five bird datasets, FishNet, and Pokémon.

### Strengths
- Ambitious and environmentally meaningful problem framing: scaling biodiversity recognition to real-world, open-vocabulary conditions.

- Clear modular system design that combines retrieval, re-ranking, and reasoning.

- Demonstrated cross-domain generality (birds, fish, Pokémon).

- The curated textual summaries for >11 k species constitute a useful resource for the community.

### Weaknesses
- Scientific novelty – The system reuses existing pretrained encoders and an MLLM without new learning mechanisms. The main novelty is the composition, not a new algorithmic contribution.

- Prompt sensitivity – The MLLM reasoning stage relies on a single fixed prompt; no ablations on phrasing, candidate ordering, or number of candidates are shown. The robustness of the reported gains is therefore uncertain.

- Intra-species visual variability – Many bird species exhibit strong sexual dimorphism (male vs. female), seasonal plumage differences, or distinct juvenile appearances. The current setup appears to use one reference image and one textual summary per species. This could bias retrieval toward a single morph and hurt generalization. Please discuss or evaluate using multiple reference embeddings per species.

- Single-object assumption – Datasets used contain one main bird per frame; it is unclear how the framework handles multi-object or occluded scenes.

- Visual re-ranking explanation – The paper attributes improvement to DINOv2’s self-distillation producing object-centric features that suppress background noise. While plausible, this claim is unverified; no attention visualizations or controlled background tests are provided.

- Scope of contribution – Since no training is performed, the paper would benefit from clearly positioning itself as a system-level integration study rather than implying a new learning framework.

### Questions
- Include prompt ablations for the MLLM reasoning step.

- Discuss or extend to multi-reference per species to address male/female, juvenile, or seasonal variation.

- Provide visual evidence or controlled tests showing that DINOv2’s self-distilled embeddings indeed suppress environmental noise.

- Clarify scientific scope—system integration vs. methodological innovation.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
- Proposes VR-RAG, a framework combining retrieval and reasoning for open-vocabulary species recognition.
- Connects encyclopedic text (Wikipedia) with visual cues for grounding unseen species.
- Builds a large benchmark of 11 k+ bird species with GPT-4o-refined summaries and curated anchor images.
- Designs a two-stage pipeline: multimodal retrieval, visual re-ranking, and MLLM-based reasoning. The method demonstrates strong cross-domain generalization across birds, fish, and Pokémon datasets

### Strengths
- The motivation of the paper is clear and significance.
- The authors construct a large-scale benchmark of more than 10k bird species and  produce concise, discriminative summaries aligned with visual evidence.
- The proposed VR-RAG framework demonstrated substantial accuracy gains over all baseline models across five bird datasets and two cross-domain settings.

### Weaknesses
- The idea of combines retrieval + re-ranking + MLLM reasoning is an intuitive and common way to bridge textual and visual knowledge bases. The contribution to novelty is limited.
- Wikipedia tends to emphasize well-studied taxa, resulting in regional and taxonomic biases that leave numerous cryptic or underrepresented species undocumented. How are species without Wikipedia pages handled? Is coverage gap quantified?
- Some design choice are not deeply analyzed, such as the number of retrieved candidates, or alternative retrieval architectures.

### Questions
- The paper mentioned the data curation using GPT-4o to filter. Have authors check whether this step will bring in bias and noise?
- The study focuses mainly on classification and retrieval accuracy, without exploring more demanding downstream tasks such as visual question answering.
- How much gain comes from re-ranking vs. GPT-4o summary quality alone?

### Soundness
3

### Presentation
2

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
The paper presents a RAG-based framework for species classification that uses wikipedia summaries of species as the knowledge database. Firstly, an LLM is used to summarize the Wikipedia articles and then are divided into chunks. Multimodal embeddings of anchor images and the text chunks are created which are used to retrieve appropriate text chunks given a query image, which is embedded with a ensemble of VLMs. Finally, an MLLM is used to rerank the retrieved articles for final prediction. The proposed RAG-based system achieves state-of-the-art performance in various species classification benchmarks.

### Strengths
1. The paper proposes a RAG-based species classification pipeline using wikipedia article summaries and multimodal ensemble-based retrieval method.

### Weaknesses
1. Limited technical novelty as it is a simple RAG system that is used everywhere now but applied to species image classification. How is the framework different than just an application (RAG + x) where x is some problem? What are some exclusive properties of the proposed system that is tailored to species classification only? Can it be applied to other problems?
2. What is the computational overhead of the proposed RAG system as compared to the other VLMs. What is the cost-benefit ratio of using this system? What is the inference time given a query? 
3. Is it possible to integrate additional modalities as done by recent works such as TaxaBind? Would incorporating additional modalities improve retrieval and the overall understanding capabilities of the model.
4. How robust is the proposed RAG framework to the database? Can the authors provide experiments for ablating the size of the database and robustness to noisy articles in the database? 
5. Can you provide some uncertainty estimates to when reranking is not useful? What are the concrete failure cases when reranking is incorrect and retrieval is accurate?
6. In general, since this is in applications track, the paper needs to have more systems level analysis and experiments. I dont see particular systems level innovations.
7. All the experiments are provided on commonly used bird classification benchmarks. Any use cases of the RAG system beyond identification of common species? Like can you provide some experiments for novel species identification?

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
