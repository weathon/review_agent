# VidHal: Benchmarking Hallucinations in Vision LLMs

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Vision Large Language Models (VLLMs) are widely acknowledged to be prone to hallucinations. Existing research addressing this problem has primarily been confined to image inputs, with sparse exploration of their video-based counterparts. Furthermore, current evaluation methods fail to capture nuanced errors in generated responses, which are often exacerbated by the rich spatiotemporal dynamics of videos. To address these two limitations, we introduce VidHal, a benchmark specially designed to evaluate video-based hallucinations in VLLMs. VidHal is constructed by bootstrapping video instances across a wide range of common temporal aspects. A defining feature of our benchmark lies in the careful creation of captions which represent varying levels of hallucination associated with each video. To enable fine-grained evaluation, we propose a novel caption ordering task requiring VLLMs to rank captions by hallucinatory extent. We conduct extensive experiments on VidHal and comprehensively evaluated a broad selection of models, including both open-source and proprietary ones. Our results uncover significant limitations in existing VLLMs with respect to video-based hallucination generation. Through our benchmark, we aim to inspire further research on i) holistic understanding of VLLM capabilities, particularly regarding hallucination, and ii) advancing VLLMs to alleviate this problem.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces VIDHAL, a new benchmark to evaluate temporal hallucinations in VLLMs. VIDHAL pairs videos covering five temporal aspects with multiple captions exhibiting varying levels of hallucination. It proposes two evaluation tasks: standard MCQA and a novel, fine-grained Caption Ordering task (scored using NDCG) where models must rank captions by accuracy. Benchmarking 23 VLLMs reveals they perform poorly on nuanced temporal aspects like Direction and Event Order, often struggling to differentiate between hallucination levels and relying heavily on static image priors.

### Strengths
1. The "caption ordering task" is a significant methodological strength. It moves beyond simple binary "is this a hallucination?" evaluations (like POPE or MMHalBench) and forces the model to demonstrate a nuanced understanding of degrees of factual inaccuracy.
2. The paper is easy to follow.
3. The paper is exceptionally thorough, testing 23 recent VLLMs from 13 different families.
4. The results demonstrate a key weakness in current VLLMs: they rely heavily on static image priors. Models perform well on "Object" and "Action" but fail on "Direction" and "Order".

### Weaknesses
1. The "hallucinations" are synthetic artifacts of GPT-4o's "imagination," which may not reflect the types of hallucinations that VLLMs naturally produce.
2. The five chosen aspects are not exhaustive. They primarily focus on descriptive perception. More complex forms of temporal reasoning, such as causality or counterfactuals , are not covered.
3. Models from the same family (like GPT-4.1) or models trained to emulate GPT-4's style may score artificially high because they align better with the data generator. While the authors use other LLMs for filtering, the generation source is still a single model.

### Questions
1. The benchmark's core data consists of synthetic hallucinations from GPT-4o. These are not organic VLLM errors but artifacts of a single generator. The study is therefore evaluating performance on an artificial task that may not reflect real-world model failures.

2. The benchmark's definition of "temporal reasoning" is limited to low-level descriptive perception (e.g., direction, order). It fails to evaluate more complex and critical cognitive abilities like causality or counterfactuals, offering an incomplete picture of model intelligence.

3. The data generator (GPT-4o) and a top-performing model (GPT-4.1) belong to the same family. How can the model rankings be considered fair when the evaluation is inherently biased?

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
5

### Summary
The paper identifies 5 categories of temporal hallucination (action, attribute, direction, object, event order), and constructs a video temporal benchmark dataset. Models are tested through caption-ordering and multi-choice tasks, scored with NDCG for further fine-grained evaluation. Results across 23 VLLMs reveal that they still rely heavily on image prior therefore scaling law does not always hold.

### Strengths
- This paper has effective presentation, figures and pipeline illustrations help understanding the overall idea and writing is very easy to follow.
- The paper introduces a new evaluation task of caption ordering, adapted NDCG gives partial credits even though the models fail to retrieve the optimal answer, and the dataset is scalable by increasing $M$ and video sources.
- Impressively extensive evaluation (23 models from 13 model families) is reported, providing actionable diagnostics.

### Weaknesses
- There is only one type of Clause Semantic hallucination, while 4 types are associated with Lexical Semantic hallucination. 
- Only one target (event order) is identified for Clause Semantic hallucination. The concern on this unbalanced assessment remains.
- Some of the findings are already revealed in previous works. For example, a preceding paper [1] suggested that models perform better on object aspects than the action category, and scaling law does not always hold [2].
- The overall pipeline heavily relies on proprietary models, as mentioned in limitation. 100 samples were verified by humans and 87% agreement is promising, the paper does not quantify how remaining synthetic noise in the dataset might affect model rankings.
- The benchmark dataset is sourced from publicly available video datasets, the resulting video distributions may be unbalanced and could advantage models trained on similar sources. This may introduce potential bias.

I will increase the score once current concerns are addressed.

### Questions
(Major)
- The motivation of this paper is to suggest a video-based hallucination benchmark that captures nuanced, video-specific hallucinations. Is this “nuance” controlled by the hallucinatory caption generation in Section 3.3?
- Can attribute, object, and direction hallucination categories be identified as “temporal” hallucinations? For example, the first example in Figure 24-Attribute asks the total count of sit-ups of a person with a white vest performed. I believe this should be classified as an action category, and the traffic light example should be classified as an event order category. Also, one can accurately determine the direction of the object only by observing the corresponding frame unless anomalies exist (dog walking backwards in Figure 24-Direction).
- How can we control the degree of hallucination as described in line 232-233? Is there a quantitative estimate to sort hallucination levels? On the left side of Figure 25, the hallucination level increases as the number of stationary metal objects increases. Can we assume that two stationary objects or is it from caption ordering in Section 4.1? If so, I believe that mentioning the caption ordering before the line 232-233 can enhance the readability.
- Are VLLMs prone to non-transitive and cyclic ranking? I believe there should be a reference or small experimental results to show the order of the given choices can highly impact the final results.

(Minor)
- Notations mentioned in Section 4.2 are already used before declaration (last paragraph in 3.3). Shouldn’t it be declared before Section 3.3?

References

[1] Wang et al. VideoHallucer: Evaluating Intrinsic and Extrinsic Hallucinations in Large Video-Language Models. arxiv:2406.16338

[2] Zhu et al. PromptRobust: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts. LAMPS 2024

### Soundness
3

### Presentation
3

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
The paper introduces VIDHAL, a benchmark designed to evaluate temporal hallucinations in Video-based Large Language Models (VLLMs), addressing a gap in prior work that has predominantly focused on image inputs. VIDHAL contains 1,000 video instances covering five temporal aspects—Action, Attribute, Direction, Object, and Event Order—with each video paired with multiple captions exhibiting varying degrees of hallucination. A caption ordering task is proposed, alongside metrics such as MCQA accuracy and a ranking-based NDCG, to capture fine-grained differences in hallucination severity. The authors evaluate 23 VLLMs, both open-source and proprietary.

### Strengths
1. The VIDHAL benchmark considers five temporally relevant aspects—Action, Attribute, Direction, Object, and Event Order—in its video selection and annotation process. This multidimensional design makes the evaluation notably more comprehensive .
2. The experiments encompass a wide range of mainstream VLLM models, including both open-source and proprietary systems, covering diverse architectures and parameter scales. This breadth of evaluation enhances the completeness and credibility of the reported results.
3. The authors further provide analyses on hallucination misalignment, image prior reliance, and input order sensitivity. These additional investigations add depth to the study’s findings and yield more nuanced insights into model behavior.

### Weaknesses
1. The hallucinated captions in the dataset are primarily generated automatically using GPT-4o, and in the sampled subset, only 87% of the data were consistent with human verification. Beyond this, the generated captions are inevitably subject to the inherent biases and stylistic tendencies of the generative model itself. This may affect the objectivity and generalizability of the benchmark evaluation.
2. While the authors have identified the tendency of models to rely on image priors in video tasks, the current analysis remains largely at the level of phenomenon description. I would encourage the authors to further investigate the underlying causes of this bias and explore systematic mitigation strategies that could be applied in practice.

### Questions
As described in Weakness.
And how large would the performance variance be if the same model were evaluated multiple times? In Table 2, are the reported results based on a single evaluation run? Clarifying this would help assess the stability and reliability of the presented comparisons.

### Soundness
3

### Presentation
2

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
This paper introduces VIDHAL, a new benchmark designed to evaluate temporal hallucinations in VLM, addressing the gap left by existing image-focused hallucination research. VIDHAL is constructed using video instances covering five temporal aspects  sourced from existing datasets. The authors benchmarked a wide range of VLM on VIDHAL, finding significant limitations in their ability to handle video-based temporal details and differentiate hallucination levels, especially concerning direction and event order.

### Strengths
1.The paper identifies and addresses a pertinent gap in current VLM evaluation by focusing specifically on temporal hallucinations in videos, which are underexplored compared to image-based hallucinations.

2.VIDHAL introduces a novel evaluation paradigm through the caption ordering task and associated ranking metrics, aiming to provide a more fine-grained assessment of a VLM's ability to discern subtle factual inconsistencies in video descriptions.

### Weaknesses
1.The benchmark construction heavily relies on GPT-4o for generating both anchor and hallucinatory captions across different predefined hallucination levels. This raises concerns about the benchmark's validity and potential biases. It's questionable whether GPT-4o can consistently generate meaningfully distinct and realistic levels of temporal hallucination across diverse videos.

2.More details regarding the human validation process are needed to assess the reliability of the annotations. Information on how annotators were recruited, the total number involved, and the specific procedure for determining the caption order is crucial but seems missing.

3.The reported human agreement rate of only 87% raises concerns, especially as the NDCG scores of top proprietary models like Gemini-2.5 approach this level. This proximity suggests that the benchmark's annotation quality might already be limiting the reliability and accuracy of the evaluation for high-performing models.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
