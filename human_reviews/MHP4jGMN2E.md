# The Impact of Element Ordering on LM Agent Performance

- Decision: Reject
- Scores: 5, 5, 5, 6

## Abstract
There has been a surge of interest in language model agents that can navigate virtual environments such as the web or desktop. To navigate such environments, agents benefit from information on the various elements (e.g., buttons, text, or images) present. However, it remains unclear which element attributes have the greatest impact on agent performance, especially in environments that only provide a graphical representation (i.e., pixels). Here we find that the ordering in which elements are presented to the language model is surprisingly impactful—randomizing element ordering in webpages compromises average agent performance to a degree comparable to removing all visible text from webpages. While web agents benefit from the semantic hierarchical ordering of elements available via the browser, agents that parse elements directly from pixels do not have access to any such ordering. Here we endeavor to derive effective orderings and investigate the impact of various element ordering methods in web and desktop environments. We find that dimensionality reduction provides a viable ordering for pixel-only environments. We train a UI element detection model to derive elements from pixels and apply our findings to an agent benchmark—OmniACT—where we only have access to pixels. Our method completes more than two times as many tasks on average relative to the previous state-of-the-art.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This manuscript investigates the influence of ordering (in which elements are presented) to the performance of language model agents on navigating virtual environments, and proposes two simple ordering methods (raster and t-SNE) that are better than random ordering.

### Strengths
1. The investigated problem (ie, ordering’s effects) is somewhat novel.
2. The findings that ordering can influence the effects are intuitive and consistent with human perception.
3. The paper is presented clearly and easy to follow.

### Weaknesses
W1. Some conclusions are not well supported by experimental results.
- 1.1 In L299, the manuscript claims that “Random ordering results in a similar performance drop to removing all HTML text descriptions”. However, random ordering results in 37.04% (Gemini 1.5) and 44.44% (GPT4-V) in Table 2, while removing all texts brings 3.7% (Gemini 1.5) and 38.89% (GPT4-V) in Table 1. The performance gap between these two is still large.
- 1.2 In the Introduction, the manuscript claims that the t-SNE ordering improves performance. However, in Table 4, t-SNE is worse than raster in (a) VWA with Llama3, and (b) OmniACT with human annotations. Here, 4 out of 9 metrics, almost half, show raster is better than t-SNE. 

W2. The analyses and explanations of ordering’s effects are not sufficient. The manuscript notices that t-SNE is worse than raster in OmniACT with human annotations and provides these results in L455. However, why raster is better than t-SNE with human annotations has not been further analyzed.

W3. The necessity of using t-SNE to reduce dimension from 2D to 1D is not well stated. It is well known that t-SNE is more suitable for high-dimensional features, but the dimension of the original space is small (ie., 2D). In such case, using t-SNE will roughly equal the process that (1) choosing the top-left one element A, (2) finding its nearest neighbor element B, (3) then finding the nearest neighbor of both element A and element B (with a weighting strategy like exponential moving average), and so forth. 

W4. The experimental settings are not clear.
- 4.1 In L704, the manuscript states that only a subset of OmniACT is used in experiments. However, in Sec. 6.2, the manuscript directly compares the results with previously reported results tested on the full OmniACT. This raises concerns about the comparison results. 
- 4.2 In Table 5, the manuscript claims state-of-the-art performance. However, the sequence score is much lower than previous methods. Also, the foundational models are different. It is hard to say whether the performance difference is from the foundational models or the designs. 

W5. There are some writing mistakes.
- In L212, the words “must find” are repeated two times.
- In L461, “on Omniact” should be “on OmniACT”.
- In L465, “a LM” should be “an LM”.

### Questions
I am not very familiar with this field, so I will adjust my rating after reading other reviewers' feedbacks. 

Please see weaknesses W1 to W4. Writing mistakes do not need a response.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper explores the effect of element ordering on LLM agent performance in virtual environments like web and desktop. Through ablation studies on the VisualWebArena and OmniACT benchmarks, they find that random element ordering significantly degrades performance, similar to removing all textual information. To address this, the authors propose using dimensionality reduction techniques, specifically t-SNE, to create effective element orderings based on spatial relationships in pixel-based environments. The authors contribute a trained UI detection model and achieve state-of-the-art performance on OmniACT, demonstrating the impact of ordering strategies for agent navigation in pixel-only environments.

### Strengths
- Originality: The paper tackles a relatively unexplored area—optimizing UI element ordering for LM agents in pixel-only environments. The use of t-SNE for ordering based on spatial relationships is a novel application, offering a fresh perspective on improving agent navigation performance.
- Quality: The research employs ablation studies on VisualWebArena and OmniACT. The methodological depth and comparison across multiple ordering methods highlight the improvement of the approach.
- Clarity: The paper presents the problem and solution clearly, with intuitive descriptions of complex concepts like dimensionality reduction for ordering.
- Significance: Achieving state-of-the-art performance on OmniACT, the paper has practical implications for LM agent design in unstructured environments. The release of the trained UI detection model will further enhances its contribution to the research community.

### Weaknesses
- The paper could clarify its discussion of the dimensionality reduction approach, specifically addressing the parameters used in t-SNE. As t-SNE can be sensitive to parameter tuning, a detailed analysis of how parameter choices affect ordering outcomes would strengthen the validity of the results.
- Another weakness is the limited exploration of alternative ordering methods. While t-SNE provides performance gains, the study would benefit from a broader examination of ordering techniques, especially methods more aligned with semantic or functional groupings, which could be valuable in complex UI layouts where spatial proximity does not equate to functional relevance. 
- While the authors release the trained UI detection model, the study would benefit from greater transparency in terms of training data diversity and model performance metrics, such as precision and recall for UI element detection, to better assess its effectiveness across domains.
- The ablation study lacks rigorous control over experimental variables, which makes it difficult to isolate the exact impact of individual attributes on agent performance. For instance, the experiments appear to vary multiple aspects of the state representation without consistently controlling. A more systematic approach to ablation studies would enhance the credibility of the findings.

### Questions
- The paper mentions using default parameters for t-SNE, but this method can be sensitive to parameter choices, such as perplexity and learning rate. Please provide a sensitivity analysis of key t-SNE parameters (e.g., perplexity, learning rate) and their impact on ordering performance.
- While t-SNE shows promising results for ordering, it may also be beneficial to explore other dimensionality reduction techniques like UMAP and MDS. Please compare t-SNE with at least one other dimensionality reduction technique (e.g., UMAP) on a subset of data, and report the relative performance in terms of agent task success rate.
- While the appendix provides training details for the UI detection model, the paper would benefit from additional performance metrics specific to this scenario. Please provide a breakdown of the training data composition and to report related object detection metrics (e.g., mAP, precision-recall curves) for the UI detection model.
- Tables 4 and 5 report key results on different ordering methods, yet they lack detailed context, such as specific configurations or experimental conditions for each row. Could the authors expand on these tables with additional descriptions or footnotes? For example, is the trained model used for UI detection consistent across all experiments, does the success rate and action score reflect the same metric on OmniACT, etc.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper investigates the impact of element ordering on the performance of language model agents. The empirical study highlights that the sequence in which UI elements are presented to the agent significantly influences performance, especially for elements represented solely by pixels, where traditional hierarchical ordering is unavailable.

The authors propose an ordering method based on dimensionality reduction, necessitating the training of a UI element detection model to extract elements from pixel data. This approach offers a viable solution for ordering in pixel-only environments. The proposed method consistently demonstrates performance improvements over random ordering and achieves a new state-of-the-art result on the OmniACT benchmark.

### Strengths
1. Reveals the Critical Impact of Element Order on Agent Performance: Through systematic experiments, the paper effectively demonstrates the significant influence of element ordering on language model agents operating in pixel-only environments. This finding presents a new perspective for developing efficient virtual environment navigation algorithms. Previous research has often concentrated on accuracy in image recognition and text analysis, overlooking the importance of element order in contextual understanding. This paper addresses this gap by introducing order optimization as a novel method for enhancing agent performance.

2. Provides an Ordering Method in Environments Lacking Hierarchical Information: In situations where structural information (e.g., HTML or DOM trees) is not available, traditional methods struggle to identify an optimal element order. By utilizing t-SNE for dimensionality reduction, the paper effectively maps pixel-level interface elements to a one-dimensional sequence, establishing a practical ordering method for pixel-only environments. Despite its simplicity, this approach achieves performance gains in pixel-based contexts, setting new performance benchmarks on platforms like OmniACT.

### Weaknesses
1. While the baseline of random ordering is understandable as detrimental to large language models in interpreting UI, it is overly simplistic and is not a strong baseline. The study would benefit from incorporating more heuristic baselines. For instance, could a vision-language model, such as GPT-4V, assist in determining an optimal ordering when seeing the UI directly?

2. In Table 6, the proposed method consistently outperforms other ordering techniques only when elements are detected using Faster R-CNN, failing to demonstrate consistent advantages in other scenarios.

3. Although the paper validates the effectiveness of various ordering methods (e.g., t-SNE ordering surpassing random ordering), it lacks a systematic analysis of why specific ordering methods enhance performance. The authors primarily illustrate the effects of ordering through experiments but do not investigate the mechanisms by which different strategies influence the language model's understanding and contextual construction.

### Questions
Please refer to the weaknesses section above for specific points that require clarification or further detail.

### Soundness
2

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
5

### Summary
This paper investigates language model agents' navigation capabilities in virtual environments and focuses on exploring how element ordering affects agent performance. The paper's main finding is that element ordering has a significant impact on agent performance, with random ordering leading to substantial performance degradation - comparable to the effect of completely removing all visible text from web pages. While web-based agents can benefit from the semantic hierarchy provided by browsers, agents that parse elements directly from pixels cannot access this ordering information. The paper proposes an ordering method based on dimensionality reduction (such as t-SNE) that performs well in scenarios where only pixel information is available.

### Strengths
1. The paper provides the first in-depth investigation of how element ordering affects agent performance and demonstrates its significance.

2. The proposed dimensionality reduction-based ordering method performs well in pixel-only scenarios and achieves new state-of-the-art performance on the OmniACT agent benchmark.

3. The paper introduces a UI element detection model which has been made publicly available for other researchers to use and improve upon.

### Weaknesses
1. While the paper focuses on the t-SNE dimensionality reduction ordering method, it lacks in-depth analysis and comparison with other ordering methods. Additionally, all ordering methods show significant performance gaps compared to Pre-ordering (Table 4).

2. The paper briefly introduces the training process of the UI element detection model but lacks more detailed specifics.

### Questions
1. How does element ordering affect results at different orders of magnitude (e.g., fewer than 10 elements vs. more than 50 elements)?

2. While the ablation section examines how different inputs affect experimental results, to draw more convincing conclusions, the authors should conduct additional ablation experiments on datasets beyond VisualWebArena.

### Soundness
4

### Presentation
3

### Contribution
3
