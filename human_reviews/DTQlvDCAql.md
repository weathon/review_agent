# HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics

- Avg Score: 4.25
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 8, 3, 3

## Abstract
Existing research often treats long-form videos as extended short videos, leading to several limitations: inadequate capture of long-range dependencies, inefficient processing of redundant information, and failure to extract high-level semantic concepts. To address these issues, we propose a novel approach that more accurately reflects human cognition. This paper introduces HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics, a model that simulates episodic memory accumulation to capture action sequences and reinforces them with semantic knowledge dispersed throughout the video. Our work makes two key contributions: First, we develop an Episodic COmpressor (ECO) that efficiently aggregates crucial representations from micro to semi-macro levels, overcoming the challenge of long-range dependencies. Second, we propose a Semantics ReTRiever (SeTR) that enhances these aggregated representations with semantic information by focusing on the broader context, dramatically reducing feature dimensionality while preserving relevant macro-level information. This addresses the issues of redundancy and lack of high-level concept extraction. Extensive experiments demonstrate that HERMES achieves state-of-the-art performance across multiple long-video understanding benchmarks in both zero-shot and fully-supervised settings.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces a vision encoder that integrates both micro-level and macro-level details before inputting the data into large language models (LLMs) for comprehensive long-form video analysis. The Episodic COmpressor (ECO) is responsible for consolidating video frames into a memory buffer, while the Semantics ReTRiever (SeTR) combines video tokens using a dot product mean approach, which is then followed by a Q-former model. In practical applications, employing a vicuna-7B model, this method has demonstrated significant performance improvements on benchmarks designed to assess long-video understanding capabilities.

### Strengths
1. The writing in the paper is clear. I strongly resonate with the introduction regarding episodes and semantics.
2. The ablation study is particularly thorough, with tables 4 and 5 highlighting the significant impact of ECO and SeTR on the model's performance.

### Weaknesses
1. Figure 1 is misleading and somewhat overstated. From my perspective, the paper primarily focuses on token merging and compression. However, Figure 1 gives the impression that this work can develop an understanding of semantics and episodes at the language level.

2. The distinction between ToMe and the discussed approach is unclear. In fact, Section 3.4 closely resembles ToMe, where ToMe is executed just once.

3. The comparisons made are somewhat outdated. There have been several recent developments in video LLMs and benchmarks [1, 2, 3], yet the author does not address or report their performance.


[1] Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis.

[2] LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding

[3] Streaming Long Video Understanding with Large Language Models.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes a new multi-modal language model (LLM) for the video question-answering problem. It follows a divide-and-conquer paradigm, which involves the following steps: First, extracting the episodic-level representation, Meanwhile, retrieving the key semantics, Finally, using this information to query the LLM.The figures in the paper help quickly understand the proposed approach. The experimental results are state-of-the-art (SOTA), and the ablation study and model analysis sections are also robust.

### Strengths
-The idea of using "Episodes and Semantics" for understanding videos is novel.

-The technical contribution of a new multi-modal LLM for video understanding is solid.

-The experimental results are convincing and state-of-the-art (SOTA).

### Weaknesses
-In Section 3, the authors claim that "the core ideas of episodic memory compression (ECO) and semantic knowledge retrieval (SeTR) can be applied to other models," but they do not conduct experiments to support this claim.

-Figure 6 only shows the good cases, while I am curious about the failure cases that cannot be handled by the ECO+SeTR approach.

-Some recent works on video understanding are missing. For example, the paradigm of 2D CNN + temporal modeling was popular between 2019-2022, and some representative approaches [1][2] should be mentioned. Additionally, video semantic compression approaches are highly related to the "semantic condensing" idea proposed in the paper.
[1]Wang L, Tong Z, Ji B, et al. Tdn: Temporal difference networks for efficient action recognition[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021: 1895-1904.
[2]Tian Y, Yan Y, Zhai G, et al. Ean: event adaptive network for enhanced action recognition[J]. International Journal of Computer Vision, 2022, 130(10): 2453-2471.
[3]Tian Y, Lu G, Yan Y, et al. A coding framework and benchmark towards low-bitrate video understanding[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

### Questions
- How to apply the idea to other models.

- Report the failure cases.

-Adding recent related works.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper aims to address long video understanding from the perspective of episodic and semantic memory. The authors provide an insightful analysis into the two memory mechanisms that motivate the architecture design, where multiple QFormers are employed to compress tokens at different granularities and token merging is applied to consolidate semantic-level memory. The experiments validate the effectiveness of the architecture.

### Strengths
1. The analysis of episodic and semantic memory makes sense and is a promising way to address long video understanding.
2. The problem definition is clear and easy to follow.
3. The architecture shows significant improvement in inference speed compared to existing memory-augmented video LLM.

### Weaknesses
1. The architecture design cannot reflect the authors' analysis of episodic and semantic memory. Simply using QFormer or token merging strategies can compress video tokens, but is not sufficient to construct the structural memory.
2. Also, the authors are expected to show some visualizations of the token merging in semantic retrieval to show the structure of the semantic memory compression.
3. The results on more recent long video benchmarks, e.g., VideoMME, MLVU, LVBench, etc, are desired.

### Questions
See weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents HERMES, a framework designed for long-form video understanding that leverages episodic memory and semantic knowledge. The authors propose two main components: the Episodic COmpressor (ECO), which aggregates temporal information, and the Semantics reTRiever (SeTR), which extracts high-level semantic cues. The model claims to outperform existing methods on multiple benchmarks, showcasing its effectiveness in handling long videos.

### Strengths
- The authors enhance Q-former-based models, such as MA-LMM, by integrating episodic and semantic information. The architecture is clearly defined and appears to be replicable.
- The paper reports impressive performance metrics across various long-video benchmarks, indicating that HERMES effectively addresses the complexities inherent in long-form video content.

### Weaknesses
- While the use of memory mechanisms and temporal information compression is valuable, the approach does not significantly advance the current state of the art. The paper primarily builds on the MA-LMM framework, with its main innovations centered around the memory components. The concepts introduced in the Episodic COmpressor and Semantics reTRiever share similarities with the memory bank and mechanisms used in MA-LMM. Furthermore, comparable memory structures are present in MovieChat, which differentiates between long-term and short-term memory—concepts that correspond to the SeTR and ECO, respectively. The authors are encouraged to discuss these distinctions in detail to better highlight the unique contributions of HERMES.
- The methods used for comparison across different datasets are inconsistent, making it difficult to draw clear conclusions about HERMES's overall performance. A standardized evaluation framework would enhance the reliability of the results.
- There are inconsistencies in the use of symbols throughout the paper. For instance, in Line 208, \( F_w \) refers to the number of frame features, while in Line 222, \( F_w \) denotes the window, with \( w \) indicating the number of frame features. This inconsistency could lead to confusion and should be clarified.

### Questions
Can the authors provide a more detailed discussion on how HERMES distinguishes itself from existing models like MA-LMM and MovieChat? Specifically, what unique contributions do the Episodic COmpressor (ECO) and Semantics reTRiever (SeTR) offer compared to similar mechanisms in these models?

### Soundness
2

### Presentation
3

### Contribution
2
