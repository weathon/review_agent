# Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Large Vision-Language Models (LVLMs) exhibit outstanding performance on vision-language tasks but struggle with hallucination problems.
Through in-depth analysis of LVLM activation patterns, we reveal two key findings: 1) truthfulness and visual perception capabilities predominantly engage different subsets of attention heads within the model architecture; and 2) truthfulness steering vectors vary significantly across different semantic contexts. Based on these observations, we propose Dynamic Multimodal Activation Steering, a training-free approach for hallucination mitigation. Our method constructs a semantic-based truthfulness steering vector database and computes visual perception steering vectors, enabling context-aware interventions during inference by dynamically selecting the most relevant steering vectors based on input semantic similarity and applying them to the most influential attention heads. We conduct comprehensive experiments across multiple models and datasets, demonstrating that our approach significantly enhances model performance, outperforming existing state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to address the hallucination problem in Large Vision-Language Models (LVLMs). Through an analysis of internal activation patterns, the authors report two key findings: (1) the attention heads responsible for truthfulness and visual perception are largely disjoint within the model; and (2) the truthfulness steering vectors that guide factual output vary significantly across different semantic contexts.
Building on these insights, the paper proposes Dynamic Multimodal Activation Steering (DMAS) — a training-free, plug-and-play inference-time intervention technique. The method consists of three main stages:
1.	Constructing a Truthfulness Steering Vector Database: Semantic clustering is applied to the data, and within each cluster, activation differences between truthful and hallucinatory samples are computed. These are stored in a database where semantic embeddings serve as keys and truthfulness steering vectors as values.
2.	Computing the Visual Perception Steering Vector: Activation differences between the model’s responses to clean and noise-distorted images are used to derive a steering vector that enhances visual attention.
3.	Dynamic Intervention: During inference, the method retrieves the most semantically relevant truthfulness steering vector from the database based on the input text and combines it with the visual perception steering vector. The combined signal is then applied selectively to the most influential attention heads, effectively reducing hallucinations.
Experimental results demonstrate that DMAS achieves significant improvements over existing approaches across multiple benchmarks and model architectures.

### Strengths
Clear problem orientation: The paper provides an in-depth analysis of hallucination issues in LVLMs, particularly revealing the functional tendencies of different attention heads and the semantic dependence of steering vectors through preliminary studies, which offers strong motivation for the subsequent method design.

Broad experimental coverage: The paper conducts comprehensive experiments across multiple models (LLaVA v1.5, QwenVL) and multiple benchmarks (MME, POPE, CHAIR), including detailed ablation studies and case analyses, demonstrating the effectiveness of the method.

Significant performance improvement: Experimental results show that DMAS achieves state-of-the-art performance across multiple tasks, with particularly substantial gains on MME and CHAIR metrics.

### Weaknesses
Reliance on synthetic counterfactuals in database construction: The method constructs “hallucinated” answers by flipping or randomly selecting incorrect options, which is relatively easy for multiple-choice or discriminative datasets. However, this approach may not reflect realistic hallucination types in open-ended generation tasks. As a result, the learned “steering vectors” may be biased toward these synthetic patterns.

Dependence on specific labeled data: As noted, the main limitation of this method is that its so-called “training-free” property relies on a pre-constructed steering vector database that depends on labeled data (true/false answer pairs). This severely limits the generality and scalability of the approach. The authors should more transparently discuss this prerequisite and its associated limitations.

High hyperparameter sensitivity: The method introduces several key hyperparameters, including intervention strengths $\alpha$ and $\beta$, the number of intervention heads $K$, and the number of clusters. Analysis in Figure 3 shows that model performance is highly sensitive to these parameters, and inappropriate settings can lead to a sharp drop in performance. This makes hyperparameter tuning costly in practical applications.

### Questions
Counterfactual generation: For open-ended datasets (e.g., CHAIR, AMBER), how are hallucinated answers generated? Are they human-labeled, randomly perturbed, or model-generated? How sensitive are the steering vectors to this generation method?

Automatic hyperparameter selection: Is there a lightweight, label-free method to choose (α, β, K) or the number of clusters, to avoid expensive grid search for each new model or domain?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Dynamic Multimodal Activation Steering (DMAS), a training-free and plug-and-play method to reduce hallucinations in Large Vision-Language Models (LVLMs). The authors find that truthfulness and visual perception rely on distinct attention heads and that truthfulness patterns vary with semantics. DMAS builds a semantic truthfulness steering vector database and computes visual perception steering vectors, dynamically applying them to the most relevant attention heads during inference.

### Strengths
1. This paper conducts an interesting analysis of attention patterns, revealing which attention heads are most sensitive to truthfulness versus visual perception.
2. This paper proposes an interesting Dynamic Multimodal Activation Steering, which incorporates steering vector to mitigate hallucination.
3. The paper demonstrates good writing quality and is easy to read.

### Weaknesses
1. Experiments were conducted on a limited set of backbones; it would be better to include experiments on more recent models.
2. Some important hyperparameters are missing from the paper. For example, the temperature, top-p, and top-k settings are not reported.
3. The paper lacks comparisons with recent decoding strategies, such as DECO [1] and DAMO [2]. As far as I know, they also perform well in hallucination mitigation.
    - [1] MLLM can see? Dynamic Correction Decoding for Hallucination Mitigation
    - [2] DAMO: Decoding by Accumulating Activations Momentum for Mitigating Hallucinations in Vision-Language Models
4. The layout of Table 3 and Table 4 could be improved for better readability and presentation.
5. The formatting of captions should be consistent throughout the manuscript. For example, Figure 1, Figure 2, and Table 1 all end with a period, but Figure 3 does not.

### Questions
1. How does the method perform on more recent models? For example, Qwen2.5-VL-7B, which was released at the beginning of 2025.
2. While MME and POPE are simple and well-known benchmarks, how does the method perform on other, more complex ones, such as MM-Vet and LLaVA-Bench? The effectiveness and improvements of the proposed method should be demonstrated across a broader range of benchmarks.
3. Could the authors release the detailed hyperparameter settings? 
4. Did the authors use greedy search? 
    - If YES, I am a bit curious about the reported results.  Based on my own experiments with LLaVA-1.5-7B on the MME benchmark, the `Existence` subtask under the regular setting should achieve a score around 190, while `Count` and `Color` should be around 160 and 165, respectively.  (Note: Here, I refer to directly using the original MME benchmark rather than data from other repositories, as those may introduce different prompts.)
    - If NOT, multiple runs should be conducted to obtain statistical results and reduce randomness or variance in the evaluation.
5. Based on the reported results, the improvements appear to be marginal on POPE on LLaVA, with performance comparable to ICT. So how to demonstrate the effectiveness?
6. It is interesting that the paper identifies which attention heads are most sensitive to truthfulness versus visual perception. I am curious whether these findings also hold for different models — for example, Qwen-VL, which was used in the main experiments. Furthermore, for the 13B model reported in Table 8, what are the corresponding findings? Since the 7B and 13B models have different numbers of layers, it would be insightful to know whether similar attention patterns are observed.

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
The goal of this work is to mitigate hallucinations in LVLMs by steering internal activations in a way that respects both visual grounding and factual correctness. The first contribution is an analysis showing that truthfulness and visual perception largely rely on different subsets of attention heads, and that truthfulness directions vary with semantic context. To address this, the paper proposes a training-free method that precomputes steering vectors for the visual and textual pathways and then applies them in a context-aware manner at inference. Concretely, the method clusters prompts into four semantic groups, builds a truthfulness vector per cluster from activation differences between factual and hallucinated answers, derives a visual-perception vector from clean versus noise-corrupted images with object prompts, and then steers the most influential heads using these vectors.

The results show that the approach improves on discriminative VQA benchmarks such as POPE and MME subsets, and on open-ended captioning with CHAIR, while also showing signs of transfer beyond the construction datasets (ScienceQA, ViQuAE). The ablations explore the choice of hyperparameters and cluster size and importance of using both visual and truthfulness vectors.

### Strengths
- A training-free and context-aware method to curb hallucinations in LVLMs by nudging a small set of attention heads tied to visual grounding and factuality. It helps on both open-ended captioning (e.g., CHAIR) and VQA-style benchmarks (e.g., POPE/MME).
- The setup is easy to follow: datasets and metrics are spelled out, baselines are sensible, and the main knobs (α, β, and the number of intervened heads K) are reported with the ranges they tried.
- Ablations show both components (truthfulness and visual) matter, dynamic retrieval beats a single fixed vector, and you get reasonable guidance for choices like cluster count and K rather than hand-wavy defaults.

### Weaknesses
- Although training-free, the method isn’t truly plug-and-play across LVLMs: the influential-head masks and truthfulness/visual steering vectors must be recomputed for each new model (with α/β/K re-tuned), so deploying on a different backbone requires non-trivial one-time setup rather than drop-in reuse.
- It’s encouraging to see gains without regressions and signs of cross-domain transfer (Table 5). To make the generality claim more convincing, reporting results against stronger baselines beyond Regular would make the narrative stronger.

### Questions
- Given that the steering database is built from AMBER/SEED (discriminative and MCQ) [lines 172-173] using label flips or random incorrect choices, clustered into four groups, I am curious to know the intuition behind how these vectors remain valid for open-ended generation. What mechanism supports transfer from discriminative supervision to generative benchmarks such as CHAIR?
- For visual steering vector, does the selecting strategy for negative objects impact this method’s performance? e.g would selection of hard negatives or commonly hallucinated objects as negatives instead of random negative object selection done in the paper impact performance? What are the takeaways from practitioners on the sensitivity of negative object/answer selection for the database generation?

### Soundness
3

### Presentation
3

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
This paper introduces Dynamic Multimodal Activation Steering (DMAS), a training-free and plug-and-play method designed to mitigate hallucinations in Large Vision-Language Models (LVLMs). The core idea is to dynamically intervene in attention head activations by separating intervention into two components: a "truthfulness steering vector" and a "visual perception steering vector." The truthfulness vector is learned via contrastive activation differences across semantically clustered data and stored in a dynamic database, while the visual perception vector is derived from clean vs. noisy image inputs. During inference, DMAS retrieves the most context-relevant truthfulness vector and applies both to the top-K most influential attention heads.

### Strengths
The paper is well-motivated and proposes a clever approach to VLM hallcunation issue. The realization that truthfulness steering vectors vary significantly across semantic contexts and the resulting dynamic, context-aware database approach is novel within activation steering literature. In addition, the training-free nature of this work is highly preferred, especially for such kind of problems where efficiency matters.

### Weaknesses
I am giving a conditional weak reject, and I think the the following issue should be carefully addressed by authors.

- A more comprehensive analysis of the constructed dataset should be provided. One of the major novelty of the method is to pre-construct a set of embeddings and select steering vectors accordingly from this dataset. As such, the property of the dataset matters a lot, but very limited analysis is given. How different the performance will be if we start from a different dataset? Will the size of the dataset matters? How random the performance is when we change the selection criteria? I think the study of dataset is largely lacking, making the understanding of DMAS very restricted.
- The overall improvement shown in the main tables are marginal, and authors are presenting them in a somewhat misleading way (the $\Delta$ should be with the best baselines, not the "Regular". This happens a lot in the paper and must be addressed). In table 2 other methods can give SOTA performance or very close performance. Without standard deviation reported, I am not convinced that this minor improvement over ICV and VTI is significant.  The authors are encouraged to include more experiments to demonstrate the effectiveness of DMAS, or otherwise, I doubt the necessity of having such a dynamic, retrival-related method.

### Questions
I think the authors do not use a correct citation format. Citations should be with brackets when the cited papers are not the subjects. please address.

### Soundness
2

### Presentation
2

### Contribution
2
