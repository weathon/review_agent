# Beyond Unified Directions: Context-adaptive Representation Steering for LLM Safety Alignment

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Large language models (LLMs) face significant generative safety risks in deployment, and representation steering has emerged as a lightweight alternative to resource-intensive training-based safety alignment methods. However, existing representation steering approaches compute a unified steering direction, which fails to leverage context-specific information critical for precise safety alignment. To address this limitation, we propose \textit{CA-Steer}, a context-adaptive representation steering method for LLM safety alignment. It computes a context-adaptive direction by retrieving contextually similar safe and unsafe representations as references. Besides, a sample-level steering gate is introduced to filter unnecessary operations, ensuring safety alignment without compromising LLM utility. Evaluations on three safety benchmarks and two utility benchmarks show that CA-Steer significantly outperforms existing baselines: it improves the vanilla LLM’s average safety score from 85.80\% to 97.09\% (surpassing the best baseline by 6.28 percentage points), and maintains nearly no utility loss. In-depth analyses further confirm the rationality of its design and its acceptable overhead.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CA-Steer, a context-adaptive representation steering method for improving safety alignment in large language models (LLMs). Unlike prior steering approaches that apply a single global direction to all contexts, CA-Steer computes token-level context-specific steering vectors by retrieving contextually similar safe and unsafe representations from pre-built banks. It also introduces (1) a sample-level padding strategy to handle retrieval imbalance or emptiness and (2) a steering gate that conditionally applies steering only when the prompt is likely to be unsafe.

### Strengths
1. The paper identifies a clear limitation in current steering methods — the “one-size-fits-all” assumption — and introduces context-adaptive steering as a principled extension.
2. Experiments are extensive and systematic, covering multiple open-source LLMs and diverse safety/utility benchmarks.
3. Writing is clear, structured, and professional.

### Weaknesses
1. Some experimental results are wrong. For example, in Table 1, ALERT benchmark for qwen with $\text{prompt}_\text{hand}$, the author reports the safety performance is 96.16%. However, the test dataset in ALERT has only 1,000 items. How can you get these results? I think the authors should check the results of your paper (Not only what I mentioned).

2. The method is not novel.  The steering method has been widely used. The authors should cite these articles and add a statement of innovation.

### Questions
See Weaknesses.

### Soundness
3

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
This paper proposes a context-adaptive representation steering framework (CA-Steer) for improving large language model (LLM) safety alignment. While prior methods typically apply a unified, dataset-level steering direction to all contexts, CA-Steer introduces a more fine-grained and dynamic approach. It retrieves token-level and sample-level contextually similar representations to compute adaptive steering vectors during inference, enabling more precise and effective safety control.

### Strengths
1. The paper presents a lightweight method for steering LLM behaviour toward safer responses.
2. The analysis showing that safe and unsafe representations form distinct clusters across different contextual settings is insightful.

### Weaknesses
1. Limited generalizability. Since the steering vectors in CA-Steer are context-dependent, the method appears potentially data-intensive, requiring diverse and comprehensive representation banks that cover various contextual settings and risk types. 
- Could the authors discuss how CA-Steer scales when contextual coverage is limited, for instance, if certain risk categories or prompt styles are underrepresented in the safety dataset? 
2. Confusing design on computing the steering vector. The method computes a steering vector by retrieving contextually similar safe and unsafe representations for the same target token representation $h$. 
- How can $h$ be simultaneously similar to both safe and unsafe examples? Representation steering typically assumes that safe and unsafe representations are linearly separable, implying that a token should be close to either safe or unsafe examples but not both. 
- The authors report that, on ALERT, 55.1% of retrieved subsets are dominated by safe samples and 22.5% by unsafe ones. Does this suggest that most retrievals are inherently imbalanced and that balanced retrievals are relatively rare? If so, does this mean that computing a steering vector based on both safe and unsafe similar representations may not always be feasible or meaningful in practice?
3. Concerns on the effectiveness of the padding strategy. In Section 4.2, the authors report that Global Padding outperforms Only Padding, even though Global Padding uses dataset-level mean representations while Only Padding uses context-specific samples. 
- Could the authors clarify why the global representation performs better than the local, context-dependent one?

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

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
The paper presents a method for LLM representation steering for inference-time safety. The key novelty behind the presented method is incorporating context information from the input prompt and partial response to compute the steering vector for each token. The paper also presents a simple gating mechanism that decides whether a token needs steering or not. Evaluation on safety and utility benchmarks shows improvements to the model safety and a minor degradation to the model utility (which is clearly demonstrated to be due to the introduced gating mechanism).

### Strengths
1. The presented method is simple and intuitive and is shown to introduce a relatively small compute overhead.

2. The evaluation clearly demonstrates the effectiveness of the method compared to previous work.

### Weaknesses
1. The safety evaluation is somehow limited: it is based on 3 benchmarks with all 3 models evaluated already are doing reasonably well on 2 of them. It would be interesting to demonstrate the effectiveness of the method against adversarial jailbreak attacks as well. That would be a stronger evidence on the robustness of the presented approach.

### Questions
Would you please elaborate on the training data? its size, source, structure, etc.

### Soundness
3

### Presentation
3

### Contribution
2
