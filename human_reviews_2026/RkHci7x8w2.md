# Semi-Supervised Preference Learning for Multi-modal Large Models via Risk Analysis

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Ensuring that multi-modal large models possess reasoning capabilities aligned with human preferences is of paramount importance. Currently, the most effective approach involves fine-tuning these models using reward models optimized towards human-aligned objectives. However, optimization of reward models typically requires large-scale human-annotated datasets, which pose a significant bottleneck for downstream tasks with limited labeled samples. To address this limitation, we propose a \textbf{S}emi-supervised \textbf{P}reference learning approach based on \textbf{R}isk \textbf{A}nalysis, denoted by \textbf{SPRA}, which can accurately assess the alignment of large model outputs with human preferences using limited labeled data. The proposed SPRA measures preference by a risk model, whose construction consists of three main steps: (1) extract risk features that encode human priors from a limited set of labeled samples; (2) construct a risk model based on risk features; (3) train the risk model. Then, SPRA uses the resulting risk model to rank model responses, with lower-risk ones prioritized as preferred outputs.  
By explicitly incorporating human priors into its modeling framework, SPRA achieves not only high interpretability but also flexibility to adapt to diverse human preference distributions by adjusting the priors. This contrasts to traditional single-preference predictors, which lack such adaptability. In particular, the SPRA risk model is parameter-efficient, containing only thousands of parameters, which significantly reduces computational overhead and simplifies reward optimization. Our empirical study on real benchmark datasets validates the efficacy of SPRA.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes SPRA, a semi-supervised framework for learning human preferences over multimodal model outputs using limited labeled data. The core idea is to construct a “risk model” that aggregates heuristic risk features (e.g., confidence, consistency, entity matching) into a preference score, which is then used to construct preference pairs for DPO fine-tuning. A CLIP-based mapping mechanism is introduced to scale risk feature generation to unlabeled data.

### Strengths
- **Practical Motivation**: Addresses a real bottleneck in preference learning, lack of large-scale human annotations.
- **Systematic Framework**: The pipeline from rule-based risk features, statistical modeling to preference fine-tuning is clearly laid out.
- **Engineering Contribution**: The activation matrix mapping trick using CLIP embeddings is a useful and efficient way to scale pseudo-label generation.
- **Competitive Empirical Results**: Outperforms several strong baselines on preference prediction tasks, despite using far fewer labeled samples.

### Weaknesses
- **Conceptual Clarity**: The definition of “risk” is ambiguous and inconsistent across sections. It sometimes refers to uncertainty, sometimes to semantic incorrectness, and sometimes to human preference deviation. The paper conflates “risk” with “preference” without clearly justifying why lower risk implies higher preference.
- **Misleading Claims of "Parameter-Efficiency"**: The paper repeatedly emphasizes that the final risk model is "parameter-efficient" with only "thousands of parameters". This claim is misleading as it conveniently ignores the massive computational overhead of the Risk Feature Generation step. As described in Figures 2 & 3 and Section 3.1.1, generating a single feature vector requires running inference on multiple large "tool" models (e.g., Florence2, Tulu, Qwen2VL-2B), which is also described as "time-consuming" in the paper. 
- **Unjustified Methodological Complexity**: The paper introduces a complex theoretical apparatus from financial risk. However, this entire framework appears to be superficial. After deriving the VaR, the authors state it cannot be used for training directly and instead pass it through a Sigmoid function and optimize it using a standard binary cross-entropy loss against "risk labels" of 0 (match) or 1 (mismatch). It is entirely unclear what this convoluted path achieves over a standard, direct binary classification setup (i.e., training the network to directly predict the mismatch probability p). The paper provides no ablation or justification for this added complexity, leaving me unconvinced that it yields any measurable benefit.

### Questions
- Could the authors provide a full computational cost analysis (e.g., total FLOPs or GPU-hours) for the entire pipeline, including feature generation, mapping, and risk model training? How does this total cost compare to previous works?
- Can the authors provide an ablation study that directly compares their VaR-based loss function (Eq. 8) against a standard binary cross-entropy loss that directly trains the risk model to predict the risk label using the same features? This is necessary to justify the financial risk framework.
- Can the authors provide a formal and concise definition of "risk" mentioned in this paper?

I am willing to raise my rating if the authors can address my points above.

### Soundness
3

### Presentation
2

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
The paper proposes a new risk-based preference modeling approach that enables model improvement with limited labeled data.
Here, the risk is defined using features derived from image-understanding questions (feature extraction questions). It is measured by how much the model’s answer changes when such features are included. A risk model is trained to approximate this risk using a Gaussian distribution, and the main model is then refined via DPO training on data labeled by the risk model, using self-corrected answer pairs generated by the model itself.

### Strengths
1. The concept of extracting additional features used in human reasoning and leveraging them for risk computation is novel. Although the paper employs a limited rule-based approach for feature extraction, this idea has the potential to be extended toward a broader and more comprehensive analysis.

2. The evaluations conducted on diverse datasets such as Animals, RealWorldQA, and MMBench demonstrate that the proposed method is not limited to specific scenarios, but can be widely applied across various contexts.

3. It is impressive that the risk model, despite its small size (only 7K parameters), achieves preference classification performance comparable to vision-language models (VLMs) that are over a thousand times larger.

### Weaknesses
1. The performance improvements reported in the paper appear marginal. For instance, in the case of Qwen2-VL-2B, the improvement is only around 1–2%, which is roughly within the range of statistical variance. It is therefore unclear whether these gains are statistically significant.
2. The paper compares its method to SCL trained on the full dataset, but it should also include a comparison with SCL trained on the same subset of data used by SPRA. Since SPRA achieves comparable performance while using only about 40% of the labels on most benchmarks, its apparent data efficiency advantage might not be as strong as claimed. A difference of around 40% may not significantly impact the overall training performance.
3. It is uncertain how effective this approach would remain when applied to more advanced models. For example, with recent models such as Qwen2.5 or 3 VL (8B-scale or even small), even simple self-correction or self-rewarding mechanisms may yield similar improvements—especially given that the performance gains reported in this paper are quite small. A fair evaluation should therefore include comparisons with these alternative, simpler methods, as well as with "reward model'' approaches, since the proposed risk model plays a similar role. The absence of such comparative analyses makes it difficult to fully assess the validity and relative effectiveness of the proposed method.

### Questions
1. Are there any additional baselines beyond SCL? For example, would it be possible to include a simple SFT baseline for comparison?

2. The process of extracting risk features is based on heuristically defined rules and involves generation through a model. Since these predefined rules cannot fully capture the model’s overall feature and are heuristically chosen, they may introduce bias. Furthermore, the model used to generate problem instances could inherently contain bias. Did you observe any such potential biases arising from either the rule definitions or the problem generation itself?

### Soundness
2

### Presentation
2

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
This paper propose a Semi-supervised Preference learning approach based on Risk Analysis（SPRA），which can measures preference by a risk model. Experimental results show that this method achieves good performance, while significantly reducing computational overhead and simplifying reward optimization. However, the method proposed in this paper does not seem to be particularly innovative. Moreover, the experimental hyperparameters and implementation details are not sufficiently described or analyzed, making it difficult to understand what the model has actually learned regarding risk handling.

### Strengths
1. This paper presents a seemingly effective approach for handling risks and validates it through experiments.
2. The authors' writing in the methodology section is relatively clear.

### Weaknesses
1. My main concern is that the proposed method in this paper appears more like a technical implementation strategy, lacking sufficient theoretical support. Although some experimental results are provided, they are not comprehensive enough. I recommend adding experiments under different hyperparameters, such as varying temperatures and learning rates, to better validate the method's effectiveness and robustness.
2. The paper compares the proposed SPRA method with SCL in experiments, and the results show that SPRA does not demonstrate a significant performance advantage. Although the authors mention that SPRA uses only a small amount of ground-truth labels compared to SCL, this advantage should primarily be attributed to the inherent effectiveness of semi-supervised learning methods. I am curious about what specific contributions the authors have made in this regard.
3. It should be clarified how CVaR is incorporated into the RiskModel, with a more detailed explanation provided. Additionally, an analysis of the impact of relevant hyperparameters should be included.
4. The operation symbols in Equations (5) and (6) are not explained or introduced.
5. The authors should present and analyze the model training curves, such as loss, policy entropy, KL divergence, and other relevant metrics.
6. The font size in Table 1 and Table 2 is too small; it is recommended to adjust it for better readability.

### Questions
Please refer to the “Weakness” section for related questions.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces a new framework for aligning vision-language models (VLMs) with human preferences using limited labeled data. Instead of relying on costly reward models trained from large-scale human feedback, SPRA models preference alignment as a risk estimation problem: outputs that deviate from human priors are treated as high-risk. The method constructs risk features—both statistical (confidence-based) and generative (consistency-based)—to quantify how well a model’s response conforms to interpretable human-like rules. A small risk model then aggregates these features into a continuous “Value-at-Risk” score that ranks outputs by preference likelihood. This enables efficient fine-tuning via Direct Preference Optimization (DPO). Experiments across classification and VQA benchmarks show that SPRA achieves accuracy comparable to or better than fully supervised methods while requiring far fewer human annotations.

### Strengths
originality: The paper introduces SPRA – a framework to align VLM with human preference leveraging a light-weighted risk model. The risk model is trained on a set of risk features marking deviation from human priors. This is a novel idea of designing a set of simple oracle rewards for VLM alignment rather than user large model judges, making preference alignment more cost-efficient. 
quality & clarity: The paper provides quite clear derivation of different risk features, detailed experimental results and ablation results in the appendix. Overall writing and figures are clear except for the point mentioned in weakness. 
significance: The developed method labels preference dataset for VLM alignment with much reduced human or compute resource as it uses a light-weight risk model. It also shows better grasp of the underlying ground truth label as well as competitive performance in align generation policy at downstream.

### Weaknesses
1. Figure clarity: Figure 2 the arrow "Bert-score" occludes part of the illustrative text; Figure 3 contains wrong spelling "Optiions" and the workflow is confusing.
2. Section 4.2 Table 2: In Table 1, the authors show that their method SPAR largely improve the prediction accuracy of the ground truth label. However, in using SPAR for fine-tuning, the gain on the downstream tasks are much less. Could authors provide more intuition regarding this?
3. Section 3.1.1 risk feature generation: I understand that the authors try to find a set of simplified oracles to define what a human would call "risk." However, I need more motivation and explanation on the statistical risk feature part. Take entropy as an example, if the model generate high entropy tokens it doesn't necessarily indicate the model is not aligned. It could be your input data is low quality? Given an ambiguous image e.g. the model naturally would have high uncertainty in generation, not necessarily because it's policy is not aligned.

### Questions
1. Regarding weakness 1: it'd be helpful to improve on figure clarity.
2. See weakness 2.
3. See weakness 3.

### Soundness
3

### Presentation
2

### Contribution
3
