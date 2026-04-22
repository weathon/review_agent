# Teaching Metric Distance to Discrete Autoregressive Language Models

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
Large language models (LLMs) operate as autoregressive predictors over discrete token vocabularies, a formulation that has enabled their adaptation far beyond natural language to vision, robotics, and multimodal reasoning. However, training against one-hot targets disregards metric relationships between tokens and limits effectiveness on tasks where distance is meaningful, such as numerical values, spatial coordinates, or quantized embeddings. We introduce DIST2Loss, a distance-aware objective for discrete autoregressive models that replaces one-hot targets with reward-weighted distributions derived from predefined token distances. DIST2Loss can be interpreted as the closed-form solution to entropy-regularized policy optimization with known per-token rewards, retaining the core mechanism of reinforcement learning while avoiding sampling, rollouts, and instability. Our experiments show that DIST2Loss improves data efficiency and downstream performance across diverse domains. It yields tighter bounding boxes in visual grounding, accelerates robotic manipulation by improving action learning, enhances reward modeling for LLM alignment, and strengthens vector-quantized image generation. These results demonstrate that distance-aware supervision offers a simple and general alternative to one-hot supervision for discrete autoregressive models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses a key limitation in adapting large language models (LLMs) to tasks involving numerical or metrically structured data. Conventional fine-tuning methods treat tokens as categorical variables, ignoring the intrinsic distance relationships among them—such as numerical proximity or spatial metrics. To overcome this, the authors propose DIST² loss, a new objective that integrates predefined distance relationships between tokens into the autoregressive loss used during training. This approach enables LLMs to better capture metric structure while remaining compatible with standard categorical token modeling. Empirical results demonstrate substantial performance improvements across tasks where distances are semantically meaningful, including object detection, object manipulation, reward modeling, and image generation.

### Strengths
* The paper is clearly written and easy to follow.
* The problem is important and timely, addressing a fundamental gap in how LLMs handle numerical and metric relationships.
* Empirical results show significant improvements across diverse downstream tasks, including those with limited data
* The proposed method integrates smoothly into existing LLM training objectives, maintaining compatibility with categorical token modeling.

### Weaknesses
The paper does not clearly discuss how different types of numerical or metric data (e.g., integers, floats, directions) are handled under the same framework. It remains unclear whether the model improves actual numerical reasoning accuracy, beyond performance metrics on downstream tasks. There is a lack of explicit evaluation of the model’s ability to reason about true numerical distances or relationships, which would more directly validate the method’s intended benefits. See the questions below.

### Questions
1. Is the same metric used for every numeral token, or are different metrics applied depending on token type (e.g., integers vs. floats vs. directions)?
2. Are metric-based losses applied to text-only tokens, or only to numerically meaningful ones?
3. Are such tokens, on which the metric/distance-based loss is applied, extracted manually?
4. While the paper shows improvements in downstream tasks, could the authors demonstrate that the model achieves more accurate numerical reasoning (e.g., better alignment between predicted and true distances)? The authors could consider a design of a controlled experiment where digits/numbers are extracted from data and the trained models are asked to evaluate the distances or any related tasks. The results are then compared with true numerical distance values to directly test metric reasoning performance.

The current score reflects the questions above. I will consider updating the rating once my concerns have been resolved.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper propose to learn the connections between discrete outputs, such the correlations between them can be captured during the learning process.

### Strengths
1. The reported results are marginally improved compared with existing methods.

2. The source code is produced.

### Weaknesses
1. The motivation of learning correlations among discrete concepts/semantics is not novel, which has been largely explored in the knowledge distillation works for a decade.

2. The proposed DIST^2 loss is just a combination of loss used in knowledge distillation and KL-divergence, which is also proposed in 'Distilling Knowledge from Graph Convolutional Networks. CVPR 2020'.

3. Overall, this work lacks of novelty and is with limited technical contribution.

### Questions
Please refer to the Weaknesses.

### Soundness
3

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
The work aims to incorporate an additional regularisation term into the loss of an LLM which more explicitly forces the model to learn a representation of distance (in terms of the metric used in the regularisation term). The distance regulariser is also phrased in terms of entropy-regularisation in policy optimisation. Extensive experiments are shown, including an ablation study, which consistently support the utility of the proposed regularisation term.

### Strengths
## Originality
The work is fairly original, particularly in the exact way they approach incorporating the distance metric. I'm particularly considering the discretised distance loss in this case which enables the KL divergence to work in a straight forward manner. The related work section at the end provides a nice overview of the literature and is very fair to other works in the space. I think contextualising the work in this manner should count towards originality, even if it does make the source of the ideas for the work more transparent.

## Quality
The experimental design and extensive nature of the experimentation is a large strength of this work for me. Particularly as the work is proposing a "framework" it is good to see such breadth of experimentation considered. The motivation and hypothesis of the work is clear and grounded well in the literature (to my knowledge). The results which are obtained do directly test the claims of the work and are interpreted fairly overall.

## Clarity
Overall the paper is well written, with clear tables and figures. The paper is structured well to support understanding and mathematical notation is clear, consistent and mostly intuitive. I also appreciate how the sections are structured in order of task complexity (as best they can be).

## Significance
Overall I think the work has the potential to inspire future work and does provide good results on the benchmarks. Once again the extensive literature review also supports the fact that this work has broad utility across a couple domains which supports its significance.

### Weaknesses
## Clarity
I find the manner that subsequences is introduces in Section 2.2 a bit unintuitive and requires some effort to parse. I'm puzzles that the subsequence needs to be sequential within the input sequence. I assume this is answered by the point: "...multiple elements are present within $s$, we limit our explanation to a singe $x$-subsequence here for clarity". I find that this is a bit too subtle of a statement to actual convey the fact that it could easily generalise (if indeed it can and so ease of explanation becomes the priority). For example, what would need to change to the formulation or equations to make it work for multiple sequences? What tasks are of this nature and why might a single sequence be a sufficient explanation? Does this require the permutation invariance of the LLM to work?

## Significance and Quality
I will ground these two sections as they share a common point. One of the primary issues for me is that the need for supervision on the distance metric is somewhat glossed over. Fundamentally the model is being given more supervision and so higher performance is expected. This limits significance to a degree, but this affect quality more for me as this should really be discussed. How easy it is to define metric spaces for a variety of problems is the determinant of the success and significance of the work and this should be more clearly acknowledged and discuss. The level of experimentation shown here does support that it is possible, but it is left to the reader to gauge and really this is where the conceptual insight of the work lies.

### Questions
I have listed a number of questions under weaknesses regarding the limitations of presenting the mathematical details using a single subsequence. I would appreciate if these could be answered.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on the field of LLM extended to non-linguistic domains (e.g., multimodal understanding, robotic manipulation, generative reward modeling) and addresses the key problem that traditional one-hot targets and cross-entropy loss ignore the metric relationships (e.g., coordinates, rotation angles, quantized embeddings) inherent in tokens. Motivated by the inefficiency of conventional fine-tuning (neglecting metric structure) and the instability of RL methods (sampling/rollout noise), the authors aim to enhance model performance in low-data regimes while maintaining compatibility with existing architectures . The core method, DIST2Loss, transforms continuous exponential family distributions derived from inherent distance metrics (e.g., Euclidean distance, RMSE) into discrete categorical targets, computes distance-aware loss via KL divergence, and fuses it with standard cross-entropy loss. Empirical results show that DIST2Loss consistently improves performance across diverse tasks with the most notable gains in data-scarce settings .

### Strengths
1. The manuscript addresses a critical limitation of discrete autoregressive models (neglect of metric token relationships) with a principled solution—DIST2Loss directly embeds metric structure into the target distribution without relying on extra data or architectural modifications. This fills a gap in extending LLMs to non-linguistic tasks where spatial/numerical relationships matter.

2. DIST2Loss is validated on five distinct tasks (meta linear regression, visual grounding, robotic manipulation, reward modeling, image generation), demonstrating its broad applicability. This cross-task consistency strengthens the credibility of the method. It also performs well in low-data regimes.

### Weaknesses
1. Insufficient details on hyperparameter $\tau$: The temperature hyperparameter $\tau$  controls the smoothness of the target distribution, but the manuscript only states "small values for digits and larger values for VQ-VAE vocabularies" without providing specific values or a systematic sensitivity analysis. 

2. I suggest the paper incorporates additional technical approaches for comparable distance perception methods. For instance, it could detail whether other methods achieve distance modeling through modifications to the loss function, adjustments to the target distribution, or the introduction of external modules. 

3. Does the default setting of $\alpha$ = 0.1 potentially lead to a scale imbalance problem? Is it possible to incorporate an adaptive scaling mechanism to prevent this?

4. The authors evaluate other capabilities solely after single-task fine-tuning (e.g., MMLU after fine-tuning for reward modeling). However, they do not validate performance in multi-task fine-tuning scenarios. 

5. The authors only validated general language abilities (MMLU) and did not cover non-linguistic fundamental abilities (e.g., image understanding capabilities). Including these aspects would significantly enhance the paper's quality.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
