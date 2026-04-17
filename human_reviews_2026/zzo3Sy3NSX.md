# Your Language Model Secretly Contains Personality Subnetworks

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Humans shift between different personas depending on social context. Large Language Models (LLMs) demonstrate a similar flexibility in adopting different personas and behaviors. Existing approaches, however, typically adapt such behavior through external knowledge such as prompting, retrieval-augmented generation (RAG), or fine-tuning.
We ask: do LLMs really need external context or parameters to adapt to different behaviors, or do they already have such knowledge embedded in their parameters?
In this work, we show that LLMs already contain persona-specialized subnetworks in their parameter space. Using small calibration datasets, we identify distinct activation signatures associated with different personas. Guided by these statistics, we develop a masking strategy that isolates lightweight persona subnetworks. Building on the findings, we further discuss: how can we discover opposing subnetworks from the model that lead to binary-opposing personas, such as introvert-extrovert? 
To further enhance separation in binary opposition scenarios, we introduce a contrastive pruning strategy that identifies parameters responsible for the statistical divergence between opposing personas. Our method is entirely training-free and relies solely on the language model's existing parameter space. Across diverse evaluation settings, the resulting subnetworks exhibit significantly stronger persona alignment than baselines that require external knowledge while being more efficient. Our findings suggest that diverse human-like behaviors are not merely induced in LLMs, but are already embedded in their parameter space—pointing toward a new perspective on controllable and interpretable personalization in large language models. Our code is available at https://github.com/Ruimeng-Ye/Persona.git.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The author introduces thousands of extra samples and then performs activation-guided pruning to obtain a sparse sub-network specialized for that persona. For opposing personas, a contrastive pruning method is designed to ensure the two sub-networks are mutually separate.

### Strengths
Compared to past prompt-based methods, this paper's approach of calculating a mask via pruning allows for the low-cost creation and switching of multiple personas within a single model.

This method possesses stronger interpretability, and I appreciate the author's detailed experiments, which explain why some personas are more difficult to separate.

### Weaknesses
However, I do not see the practical benefits. For example, I do not wish to obtain a fully introverted LLM. In fact, every user's needs are diverse. The method proposed in this paper lacks sufficient flexibility and cannot achieve dynamic, fine-grained control. I suggest the author could perhaps try a set of special synthetic persona experiments, such as "70% introversion + 30% thinking," to see if the current method would still be effective. Additionally, users often prefer to align personas on the newest, state-of-the-art models. However, this paper's method is not applicable to closed-source models, and I am unsure if it's possible to adjust the personality traits of these closed-source models via API-based control.

The author lacks discussion on whether this method affects the LLM's original performance on common tasks like AIME, HumanEval, MMLU, etc.

This paper's method is constrained by hyperparameters and calibration data. Different sparsity ratios exhibit varied performance, requiring additional costs to select the optimal sparsity. Furthermore, this method is not zero-shot; it requires thousands of samples, which is an extra cost. The reader is left unclear as to how sensitive this method is to the quality, quantity, and bias of the calibration data. Is it possible that the imperfect separation of different persona types is due to the data itself?

### Questions
Have you considered using mechanistic interpretability to find the LLM's identifiable internal computation paths?

### Soundness
2

### Presentation
1

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
This paper investigates whether LLMs already contain latent persona-specific capabilities embedded in their parameter space without requiring external knowledge. Inspired by the lottery ticket hypothesis, the authors propose a training-free method to extract lightweight persona subnetworks via structured activation-guided pruning. They also introduce a contrastive pruning strategy to enhance separation between opposing personas. The resulting subnetworks demonstrate improved persona alignment across several benchmarks, outperforming prompt and retrieval-based baselines. They find that diverse human-like behaviors are not merely induced in LLMs, but are already embedded in their parameter space.

### Strengths
* The paper is well-motivated. It is intuitive that pretraining can embed personality subnetworks in LLMs, and the proposed training-free pruning provides a practical way to approximate the upper bound of persona knowledge already encoded in the parameters.
* The proposed activation-guided and contrastive pruning framework is theoretically grounded in the lottery ticket hypothesis and activation-based interpretability, making it a principled way to isolate latent persona subnetworks already embedded in pretrained LLMs.
* The paper is validated across diverse persona benchmarks such as MBTI, AI Persona, RoleAgentBench, demonstrating consistent improvements over prompt- and RAG-based methods, with interpretable analyses of mask separability and sparsity ratios.

### Weaknesses
* The proposed framework essentially functions as an interpretability probe rather than a generative alignment method. Its real contribution lies in exploring the upper bound of persona encoding already latent in LLMs, not in improving persona expression. Therefore, directly comparing it with SFT is conceptually inconsistent. For an interpretability-oriented method, the most crucial evaluation should concern faithfulness—whether the discovered subnetworks truly correspond to the model's intrinsic persona representations.
* The paper does not explore how instruction tuning or model size might influence the encoding and separability of personas in LLMs.
If pruning is meant to expose existing persona structures, then it is essential for understanding how these structures vary before and after instruction tuning, or across different model scales of the same architecture.

### Questions
See Weaknesses.

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
This work discovers that an LLM subnetwork may represent a specific persona. By applying an extracted binary mask on the linear weights, the target persona can be emphasized in the outputs. A contrastive pruning algorithm is also proposed to disentangle the personas in the parameter space.

### Strengths
- The idea of identifying a subnetwork that represents a target persona is interesting.
- The method does not require explicit gradient-based training, which makes the overall process simple and interpretable.
- The provided analyses on the persona evaluation are extensive.
- The manuscript is well written and easy to follow.

### Weaknesses
### 1. Affect of Pruning on General Performance
 While the approach for identifying sub-networks linked to specific personality traits is compelling, the work does not address how pruning affects overall model performance. Including an evaluation of whether important downstream capabilities are improved -- or at least preserved -- would significantly strengthen the contribution.

### 2. Precise Mechanism of the Contrastive Pruning Algorithm
I am skeptical about the contrastive pruning algorithm because even when two personas are seemingly opposite to each other, I do not think that means that the subnetwork neuron set should be orthogonal. That said, it would help to understand the effect of this algorithm when the pair of personas is similar. For instance, try Power-Seeking vs. Wealth-Seeking (or maybe "desire-for-discreetly-acquiring-power" in the dataset) and compare the Power-Seeking performance with the performance reported in the manuscript. 

### 3. Figure Clarification
Figure 3 is a bit confusing when comparing the MBTIs with the base model. For example, for the "N" trait, why do the INFP, INFJ, ... traits have lower "N" dimension scores compared to the base model? Are the scores relative values?

### 4. Minor Points
- I suggest moving Figure 1 to page 3 or 4.

### Questions
- Why are the experiments done on the base model, and not the instruction-tuned models?

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
The paper proposes a novel framework for isolating persona-specialized subnetworks in LLMs via activation-guided pruning, without the need for additional training. The method demonstrates that distinct personas can naturally emerge as separate activation patterns within pretrained models. The authors employ a pruning strategy to extract persona-specific subnetworks, which leads to more efficient persona switching. Experiments show that the pruning method outperforms traditional techniques.

### Strengths
1. The idea that personas are embedded within the parameters of pretrained LLMs and can be extracted without additional training provides a fresh perspective on LLM personalization.

2. The contrastive pruning technique proves to be particularly effective in distinguishing opposing personas, which is a challenging aspect in persona modeling.

3. The method offers a training-free solution that is more computationally efficient than alternative techniques such as fine-tuning or RAG, requiring minimal additional resources.

### Weaknesses
1. While the method works well for some personas, there are instances where certain personality dimensions, like N/S and J/P from the MBTI dataset, show weaker separation, leading to less distinct personas. This limitation could be addressed with more dimension-aware or layer-aware techniques.

2. Results on Llama models show that the scalability of models to other architectures or domain-specific tasks is not fully explored. The authors should clarify how well this approach might generalize to other pretrained LLMs or tasks.

3. The method relies heavily on small calibration datasets, and it is better to focue on some larger.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
