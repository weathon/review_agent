# Diversity Boosts AI-Generated Text Detection

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2

## Abstract
Detecting AI-generated text is an increasing necessity to combat misuse of LLMs in education, business compliance, journalism, and social media, where synthetic fluency can mask misinformation or deception. While prior detectors often rely on token-level likelihoods or opaque black-box classifiers, these approaches struggle against high-quality generations and offer little interpretability. In this work, we propose DivEye, a novel detection framework that captures how unpredictability fluctuates across a text using surprisal-based features. Motivated by the observation that human-authored text exhibits richer variability in lexical and structural unpredictability than LLM outputs, DivEye captures this signal through a set of interpretable statistical features. Our method outperforms existing zero-shot detectors by up to $33.2$% and achieves competitive performance with fine-tuned baselines across multiple benchmarks. DivEye is robust to paraphrasing and adversarial attacks, generalizes well across domains and models, and improves the performance of existing detectors by up to $18.7$% when used as an auxiliary signal. Beyond detection, DivEye provides interpretable insights into why a text is flagged, pointing to rhythmic unpredictability as a powerful and underexplored signal for LLM detection.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes DivEye, a lightweight and interpretable framework for detecting AI-generated text using surprisal-based statistical features.
By modeling how token-level unpredictability fluctuates within text, DivEye achieves strong zero-shot detection performance across domains and language models, outperforming prior detectors while remaining efficient and model-agnostic.

### Strengths
DivEye achieves competitive AI-text detection performance across models and domains.
It introduces an interpretable detection method based on surprisal diversity, capturing rhythmic and temporal fluctuations in unpredictability —a cognitively meaningful signal that distinguishes human and AI text.
It achieves strong zero-shot, model-agnostic, and cross-linguistic generalization, functioning efficiently without fine-tuning and even enhancing the performance of other detectors by up to 18.7%.

### Weaknesses
- The features used in this study have already been identified in prior research, and the performance gains mainly come from combining them — the work offers limited new scientific insight.
- Whether “interpretable statistical features” are truly interpretable is highly subjective. If the features in this paper are considered interpretable, then existing methods using probability or curvature should also qualify as interpretable.

### Questions
- Why are the commercial and open-source models used in the experiments generally outdated?

### Soundness
2

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
4

### Summary
The author’s propose DivEye, a detection approach which goes beyond the simple aggregate metrics of most zero-shot detection methods. Instead, they look at distributional features, first order, and second order features and they show that an XGBoost classifier trained on these features is good across a range of testing scenarios.

### Strengths
* S1 - The paper goes beyond the simple aggregate metrics of most zero-shot detection methods. Instead, they look at distributional features, first order, and second order features and they show that an XGBoost classifier trained on these features is quite good.
* S2 - Evaluations across many different scenarios is performed, notably using the testbeds of the MAGE dataset and the RAID dataset.

### Weaknesses
* W1 - In the “Relevance of DivEye’s Features” ablation, what would’ve been more convincing is training XGBoost models only on subset of features. The post-hoc importance of the full model (the one that uses all features) doesn’t necessarily imply that all the features are indeed necessary, perhaps the model could’ve found a solution just as good without them.
* W2 - It is uncertain whether the approach improves upon the baselines when the tolerance of false-positives is low, a realistic scenario when detectors are applied. To evaluate this, the authors should evaluate the AUROC at lower FPR values, such as 1%, this has become standard in various detection works such as  https://arxiv.org/pdf/2405.07940, https://arxiv.org/pdf/2401.12070 and https://arxiv.org/pdf/2401.06712
* W3 - In Table 1, all the methods compared are zero-shot in that they haven’t been trained, but DivEye does train an XGBoost classifier, correct? It would’ve been interesting to see how the MAGE Longformer (which is trained) fairs in these testbeds. Looking at the MAGE Longformer baselines in the original paper (https://arxiv.org/pdf/2305.13242) it looks like it out-performs DivEye on most testbeds. This is my main concern with this paper, as it does seem that the MAGE Longformer baseline was left out unfairly.

### Questions
* Q1 - In Table 2 DivEye shouldn’t be bolded right? Moreover, DivEye requires training a classifier and hence is not “zero-shot” in the true sense of not having done any training on relevant data. Binoculars and FastDetectGPT are true zero-shot methods.
* My main concern comes from taking a look at the original MAGE paper and looking at the results across the testbeds it looks like their Longformer approach performs quite well, and even outperforms Diveye. It's unclear why the author's left this method out. Addressing this concern would significantly raise my score.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors claim to introduce a new method for enhancing existing LLM detectors using the token-level surprisal high-level statistics, computed according to a proxy LLM, combined with an existing LLM detection method, with the aggregate statistics from both the author's method and the existing LLM detection methods being fed into a trained binary classifier providing the output text detection. Authors then demonstrate the superior performance of their method compared to the SotA LLM detectors.

### Strengths
LLM detector enhancement work remains relevant to the real-world use and misuse of LLMs. Authors demonstrate empirically an improvement in the detection performance of their method on a known, hard real-world benchmark - RAID, indicating that their method is an improvement. Similarly, the proposed method's resilience to attacks and multilingual performance are both impressive.

### Weaknesses
I am not convinced of the novelty of the author's contribution. The proposed metric authors use - "per-token surprisal" - seems to be strictly equivalent to per-token perplexity, a metric that has been historically the most widely used for LLM text detection, introduced as early as 2019 in [1]. The authors seem to suggest that perplexity of text is calculated as an average (L184-185; citation is off as well), which is not the case. It does not help that the paper authors link in their mathematical definition of surprisal - (Kuribayashi et al., 2025) - does not actually define a suprisal metric or compare it to the perplexity.

As such, their proposed method becomes an ensemble classifier of an existing detection method and a perplexity-based detector, which are indeed expected to perform better than single detectors. However, even the better performance of the ensemble detector does not seem to be properly proven, due to the use by authors of the AUROC and custom metrics, rather than TPR @ fixed low FPR, which is representative of real-world use of LLM detectors [2]. It is commonly adopted in the existing literature. In fact, most recent zero-shot classifiers are optimized for it, potentially sacrificing AUROC and other less relevant metrics.

[1] Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019). The Curious Case of Neural Text Degeneration. ArXiv, abs/1904.09751.
[2] Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramèr, F. (2021). Membership Inference Attacks From First Principles. 2022 IEEE Symposium on Security and Privacy (SP), 1897-1914.

### Questions
What is the difference between the per-token surprisal and per-token perplexity?

### Soundness
2

### Presentation
3

### Contribution
2
