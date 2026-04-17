# HAPDA: A Human-Machine Predictive Discrepancy Adapter for AI-Generated Text Detection

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 4, 2

## Abstract
Recent advances in large language models (LLMs) have enabled them to generate text with increasingly human-like linguistic styles, posing significant challenges for AI-generated text detection (AGTD). Mainstream zero-shot AGTD methods primarily compute token-level AI-likeness scores from a machine-centric perspective represented by proxy models, and treat all tokens equally in the overall detection score calculation. However, these methods overlook predictive discrepancies between humans and LLMs in interpreting the same text. Our key intuition is that tokens exhibiting greater divergence in human and machine predictions offer stronger cues for authorship attribution. To address this limitation, we propose \textbf{HAPDA}, a \underline{h}uman-m\underline{a}chine \underline{p}redictive \underline{d}iscrepancy \underline{a}dapter for the AGTD task. HAPDA consists of (i) a joint fine-tuning strategy for training paired human and machine preference models, and (ii) a discrepancy-aware reweighting mechanism to calibrate token-level detection scores in downstream detectors. Extensive experiments across multiple datasets demonstrate that HAPDA consistently and significantly improves the performance of five representative baselines under diverse evaluation settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a novel framework called HAPDA, a human-machine predictive discrepancy adapter designed to improve the accuracy of zero-shot AI-generated text detection  methods. Current detection methods, often referred to as MeanZero, average token-level scores but overlook the significant predictive discrepancies between human- and machine-generated text interpretations. HAPDA aims to address this by introducing two main components: a joint fine-tuning strategy, which  trains models favoring human- and machine-written text, and a discrepancy-aware reweighting mechanism, which  assigns higher importance to tokens where human and machine predictions diverge. Experiments across various settings, including white-box, black-box, and adversarial attacks, demonstrate that integrating HAPDA consistently and significantly enhances the performance of several representative  AI-generated text detection  baselines.

### Strengths
- Addresses AI-generated text detection (AGTD) from a joint human and machine predictive perspective.

- Models predictive discrepancies between human and machine predictions to provide essential token-level reweighting for detection scores

- Improves the detection performance (AUROC scores) of five representative MeanZero baselines under diverse evaluation settings

-  HAPDA-enhanced zero-shot detectors achieve AUROC scores comparable to or even surpassing the latest advanced zero-shot detectors (like DNA-GPT, Raidar, and Lastde)

### Weaknesses
1.  HAPDA requires two auxiliary models, to compute necessary aspects like token probability disagreements and uncertainty , which can lead to increased  runtime and memory overhead during training and inference. It is recognized by the authors that techniques like quantization and LoRA are needed to mitigate this issue, which however might affect performance accuracy and flexibility.  

2. A performance comparison between the enhanced zero-shot detectors with HAPDA and a fine-tuned baseline detector  on the same human and machine text pairs used for HAPDA-finetune would be important.  
Although the   focus of the work is on HAPDA’s utility strictly on zero-shot detectors, HAPDA uses a joint fine-tuning strategy (HAPDA-Finetune) based on a labeled corpus (human-machine text pairs) to train the preference models. But the paper does not provide results comparing HAPDA-enhanced zero-shot detectors against a common, fully supervised detection baseline (like a fine-tuned RoBERTa or another transformer model) trained and evaluated on the same fine-tuning data.  Therefore, while the comparison may be outside the scope defined by the authors (enhancing MeanZero methods), its absence is a valid point of critique regarding the method's overall comparative standing against all available detection paradigms.

### Questions
See above Wekanesses and consider addressing point 2.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The author’s introduce HAPDA, a framework for re-weighting the token-level detection scores of zero-shot detectors. The strategy involves fine-tuning two LLMs, M_{human} and M_{machine}. M_{human} is fine-tuned to prefer human-written text, while M_{machine} is fine-tuned to prefer machine-generated text. Moreover, the authors include a loss to encourage distinctiveness on the same input. These models are then used to re-weigh the token-level detection scores of zero-shot detectors.

### Strengths
* S1 - The authors propose a detector agnostic framework by which to improve the performance of MeanZero detectors.
* S2 - The approach is interesting, in particular moving from naive averages of a MeanZero metric to more informed averages is a good idea, and it’s well motivated.

### Weaknesses
* W1 - The approach requires fine-tuning of two LLMs, and at inference it requires that one evaluates three LLMs.
* W2 - Baselines - The paper would’ve benefited from stronger baselines to HAPDA. (1) One could consider a baseline (or ablation) where M_{machine} = M_{proxy}. That is, only train M_{human}. (2) Another possibility is to let the weights be the inverse of the softmaxed perplexities. That is, first run the LLM through the sample, get the perplexity for each token, apply the softmax operator so that every value lies between 0 and 1, and then take the inverse (1 - softmax(PPL_i)) which would up-weight the score for tokens which are unlikely under the LLM.
* W3 - It is uncertain whether the approach improves upon the baselines when the tolerance of false-positives is low, a realistic scenario when detectors are applied. To evaluate this, the authors should evaluate the AUROC at lower FPR values, such as 1%, this has become standard in various detection works such as  https://arxiv.org/pdf/2405.07940, https://arxiv.org/pdf/2401.12070 and https://arxiv.org/pdf/2401.06712
* W4 - The authors should evaluate the robustness against paraphrasing attacks which are well known to degrade detector performance: https://arxiv.org/abs/2303.13408. I believe the RAID benchmark already contains paraphrases, and even more adversarial attacks that the authors could evaluate on. 
* W5 - Evaluations with Binoculars (https://arxiv.org/pdf/2401.12070) are missing. This approach normalizes perplexity scores by re-weighting them with another LLM, which seems related to this approach.

### Questions
* Q1 - In 4.2.4, was the value of lambda chosen on a separate validation set?
* My main concerns are W2, W4, and W5, if those concerns are adressed my score would increase.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces HAPDA, a novel adapter for improving AI-generated text detection. HAPDA consists of (i) a fine-tuning strategy for training human-/machine-preference models  that exhibit stronger preferences for human, resp. machine, generated texts, and (ii) a calibration method for assigning higher weights to more discriminative tokens during the detection process. HAPDA has been evaluated on multiple white-box and black-box settings, showing performance improvements for the underlying detectors.

### Strengths
- The proposed approach overcomes the uniform token weighing "issue" by better leveraging single-token informativeness in machine-generated text detection.
- Proposing an adapter that works with existing detectors is valuable as it might make them stronger rather than obsolete.
- The experimental validation of HARPDA suggests it leads (in most cases) to tangible improvement on the base performance of the underlying detectors. Furthermore, HARPDA turns out to be robust to a set of adversarial attacks.

### Weaknesses
- HARPDA requires training two preference models, this is costly, and might affect the practical usability of the proposed approach as well as latency.
- Related to the previous point, the training data might introduce some domain dependencies that could affect out-of-domain generalization of the overall framework. Some additional investigations on this point are needed. 
- As training the preference models requires learning stylistic differences between humans and machines, I wonder if simply fine-tuning encoder models for the detection task would provide a stronger baseline than HARPDA. These are missing and are often reported as state-of-the-art detectors. 
- The set of statistical baselines lacks some relevant works like Binoculars (Hans et al., 2024), which is cited as a related work but not used in the experimental setup.

### Questions
See W2 to W4.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a framework for AI generated text detection which identifying the more discriminative feature between human and AI text.

### Strengths
The paper proposes a framework for AI generated text detection to improve its performance. The problem addressed in this paper is extremely important and timely. The proposed method demonstrates good performance.

### Weaknesses
The proposed framework is quite complicated, and the paper does not clearly justify the purpose or necessity of its different components. The paper claims to incorporate the human perspective, but it is unclear what this means in the context of AI-generated text detection or why it is necessary for the problem. Overall, the writing requires significant improvement to enhance clarity and readability. The proposed framework also appears incremental and is heavily inspired by DPO. In addition, the paper should include more baselines and different categories of AI-generated text detectors to make the results and comparisons more generalizable.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2
