# SaFT: Spotting Style Imitation and Filtering Content Interference for Zero-Shot LLM-Generated Text Detection

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6, 2

## Abstract
Large language models (LLMs) have achieved advanced text generation capabilities, necessitating the development of reliable LLM-generated text detection to prevent potential misuse.
However, current probability-based zero-shot detection methods face two critical challenges that reduce the detection accuracy of LLM-generated texts: the $\textit{style imitation challenge (SIC)}$ and the $\textit{content interference challenge (CIC)}$. 
The SIC arises as LLMs develop increasingly stronger abilities to mimic human writing styles, while the CIC occurs when surprising content characteristics interfere with probability analysis.
To address these challenges, we propose $\textbf{\textit{SaFT}}$, a novel framework built upon $\textit{Style-Oriented Instruction Prefix (SOIP)}$ to guide probability analysis for spotting style imitation and filtering content interference. Our framework introduces $\textit{SIC-Detection (SIC-D)}$ that spots style imitation by making style-imitating texts less unexpected through probability analysis conditioned on human-style instructions, and $\textit{CIC-Detection (CIC-D)}$ that filters content interference by difference analysis between probability distributions conditioned on contrasting style instructions, exploiting the insight that identical models exhibit equivalent content-related surprises.
The final detection score is composed of SIC-D and CIC-D components.
Extensive experiments demonstrate that SaFT consistently outperforms existing state-of-the-art methods, achieving improvements of 4.9\% in average AUROC and 20.4\% in average TPR @ 10\% FPR.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Large language models (LLMs) exhibit advanced text generation abilities, underscoring the need for reliable detection to mitigate misuse. However, existing zero-shot detectors struggle with two key issues: style imitation (SIC), where LLMs successfully mimic human writing styles, and content interference (CIC), where surprising content distorts probability signals. To tackle these challenges, the authors propose SaFT, a novel framework that employs a Style-Oriented Instruction Prefix (SOIP) to guide probability analysis. SaFT integrates two modules: SIC-Detection (SIC-D), which identifies style imitation through human-style conditioning, and CIC-Detection (CIC-D), which mitigates content interference via differential probability analysis between contrasting style instructions. The final score combines both components. Extensive experiments show that SaFT surpasses existing methods, improving average AUROC by 4.9% and TPR@10% FPR by 20.4%.

### Strengths
- Proposes a new detection approach that prefixes style-related instructions when computing probabilities in zero-shot detectors, allowing correct identification of LLM-generated texts even when they are human-mimicked or generated from special or surprising prompts.
- Comprehensive evaluation across a wide range of models and domains, as well as ablation studies of their proposed components
- Robustness analyses (top-p, text length, paraphrasing), although each evaluation is not fully comprehensive.

### Weaknesses
- **Questionable fairness of evaluation:** The detection setting in this paper is black-box detection, where it needs to select a proxy model to get probability distributions. From Appendix A.4, this paper seemingly adapts a default setting of each detector, and thus the proxy model is different for each detector. Based on previous findings [1,2], selecting which models for the proxy would heavily affects the detection performance. Therefore, for fair comparison, it would be necessary to use a common proxy model consistently for each detector. (For instance, why does the proposed method choose Llama-3.1-8B-instruct as a proxy?)
- **The unclear necessity of CIC-detection:** It is unclear how the value of β varies across samples. For instance, does β change for texts generated with special prompts compared to more general ones? If it remains constant, the necessity of introducing β itself is questionable. Moreover, Table 3 shows that the detection performance is substantially low when using only CIC-D, yet combining it yields higher performance than SIC-D. It would be helpful to clarify why this occurs.
- **The choice of instructions:** The motivation of using “Express ideas using concise sentences” and “Express ideas using detailed sentences” as instructions to represent human and LLM writing style is ambiguous. Moreover, the human (or LLM)-style instruction in Appendix C is not always the case for real-world scenarios. People will use more tailored and special prompts to mimic human-writing or prompting with few-shot human-written texts, for instance.

---
### References

[1] Mireshghallah et al. Smaller Language Models are Better Zero-shot Machine-Generated Text Detectors. EACL 2024.

[2] Dubois et al. MOSAIC: Multiple Observers Spotting AI Content. Findings of ACL 2025.

### Questions
There is a gap between the motivation of this study and the dataset construction. The motivating start point is that zero-shot detectors will degrade their performance on human-mimicked texts or creative and surprising texts via special prompts. However, this paper builds their test set by simply generating a continuation as a LLM-generated text from the prefix of a corresponding human-written text. It would be a better evaluation for this proposed method to use a dataset that aligns with the original motivation.

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
3

### Summary
- The paper proposes SaFT, a novel zero-shot LLM text detector designed to overcome two specific advanced failure modes: the SIC and the CIC, both of which attempt to make AI text more human-like.

### Strengths
- The paper's primary strength is its novel motivation - it clearly identifies and names two challenges that cause existing SoTA detectors to fail.
- The paper is also very easy to read and understand.

### Weaknesses
- In L236, the paper claims that "regular LLM-generated texts... are less affected by this conditioning". This assertion appears questionable. Conditioning an LLM on a human-style instruction $I^h_{SIC}$ should not intuitively make its "regular" outputs less probable or more surprising. Since regular LLM text already reflects an average human style inherent in its pretraining distribution, such conditioning would likely either decrease or increase perplexity (based on how well the model has been pre-trained); but it would not remain unaffected by this conditioning. If the goal is to identify differences between human text, human-style LLM text and regular LLM text, this method would be flawed.
- The hypothesis behind SIC-C appears fragile and highly dependent on the specific choice of $I^h_{SIC}$. If model developers were to craft or optimize this instruction carefully, it could yield even lower perplexity outputs, even for human-written text, thus invalidating the detector’s assumption. Since $I^h_{SIC}$ is realistically a black-box component, this would reduce robustness of the proposed method. The authors should also provide sensitivity analysis across diverse instruction variants.
- Formally, human-style LLM text can also be thought of the model sampling from regions of the probability space that are uncommon but represent human-like diversity. Prior work [1] demonstrates that such diversity can be achieved through alternative means, such as diverse prompts or high-temperature sampling. Since high-temperature sampling also increases diversity and reduces predictability, it could interfere with the detectors. Evaluating the detector's behavior on texts generated under varying temperature settings would be helpful here.
- In Eqn. 5, the inequality direction appears incorrect. Given that $\alpha(t)$ is expected to be low and $\beta(t)$ is expected to be high for LLM-generated text, $\text{SaFT}(t)$ should naturally be lower for such samples. Therefore, the condition should likely be $< \tau$, not $> \tau$. The authors should revisit and verify this formulation.
- It is unclear how the two proposed detectors collectively contribute to distinguishing between LLM-generated and human-written texts. The paper suggests they are primarily effective in identifying human-style LLM outputs, but not for standard LLM vs. human text discrimination. Clarification or additional justification is required.
- The experimental setup (Appendix B.2) evaluates only human-style LLM texts and genuine human texts. The authors should include pure LLM-generated text in order to avoid any bias. 


---

Minor Errors:
- L88: "poduce" to "produce"

[1]  Zhang et al. Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity. arXiv:2510.01171.

### Questions
- It is unclear why the authors specifically chose to focus on SIC and CIC. There exist many other possible forms of evasion - such as hybrid (AI and human text combined) that could similarly challenge detection systems. Can the authors justify why they focus specifically on these problems?

All the questions and suggestions have been listed in the weaknesses. I am willing to increase my score if the authors address these concerns.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Spotting Style Imitation and Filtering Content Interference, a novel zero-shot LLM-generated text detection framework designed to overcome two major challenges faced by existing probability-based detection methods --- Style Imitation Challenge and Content Interference Challenge.  Experiments across six LLMs and four text domains (news, scientific abstracts, biomedical QA, and reviews) show SaFT consistently outperforms previous baselines.

### Strengths
1. The decomposition into SIC-D and CIC-D is well-motivated and theoretically grounded in content–style disentanglement.
2. The paper was well organized. Figures and equations (especially Fig. 2) provide clear intuition for the SOIP mechanism.
3. The empirical results are strong, with consistent improvements across diverse datasets and model families.

### Weaknesses
1. The method in [1] is highly related, as it shares the same motivation of distinguishing LLM- vs. human-style writing. Adding it as a baseline would strengthen the evaluation.

2. While the paper motivates its instruction design based on cognitive load theory (Appendix C), the conceptual leap from “humans write concisely” versus “LLMs write elaborately” to the specific instruction templates used is insufficiently justified. Is there any quantitative metric supporting this design choice?

3. In Equation (4), the paper combines α(t) and β(t) using a ratio. Why was division chosen over alternatives such as subtraction, weighted summation, or other combination strategies? A short explanation would clarify the design rationale.

4. Discussion about failure cases. It would significantly strengthen the paper to include a statistical analysis of failure recovery — for example: 1. How many samples that other baseline methods misclassified were correctly detected by SaFT?  2. What types of samples still remain difficult for SaFT to classify correctly?

[1] Wu, Junchao et al. “Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore.” Proceedings of the 31st International Conference on Computational Linguistics, 2025.

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
4

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
The paper proposes SaFT, a probability-based zero-shot detection method for detecting Machine generated text which is designed to address two challenges: style imitation challenge and content interference challenge. The paper proposes SIC-Detection , aiming to targets texts that mimic human writing through style imitation by conditioning evaluation on explicit style instructions, and a  CIC-Detection to filter content interference.

### Strengths
The paper addresses the important and timely challenge of detecting AI-generated text. The problem is well-motivated, the proposed approach is clearly explained, and the paper is overall well-structured and easy to follow.

### Weaknesses
The paper relies on predefined “style-oriented instruction prefixes” to represent human and machine styles. This assumes such styles can be described accurately, which will not hold in diverse or nuanced real-world writing contexts. Note that if the human and machine styles were to be accurately characterized, they could have been directly used to detect AI generated texts.
The proposed approach is also based on prompting, making it susceptible to prompt design and potentially not robust. Finally, in the experiments, more baseline detectors should be included from different categories of AI-generated text detection to make the results more comprehensive.

### Questions
Please refer to the weakness section

### Soundness
1

### Presentation
3

### Contribution
2
