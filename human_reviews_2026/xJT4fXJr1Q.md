# Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Large vision-language models (LVLMs) have achieved substantial advances in multimodal understanding. However, when presented with \textcolor{black}{challenging or distribution-shifted inputs}, they frequently produce unreliable or even harmful content, \textcolor{black}{such as hallucinations or toxic responses. We refer to such misalignments with human expectations as \emph{misbehaviors} of LVLMs, which} raise serious concerns for their deployment in critical applications. \textcolor{black}{Existing research have disclosed that such misbehaviors are closely linked to model uncertainty. We find they primarily stem from two distinct sources of epistemic uncertainty: internal contradictions (conflict) and the absence of supporting information (ignorance).} While existing uncertainty quantification methods typically capture only total predictive uncertainty, they struggle to distinguish between these underlying causes. To address this gap, we propose Evidential Uncertainty Quantification (EUQ), \textcolor{black}{a training-free framework that explicitly decomposes epistemic uncertainty into conflict (CF) and ignorance (IG)}. Specifically, we interpret features from the model output head as either supporting (positive) or opposing (negative) evidence. Leveraging Dempster-Shafer Theory of belief functions, we aggregate this evidence to quantify internal conflict and knowledge gaps within a single forward pass. We extensively evaluate EUQ across four misbehavior categories, including hallucinations, jailbreaks, adversarial vulnerabilities, and out-of-distribution (OOD) failures using state-of-the-art LVLMs. Experimental results demonstrate that EUQ consistently outperforms strong baselines, \textcolor{black}{achieving relative improvements of up to 10.5\% in AUROC.} \textcolor{black}{Our evaluation further reveals} that hallucinations correspond to high internal conflict and OOD failures to high ignorance. \textcolor{black}{Furthermore, a layer-wise evidential uncertainty dynamics analysis provides a novel perspective on the evolution of internal representations.} The source code is available at \url{https://github.com/HT86159/EUQ}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Evidential Uncertainty Quantification (EUQ), a method to detect misbehaviors in Large Vision-Language Models (LVLMs) by quantifying two types of epistemic uncertainty: conflict (CF) and ignorance (IG). Using Dempster-Shafer Theory, EUQ models output-layer features as evidence, enabling efficient detection of hallucinations, jailbreaks, adversarial attacks, and OOD failures in a single forward pass. Extensive experiments on four LVLMs show that EUQ outperforms existing baselines in AUROC and AUPR, with insights into layer-wise uncertainty dynamics.

### Strengths
1. Good writing.
2. The proposed method appears to have promising performance.
3. The proposed indicators were effective across different series, validating the effectiveness of the method.

### Weaknesses
1. The different types of epistemic uncertainty (CF and IG) quantified are all effective for hallucination detection.
2. There is a lack of baseline data for some hallucination detection in 2025.
3. Some work on the detection of LVLMs using evidence theory has not been discussed fully.
4. It is recommended to test on a larger model, such as 72B, because there are often inconsistencies between small and large models.
5. Can CF and IG complement each other to improve the final detection performance?
6. It's best to separate citations from the main text, for example, using \citep.

### Questions
See Weaknesses.

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
This paper proposes Evidential Uncertainty Quantification (EUQ), a method for detecting misbehaviors in LVLMs including hallucinations, jailbreaks, adversarial vulnerabilities, and out-of-distribution failures. The authors also investigate the conflict and ignorance uncertainty, and argue these two forms of epistemic uncertainty are sources for the misbehaviors.

### Strengths
Interesting paper to read as it classifies different types of misbehaviors in VLMs and it is observed that CF/IG can be used to distinguish different types of misbehaviors in VLMs.

### Weaknesses
1. Although Figure 4 and the appendix visualizations distinguish misbehavior types, there is no deeper linguistic or visual semantic analysis explaining why certain errors yield high CF or IG.
2. Thresholding (which could vary across LVLMs, datasets, or misbehavior categories) would have to be determined externally. Additionally, since the authors propose metrics to evaluate misbehaviors in VLMs and make observations, the size of the datasets and the chosen VLMs (four VLMs with ≤ 8B parameters) appear somewhat small. What model is evaluated in Figure 5 for model size analysis is not clear.
3. I also think this paper would benefit from including more baselines for hallucination/jailbreak detection. Predictive entropy is a quite simple baseline.

Minor: typos on line 428 for 'adn'

### Questions
See weakness.
1. Can you provide more linguistic or visual semantic analysis explaining why certain misbehaviors yield high CF or IG?
2. Can you compare IG and CF with more baselines besides SE (e.g., POPE, HiddenDetect...) / add more datasets/LVLMs to validate the generalization ability of these two metrics and the validity of the observations?

[Evaluating Object Hallucination in Large Vision-Language Models](https://aclanthology.org/2023.emnlp-main.20/) (Li et al., EMNLP 2023)

[HiddenDetect: Detecting Jailbreak Attacks against Multimodal Large Language Models via Monitoring Hidden States](https://aclanthology.org/2025.acl-long.724/) (Jiang et al., ACL 2025)

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors present a training-free method to interpret the uncertainty of a VLM by computing the per-token conflict and ignorance uncertainty in the model's predictions. The authors show that their method outperforms other uncertainty interpretation methods with moderate compute overhead. The paper's methods are evaluated across hallucination, jailbreaking, and out-of-distribution generalization tasks at different model scales.

### Strengths
The main strengths of the paper are:
1) Disaggregating the uncertainty into conflict and ignorance uncertainty to interpret the uncertainty of a VLM. This allows the authors to measure uncertainty in different contexts (e.g. hallucination, jailbreaking, out-of-distribution generalization).
2) The experiments are thorough and conducted on multiple model families and expressly evaluated at many scales.

### Weaknesses
There are no major weaknesses in the paper. However, in Figure 1 in the paper, the authors show an illustrative example of measuring uncertainty in chain-of-thought reasoning. It would be useful to see examples of how the authors' proposed method can identify uncertainty in these reasoning traces. Currently the authors only evaluate their method on benchmarks which often only measure uncertainty on shorter token sequences.

### Questions
1) The paper does not explicitly introduce any mechanism to focus uncertainty on key or semantically important tokens. Would this make it unsuitable for tasks such as dense image captioning or long reasoning traces where only some tokens are key to measuring uncertainty?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies detecting the misbehaviors of VLMs, in particular, adversarial inputs, Jailbreak inpus, OOD inputs, and model hallucination. The method then extracts the logits from the VLM and process it using "Evidence Theoryt" and get an uncertainty quantification score. The authors then utilize this score to perform misbehavior detection and observe improved performance compared with baseline approaches on. a variety of methods.

I have to say I really cannot understand the method. The paper's technique involves too many concepts I have never heard about:

- Dempster-Shafer Theory
- Aasic belief assignment, BBA
- Degree of conflict
- conflict (CF) and ignorance (IG)
- Least Commitment Principle (LCP) 

I don't think this is the author's problem since this is all due to my lack of knowledge. However, my reviews may not provide lots of useful inputs. I would also appreciate it if the authors could explain things in simpler language, e.g. through an algorithm block, such that I could understand how things are implemented in practice.

### Strengths
- The method seems to be very mathematically rigorous

- The paper considered lots of datasets and models.

### Weaknesses
- The proposed method does not seem to have too much novelty; I can't see why the proposed method is specific to VLM or why it cannot be applied on e.g. BERT, ResNet, LLM.

- It is unclear if the method is applicable to closed-source model since the method requires access to logits.

- The choice of baseline is little bit confusing, semantic entropy is for uncertainty quantification over free form generation, but many tasks here only require a single word as the output (if I understand correctly). What is the point of performing clustering here?

### Questions
- Why is the column title near lines 393 and 394 "Method" rather than "model"?

- "For multiple-choice and yes/no tasks, correctness is assessed using ROUGE-L Lin (2004) (threshold > 0.5)." I don't understand why ROUGE is needed here; isn't accuracy applicable?

- The prompt forces the model to hallucinate (line 1229)
```
Please check whether the following description matches
the picture content. Just answer yes or no without explanation.
<image caption>, 
```
Has the author tried providing the model with the option to reject? Some recent work (e.g. https://arxiv.org/abs/2505.11804) shows that if prompted properly.

### Soundness
3

### Presentation
3

### Contribution
3
