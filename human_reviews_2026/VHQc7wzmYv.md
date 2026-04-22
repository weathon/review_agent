# TokUR: Token-Level Uncertainty Estimation for Large Language Model Reasoning

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
While Large Language Models (LLMs) have demonstrated impressive capabilities, their output quality remains inconsistent across various application scenarios, making it difficult to identify trustworthy responses, especially in complex tasks requiring multi-step reasoning.
In this paper, we propose a **Tok**en-level **U**ncertainty estimation framework for **R**easoning (**TokUR**) that enables LLMs to self-assess and self-improve their responses in mathematical reasoning.
Specifically, we introduce low-rank random weight perturbation during LLM decoding to generate predictive distributions for token-level uncertainty estimation, and we aggregate these uncertainty quantities to capture the semantic uncertainty of generated responses.
Experiments on mathematical reasoning datasets of varying difficulty demonstrate that TokUR exhibits a strong correlation with answer correctness and model robustness, and the uncertainty signals produced by TokUR can be leveraged to enhance the model’s reasoning performance at test time.
These results highlight the effectiveness of TokUR as a principled and scalable approach for improving the reliability and interpretability of LLMs in challenging reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the TokUR method, which injects low-rank weights to approximate the weight posterior distribution in Bayesian modeling of LLMs. During the decoding stage, it captures uncertainty in the model, leveraging quantized uncertainty to simultaneously estimate the semantic uncertainty of generated responses. Experimental results demonstrate that the proposed method produces uncertainty estimates that correlate well with task difficulty, and further show its effectiveness in detecting erroneous reasoning paths and improving answer quality.

### Strengths
1. This paper introduces a novel approach for estimating uncertainty in long-form reasoning and demonstrates its effectiveness.
2. The paper is clearly written and easy to follow.
3. The experimental design is fairly comprehensive.

### Weaknesses
1. The authors decompose uncertainty into Aleatoric Uncertainty (AU) and Epistemic Uncertainty (EU). However, the definition of AU appears to correspond more to uncertainty in the input, while EU is more related to uncertainty in the model output. Since EU focuses on the model’s prediction behavior, it should theoretically better reflect the model’s capability issues. Nevertheless, the experimental results do not support this, and the paper does not provide analysis or interpretation of the three uncertainty estimates. I believe such analysis should be included.

2. The expressions for Query-Level Uncertainty and Response-Level Uncertainty both involve (y), and the definitions appear highly similar, which may cause confusion or misunderstanding. I suggest the authors explicitly clarify the difference and explain the rationale behind these definitions to avoid ambiguity for readers.

3. Assumption 3.1 ignores temporal correlations among parameters. The impact of this approximation needs to be justified either qualitatively or quantitatively to demonstrate the validity of the assumption.

4. Some important related work is missing in the discussion. Many existing studies on uncertainty estimation and LLM reasoning theory are not referenced. For example:
    *  *Position: Uncertainty Quantification Needs Reassessment for Large Language Model Agents* (ICML 2025), which discusses the relevance of uncertainty estimation.
    * *A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning* (NeurIPS 2025), which provides theoretical insights into reasoning errors in LLMs to improve reasoning accuracy.
    * *From Calibration to Collaboration: LLM Uncertainty Quantification Should Be More Human-Centered* (arXiv:2506.07461), which analyzes the limitations of existing LLM uncertainty estimation approaches.

### Questions
See weaknesses above.

### Soundness
2

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
3

### Summary
This paper introduces the TokUR framework for uncertainty quantification in large language models (LLMs), targeting complex, multi-step mathematical reasoning tasks. TokUR estimates uncertainty at the token level by introducing low-rank random weight perturbations during decoding, enabling principled decompositions into aleatoric and epistemic uncertainty for each token and entire responses.

### Strengths
- Most technical concepts are well-explained and visualized, aiding algorithmic understanding.

- The approach itself is elegant.

- It addresses a key practical need for safe and reliable LLM deployment, especially where confidence calibration is critical.

### Weaknesses
- Experiments focus on mathematical reasoning; application to verbal reasoning, open-ended tasks, or black-box LLMs is not shown.

- Repeated sampling during inference, while compatible with vLLM, is still non-trivial and could limit deployment in latency-sensitive use cases.

### Questions
- How does TokUR perform on open-domain, verbal, or dialogue tasks?

- How much cost does in incur to get perturbed weights?

- How do these perturbed weights affect the performance of the model?

### Soundness
3

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
3

### Summary
The paper introduces TokUR, a framework that lets LLMs self-assess and self-improve on multi-step mathematical reasoning by estimating token-level uncertainty during decoding. TokUR applies low-rank random weight perturbations, producing predictive distributions for each generated token. These per-token signals are decomposed into aleatoric and epistemic uncertainty and then aggregated to the sequence level to capture a response’s semantic confidence. Compared to prior query-level methods and response-level methods, TokUR offers a more fine-grained and accurate uncertainty estiomation. Experiments show that TokUR’s uncertainty is strongly correlated with correctness and robustness, flags incorrect reasoning paths, selects higher-quality solutions among candidates.

### Strengths
1. Strong theoretical ground. 

The paper’s theoretical backgrounds appear solid and thoughtfully developed. Assumptions are stated clearly.

2. Clear positioning vs. prior work. 

The manuscript does an excellent job for introducing the limitations of prior methods (query-level and response-level approaches), making it easy to understand the gap this work fills. 

3. Readable and well organized. 

The paper is clearly written and easy to follow, with clear notations.

### Weaknesses
1. On task scope

It seems that the framework is “not limited to mathematical reasoning”. Is there a reason that the paper should focus especially on mathematical reasoning? A few non-math settings would support the generalizability of the framework.

2. On cost analysis

While the cost is the common challenge of uncertainty estimation, it seems that there is no thorough cost analysis in the paper, compared to prior methods like query-level uncertainty estimation. Also, how does the estimation technique used in Section 3.3 help reduce cost? 

3. On model diversity

Evaluating two models from the same Llama family seems insufficient to support the correlation between accuracy and token-level uncertainty estimation. Including a few more models from different family and sizes is encouraged to reduce bias in experiments and to strengthen the claim.

### Questions
1. It would be helpful to analyze the length-robustness of your token-level uncertainty estimates. As responses grow longer, how do the estimation errors (or bounds) evolve?

### Soundness
3

### Presentation
3

### Contribution
2
