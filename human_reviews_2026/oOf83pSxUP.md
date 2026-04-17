# Old N-Grams Never Die:Towards Identifying LLMs-Generated Text using Antique N-Grams

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 6

## Abstract
The proliferation of large language models (LLMs) has triggered an influx of AI-generated content, making robust detection of such content paramount for maintaining academic, journalistic, and regulatory integrity. However, the community has largely overlooked a time-tested resource that classical n-gram models, trained exclusively on human-authored corpora, may serve as a de facto gold standard for identifying machine-generated writing. In this paper, we build upon well-trained pre-AI N-Gram models to form the backbone of a lightweight AI-text detection system called \textbf{GramGuard}. Specifically, by generating paraphrased variants via temperature-controlled decoding from LLMs, we measure the shifts in log-likelihood, entropy, and token frequency variance between original texts and perturbed versions. These \emph{delta} features then feed into an ensemble classifier to yield interpretable decisions about authorship. Extensive experiments on PubMed, WritingPrompts, and XSum demonstrate that \textbf{GramGuard} matches or exceeds state-of-the-art detectors in performance and robustness. Our findings reaffirm the enduring value of pre-AI n-gram models and introduce a scalable, transparent solution for AI-text detection.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces GramGuard, a supervised AI text detector that uses pre-LLM n-gram models as a gold standard for human writing. The method works by paraphrasing a given text and measuring the shift in its log-likelihood, entropy, and token variance - which is then fed into a classifier.

### Strengths
- This work is very well motivated and highlights a curious question in this domain.

### Weaknesses
- The submission violates the double-blind review policy. The authors should anonymize their code and model releases (AnonymousGitHub) and avoid providing GitHub and HuggingFace repository links that can potentially reveal the authors' identities.
- The mathematical notation is often incorrect -- eqn. 3 and 4 is mathematically inaccurate and difficult to interpret. Moreover, I believe the authors must clarify the meaning of $p_{\mathbb{N}}(g_{\mathbb{N}}^i)$ and ensure that all symbols are clearly defined and used consistently across the paper.
- Is there any formal / empirical proof for Corollary 1.1? I don't believe Figure 1 is sufficient for proving this claim.
- The framework is described as efficient, yet the implementation contradicts this claim. The authors generate 60 paraphrased samples per instance per LLM for feature generation, which is computationally expensive, particularly during inference.
- Based on the described training flow of GramGuard, it appears that the model may overfit to the specific datasets and LLMs it was trained on. As a result, I suspect that the performance on unseen domains or models is likely to degrade substantially. The authors should include OOD evaluation results, testing the detector on the dataset from these papers [1, 2].
- The paper repeatedly claims robustness to paraphrasing; however, since paraphrasing is explicitly used as part of the training process, this robustness is not particularly surprising or novel.
- The citation formatting throughout the paper is incorrect, significantly compromising readability. Please use the correct citation command.

---

[1] Li et al. MAGE: Machine-generated Text Detection in the Wild. arXiv: 2305.13242.

[2] Dugan et al. RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors. arXiv: 2405.07940.

### Questions
Several questions have already been raised in the Weaknesses. I am willing to increase my score if the authors address the concerns above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The author’s propose GramGuard, an approach which leverages n-gram LMs on pre-LLM human corpora to detect machine-generated text. Given the ngram-LLM, the authors extract the log-likelihood, the entropy, and the variance which get fed into XGBoost (which is trained) for classification. They show that GramGuard is robust under their testing scenarios.

### Strengths
* S1 - The author’s show the power of simple N-Gram LM models trained on pre-LM human corpora. 
* S2 - The author’s not only consider the likelihood of the model as usual with many detectors, but also look at the entropy and variance. One could imagine using these features in other settings as well.
* S3 - The method doesn’t require training on machine-text.

### Weaknesses
* W1 - It’s unclear whether the experiments in 5.2 were performed on a separate validation set. If not, it’s difficult to interpret the results as the best model would’ve been picked on test data. 
* W2 - The finding that a detector (in this case n-gram-based) trained on human corpora is effective at identifying LLMs seems related to this detector that makes a similar point: https://arxiv.org/pdf/2401.06712 The authors should compare GramGuard to it given its similarity.
* W3 -  Most of the models compared against seem to be from the ChatGPT family, except for Gemini. The authors could evaluate across the various test-beds of the MAGE dataset (https://arxiv.org/pdf/2305.13242) for a more complete evaluation. This dataset contains the OpenAI family, the Llama family, GLM, Flan-T5, and others. Moreover, they have test beds that control for various interesting factors.
* W4 - While the N-Gram model was trained only on pre-LLM data, the XGBoost model was trained on LLM and Human data, correct? If so, were the models only evaluated in in-domain settings where the testing data matches the training data? Or was it set up so that XGBoost was trained on PubMed and evaluated on XSum, for example. There seem to be many details regarding the training of XGBoost missing. 
* W5 - There really only is one experimental result and an ablation. More things could’ve been evaluated, as for example the performance when XGBoost is trained on a different domain, the performance as the number of tokens grows, etc.

### Questions
* Q1 - My first and most pressing concern is that it's not clear whether the hyper-parameters were chosen on a separate validation set or not (W1)
* Q2 - Supposing my concern above (Q1) is addressed, then my main concern is W4, followed by W5 and W2. The cross-domain and cross-LLM robustness should've been evaluated, otherwise the results aren't very significant. If these concerns are addressed, I am willing to raise my score.

### Soundness
1

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
4

### Summary
This paper proposes a method for the supervised detection of LLM-generated text. The proposed method, GramGuard, involves (1) paraphrasing the target text many times with an LLM, (2) scoring the perplexity of the original text and paraphrased texts under an LLM, (3) computing deltas between these values, and (4) feeding the deltas into an XGBoost classifier.

### Strengths
1. To the best of my knowledge, the idea of using n-gram models in a DetectGPT-style detector is novel
2. While the paper would benefit from some additional proofreading for typos and consistency, it is overall pretty clear and easy to follow

### Weaknesses
1. However, the idea of using n-gram models in LLM-generated text detection is not novel. See, for example, Ghostbuster (Verma, et al. 2024) which uses probabilities from a range of unigram and trigram models as features in supervised classification

2. My primary concern is that the baselines tested in this paper are relatively weak. In particular, many of the methods in Table 1 are unsupervised detectors, which are known to struggle when the scoring and target model differ. While the paper has some supervised methods (e.g., RoBERTa), these are relatively weak supervised baselines, cf. Verma, et al. 2024 for a comparison. I would recommend adding some stronger baselines like Ghostbuster, Binoculars, or some of the best models from the RAID benchmark. Additionally, if you are going to compare with a closed-source commercial detector like GPTZero, I would consider replacing that with the Pangram Labs detection model, which is substantially stronger.
 
3. The paper seems to be missing some important experimental details about the baseline models. For example: how were the RoBERTa-based models trained? What scoring model was used on DetectGPT? Were classification thresholds tuned independently on each of your three domains? 

4. The paper would benefit from an additional round of proofreading. For example, there is a typo in the title (“LLMs-generated” -> “LLM-generated”), n-gram models are referred to as “N-Gram”, “n-gram”, and “Ngram” (lack of consistency), and \citet is used in place of \citep throughout the paper.

### Questions
N/A

### Soundness
2

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
5

### Summary
This paper proposes to leverage pre-AI N-Gram models exclusively trained on human corpora as the “gold standard” for AI text detection. Building on this foundation, this paper introduces GramGuard, a lightweight, interpretable framework that identifies machine-generated text through shifted statistics of paraphrastic variants. Specifically, by generating paraphrased variants via temperature-controlled decoding from LLMs, this paper measures the shifts in log-likelihood, entropy, and token frequency variance between original texts and paraphrased versions. These “shifts” features are then fed into an XGBoost model to yield interpretable decisions about authorship. Extensive experiments on PubMed, WritingPrompts, and XSum demonstrate that GramGuard matches or exceeds state-of-the-art detectors in performance and robustness.

### Strengths
1.	The idea of using pre-AI N-gram models trained solely on human-written corpora as a scoring model for AI text detection is interesting.
2.	The adopted XGBoost model helps improve the interpretability of the overall approach.
3.	The method achieves state-of-the-art detection accuracy across three datasets and maintains robustness under paraphrastic attacks.

### Weaknesses
1.	The proposed framework is not as lightweight as claimed in the paper since it involves large language models like GPT4 or Genimi for paraphrasing input samples.
2.	The paper only uses responses from GPT-4_1-mini as AI responses and paraphrases with GPT3.4, GPT 4 model, and Gemini model. Note that GPT-4_1-mini, GPT3.4 and GPT 4 are from the same model family. Experiments with synthesized AI responses from one more model will help strengthen the paper to have a more robust and diverse evaluation

### Questions
1.	The citation format is not correct. Using \citep{} and \citet{} appropriately.
2.	In Corollary 1.1, it says “Under perturbation, the process of rephrasing machine text tends to sample the tokens with lower probabilities compared with their original sample”. Is it true? And why?
3.	What is L in equation (6)?

### Soundness
3

### Presentation
3

### Contribution
3
