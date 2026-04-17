# SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Large Language Models (LLMs) have impressive multilingual capabilities, but they suffer from unexpected code-switching, also known as language mixing, which involves switching to unexpected languages in the model response. This problem leads to poor readability and degrades the usability of model responses.
However, existing work on this issue lacks a mechanistic analysis and shows limited effectiveness.
In this paper, we first provide an in-depth analysis of unexpected code-switching using sparse autoencoders and find that when LLMs switch to a language, the features of that language exhibit excessive pre-activation values. Based on our findings, we propose $\textbf{S}$parse $\textbf{A}$utoencoder-guided $\textbf{S}$upervised $\textbf{F}$ine$\textbf{t}$uning (SASFT), which teaches LLMs to maintain appropriate pre-activation values of specific language features during training. Experiments on five models across three languages demonstrate that SASFT consistently reduces unexpected code-switching by more than 50\% compared to standard supervised fine-tuning, with complete elimination in one case. Moreover, SASFT maintains or even improves the models' performance on six multilingual benchmarks, showing its effectiveness in addressing code-switching while preserving multilingual capabilities. The code and data are available at https://github.com/Aatrox103/SASFT.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of unexpected code-switching in multilingual Large Language Models (LLMs). The authors first conduct a mechanistic analysis using sparse autoencoders, identifying that the issue is caused by the excessive pre-activation of language features. Based on this finding, they propose Sparse Autoencoder-guided Supervised Finetuning (SASFT), a novel method that trains LLMs to control these pre-activation values. Experimental results on five models and three languages show that SASFT reduces unexpected code-switching by over 50% (eliminating it entirely in four cases) while maintaining or even improving performance on standard multilingual benchmarks.

### Strengths
This work's key strength lies in its mechanistic, root-cause analysis of the code-switching problem, moving beyond superficial fixes. By using sparse autoencoders, it identifies the core issue: excessive pre-activation of language features.
The proposed solution, SASFT, is highly effective, consistently reducing unexpected code-switching by over 50% and even eliminating it entirely in several cases. Crucially, it achieves this without compromising performance, as it maintains or even improves the models' capabilities on standard multilingual benchmarks.
In short, its main strengths are its diagnostic depth, highly effective solution, and ability to fix the problem without sacrificing general multilingual proficiency.
The paper is well-written and easy to follow.

### Weaknesses
1.Limited scope of evaluation: The solution is only tested on three languages and five models. Its effectiveness across a wider range of languages, especially low-resource ones, remains unverified.

2.Uncertain generalizability: The method's performance is demonstrated on "six multilingual benchmarks," but it is unclear if it generalizes well to other critical tasks like reasoning, complex translation, or creative writing.

3.Computational overhead: The approach relies on sparse autoencoders, which likely introduce significant computational cost and complexity compared to standard fine-tuning, a trade-off not mentioned.

4.Lack of comparative baselines: While it outperforms standard supervised fine-tuning, it is not compared against other specialized techniques aimed at reducing code-switching, making its relative advancement unclear,like the works listed in related works.

5.Superficial treatment of intentional Code-Switching: The method focuses on "unexpected" code-switching but may risk suppressing intentional and culturally appropriate code-switching (e.g., in bilingual communities), potentially reducing linguistic flexibility.

### Questions
1.how were these code-swithing issues for testing are constructed or prompted? Overall, it seems that this issue occurs relatively infrequently and is difficult to reproduce, especially across different models. Such cases seldom happen, making it challenging to trace and resolve the problem on a large scale.

2.Page 4 Line 184， how many unexpected code-switching responses are collected？

3.Have you analyzed for different languages which are prone to code-switch with each other?

4.On which layers were the main supervised SFT experiments conducted? Regarding the time efficiency of autoencoder computation, does using more layers lead to higher computational efficiency, and how was this trade-off balanced in the ablation studies?

5.Why compare with RL (GRPO), it is more likely to make alignment during the RL phase, not for multi-lingual extension.

6.According to Table2, the proposed method does not show a significant improvement in effectiveness.

7.Figure 6, pls give detailed description for multi-layer, like the specific number of layers and which layer.

### Soundness
3

### Presentation
4

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
This paper proposes a novel method, SASFT, to mitigate the issue of unexpected code-switching in large language models (LLMs). The method is grounded in Sparse Autoencoders (SAEs), which are used to identify language-specific feature. An auxiliary loss is introduced during supervised fine-tuning to suppress pre-activation values of irrelevant language features, thereby reducing unexpected code-switching. The method is evaluated on five multilingual LLMs across six benchmarks and three code-switching target languages, and shows strong reduction in code-switching rates, while mostly maintaining or improving performance on standard benchmarks.

### Strengths
1. The paper addresses an underexplored yet practically important problem, unexpected code-switching, which impacts user experience and model usability.

2. Demonstrates consistent reductions in code-switching across various models and languages, outperforming previous methods (e.g., GRPO) in most settings.

3. The paper offers detailed analysis on factors such as layer depth and feature selection.

4. The paper is clearly written and easy to follow.

### Weaknesses
1. The method for identifying language-specific features relies on rankings without justification. This introduces sensitivity to hyperparameter selection and limits the interpretability of the results, especially in multilingual settings where features may vary across tasks.

2. The paper reports a substantial +327% increase in Korean code-switching under the GRPO method, but does not provide sufficient explanation for this anomaly. A deeper analysis is needed to clarify the cause of such a drastic change.

3. It remains unclear why SASFT underperforms on certain benchmarks, such as MMLU and HellaSwag, for the Qwen3-8B model. This raises questions about robustness and generalizability across tasks.

### Questions
1. How does the proposed method perform on multilingual mathematical reasoning tasks, such as MGSM？

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the problem of unexpected code-switching in multilingual large language models (LLMs). The authors first employ Sparse Autoencoders (SAEs) to analyze the internal representations of LLMs and discover that unexpected language switches are correlated with over-activation of target-language-specific features. Based on this finding, they propose a novel fine-tuning method, SASFT (Sparse Autoencoder-guided Supervised Finetuning), which introduces an auxiliary loss term during supervised fine-tuning to constrain the pre-activation values of specific language features below a threshold, thereby reducing activations of irrelevant languages. Experiments on five models and three target languages (Chinese, Russian, and Korean) show an average >50% reduction in unexpected switching, including four cases of complete elimination, while maintaining or improving performance on six multilingual benchmarks.

Main Contributions:
- Using sparse autoencoders, the paper reveals that when a model is about to unexpectedly switch to language L, the language-specific features of L show significantly elevated pre-activation values in the residual stream.
- Proposes SASFT, a training-stage approach that suppresses the activation of irrelevant language features without requiring inference-time intervention.
- Provides comprehensive experiments demonstrating that SASFT is effective, robust, and preserves multilingual capabilities.

### Strengths
1. The paper proposes a Sparse Autoencoder-guided Supervised Finetuning (SASFT) approach that combines sparse autoencoders with supervised fine-tuning. 
2. The effectiveness of SASFT is validated through extensive experiments across multiple language pairs and model families. The results demonstrate consistent mitigation of code-switching phenomena in diverse multilingual settings.
3. Experimental evidence indicates that the proposed method effectively reduces unintended language switches, thereby improving the accuracy, consistency, and usability of multilingual model outputs.

### Weaknesses
1. Limited Baseline Comparison：The paper only compares SASFT with GRPO, which, although relevant, is insufficient to establish the method’s relative advantage.
2. Lack of Mechanistic or Causal Analysis：While the paper empirically observes that the pre-activation values of target-language features increase prior to code-switching and validates this via directional ablation, this evidence remains correlational. The work does not provide a mechanistic explanation of why such activation leads to language switching, nor tests the causal hypothesis.
3. Potential Model-Specific Bias. The reported improvements may partially stem from inherent differences in model multilingual balance, rather than from SASFT’s general efficacy. The paper does not control for or analyze how such model-specific language priors influence the observed reductions in code-switching.

### Questions
1. Could the authors provide a more comprehensive comparison to strengthen the empirical validity of SASFT?
2. The paper suggests that increased pre-activation of language-specific features precedes unexpected code-switching. Have the authors examined whether artificially increasing the activation of a non-target language feature can induce code-switching?

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
The paper studies unexpected code-switching in multilingual LLMs and links the phenomenon to unusually high pre-activation on language-specific SAE features. Building on this observation, it proposes SASFT: add an auxiliary loss during SFT that penalizes pre-activation of irrelevant language features across several layers.

### Strengths
1. Clear story from SAE analysis to a concrete training modification.
2.  The results are good to show the proposed method's effectiveness.
3. The paper is well-writen and easy to follow.

### Weaknesses
1. GRPO is run with only 10k samples (1k per language). That seems light for a control-behavior objective. Consider stronger RL baselines , or simple supervised baselines that directly penalize language-ID tokens . Without stronger baselines, it’s hard to attribute gains purely to SASFT. Does the author could ensure the GRPO have true convergence?
2. The study focuses on zh/ru/ko. It’s unclear if results hold for low-resource scripts (e.g., Amharic, Khmer), closely-related Latin languages where CS is subtler (es/pt/fr/it)? 
3. SASFT relies on high-quality SAEs and on accurate language-feature identification; for Qwen the authors train their own SAEs, for others they reuse published ones. How sensitive are results to (1)SAE training corpora, dimensionality, sparsity target? (2)Which layer(s) the features are extracted from? (3) Feature drift after fine-tuning (do features stay monosemantic)?

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
3
