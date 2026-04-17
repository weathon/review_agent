# MulVuln: Enhancing Pre-trained LLMs with Shared and Language-Specific Knowledge for Multilingual Vulnerability Detection

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Software vulnerabilities (SVs) pose a critical threat to safety-critical systems, driving the adoption of AI-based approaches such as machine learning and deep learning for software vulnerability detection. Despite promising results, most existing methods are limited to a single programming language. This is problematic given the multilingual nature of modern software, which is often complex and written in multiple languages. Current approaches often face challenges in capturing both shared and language-specific knowledge of source code, which can limit their performance on diverse programming languages and real-world codebases. To address this gap, we propose MULVULN, a novel multilingual vulnerability detection approach that learns from source code across multiple languages. MULVULN captures both the shared knowledge that generalizes across languages and the language-specific knowledge that reflects unique coding conventions. By integrating these aspects, it achieves more robust and effective detection of vulnerabilities in real-world multilingual software systems. The rigorous and extensive experiments on the real-world and diverse REEF dataset, consisting of 4,466 CVEs with 30,987 patches across seven programming languages, demonstrate the superiority of MULVULN over thirteen effective and state-of-the-art baselines. Notably, MULVULN achieves substantially higher F1-score, with improvements ranging from 1.45% to 23.59% compared to the baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a method for enhancing software vulnerability detection by generalizing across multiple programming languages. The proposed solution utilizes CodeT5, a pretrained language model (PLM) as a backbone that includes general parametric knowledge of multiple languages. The input to the PLM is then augmented by several extra tokens designed to signal which programming language is being used. The embeddings for those extra tokens are learned during the training of the model, and the model further incentivized to select the embeddings associated with the correct language through two proposed approaches: key parameter query, and parameter masking.

### Strengths
- The proposed method seems like a lightweight add-on to pretrained language models that slightly improves their performance on vulnerability detection, and may be relevant for other software-related tasks as well.
- The “parameter selection via key-parameter query” method seems interesting as it bears resemblance to the attention mechanism in transformer models, despite the fact that it showed less impressive results than the “language-aware parameter masking”

### Weaknesses
- The related work section did not cite any of the existing works targeting multi-lingual software vulnerability detection (e.g., [A]. [B], [C]. These paper should have been discussed in the related work, and even used as baselines for comparison.
    
- The authors mentioned polyglot applications (i.e., projects including multiple languages) to motivate their proposed solution, but this type of software was not present in the considered dataset and was thus never evaluated in the experiments.
    
- The external validity of the work is questionable. Although the proposed detector was trained on the training set of the REEF dataset, and was evaluated on the test set of the same dataset, this might not be enough. Other datasets (from the plethora of available datasets on vulnerability detection) should have been considered to prove the efficacy of the proposed method in various settings.
 
[A] Zhang, Boyu, Triet HM Le, and M. Ali Babar. "MVD: A Multi-Lingual Software Vulnerability Detection Framework." *arXiv preprint arXiv:2412.06166* (2024).

[B] Zhang, Ting, et al. "Benchmarking Large Language Models for Multi-Language Software Vulnerability Detection." *arXiv preprint arXiv:2503.01449* (2025).

[C] Yu, Junji, et al. "A Preliminary Study of Large Language Models for Multilingual Vulnerability Detection." *Proceedings of the 34th ACM SIGSOFT International Symposium on Software Testing and Analysis*. 2025.

### Questions
- Why did the authors not cite related works on multi-lingual vulnerability detection in their related work? and why were performance comparisons not conducted with those recent works?
    
- In section 3.1 you mention that you consider function-level binary classification of vulnerabilities. However this exact problem has come under strong criticism from recent work [A], namely it fails to incorporate contextual information which are crucial to deciding whether software is vulnerable. How would you defend this choice?
    
- For multi-lingual projects, could a weighted sum of language-specific parameters $P_X$ be used instead of the argmax in equations (1) ad (2)?
    
- For pretrained language models, have you experimented with chain-of-thought prompting? You could  elicit the model to first output the used programming language, which would naturally condition the upcoming generated tokens deciding whether the code is vulnerable. It would be interesting to compare this to the learnable conditioning proposed by your solution.
    

[A] Risse, Niklas, Jing Liu, and Marcel Böhme. "Top score on the wrong exam: On benchmarking in machine learning for vulnerability detection." *Proceedings of the ACM on Software Engineering* 2.ISSTA (2025): 388-410.

### Soundness
2

### Presentation
3

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
This paper addresses the problem of multilingual software vulnerability detection by proposing MulVuln, a framework that augments pre-trained language models with a parameter pool specifically designed to capture both shared (cross-lingual) and language-specific knowledge in source code. The approach combines the strengths of PLMs for general semantic representation with dynamic selection of language-tailored parameters via either key-parameter querying or language-aware masking. The method is rigorously evaluated against thirteen baselines on the challenging REEF dataset, demonstrating superior F1 and recall, as well as strong performance on top vulnerability types (CWEs) and across seven diverse programming languages.

### Strengths
- The proposed method consistently outperforms a wide range of strong baselines and modern LLMs.
- The paper provides explicit interpretability through visualizations that offer valuable insights into the inner workings of the model.
- The research ensures reproducibility and offers practical value by providing sufficient implementation details and using a realistic dataset.

### Weaknesses
- Although the performance of multilingual code vulnerability detection is commendable, some related work on vulnerability detection based on LLMs  (beyond merely utilizing LLMs themselves) has not been cited or thoroughly analyzed. Due to the lack of a clear comparison with LLM-based vulnerability detection work, its originality is limited.
- There is limited analysis of the computational overhead. The design introduces language-specific parameter matrices on top of parameter-rich PLMs, which may incur additional computational costs.
- Although recall and F1 are highlighted (arguably reasonable in vulnerability detection due to prioritizing recall), MulVuln’s precision often increases less than recall. In Table 1, some baselines achieve comparable or even higher precision. Given the application (where false positives may incur costs), more discussion or tuning toward precision-oriented use cases is warranted.

### Questions
- The main benefit is claimed around the dynamic query mechanism for associating code (possibly ambiguous or mixed-language) with the right parameter. How does the model behave for code samples with mixed or embedded scripting languages, or where the language cannot be reliably determined upfront? Are there empirical results for such edge cases?
- Results currently lack confidence intervals/statistical significance analysis. Are the observed improvements in F1-score over the best PLM/LLM baselines robust to different seeds or test splits? Could the authors report mean and variance over multiple runs?

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
4

### Summary
This paper proposes MulVuln, a novel approach for multilingual software vulnerability detection. The authors highlight that most existing AI-based detection methods are limited to a single programming language, which is insufficient for modern software systems that are often complex and written in multiple languages. MulVuln is designed to capture both shared knowledge that generalizes across different languages and language-specific knowledge that reflects unique coding conventions. The approach was evaluated on the real-world REEF dataset, which includes 4,466 CVEs across seven different programming languages. The experiments demonstrated that MulVuln significantly outperformed thirteen state-of-the-art baselines, achieving an F1-score improvement of 1.45% to 23.59%.

### Strengths
Important Problem: The paper correctly identifies and addresses a significant, practical gap in SVD research: the lack of effective, multilingual models for real-world codebases.

Clear Methodology: The proposed MulVuln approach is simple, intuitive, and clearly explained. The two selection mechanisms (instance-based query vs. language-aware training) are sensible explorations of the design space.

### Weaknesses
**Limited and Unclear Empirical Evaluation**: The experimental design suffers from two significant gaps regarding contemporary baselines: (1) While the paper states the use of various prompting strategies (zero-shot, few-shot, and instruction-based few-shot prompting), the final result in RQ1 is an aggregate, single score for all LLMs. (2) The paper overlooks several highly relevant and recently published baselines based on both PLMs [1-2] and LLMs [3-4], which significantly weakens the claim of achieving state-of-the-art performance. 

[1] Distinguishing Look-Alike Innocent and Vulnerable Code by Subtle Semantic Representation Learning and Explanation

[2] SCALE: Constructing Structured Natural Language Comment Trees for Software Vulnerability Detection

[3] Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data

[4] Collaboration to Repository-Level Vulnerability Detection


**Limited Scalability Discussion**: The model's design, especially Eq. 2, assumes a closed set of $S$ languages, with $S$ parameter matrices. This does not scale well to dozens of languages and offers no clear path for handling languages unseen during training (a critical aspect of true multilingual generalization).

### Questions
1. The authors state that "zero-shot, few-shot and instruction-based few-shot prompting were adopted for DeepSeek-Coder, Code Llama, Llama 3, GPT-3.5-Turbo and GPT-4o". However, the authors later claim that LoRA fine-tuning was also "applied" (l. 365). Please explicitly specify which of the above models were actually LoRA-fine-tuned and which were only prompt-engineered. Besides, RQ-1 reports a single bar per model; it is impossible to tell whether the number comes from zero-shot, few-shot or instruction-based few-shot. Clarify the exact prompting protocol used for each reported result.

2. Does LoRA fine-tuning of LLMs surpass the performance of MulVuln? Present a head-to-head comparison (MulVuln vs. LoRA-LLM) on the same test split so that the benefit of your adaptation strategy can be quantified.

3. If CodeT5-base already delivers strong results, have you experimented with larger checkpoints of the CodeT5+ family (e.g., 2B or 16B parameters)? 

4. Can MulVuln generalise to unseen programming languages, or at least adapt from a handful of training samples in a new language? Report zero-/few-shot transfer results on at least one language never seen during training to validate the claim of language-agnostic vulnerability detection.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes MulVuln, a multilingual vulnerability detection approach that augments pre-trained language models with a learnable parameter pool to capture both shared and language-specific knowledge. The method selects appropriate parameters via key-based matching or language masking, concatenates them with input embeddings, and processes them through CodeT5's encoder. Experiments on the REEF dataset covering 7 programming languages show F1-score improvements of 1.45-2.81% over fine-tuned CodeT5/CodeT5+.

### Strengths
The paper addresses a important problem of multilingual vulnerability detection with a clear and intuitive approach. The experimental evaluation is comprehensive, covering 7 programming languages on the REEF dataset with 4,466 CVEs and comparing against 13 diverse baselines spanning deep learning models, pre-trained language models, and large language models. The proposed two-component design balances shared cross-language knowledge with language-specific features, and the visualizations provide useful insights into how the parameter pool operates. The writing is generally clear and the methodology is well-explained.

### Weaknesses
The most critical flaw is the absence of parameter-efficient fine-tuning baselines like Prefix-Tuning, LoRA, and Adapters, which are directly comparable to the proposed approach and essential for establishing novelty—without these comparisons, the contribution reduces to applying existing prefix-tuning techniques to vulnerability detection. All experimental results lack statistical rigor with single-run evaluations, no error bars, and no significance testing, making it impossible to determine whether the improvements are meaningful or simply noise. The claims about learning "language-specific knowledge" are inadequately supported, with no analysis of what the parameter pool actually encodes, no parameter similarity matrices across languages, and no cross-language transfer experiments to validate the separation of shared versus specific features. Critical ablation studies are missing, particularly varying the parameter pool size S beyond the default value of 7 and testing different query functions beyond the [CLS] token. The generalization capabilities remain completely untested through leave-one-language-out experiments, temporal splits, or cross-project evaluation, which is problematic given the dataset's severe imbalance that goes unaddressed. Several experimental results are unexplained, such as why DeepSeek-Coder with 6.7B parameters achieves only 48.61% F1 while the much smaller CodeT5 performs better, and why GPT-4o's precision (74.54%) vastly exceeds MulVuln's (57.51%). The theoretical justification is entirely absent, with no explanation for why prepending 5 learnable tokens should capture language-specific knowledge or why the particular loss formulation in Equation 3 is appropriate.

### Questions
see in the Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
