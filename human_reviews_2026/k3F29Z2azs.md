# Evaluating and Steering Modality Preference in Multimodal Large Language Model

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Multimodal large language models (MLLMs) have achieved remarkable performance on complex multimodal tasks. 
However, it remains insufficiently explored whether they exhibit \textit{modality preference}, a tendency to favor one modality over another when processing multimodal contexts.
To study this question, we introduce $MC^2$ benchmark, which constructs controlled evidence-conflict scenarios to systematically evaluate modality preference in decision-making.
Extensive experiments reveals that all 20 tested MLLMs generally demonstrate clear modality preferences, and such preferences can serve as a useful indictor of downstream task performances of MLLMs. 
Further analysis shows that modality preference can be controlled by instruction guidance and be captured within the latent representations of MLLMs.
Built on these insights, we propose a probing and steering method based on representation engineering to explicitly control modality preference without requiring additional fine-tuning. 
This method effectively amplifies modality preference toward a desired direction and demonstrates promising improvements across multiple downstream applications, including multimodal visual understanding and multimodal machine translation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MC2 (Multimodal Context Conflict), a controlled benchmark designed to evaluate modality preference in multimodal large language models (MLLMs). The benchmark presents tasks where textual and visual contexts intentionally provide conflicting evidence, enabling the measurement of whether a model relies more on text or vision. The authors further propose a representation-engineering-based steering method that identifies modality preference directions in the latent space and manipulates them at inference time, allowing for training-free control of modality bias. Experiments on 20 MLLMs demonstrate measurable modality preferences and show that steering can improve performance on multimodal visual understanding and multimodal machine translation tasks.

### Strengths
- **Novel diagnostic perspective:**
The work introduces an insightful framing of modality preference as a measurable and steerable property of MLLMs, which is conceptually fresh and highly relevant to understanding multimodal reasoning.

- **Well-controlled benchmark (MC2):**
MC2 is carefully designed to minimize confounding factors such as model knowledge and unimodal competence. The dual-version, consistency-filtered binary QA format enhances measurement reliability.

- **Methodological innovation:**
The application of representation engineering to multimodal preference is new. The linear probing–steering framework offers an elegant, training-free way to interpret and modulate model behavior.

### Weaknesses
1. **The MC2 benchmark is relatively small in scale.**
   Since *Modality Preference Steering* depends on the total sample number (N), I believe that enlarging MC2 would not only enable a more comprehensive evaluation but also improve the stability of the *Modality Preference Steering* procedure itself.

2. **Lack of ablation on (N) — stability analysis is missing.**
   The paper does not analyze how the steering results vary with the number of samples used to compute the modality preference direction. An ablation or sensitivity study on (N) would provide valuable insights into the robustness of *Modality Preference Steering*.

3. **Limited evaluation scope in Table 3.**
   Table 3 only reports results on the PhD benchmark for Qwen2VL-7B and OneVision-7B. It would be informative to include results for models that exhibit strong text preference—such as LLaVA-1.6-7B—and to extend the evaluation to additional benchmarks, which is strongly encouraged.

### Questions
1. How sensitive are the results to the number and diversity of samples used for computing (u_ℓ)?
2. Could the authors provide qualitative attention maps or token-level analyses showing how steering changes modality reliance?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores how multimodal LLMs exhibit modality preference: favoring either text or vision when resolving conflicting multimodal inputs. The authors build a new benchmark, MC², to quantify such preferences under controlled text-image conflict settings. The paper finds that most models are text-biased, though larger models shift toward vision. The proposed representation-based steering method identifies latent directions corresponding to modality bias and adjusts them without finetuning, improving both controllability and performance on visual understanding and translation tasks.

### Strengths
1. Novel perspective: The paper tackles modality preference in MLLMs from a fresh and underexplored angle, offering new insights into how models balance visual and textual information.
2. High-quality benchmark and analysis: The proposed MC² benchmark is carefully constructed with controlled modality conflicts and strong human validation, accompanied by clear and convincing visual analyses that reveal consistent patterns across models.
3. Good downstream performance: The representation-based steering method not only provides interpretable control of modality bias but also delivers solid improvements on multimodal translation and visual understanding tasks.

### Weaknesses
1. Generalization of improvements. The method is tested mainly on multimodal translation and visual understanding. Whether the observed gains extend to broader multimodal reasoning, grounding, or dialogue tasks remains uncertain.
2. Prompt sensitivity and robustness. The benchmark uses conflict-style prompts, which might behave differently depending on how each model was trained to follow instructions. This means whether some of the performance differences are come from how well a prompt fits a model’s style rather than from real differences in modality bias.
3. Future insights for model training. The work provides limited discussion on how the findings could guide future training strategies for balanced or adaptive multimodal learning.

### Questions
See weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates **modality preference in MLLMs** (tendency to favor text/vision under conflicting multimodal inputs) and addresses limitations of existing methods (e.g., isolating modalities).

## Core Benchmark & Findings
- **MC^2 Benchmark**: 2k samples with controlled text-vision conflict scenarios (perception-level tasks, >95% single-modality accuracy to eliminate confounders) to force modality prioritization.  
- **Key Results**: All 20 tested MLLMs show clear bias (most favor text; Qwen2.5VL/InternVL3 favor vision); larger models strengthen visual preference; modality preference correlates with downstream performance (Spearman’s \(\rho=0.964\)) and is steerable via instructions.


## Method & Contributions
- **Training-Free Steering**: Probe latent modality preference directions and scale inject them into latent states to adjust bias.  
- **Contributions**: 1) Introduce MC^2 for rigorous preference evaluation; 2) Reveal MLLMs’ preference is identifiable/steerable via latent representations; 3) Propose a training-free method improving multimodal understanding/translation.

### Strengths
## 1. Well-Justified Research Motivation  
The paper targets a critical understudied gap in MLLMs: **modality preference under conflicting text-visual inputs**—a gap ignored by prior work that either isolates modalities or overlooks real-world clashes. This focus is impactful, as resolving it directly boosts MLLM reliability in applications like VQA. The authors further sharpen the motivation by focusing on perception-level tasks (to exclude external knowledge confounders), ensuring the problem is well-scoped.  


## 2. Strong Originality  
- **\(MC^2\) Benchmark**: Unlike existing multimodal benchmarks (which test fusion), \(MC^2\) is novelly designed for conflict-driven modality prioritization, with rigorous controls (e.g., >95% single-modality accuracy) to rule out comprehension errors.  
- **Training-Free Steering**: The authors propose a creative inference-only method using representation engineering—probing latent preference directions and injecting scaled vectors—to avoid fine-tuning costs, a novel application to multimodal bias control.  


## 3. Rigorous Experiments  
- **Broad Evaluation**: Tests 20 MLLMs (open/closed-source), ensuring generalizability.  
- **Quantitative Support**: Key claims are backed by hard metrics (e.g., Spearman’s \(\rho=0.964\) for performance correlation; 2.68% discrepancy with human annotations).  
- **Confounder Control**: Validates critical choices (e.g., layer selection for probing) and filters non-perception tasks, enhancing result credibility.

### Weaknesses
## 1. \(MC^2\) Benchmark’s Limited Real-World Relevance Hurts Generalizability  
While \(MC^2\) isolates modality conflict via abstract, perception-level mismatches (e.g., counting/object recognition errors), it lacks coverage of practical scenarios where modality preference matters—such as multi-turn dialogues (historical text vs. new images) or long-chain reasoning (text inferences contradicting visuals). This over-simplification means the observed preference patterns may not generalize to dynamic, context-rich real-world MLLM use cases.  


## 2. Steering Method Lacks Adaptiveness, Reducing Practicality  
The training-free method requires **a priori specification of target modality preference** (text/vision) to inject vectors. It cannot enable models to autonomously prioritize modalities based on input quality or task needs—an essential capability for real applications (e.g., chatbot, general assistant). This shifts judgment burden to humans, limiting usability in low-supervision scenarios.

### Questions
Same as the section of **Weakness**.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates modality preferences in Multimodal Large Language Models (MLLMs) by introducing the MC² benchmark, which uses controlled evidence-conflict scenarios to systematically evaluate whether models favor vision or text modalities. The authors evaluate some MLLMs and find that most exhibit text preference, with modality preference correlating with downstream task performance. They propose a representation engineering method to steer modality preferences without fine-tuning, demonstrating improvements on visual understanding and machine translation tasks.

### Strengths
- Well-motivated research question: The investigation of modality preference in MLLMs addresses a fundamental yet understudied aspect of multimodal reasoning, with clear practical implications for model design and application.

- The MC² benchmark is carefully constructed with controlled confounding factors (question comprehension, single-modality perception, internal knowledge). The semi-automated pipeline with human verification ensures data quality.

- The representation engineering approach for steering modality preference is training-free, computationally efficient, and demonstrates measurable improvements on downstream tasks (e.g., +1.33 BLEU on MMT).

### Weaknesses
- The MC² benchmark focuses exclusively on perception-level tasks (counting, color, object recognition, etc.) requiring minimal reasoning. This limits generalizability to more complex multimodal reasoning scenarios involving inference, common sense, or abstract reasoning. The findings may not transfer to tasks requiring deeper cross-modal integration.

- The benchmark relies on artificially constructed conflicting contexts, which may not reflect real-world scenarios where modalities typically provide complementary rather than contradictory information. The ecological validity of these conflict scenarios is questionable.

- While the paper demonstrates the effectiveness of the proposed steering method, it lacks thorough analysis of when and why the method might fail. What are the failure modes? How does performance degrade with increased steering intensity? Are there tasks where steering is ineffective or harmful?

- The method requires two inference passes (probing and steering). The computational overhead, memory requirements, and latency implications are not discussed, which is important for practical deployment.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
