# Contrastive Weak-to-Strong Generalization

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Weak-to-strong generalization provides a promising paradigm for scaling large language models (LLMs) by training stronger models on samples from aligned weaker ones, without requiring human feedback or explicit reward modeling. However, its robustness and generalization are hindered by the noise and biases in weak-model outputs, which limit its applicability in practice. To address this challenge, we leverage implicit rewards, which approximate explicit rewards through log-likelihood ratios, and reveal their structural equivalence with Contrastive Decoding (CD), a decoding strategy shown to reduce noise in LLM generation. Building on this connection, we propose \textbf{Contrastive Weak-to-Strong Generalization (ConG)}, a framework that employs contrastive decoding between pre- and post-alignment weak models to generate higher-quality samples. This approach enables more reliable capability transfer, denoising, and improved robustness, substantially mitigating the limitations of traditional weak-to-strong methods. Empirical results across different model families confirm consistent improvements, demonstrating the generality and effectiveness of ConG. Taken together, our findings highlight the potential of ConG to advance weak-to-strong generalization and provide a promising pathway toward AGI. Our code is available at: \url{https://anonymous.4open.science/r/ConG/}

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a weak-to-strong framework that formalizes a structural equivalence between implicit rewards and Contrastive Decoding. It uses a two-stage pipeline—SFT followed by DPO with CD positives—to transfer preferences from a weaker model to a stronger one. Experiments show consistent gains with minimal downstream degradation.

### Strengths
1. Thorough theoretical analysis. Starting from the log-likelihood–ratio form of implicit rewards, the paper derives a structural equivalence to contrastive decoding, provides a token-level decomposition, and offers behavioral predictions for the coefficient α; these align with the empirical results and improve interpretability and persuasiveness.

2. Clear writing. The paper is well structured and easy to follow, with consistent notation that aids reproduction and implementation.

3. Broad experimental coverage. Evaluations span two model families, both weak-to-strong and self-alignment settings, and varying capability gaps, with ablations on α and downstream task checks; results are stable, supporting generality.

### Weaknesses
1. Lack of end-to-end efficiency metrics. Contrastive decoding introduces additional latency, yet the main text and experiments do not provide quantitative training-time results for the proposed method and baselines to support judgments about deployability and fair comparison.

2. Both benchmarks use LLM-as-judge for preference evaluation, with no validation against more reliable metrics and no cross-judge checks (e.g., different judge models) or small-scale human audits to verify robustness.

3. Limited novelty/insight. The contribution feels incremental: migrating Contrastive Decoding (CD) to this setting is intuitive—simple and effective—but it does not provide further insight into why or when it should most help beyond the reported gains.

### Questions
See Weaknesses.

### Soundness
3

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
This paper proposes Contrastive Weak-to-Strong Generalization, or ConG, a novel framework designed to improve the efficacy of the weak-to-strong generalization (W2S) paradigm for training large language models . The core contribution is the realization of a structural equivalence between implicit rewards, which approximate explicit rewards via log-likelihood ratios, and the technique of Contrastive Decoding (CD). Leveraging this insight, ConG utilizes CD between aligned and unaligned weak models to generate demonstrably higher quality samples for training a stronger model. The framework is implemented in two stages: ConG-S for supervised fine-tuning initialization and ConG for subsequent preference optimization via DPO. Experiments across different model families show that ConG significantly outperforms existing W2S baselines, achieving an average performance improvement of approximately 16.5\% on specific evaluation benchmarks.

### Strengths
The paper presents a significant and original contribution by establishing a theoretical link between implicit rewards and Contrastive Decoding. This insight is highly original and provides a solid, principled foundation for the proposed ConG method, moving beyond heuristic data filtering techniques. The quality of the method is demonstrated by consistent and substantial empirical gains over conventional W2S approaches, showcasing the effectiveness of the CD-based sample generation in reducing noise and mitigating biases from the weak model. The clarity of the problem formulation, particularly the limitations of current W2S methods due to weak-model output noise, is excellent, making the motivation for ConG clear. The significance lies in offering a more robust and scalable approach to model alignment and capability transfer, which is a critical area for safely developing more powerful AI systems.

### Weaknesses
The primary weakness of this work lies in the scope and scale of its experimental validation, particularly regarding the concept of weak-to-strong generalization. The main experiments utilizing a 3B weak model to train a 7B or 8B strong model feature a relatively small capacity gap. This difference may not be sufficiently challenging to test the limits of W2S generalization, especially compared to foundational works in this area that consider gaps spanning tens of billions of parameters (e.g., training a 70B model from a much smaller one). To truly substantiate the claim of robust W2S generalization, the method must be validated on much larger model scales and greater capacity differentials. Furthermore, the evaluation is narrowly focused on instruction-following and preference alignment benchmarks such as AlpacaEval and Arena-Hard. While these are important, the paper fails to demonstrate the method's effectiveness in complex reasoning domains like mathematical problem-solving or code generation. If ConG's mechanism is primarily tailored to improving stylistic alignment and preference quality, its generalizability to domains requiring logical correctness and factual accuracy, rather than just high-quality human preference, remains unproven.

### Questions
First, what are the results when ConG is applied to a significantly larger model capacity gap? Specifically, please provide results for training a much larger strong model, such as a 13B or 70B parameter model, using the current 3B weak model. This is necessary to confirm that the method scales effectively under the conditions of extreme W2S generalization. Second, given the reliance on preference-based metrics, could the authors provide an analysis of ConG's performance on complex, domain-specific reasoning benchmarks, such as those evaluating mathematical reasoning (e.g., GSM8K) or code generation (e.g., HumanEval)? This would definitively confirm the method's transferability beyond general instruction-following. Third, please elaborate on the mechanism of Contrastive Decoding in this context. Does the CD process primarily function to remove stochastic noise inherent in the weak model's generation, or does it also amplify the specific stylistic and moral biases that the weak model acquired during its initial alignment stage? A more nuanced discussion of CD's filtering properties is needed.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes the Contrastive Weak-to-Strong Generalization framework, aiming to improve weak-to-strong model alignment. The core idea is to combine implicit rewards with Contrastive Decoding to extract high-quality samples from weak models for training stronger models. Extensive empirical evaluations are conducted on the Qwen2.5 and Llama3 model families, covering self-alignment, weak-to-strong alignment, and downstream task fidelity. Results show that ConG significantly outperforms existing methods in both self-alignment and weak-to-strong alignment, while causing negligible degradation on downstream tasks.

### Strengths
1. The ConG algorithm is proposed in a novel way, structurally connecting implicit rewards with the contrastive decoding mechanism, providing support for weak-to-strong generalization.

2. Systematic ablation experiments are conducted to validate the impact of the contrastive coefficient α and the effect on downstream tasks.

### Weaknesses
1. The algorithm relies on the assumption that the weak model can reasonably express preferences and that implicit rewards are a good approximation; if the weak model is of low quality, it may not adequately support weak-to-strong generalization.

2. The paper provides limited weak-to-strong experimental validation on mathematical reasoning and code generation tasks, leaving its effectiveness in these domains less explored.

### Questions
Please refer to the relevant points in the Weaknesses section. If the authors can provide clarification and improvements, I would be very happy to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors studied the weak-to-strong generalization problem in large language models, where a stronger model is trained using data generated by a weaker aligned model, instead of human feedback. They proposed Contrastive Weak-to-Strong Generalization (ConG), a method that leverages contrastive decoding between pre- and post-alignment weak models to generate higher-quality supervision signals. The core idea is that contrastive decoding denoises weak model outputs by using implicit reward signals based on log-likelihood ratios, which are theoretically shown to be related to implicit reward maximization. The authors demonstrate through experiments on Qwen2.5 and Llama3 model families that ConG improves both weak-to-strong alignment and self-alignment, achieving performance gain over baselines and delivering better robustness and transfer of preferences across models.

### Strengths
- The idea of leveraging contrastive decoding to generate higher-quality supervision signals from weak models is interesting and novel.
- The experimental results are comprehensive and convincing, demonstrating the effectiveness of the proposed method in both weak-to-strong alignment and self-alignment settings.

### Weaknesses
- The word 'equivalence' is exaggerated for the relationship between contrastive decoding and implicit reward. They are related, but not equivalent.
- The baselines are not comprehensive. For example, Tao and Li (2025) proposed a two-stage framework for weak-to-strong alignment that is similar to the proposed method. The authors should compare with it.  
Leitian Tao, Yixuan Li. Your Weak LLM is Secretly a Strong Teacher for Alignment. ICLR 2025.

### Questions
- Questions
  - In the experiments of self-alignment, what is the $\pi_{r}^{w}$? And how do their performance compare with methods in Table 1?
  - What is the amount of GPU hours consumed for training models with different methods?
  - In the bottom part of Table 1, what is the reason that the performance of WSPO worse than Base?
  - In Weak SFT, which model is used as the weak model, the pre-alignment or post-alignment one?
- Suggestions
  - In Line 355, the authors should add description of WSPO.

### Soundness
3

### Presentation
3

### Contribution
3
