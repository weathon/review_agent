# Knowledge Exchange with Confidence: Cost-Effective LLM Integration for Reliable and Efficient Visual Question Answering

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Recent advances in large language models (LLMs) have improved the accuracy of visual question answering (VQA) systems. However, directly applying LLMs to VQA still presents several challenges: (a) suboptimal performance when handling questions from specialized domains, (b) higher computational costs and slower inference speed due to large model sizes, and (c) the absence of a systematic approach to precisely quantify the uncertainty of LLM responses, raising concerns about their reliability in high-stakes tasks. To address these issues, we propose an UNcertainty-aware LLM-Integrated VQA model ($\texttt{Uni-VQA}$). This model facilitates knowledge exchange between the LLM and a calibrated task-specific model (\ie \texttt{TS-VQA}), guided by reliable confidence scores, resulting in improved VQA accuracy, reliability and inference speed. Our framework strategically leverages these confidence scores to manage the interaction between the LLM and $\texttt{TS-VQA}$: the specialized questions are answered by the $\texttt{TS-VQA}$ model, while general knowledge questions are handled by the LLM. For questions requiring both specialized and general knowledge, the $\texttt{TS-VQA}$ provides candidate answers, which the LLM then combines with its internal knowledge to generate a more accurate response. Extensive experiments on VQA datasets demonstrate the theoretically justified advantages of $\texttt{Uni-VQA}$ over using the LLM or $\texttt{TS-VQA}$ alone.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a calibrated, rule-based router to combine TS-VQA and LLM to balance power consumption and performance. The first step is to apply the existing DRO method to calibrate confidence. Then the calibrated confidence is used to route questions to different decision models with varying capability vs power consumption combinations.

### Strengths
- The paper offers a practical cost-aware collaboration between TS-VQA with an LLM to save energy consumption.
- The motivation and trade-off framing are convincing.

### Weaknesses
- Modest novelty. The proposed system is essentially a confidence-controlled routing. 
- There seems to be a lack of comparison between ensemble-aware fusion vs a distilled single model, to fully understand the tradeoff between reliability and latency.
- The routing is hard-coded. There is no comparison to a learned router or agentic alternatives (that add a light verifier before calling the LLM).
- Sustainability is a central motivation. But the benefit is not sufficiently evidenced with proper accounting.
- The writing can be improved. For example, the DRO paragraph introduces $\lambda$ abruptly, leaving the mechanism unclear.

### Questions
Please give the exact form of $w$ and its dependence on $\lambda$.

### Soundness
3

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
This paper proposes Uni-VQA, a hybrid VQA framework that uses calibrated confidence scores from a task-specific VQA model to decide when and how to involve an LLM. High-confidence questions are answered locally, low-confidence ones are delegated to the LLM, and intermediate cases use candidate answers for collaboration. The approach improves accuracy while reducing reliance on costly LLM inference.

### Strengths
- The related work section is well organized and clearly positions this framework among prior VQA, calibration, and LLM-augmented systems, making it easy to understand its practical motivation.

- The theoretical derivations and more complex details are moved to the appendix, which helps maintain good readability in the main paper.



- The experimental evaluation is comprehensive, including multiple backbones, datasets, and ablation studies.

### Weaknesses
- Confidence threshold decisions could use more clarification
  - The routing strategy depends on two confidence thresholds (𝑙,𝑢). It would be helpful to include more explanation on how these values are selected and how sensitive the method is to different threshold settings across datasets or models.

- The effectiveness of TS-VQA candidates may vary depending on the confidence level
  - When the TS-VQA model has low confidence, its proposed candidates may negatively influence the LLM’s reasoning. The current strategy of suppressing candidates only in low-confidence cases is reasonable, but additional analysis on when candidates are beneficial vs. harmful would provide deeper insight into this interaction.

- LLM output reliability is not fully addressed
  - The reliability of LLM outputs is not modeled. Since the LLM handles the most uncertain samples, having some form of uncertainty estimation or error check on the LLM side could further improve the robustness of the overall system.

- Some figures have relatively small, which affects readability. For example, Figure 3 and Figure 14.

- There is no Ethics Statement or Reproducibility Statement in the paper, which are required by the ICLR submission guidelines. Authors may miss this information.

### Questions
See weaknesses please

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Uni-VQA, a method employing an LLM and a task-specific VQA model to efficiently answer questions. The task-specific VQA model is calibrated to provide reliable confidence scores. With the confidence score, the framework whether the answer from the task-specific VQA model should be processed by the LLM. Extensive experiments show the effectiveness of the method.

### Strengths
1. The motivation for introducing task-specific VQA model is reasonable and practical.
2. The calibration of task-specific VQA model provide reliable confidence score, enabling the interaction between the two VQA models.
3. Extensive experiments show the effectiveness of the method.

### Weaknesses
1. RAG (Retrieval-Augmented Generation) methods are not discussed and compared. The task-specific VQA model serves as a role of providing specific knowledge for LLM, which is similar to external and up-to-date information in RAG methods. The advantages and disadvantages of the proposed method compared to RAG methods should be discussed and compared.
2. The training cost of the task-specific VQA models and the distillation is not reported. It is unclear whether it would be the bottleneck of the framework.

### Questions
1. What are the advantages and disadvantages of the proposed method compared to RAG methods?
2. What is the training costs of the task-specific VQA models and the distillation?

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
5

### Summary
This paper proposes a novel framework called Uni-VQA for visual question answering, aiming to address challenges in directly applying large language models to VQA, such as suboptimal performance in specialized domains, high computational costs, and lack of uncertainty quantification. The key contributions include: (1) developing a calibration technique based on a diverse ensemble to improve the reliability of confidence estimates for task-specific VQA models; (2) introducing a confidence-guided knowledge exchange mechanism that dynamically decides whether to delegate to an LLM and how to provide candidate answers based on TS-VQA confidence scores, optimizing accuracy and efficiency; and (3) validating the framework through theoretical analysis and extensive experiments on VQA-v2 and COCO-QA datasets, showing superior performance over using LLM or TS-VQA alone while significantly reducing computational overhead. The paper also explores knowledge distillation for faster inference and provides analysis on carbon emissions and latency, highlighting environmental sustainability benefits.

### Strengths
The paper novelly integrates confidence-guided mechanisms with LLM-VQA collaboration, differing from traditional model cascades or simple delegation. The dynamic candidate answer selection based on confidence intervals is a creative combination, underexplored in existing work.

The method has theoretical depth and experiments are comprehensive, covering multiple VQA modelsand datasets with consistent results. Ablation studies validate component importance. Writing is concise, and figuresintuitively explain complex concepts, with clear mathematical derivations.

The work directly targets AI scalability and sustainability, reducing LLM carbon footprint, with potential impact on high-stakes applications , aligning with green AI trends.

### Weaknesses
While tested on VQA-v2 and COCO-QA, the paper does not include more diverse datasets (e.g., medical VQA or long-tailed distributions), potentially limiting generalizability. Also, LLM usage is limited to Mistral-7B and LLaVA, without extension to larger models , failing to fully assess scale effects.

The calibration technique relies on diverse ensembles, increasing training overhead, and although distillation mitigates this, it may affect deployment ease. 

Comparison with recent VQA methods (e.g., Transformer-based variants) is limited; the paper focuses on traditional baselines.

Theorem 4.2 relies on inverse relationship between entropy and confidence, but strict proof in multi-class settings depends on uniform distribution assumptions, which may deviate in practice.

### Questions
1.How does Uni-VQA handle modal missingness or distribution shifts? For example, if image quality is poor or questions are out-of-distribution, does confidence calibration remain reliable? 

2.The paper mentions that confidence thresholds are determined using a validation set. Could you provide more specifics about the optimization process? What objective function was used to balance accuracy and efficiency during threshold selection?

3.While the paper demonstrates inference efficiency, could you provide more details about the training computational costs of the diverse ensemble approach? How does the training time scale with the number of ensemble members?

4.How does the framework handle concept drift or distribution shifts over time? Have you considered mechanisms for continuous adaptation of the confidence thresholds?

### Soundness
3

### Presentation
3

### Contribution
3
