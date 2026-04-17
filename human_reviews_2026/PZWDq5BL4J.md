# Reasoning-Based Personalized Generation for Users with Sparse Data

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Large Language Model (LLM) personalization holds great promise for tailoring responses by leveraging personal context and history. However, real-world users usually possess sparse interaction histories with limited personal context, such as cold-start users in social platforms and newly registered customers in online E-commerce platforms, compromising the LLM-based personalized generation. To address this challenge, we introduce **GraSPeR** (**Gra**ph-based **S**parse **Pe**rsonalized **R**easoning), a novel framework for enhancing personalized text generation under sparse context. GraSPeR first augments user context by predicting items that the user would likely interact with in the future. With reasoning alignment, it then generates texts for these interactions to enrich the augmented context. In the end, it generates personalized outputs conditioned on both the real and synthetic histories, ensuring alignment with user style and preferences. Extensive experiments on three benchmark personalized generation datasets show that \method achieves significant performance gain, substantially improving personalization in sparse user context settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a reasoning-centered framework for LLM personalization. The proposed method decomposes user alignment into two phases: (1) Reasoning-Based Personalization (RBP), where the model generates step-level reasoning conditioned on user attributes, and (2) Self-Reflective Updating, where feedback from past interactions is distilled into a reasoning memory bank. Experiments across multiple agentic personalization benchmarks—including dialog, recommendation, and open-domain task settings—demonstrate that RBP outperforms retrieval-based and instruction-tuning baselines in consistency, adaptability, and reasoning coherence.

### Strengths
1. The paper makes a compelling case that personalization can be viewed as reasoning adaptation rather than static fine-tuning, bridging interpretability, reasoning, and user modeling in a unified framework.
2. Extensive experiments on three benchmarks show consistent gains, and the ablation studies convincingly isolate the contribution of reasoning-based updates.

### Weaknesses
1. While the proposed framework is well-executed, its main components: reasoning trace modeling, memory-based personalization, and reflective updating, are each adapted from prior works. The contribution largely stems from combining these existing techniques rather than introducing a fundamentally new learning principle.
2. Most evaluations are short-term (few-shot) personalization; it remains unclear how the reasoning memory behaves in long-horizon or multi-user settings.
3. The evaluation omits several widely used personalization and reasoning benchmarks such as LAMP, which weakens claims of general applicability.

### Questions
1. How sensitive is the reasoning-based personalization process to the quality and correctness of generated reasoning traces?

2. What are the potential strategies to prevent error accumulation or bias drift in the reasoning memory over long-term interactions?

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
4

### Summary
The paper proposes GRASPER (Graph-based Sparse Personalized Reasoning), a two-stage framework. A pretrained graph link predictor augments a user’s sparse history by predicting plausible future interactions and generating synthetic reviews for them. A fine-tuned LLM generates both an explicit reasoning path and the final output, ensuring that synthetic and real contexts are coherently integrated and aligned with the user preferences.

### Strengths
1.	The focus on sparse user personalization aligns well with real-world deployment scenarios.
2.	GRASPER combines graph-based context expansion with explicit, trainable reasoning paths for personalized generation.
3.	The experiments are conducted on three benchmarks and evaluated under conventional NLP metrics and under LLM-as-a-Judge evaluation.

### Weaknesses
1.	GRASPER requires training a graph encoder and fine-tuning an LLM with multi-stage prompting. This may increase computational cost compared to prompt-only or retrieval-only baselines like LaMP or PGraph.
2.	The method relies on generating faithful synthetic reviews for predicted user–item interactions. Errors in link prediction or reasoning could mislead the final generation. While ablations show robustness, the quality of synthetic data is not assessed.
3.	The procedure for selecting the “golden” reasoning path uses conventional metrics (ROUGE+METEOR) that may not reflect personalization quality. Why not use LLM-as-a-Judge for path selection?

### Questions
All metrics do not evaluate the personalized alignment between the generated text and reference text, which only assess the language quality, similarity or accuracy. The evaluation may not prove the responses are personalized.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes GRASPER, a framework for personalized text generation under sparse user data. GRASPER first predicts likely user–item links to augment limited user histories using a graph-based link predictor, then generates synthetic texts for these predicted items with explicit reasoning steps, and finally produces personalized outputs based on both real and augmented data. Experiments on multiple benchmarks show that GRASPER outperforms several personalization baselines, and qualitative analyses demonstrate that it is able to capture both the semantic and stylistic preferences of users.

### Strengths
1. The paper tackles a practical problem of personalization under sparse user data, which is common in real-world applications.

2. The proposed GRASPER framework is conceptually clear, combining user history augmentation with reasoning-based generation to mitigate the potential noise from predicted items.

### Weaknesses
1. For experimental results in Tables 1-3, it is not clear to me how GRASPER is applied with GPT-4o mini. As described in Section 3.2, GRASPER requires fine-tuning the LLM for reasoning alignment.

2. In the main paper, it is not clear how sparse the benchmarks are (only some statistics are provided in the Appendix), and there is no analysis on how robust the method is to different sparsity levels.

3. For equation (9), the loss only involves $t_{u, j}$, however it is mentioned that the model is fine-tuned to jointly generate the reasoning path $Z^*$ and the text $t_{u, j}$. 

4. Is the link predictor trained on sparse or non-sparse user data? If trained on sparse user data, each user only has very few ground-truth links, could the trained link predictor potentially suffer from high uncertainty in its predictions?

### Questions
1. For Figure 3, is the user profile actually empty or shown as {} for visualization? If actually empty, is the example generation based on only predicted user-item interactions?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents GRASPER (Graph-augmented Reasoning-Aligned Sparse Personalized gEneration), a framework designed to enhance personalized text generation for users with limited historical data. The approach combines two key ideas: (1) graph-based context expansion, which leverages user–item interaction graphs to enrich sparse user histories via link prediction and synthetic texts; and (2) reasoning-aligned generation, which guides the model to generate personalized content by first producing an explicit reasoning path and then the final output, aligning synthetic and real user data. The method is evaluated on three domains (Amazon, Hotel, and Stylized Feedback) and across multiple tasks including long-text generation, short-text generation, and rating prediction. Experiments demonstrate consistent improvements over baselines such as LaMP, PGraphRAG, and REST-PG, supported by both automatic metrics and LLM-as-a-Judge evaluations.

### Strengths
1.By integrating graph-based expansion with reasoning-aligned fine-tuning, GRASPER provides a coherent pipeline that balances coverage (via graph expansion) and precision (via reasoning alignment). The modular design makes it interpretable and extendable to different backbone models.
2.Experiments cover multiple datasets, tasks, and backbones, demonstrating robustness and steady performance gains. Both automatic metrics (ROUGE, METEOR, MAE/RMSE) and LLM-as-a-Judge evaluations support the claimed improvements.

### Weaknesses
1.Limited conceptual novelty (A+B composition): The proposed framework essentially combines two established ideas — graph-based personalization for sparse users and reasoning-based generation alignment — into a single pipeline. Both components have been independently explored in prior works (e.g., Au et al., 2025; Salemi et al., 2025). While the integration is well-executed and empirically validated, the conceptual novelty is limited. The paper contributes mainly at the system level rather than proposing a fundamentally new learning principle or optimization mechanism.

2.Questionable personalization authenticity: The “similar user text” retrieved from the user–item graph provides co-occurrence and population-level preference signals rather than true individual-specific information. As such, the personalization improvement may stem more from global correlations (shared neighbor statistics) than from a deeper understanding of each user’s distinct style or intent. The paper does not include qualitative or quantitative analyses demonstrating that the generated content reflects unique user preferences instead of generic consensus patterns.

3.Potential noise amplification from graph augmentation: The link prediction model used for graph-based context expansion inevitably introduces spurious or noisy edges, especially in sparse or biased user–item graphs. These errors may propagate through the synthetic history and distort the personalized style, particularly when the predicted neighbors are semantically inconsistent.

### Questions
1.How do the authors ensure that the “similar user texts” extracted from the user–item graph provide personalized rather than population-level signals? Have the authors conducted any analyses (e.g., user-level style divergence, lexical overlap, or embedding similarity) to confirm that the generated outputs are truly distinctive to each user?
2.How sensitive the generation quality is to incorrect or noisy user–item edges?

### Soundness
3

### Presentation
3

### Contribution
3
