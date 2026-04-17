# SALMAN: Stability Analysis of Language Models Through the Maps Between Graph-based Manifolds

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Recent strides in Pretrained Transformer-based language models have propelled state-of-the-art performance in numerous NLP tasks. Yet, as these models grow in size and deployment, their robustness under input perturbations becomes an increasingly urgent question. Existing robustness methods often diverge between small-parameter and large-scale models (LLMs), and they typically rely on labor-intensive, sample-specific adversarial designs. In this paper, we propose a unified, local (sample-level) robustness framework (SALMAN) that evaluates model stability without modifying internal parameters or resorting to complex perturbation heuristics. Central to our approach is a novel Distance Mapping Distortion (DMD) measure, which ranks each sample’s susceptibility by comparing input-to-output distance mappings in a near-linear complexity manner. By demonstrating significant gains in attack efficiency and robust training, we position our framework as a practical, model-agnostic tool for advancing the reliability of transformer-based NLP systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a novel way to assess the robustness of an individual sample. To achieve this, the authors estimate the geometry of the input space and output space and compare the distance samples move in the input space versus the output space. Based on this idea, they derive a score they call the SALMAN score, which estimates how a sample p moves with respect to samples in its neighborhood. To compute this score efficiently, the authors propose a novel approach to estimate a precision matrix that captures local connectivity. Next, they show that their score outperforms simple baselines, can be used to guide adversarial attacks, and achieves more efficient adversarial fine-tuning.

### Strengths
- The problem statement is clearly motivated.
- The claims are mostly supported theoretically and with experiments.
- The new method for optimizing the precision matrix is both elegant and well justified
- Connecting input and output geometry using a graph-derived geometry to estimate sample-specific robustness is novel and, given the results, promising.
- Showing that the geometry of the input space is a valuable contribution.

### Weaknesses
Weaknesses

- The paper would benefit greatly from a dedicated preliminaries section that introduces the necessary concepts, such as PGMs and the metrics used in the paragraph “Effective-resistance distance.” Additionally, notation such as e_{p,q} is not
introduced.


- The objective for the PGM is concave and can be solved efficiently with projected gradient descent. It would be important to evaluate how well your algorithm approximates the optimal solution. This can also happen in a smaller setting where the typical O(N^3) algorithm still terminates in a reasonable time.

- It is not clear how using the embeddings allows you to make statements about the input–output manifold. The embeddings after MHSA are not really inputs anymore. This requires more explanation and/or proofs.

- Reporting the cosine similarity makes sense, yet it underballs the difference between the embeddings. The arccos is not linear in that domain. To give an example, arccos(0.9772)$\approx$12.26° and 
arccos(0.9981)=3.53°, which shows quite a difference, while the cosine similarities that you report do not. This would both emphasize your contribution and simplify the interpretation of the results.

- In my opinion, the section: Salman-Guided attack, takes the wrong storyline. In an attack scenario, you cannot necessarily choose the sample you want to attack. This section would be better positioned as verifying that your ranking correlates strongly with how easy it is to attack a sample.

- Another simple baseline would be to estimate the local curvature in embedding space by sampling in an epsilon ball around the embedding and assessing how the output changes. This can be very efficiently computed with appropriate caching and batched inference.

Minor:
- The Theorems are too informal; it would be better to have more detailed statements to know exactly what is proven.

- It would be better to first introduce the intuition of your SALMAN score and then show the formulas.

Note: The paper introduces an interesting novel method that is theoretically well-founded and shows promising results. Yet, it does not properly present these and lacks the necessary details to understand the method and experiments in full detail. Thus, I am leaning toward a reject, but with appropriate improvements, clarifications, and extra baselines, I am inclined to raise my score without making any promises to do so.

### Questions
How does your approach solve the problem of the discrete nature of token embeddings if you still use embeddings?

How exactly do you generate the PGM? And what is probabilistic here?

Can you comment on how your approach relates to estimating the local curvature around a sample?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
In this work, a new approach to analyze the language model's stability, is proposed. It's a training free approach, without modifying internal parameters or resorting to complex perturbation heuristics. The key idea is  Distance Mapping Distortion (DMD) measure, which ranks each sample’s susceptibility by comparing input-to-output distance mappings in a near-linear complexity manner. Multiple models are evaluated with this metric, and multiple finetuning experiment showed  this sample-level perspective leads to (i) more efficient and higher
success-rate adversarial attacks, and (ii) improved robust fine-tuning outcomes.

### Strengths
- The paper is well-structed and easy to follow.

### Weaknesses
- One concern is about the alignment between the new proposed stability metric and the existing proposed stability metrics. The definition is reasonable and the results looks promising, but it's not compared with existing stability metrics. Ideally in most cases the newly proposed approach align with existing metrics in most cases, and stands out for some special tasks to highlight the value.

### Questions
- Embedding aggregation stage. To represent the input & output sequence, an attention based aggregation approach is proposed. Have we tried simpler approaches like sum pooling or average pooling to get the aggregated embedding? In representation learning, different aggregation approaches do not make a huge difference.

- One concern is about the alignment between the new proposed stability metric and the existing proposed stability metrics. The definition is reasonable and the results looks promising, but it's not compared with existing stability metrics. Ideally in most cases the newly proposed approach align with existing metrics in most cases, and stands out for some special tasks to highlight the value. Is it possible to compare with existing stability metrics? 

- Deterministic hidden state embedding. According to some recent work like [1] even the decoding approach is fixed, the LLM inference is still non-deterministic. Have we observe similar behavior? 

https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

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
4

### Summary
The paper proposes a robustness measure for an LLM. It compares representations from the first and last layers, using DMD. A wide range of experiments demonstrates that the introduced measure is related to variants of LLM robustness, including vulnerability to adversarial attacks and LLM fine-tuning.

### Strengths
- The article proposes numerous ways to evaluate a robustness measure for an LLM
- The authors completed a large body of numerical experiments, with some of them that I had never encountered. Maybe, focusing on specific ones should be of a separate interest (and the overall positioning of a paper as a framework for LLM robustness evaluation from multiple points of view)
- Clear structure, easy to understand the contributions, easy to read.

### Weaknesses
The main concern for me in this paper is a weak answer to the question: Is the introduced SALMAN measure better than other possible measures (and the best overall)? Specifically,
- Weak connections to topological methods are presented. See [1] for a review and [2] for a specific MTop-Div similarity measure (also CKA can be used this way, I believe), suitable for comparing "input" and "output" embeddings, as applied previously to LLMs in e.g. [3]. See also another review on the similarity measure between neural networks, suitable for your problem [4], to mine for additional baselines.
- No connection to the body of literature related to uncertainty estimation in LLMs. They naturally provide OOD detection capabilities, look at the problem from different perspectives [5], decompose uncertainty into epistemic and aleatoric components [6], develop frameworks for evaluating robustness estimates across various scenarios [7], and include adaptive RAG [8].
- A sensitivity study that focuses on the usability of a specific metric with few ablation studies. No comment on whether we can ignore \gamma^3 of \gamma^{-3} (see Formula 4), what one should consider as input and output (a questions that makes sense given the existing works [9, 10] that states that you need to focus some layers in the middle for hallucination detection or specific pairs of heads for efficient robustness evaluation). 
- Overall, the work looks like a derivative of SAGMAN. A better positioning of why it should be a separate work would be helpful.
- The formulation of theorems can be improved. For example, consider Theorem 3.5. What is $∝$? In other theorems, I feel the need to understand what exactly is proved (e.g., what exact algorithm does Theorem 3.1 consider? Theorem 3.2 - why use words with the root "effective" three times in the statement without explicitly defining what you mean).
- Baselines (1) Euclidean distance (ED) and (2) Jacobian-based sensitivity (JBS) are not defined. It is hard to understand what they do, what their computational complexity is, and how to reproduce results from these baselines (e.g., which hyperparameters to tune?).
 - Protocols and results for fine-tuning were not clear to me. Maybe it can be improved. Also, in Table 6, please correct highlighting: 94.84 is not better than 94.84 (a similar observation holds for 78.34)

Minor misprints:
- Equation equation 1 -> Equation 1
- Then. We apply -> Then we apply (I feel that paragraph 3.4. Complexity deserves another round or two of proofreading)
- Recent work (Dong, 2019) - in my opinion, 2019 is not recent given the pace of the development of deep learning

1. Uchendu, Adaku, and Thai Le. "Unveiling topological structures in text: A comprehensive survey of topological data analysis applications in nlp." arXiv preprint arXiv:2411.10298 (2024).
2. Barannikov, Serguei, et al. "Manifold Topology Divergence: a Framework for Comparing Data Manifolds." Advances in neural information processing systems 34 (2021): 7294-7305.
3. Tulchinskii, Eduard, et al. "Intrinsic dimension estimation for robust detection of ai-generated texts." Advances in Neural Information Processing Systems 36 (2023): 39257-39276.
4. Klabunde, Max, et al. "Similarity of neural network models: A survey of functional and representational measures." ACM Computing Surveys 57.9 (2025): 1-52.
5. Shorinwa, Ola, et al. "A survey on uncertainty quantification of large language models: Taxonomy, open research challenges, and future directions." ACM Computing Surveys (2025).
6. Hüllermeier, Eyke, and Willem Waegeman. "Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods." Machine learning 110.3 (2021): 457-506.
7. Fadeeva, Ekaterina, et al. "LM-Polygraph: Uncertainty Estimation for Language Models." Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. 2023.
8. Moskvoretskii, Viktor, et al. "Adaptive retrieval without self-knowledge? bringing uncertainty back home." arXiv preprint arXiv:2501.12835 (2025).
9. Kostenok, Elizaveta, Daniil Cherniavskii, and Alexey Zaytsev. "Uncertainty estimation of transformers' predictions via topological analysis of the attention matrices." arXiv preprint arXiv:2308.11295 (2023).
10. Sky, CH-Wang, et al. "Do Androids Know They're Only Dreaming of Electric Sheep?." Findings of the Association for Computational Linguistics ACL 2024. 2024.

### Questions
- How your similarity measure corresponds to other baselines that come fron uncertainty estimation, sensitivity analysis and topological data analysis domains?
- What is the exact defintion of the used baselines ED and JBS?
- Is it possible to improve your similarity measure by turning on of off some of its components, adjusting hyperparameters, etc.?
- What is the difference with SAGMAN?

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
This paper proposes a new measure for understanding the robustness of different inputs to adversarial perturbations, based on constructing a graph-based distance metrics between input and output embeddings. It does so using a near-linear complexity method for computing the manifold.

It then empirically analyzes the usefulness of the ratio between input and output distances as a method for detecting robust and non-robust regions of the input space. First, it evaluates how well the DMD metric corresponds to susceptibility to input perturbations (deletion, edit, swap, spaCy, and TextAttack), comparing against euclidean distance rankings, and jacobian-based sensitivity analysis. Then, it shows that non-robust samples as predicted by DMD are easier to adversarially attack with GCG and AutoDan. Finally, it shows that downweighting robust samples and upweighting non-robust samples increases the robustness of the model.

### Strengths
Originality: The paper proposes an original method for evaluating the per-sample stability of LLM outputs.
Quality: The paper derives a novel algorithm, then investigates its core claims with empirical experiments.
Clarity: The paper is generally clearly written.
Significance: LLM robustness is an important area.

### Weaknesses
W1 (Significance): It's unclear how to really interpret the cosine similarities in section 4.1 -- for the most part the similarities are high for robust and non-robust examples, and while the experiment shows that the metric better corresponds to sensitivity to changes in the input space, it's not really clear what the implications are.
W2 (Significance): The paper combines motivations from both adversarial robustness, and robustness to random text modifications. It's unclear to me what benefits there are to being robust to, for instance, randomly dropping words, and the paper doesn't spell this out very much.

### Questions
Q1) Can you add standard errors to the different similarity scores? The numbers are often close-ish to each other, and it would be helpful for understanding the statistical significance of the results.
Q2) Are the models undergoing robust training more robust to adversarial attacks?

### Soundness
3

### Presentation
3

### Contribution
2
