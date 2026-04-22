# Contrastive Pre-Training for Multimodal Multi-Hop Question Answering Representations

- Avg Score: 2.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2

## Abstract
The multimodal multi-hop question-answering(MMQA) task is the most representative multimodal reasoning task, with its primary goal being to perform multi-step logical reasoning based on multimodal questions to obtain accurate answers. Existing MMQA methods based on large language models (LLMs) have progressed; however, this study still faces challenges in fusing multimodal reasoning features and reasoning with multi-hop questions. To address the above issues, we propose a multimodal multi-hop representation-based contrastive pre-training (MMRCP) approach, which can effectively fuse multimodal and multi-hop question-answering features to enhance the reasoning performance of question-answering tasks. It employs two loss functions for contrastive learning training: cross-modal contrastive learning and reasoning-aware contrastive learning, which effectively obtain basic multimodal semantic and question-answering reasoning features. Subsequently, we construct a multi-hop representation fusion module that combines multimodal reasoning features to perform lightweight adaptation for multi-hop question answering reasoning tasks. Ex-tensive experiments on three real-world multi-hop question-answering datasets demonstrate that MMRCP outperforms multi-hop question-answering baselines by 3% and 4% in precision and error rate, respectively. MMRCP provides a promising direction for future multimodal reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes MMRCP, a multimodal multi-hop representation approach using dual contrastive learning to integrate multimodal and multi-hop QA features. It introduces a multi-hop representation fusion module for lightweight adaptation, achieving improved reasoning performance on three multi-hop QA datasets.

### Strengths
The paper is original in combining cross-modal and reasoning-aware contrastive learning for multimodal multi-hop QA. Experiments demonstrate performance gains.

### Weaknesses
The paper suffers from unclear notation in Figure 2 and formulas, making the methodology hard to follow (e.g., “embedding of …”, $N_{rand}$, $\mathcal{V}_{rel}$). The writing format deviates from standard templates, with significant structural issues. Experimental results are underwhelming, and method descriptions lack sufficient clarity and rigor, limiting reproducibility and impact.

### Questions
1. Some symbols in Figure 2 are difficult to interpret, such as “embedding of …”, and require clearer representations.

2. The writing template used in the paper deviates significantly from the standard, and the formatting has major issues.

3. The methodology section is poorly described; many symbols are undefined, making formulas hard to understand (e.g., $N_{rand}$ and $\mathcal{V}_{rel}$ etc.).

4. The experimental results are not very convincing (Table 1), raising concerns about the effectiveness of the proposed method.

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
5

### Summary
This work proposes contrastive pre-training with cross-modal contrastive learning loss and reasoning-aware contrastive learning loss, in order to more effectively integrate multimodal features for multimodal multi-hop QA reasoning. Also, to establish causal relationships between multimodal semantic representations, the work integrates a multi-hop QA reasoning representation based on the multimodal semantic representation and performs lightweight adaptation for downstream multi-hop QA task. Experimental results show that the resulting MMRCP method outperforms baselines on three real multi-hop QA datasets by 3% and 4% in precision and error rate.

### Strengths
1.	For contrastive pre-training, the cross-modal contrastive learning loss is a common method that has been explored in many works; however, as the authors pointed out, it is insufficient for structured dependency modeling and learning compositional semantics. The reasoning-aware contrastive learning loss is hence clearly motivated and has valuable novelty to the research topic. Three reasoning-aware tasks are designed for learning multimodal compositional reasoning structure. Specifically,  chain-aware semantic alignment learns fine-grained correspondence between textual components and visual concepts along reasoning chains. Reasoning path invariance helps model distinguish core reasoning signal and less relevant ones and improves robustness. Cross-modal graph matching learns preserving relational semantics across modalities.  The three losses are integrated through adaptive weighting.  Overall, reasoning-aware contrastive learning is helpful to inspire future multimodal reasoning research.

2.	Note that the three reasoning-aware tasks generate heterogeneous but complementary representations, while multimodal multi-hop QA requires context-aware integration of these representations. Hence, the paper introduces the Multi-hop Representation Fusion module through attention-based gating on question semantics. 

3.	The ablation study verifies that all four modules contribute to the performance, with cross-modal CL having the greatest impact, reasoning-aware contrastive learning the second greatest, and multi-hop representation fusion the least.

### Weaknesses
1.	Some citations are missing. For example, Elastic Weight Consolidation (EWC) is developed in the prior work and needs to be cited.

2.	For evaluation metrics, it is not clear how the distances are measured between predicted answers and reference answers to reliably reflect semantic relevance between them. 

3.	The rationale of selecting the evaluation datasets is not clear. Also, the rationale of selecting the baselines (including MHCQA for reasoning the answer) needs to be provided.

4.	Many gains from MMRCP over the best baseline in Table 1 are quite small, the standard deviation from the three random runs and statistical significance test results need to be reported.

5.	It would also be useful to evaluate top open-sourced and closed-source MLLMs such as Qwen2.5-VL, Gemini 2.5 pro and GPT-4o to gain more insights about the positioning of the performance of MMRCP w.r.t. these top-tier  open-sourced and closed-source MLLMs for multimodal multi-hop QA.

6.	The authors argue that the architecture supports strong zero-shot transfer to new QA formats. Yet it is not clear how the current lightweight finetuning-based evaluation setup supports evaluation of zero-shot performance on emergent multimodal multi-hop QA scenarios.

### Questions
1.	There are some presentation issues. 

a.	For example, the empty lines of Line 040-041 shouldn’t be there. Line 179 is ungrammatical. Line 187, there should be a period before “However”.

b.	Please make sure all math symbols are defined. For example, Line 251.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper propose a multimodal multi-hop representation-based contrastive pre-training method to fuse multimodal and multi-hop question-answering features to enhance the QA tasks. It provides a dual contrastive learning method and integrates multi-hop question-anwering representation. The results demonstrate the MMRCP outperforms baselines.

### Strengths
1. the proposed MSE_K and NRMSE_K can quantifies the difference between predicted answer and actual answer.

2. the pretraining work needs more effort than prompting-based work.

3. the key contributions of this submission are easy for reader to understand and follow.

### Weaknesses
1. The fomat of this submission is bad. It is more like a blog not a paper. Authors use lots of vspace to compress the main contents and make the paper very difficult to read. For example:

Line 39 - Line 42, I think it is very obvious in first page.

Line 223 - Line 224, Line 272 - Line 273, Line 316 - Line 317, Line 392 - Line396.

Line 266, Line 273 and Line 280 are not aligh with each other. 

2. it is not clear for the definition of distance in evaluation metrics.

3. The framework figure is difficult to follow. It will be better to change the legands.

4. some baselines seem unrelated to multi-hop, multi-modal QA.

5. section 4.5 seems like totally AI generated contents, especially the third paragraph from Line 437 - 446. 

6. no reproducibility details.

### Questions
please refer to the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
3
