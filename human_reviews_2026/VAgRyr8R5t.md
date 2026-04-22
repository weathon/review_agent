# From Broad Recall to Exact Distinction: Adversarial Curriculum Learning for Knowledge-Based VQA

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
Knowledge-based Visual Question Answering (KBVQA) aims to answer image-related questions by retrieving relevant facts from an external knowledge base, making the accuracy of knowledge retrieval crucial.
However, a dominant bottleneck in existing systems is that inaccurate facts are fed to the answer generator.
This issue stems from two key deficiencies: (i) an initial retrieval stage that relies on global visual features, often overlooking fine-grained evidence, 
and (ii) a reranking stage that lacks the ability to differentiate between confusing candidates, making the correct answer a lower priority.
To address this, we propose the **Adv**ersarial **C**urriculum **L**earning (**Adv-CL**) framework, which tackles these two challenges sequentially. 
First, we design a Query-guided Multi-grained Recalling (QMR) strategy that leverages both global and query-guided local features to improve the recall quality and provide a diverse set of challenging negatives for reranker training.
Subsequently, to enable exact distinction, we introduce an Adversarial Reranker Training (ART) paradigm, which compels the reranker to discern fine-grained distinctions among highly similar candidates.
It employs a minimax game where a modulator network acts as an adversary against the reranker, dynamically creating a curriculum of hard negatives by up-weighting candidates that most confuse the reranker. This forces the model to develop its discriminative capability.
In addition, we further introduce a Guarded Answer Generation (GAG) mechanism to mitigate the risk of retrieval failure exacerbating the system hallucination.
Extensive experiments on public knowledge-based VQA benchmarks show that our method achieves state-of-the-art performance, validating the effectiveness and synergistic effect of broad recall and exact distinction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Adv-CL (Adversarial Curriculum Learning) for knowledge-based visual question answering (KBVQA).

Adv-CL includes three modules:
QMR (Query-guided Multi-grained Recalling): combines global and local query-guided features to improve recall.
ART (Adversarial Reranker Training): uses a modulator network in a minimax game to dynamically generate hard negatives.
GAG (Guarded Answer Generation): adds a check so the model can abstain when retrieved knowledge is unreliable.

On E-VQA and InfoSeek, Adv-CL achieves state-of-the-art results and improves both retrieval accuracy and answer reliability.

### Strengths
1. The adversarial curriculum dynamically adapts hard negatives and improves reranker learning.
2. Strong and consistent improvements on multiple benchmarks.
3. Clear visualizations of query-guided features and modulator behavior.

### Weaknesses
1. Each module builds on known techniques; the main innovation lies in the overall framework.
2. The joint contribution of QMR, ART, and GAG is not clearly separated. There should be a more clear table showing how each module contributes to the overall improvement
3. ART introduces an additional modulator transformer and adversarial optimization; the extra training cost and inference overhead (if any) are not clearly reported.
4. Only E-VQA and InfoSeek are tested. It would strengthen generalizability to include other KBVQA or open-domain RAG datasets (e.g., OK-VQA).
5. It seems that the literature review is not comprehensive. There are more works in OK-VQA (a popular KBVQA dataset as well) that should be mentioned and discussed.

### Questions
How sensitive is ART to the λ parameter in Eq. 3?

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
This paper proposes a three-stage framework called Adv-CL, which aims to improve the reliability and precision of KBVQA. The framework consists of three main components: Query-guided Multi-grained Recalling, Adversarial Re-ranker Training and Guarded Answer Generation.
Experiments on E-VQA and InfoSeek show that Adv-CL achieves state-of-the-art performance without fine-tuning large language models.

### Strengths
1. Training with a small model has relatively low costs.
2. GAG enhances the safety of the method.
3. Clear description of the method.

### Weaknesses
The methods seem to have no major issues, but there are several severe problems in the experiment.
1. The authors conduct experiments on E-VQA and InfoSeek and they provided the details of the InfoSeek in A.1.1. As described, the InfoSeek dataset consists of a training set and three evaluation sets. The authors did not specify which evaluation set they used for evaluation. It seems reasonable to report the results of each evaluation set separately.
2. ReflectiVA reports two settings on InfoSeek (28.3 and 40.1). Please clarify why your table uses 28.3 while ignoring 40.1?
3. For mR2AG, they conducted experiments on each evaluation set of InfoSeek. Why did the authors only use the worst-performing InfoSeek-Human as the baseline result? As baselines, both ReflectiVA and EchoSight claim their results are from InfoSeek's validation set. It seems the authors should use mR2AG's validation set results as the baseline instead of results of InfoSeek-Human.
4. The ablation study was conducted incompletely. The integration between the three modules in this article is not tight, so performing ablation on the full dataset should be relatively straightforward. An experimental setup similar to the main experiment is expected.
5. The case study is too brief. It would be better for the examples to demonstrate the complete workflow, including adversarial training and GAG.

Minor issues:
- Citation Format: arXiv:2411.15041,2024a. & arXiv:2411.15041,2024b.
- Typos: "repectively", A.1.4.

### Questions
See Weaknesses.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the challenge of accurate knowledge retrieval in Knowledge-based Visual Question Answering (KBVQA), where existing systems often feed incorrect facts into the answer generator. To improve both recall quality and reranking discrimination, the authors propose the Adversarial Curriculum Learning (Adv-CL) framework, composed of three components: (1) Query-guided Multi-grained Recalling (QMR), (2) Adversarial Reranker Training (ART), and (3) Guarded Answer Generation (GAG).  Experiments on two KBVQA benchmarks demonstrate improved performance.

### Strengths
1. The paper provides a new framework that integrates multiple stages (recall, reranking, and answer generation) into a unified training paradigm.
2. The experiments cover two public KBVQA benchmarks and show the incremental effects over the current approach.

### Weaknesses
1. The framework introduces three new modules (QMR, ART, and GAG) built on top of existing methods such as FAISS and EVA-CLIP. This significantly increases pipeline complexity and may amplify error propagation between stages. The benefit-to-complexity ratio is unclear.

2. While the full model achieves performance gains, it is not convincingly shown whether all three modules are necessary. A simplified or modular version might achieve comparable performance. The improvement margins in Table 1 are modest and not consistently significant (ps. the best result on E-VQA is mR2AG which should be bolded other than the proposed methods).

3. The paper does not isolate the specific contributions of the retriever and reranker components. It remains unclear how much each stage contributes to the final performance or whether the current paradigm is inherently limited by the capacity of the base models (i.e.,  LLM backbones).

4. I'd like to see computational comparison over existing approaches, which is misleading in the current scope.

### Questions
See above.

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
3

### Summary
This paper identifies retrieval quality as the main bottleneck in knowledge-based VQA, by observing the significant gap between accuracies of a same generator with ground-truth knowledge vs. retrieved knowledge. To address this, authors propose Adv-CL which is a three-stage pipeline. At stage one, the method uses a VLM to select most relevant patches to the input image, query-guided multi-grained Recalling, combines global image features with local patch features to improve knowledge recall. At stage 2, Adversarial Reranker Training through a minimax optimization between a modulator and reranker, in which modulator tries to assign higher scores to the most challenging negative samples, while reranker learns to minimize the contrastive loss, facilitating learning from truly challenging negative examples. At stage 3, guarded answer generation mechanism, to assess reliability of the retrieved knowledge and enable abstention when evidence is not reliable. Evaluations on two datasets, shows that the proposed method achieves higher VQA accuracy and better recall quality than the state of the art.

### Strengths
- Importance of the problem & key issues: The authors have identified a critical bottleneck in KBVQA models, where poor retrieval 
significantly deteriorates VQA performance. 
- Paper's motivation is sound. The paper grounds its design in three empirical observations: retrieval–generation gap, negative‑signal decay, and factual contamination
- The proposed is a plug-and-play method with frozen VLM/LLMs, and does not require LLM fine-tuning.
- The evaluations demonstrate that the proposed method achieves higher performance compared to baselines, and the method performances are consistent across three different LLMs.

### Weaknesses
- The paper do not provide a cost analysis of the proposed method in terms of the retrieval latency, reranking cost, and end-to-end costs.
- Details of the GAG stage are not provided. For example, details of the prompt inspection and discriminator are missing (this is important for reproducibility). Additionally, AP/AR/VAR are reported, but the trade‑off curve (against threshold) and its impact on final accuracy under different abstention policies aren’t shown.
- In table 2, mR²AG shows a higher score (55.9) on E-VQA dataset, than the proposed. Additionally, the table mixes methods & generators, hence the baseline comparisons are not apple-to-apple comparisons. The paper should present the results in a way to facilitate comparison of methods on the same generator, for a fair comparison.
- Moderate novelty: Multi‑grained retrieval and dynamic hard‑negative mining are known ideas. Paper's novelty lies in (a) operationalizing query‑guided patch features for the retrieval stage and (b) casting negative weighting as an adversarial curriculum with an entropy‑regularized budgeted modulator. Additionally, GAG abstention mechanism is pragmatic rather than novel.

### Questions
- What is the cost and latencies of the method (end-to-end) and per component? 
- How is the prompt inspection for GAG designed? Is the abstention decision a binary decision made by LLM, or does the LLM return an abstention score?
- Plot AP, AR, VAR trade-off curves across thresholds and show overall accuracy change under different abstention policies (e.g., fixed refusal budget).

### Soundness
3

### Presentation
3

### Contribution
2
