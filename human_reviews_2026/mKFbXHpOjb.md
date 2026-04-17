# Tabular Learning with Background Information: LLMs, Knowledge Graphs, or Both?

- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
Tables have their own structure, calling for dedicated tabular learning methods with the right inductive bias. These methods outperform language models. Yet, many tables contain text that refers to real-world entities, and most tabular learning methods ignore the external knowledge that such strings could unlock. Which knowledge-rich representations should tabular learning leverage? While large language models (LLMs) encode implicit factual knowledge, knowledge graphs (KGs) share the relational structure of tables and come with the promise of better-controlled knowledge. Studying tables in the wild, we assemble 105 tabular learning datasets comprising text. We find that knowledge-rich representations, from LLMs or KGs, boost prediction, and combined with simple linear models they markedly outperform strong tabular baselines. Larger LLMs provide greater gains, and refining language models on a KG boosts models slightly. On datasets where all entities are linked to a KG, LLMs and KG models of similar size perform similarly, suggesting that the benefit of LLMs over KGs is to solve the entity linking problem. Our results highlight that external knowledge is a powerful but underused ingredient for advancing tabular learning, with the most promising direction lying in the combination of LLMs and KGs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a large-scale empirical study to determine the best source of background knowledge for tabular learning tasks, specifically focusing on textual data within tables. The authors compare representations derived from pure Large Language Models (LLMs), structured Knowledge Graphs (KGs), and hybrid models that refine LLMs on KGs. Using a newly assembled benchmark of 105 tabular datasets containing text, the study finds that knowledge-rich representations significantly boost predictive performance, even allowing simple linear models to outperform strong tabular baselines like XGBoost. A key finding is that larger LLMs provide greater gains. Furthermore, on a controlled subset of 15 tables with pre-solved entity linking, pure KG embeddings perform on par with LLMs of similar size. This leads the authors to conclude that the primary advantage of LLMs is not superior knowledge storage but their innate ability to solve the "symbol grounding" or entity linking problem. The paper advocates for hybrid LLM+KG approaches as the most promising future direction.

### Strengths
1. This paper addresses the important and practical problem of leveraging external knowledge from textual features in tabular data . The assembly of a new, large-scale benchmark of 105 datasets is a significant contribution to this area of research.
2. The study provides a large-scale, systematic comparison across a wide spectrum of models. This includes multiple LLM families (e.g., Llama-3, Qwen3, ROBERTa, T5) and sizes , various hybrid LLM+KG models (e.g., ERNIE, KGT5, Knowledge Card) , classic pure KG embedding methods, and several distinct downstream tabular learners (Ridge, XGBoost, TabPFNv2).
3. The paper clearly articulates the conceptual gap between traditional tabular learning and knowledge-rich modeling. This framing highlights why background knowledge matters for textual tables and situates the problem within the broader challenge of symbol grounding and structured reasoning, all presented with clear and precise exposition.

### Weaknesses
1. My most significant concern is the paper's narrow definition of its own problem domain. In Section 3.1, the authors state, "we remove all numerical columns to focus our study on text-based knowledge". This single methodological choice fundamentally changes the problem from heterogeneous tabular learning (the domain of XGBoost, TabPFNv2, and real-world tables) to a short-text prediction problem. This makes the central claim (Finding 1) that their method "outperforms strong tabular baselines" deeply problematic. The SOTA baselines (XGB, TabPFN) are being benchmarked in an artificial setting they were not designed for (text-only). A fair comparison would require a heterogeneous setup.
2. The SOTA comparison in the paper is incomplete. The authors claim to “markedly outperform strong tabular baselines,” yet their evaluation is limited to models such as XGBoost and TabPFNv2. More critically, the study omits comparisons with genuine state-of-the-art methods for deep tabular learning—such as **FT-Transformer**, **SAINT, CARTE**, or competitive **MLP-based baselines**—which are explicitly designed to handle the kind of heterogeneous tabular data that this paper excludes. Additionally, **LLM-based approaches** like **TabLLM** are not considered, further weakening the strength of the claimed performance advantages.
3. The paper's conclusions are drawn from an experimental setup heavily skewed towards small-data regimes. The methodology explicitly states, "To simulate small-data scenarios... we sample training sets of varying sizes, $n_{train}\in\{64, 256, 1024\}$"14. All major results and ranking diagrams (e.g., Fig. 7, 18) are reported at $n=1024$. While the rationale (external knowledge is most critical here) is valid, this is a "cherry-picked battlefield" that maximizes the utility of external knowledge and disadvantages GBDTs, which are known to excel as $n_{train}$ grows. The paper provides no evidence that these conclusions generalize to larger, more realistic training sets (e.g., $n_{train} > 10,000$).
4. The novelty of the second finding ("Refining LLMs on KGs is a promising combination") is limited. As the paper's own related work section details, this is a well-established research direction. The paper's contribution here is to validate this on their benchmark, but the finding that this refinement only "boosts models slightly"  makes this contribution feel incremental rather than a significant breakthrough.

### Questions
1. To make the comparison to SOTA tabular learners fair and the conclusions relevant to *tabular learning*, the authors should include experiments on the original, heterogeneous data. The most direct approach would be to **concatenate** the numerical features (which were removed) with the new text-based row embeddings. How do the results change in this (more realistic) setting?
2. Can the authors provide results on at least a medium-sized training set (e.g., $n_{train}=10240$) to demonstrate the generalizability of their findings?
3. The authors are encouraged to extend their SOTA comparison by including modern deep tabular and LLM-based baselines—such as FT-Transformer, SAINT, DANet, TabNet, or TabLLM. Including these would greatly strengthen the validity of the paper’s claims and situate the proposed method more clearly within the current landscape of tabular learning research.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a large-scale empirical study on leveraging external background knowledge—specifically from large language models (LLMs) and knowledge graphs (KGs)—to improve tabular learning on datasets containing textual entity mentions. The authors assemble a benchmark of 105 real-world and semi-synthetic tabular datasets, systematically evaluate a wide range of representation methods (including pure LLMs, pure KG embeddings, and LLMs refined on KGs), and analyze performance under controlled settings—most notably using a subset of 15 tables with pre-linked Wikidata entities. Key findings include: (1) knowledge-rich representations significantly outperform traditional text encodings (e.g., TF-IDF); (2) larger LLMs yield consistent gains; (3) refining LLMs on KGs provides modest but reliable improvements in performance and parameter efficiency; (4) when entity linking is solved, KG embeddings perform on par with same-sized LLMs, suggesting that LLMs’ main advantage lies in implicit symbol grounding rather than superior knowledge quality.

### Strengths
- Comprehensive and well-structured benchmark: The collection of 105 diverse datasets from three distinct sources (TextTabBench, CARTE, WikiDBs) provides strong external validity. The construction of a 15-table “entity-linked” subset is a methodological highlight that enables clean isolation of the entity linking factor.  

- Clear, actionable insights: The paper convincingly demonstrates that representation quality—not just model architecture—is the bottleneck in text-rich tabular learning. The conclusion that LLMs primarily help by solving symbol grounding is both nuanced and valuable for the community.  

- Reproducibility: Experimental protocols (train sizes, random seeds, PCA dimensionality, estimator choices) are thoroughly documented, and runtime costs are reported—enhancing practical utility.  

- Balanced model coverage: The evaluation spans a wide spectrum of models, from classic KG embedders (RotatE, ComplEx) to modern LLMs (Llama-3, Qwen3) and hybrid approaches (Knowledge Card, ERNIE).

### Weaknesses
- Narrow scope of KG evaluation: Pure KG models are only evaluated on the 15 linked tables, which represent a best-case scenario. The paper does not assess how KG-based methods degrade under realistic, noisy, or partial entity linking—thus overestimating their practical applicability.  

- Lack of privacy or robustness considerations: Given the focus on external knowledge, the paper overlooks critical issues such as leakage of sensitive entities via embeddings, vulnerability to adversarial entity perturbations, or compatibility with privacy-preserving learning (e.g., federated or differentially private settings).  

- Downstream estimator mismatch: The use of PCA to compress high-dimensional LLM/KG embeddings before feeding them to XGBoost or TabPFNv2 may discard useful structure. The paper does not explore alternative integration strategies (e.g., late fusion, attention-based conditioning).

### Questions
- Extend KG experiments to realistic linking scenarios: Include experiments with automatic entity linking (e.g., using BLINK or OpenTapioca) and report performance as a function of linking accuracy. This would bridge the gap between idealized and real-world deployment.  

- Discuss privacy and security implications: Even a brief discussion of risks (e.g., membership inference from KG-enhanced embeddings) would align the work with contemporary concerns in data-centric AI.  

- Explore alternative integration mechanisms: Beyond simple embedding + linear model, consider lightweight adapters or cross-attention modules that preserve the geometry of knowledge-rich representations when used with tabular foundation models.  

- Clarify the role of column context: While row serialization includes column names, the ablation on contextual disambiguation (e.g., “Cambridge, UK” vs. “Cambridge, MA”) is only mentioned qualitatively. A quantitative analysis would strengthen the claim.

### Soundness
3

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
This paper investigates how external knowledge can enhance tabular learning, particularly when tables include textual fields referencing real-world entities. The authors systematically compare two major sources of background knowledge:
(1) LLMs that implicitly encode extensive factual and semantic information; and
(2) Knowledge Graphs (KGs) that provide explicit, curated relational structures but depend on entity linking.
To enable a controlled comparison, the paper introduces a benchmark of 105 tabular datasets (drawing from TextTabBench, CARTE, and WikiDBs), encompassing both classification and regression tasks. A wide range of representation strategies are evaluated, e.g. non-pretrained encoders, LLM-based embeddings, KG embeddings, and hybrid LLM+KG refinements—paired with several tabular predictors, including ridge regression, XGBoost, and TabPFNv2. The study reveals several key findings.

### Strengths
- The work addresses an underexplored yet practically important research question—how to inject background knowledge into tabular learning—bridging symbolic reasoning (KGs) and neural representation learning (LLMs).
- The experiments cover a broad spectrum of models, from lightweight text encoders to multi-billion parameter LLMs, as well as multiple downstream learners. This comprehensive setup strengthens the credibility of the conclusions.
- The empirical observation that LLMs and KGs converge in performance after entity linking offers an interesting theoretical insight into how implicit and explicit factual knowledge may complement each other.

### Weaknesses
- The study mainly explores feature-level embeddings followed by downstream predictors. It omits more advanced fusion methods (e.g., cross-attention, joint training, or representation alignment) that could yield richer interactions between tabular and knowledge-based features.
- The finding that ridge regression outperforms more complex learners may stem from dimensionality reduction artifacts (e.g., PCA bottlenecks), potentially underestimating the capabilities of non-linear models like XGBoost and TabPFNv2.
- The paper reports aggregate metrics but does not provide qualitative case studies to illustrate where LLMs or KGs perform particularly well or poorly (e.g., domain-specific tables, rare entities, or ambiguous text).
- The definition of “refinement” is inconsistent across baselines (ERNIE, Knowledge Card, KGT5). Without further controlled ablations isolating architecture, scale, and pretraining data, it remains unclear what drives the observed gains.
- Some Wikipedia-derived datasets may resemble document classification rather than genuinely heterogeneous tabular learning, weakening claims about handling tabular structure.
- The paper could better position itself relative to: 1) Retrieval-Augmented Generation (RAG) approaches that dynamically incorporate knowledge. 2) Multimodal table encoders such as TaBERT, TURL, and TAPAS, which explicitly integrate table structure and text.

### Questions
- Could retrieval-based approaches (e.g., RAG-style KG or LLM lookups) outperform static embeddings while retaining interpretability?
- How does the method behave under noisy or partial entity linking? Could uncertainty in linking be explicitly modeled?
- Have the author(s) considered multi-column or relational dependencies, such as type hierarchies or foreign-key relationships, beyond row-level concatenation?
- Would fine-tuning smaller models with knowledge-based pretraining narrow the performance gap with large LLMs?
- Could benchmarking against retrieval-based tabular models (e.g., RAG-TAB, KnowTab) provide deeper insight into dynamic vs. static knowledge integration?
- Why does ridge regression outperform TabPFNv2 after embedding projection—does PCA distort representation geometry or reduce model flexibility?
- Are the performance improvements primarily due to semantic enrichment (better factual knowledge) or dimensional expansion (higher embedding capacity)?


I would consider raising my score if the authors can adequately address these questions.

### Soundness
3

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
4

### Summary
This paper investigates how external background knowledge can be integrated into tabular learning, focusing on textual columns that reference real-world entities (e.g., drugs, companies, locations).
The authors benchmark 105 text-containing tabular datasets and compare representations derived from **large language models (LLMs)**, **knowledge graphs (KGs)**, and **hybrid LLM + KG models**.
They find that knowledge-rich representations substantially improve downstream prediction—often more than using sophisticated tabular learners—and that larger LLMs provide greater gains. Refining LLMs on KGs improves parameter efficiency, and in the idealized case where all entities are perfectly linked, KG embeddings perform on par with LLMs.
The study concludes that combining LLMs and KGs is a promising direction for future tabular foundation models.

### Strengths
* The paper tackles a novel and meaningful problem—bridging tabular learning and external knowledge.
* Comprehensive empirical study across 105 datasets, with diverse textual attributes.
* Systematic comparison of LLM, KG, and hybrid models; clear identification of the entity-linking bottleneck.
* Results suggest interesting scaling trends and show that “representation quality > model complexity” in importance.

### Weaknesses
* **Over-restrictive assumptions:** removing all numerical features and focusing solely on text columns creates an artificial setting; results may not generalize to realistic multi-modal tables.
* **Limited real-world relevance:** experiments are confined to small-data regimes (64 / 256 / 1024 samples), which are uncommon in industrial tabular tasks.
* **Lack of raw quantitative results:** only normalized gains are reported; absolute AUC/R² improvements may be modest.
* **No discussion of KG construction or cost:** while KG embeddings are used, the paper does not analyze the effort required for entity linking or graph maintenance, undermining claims of practical benefit.
* **Reproducibility issues:** key configuration details (exact sampling splits, variance across seeds, hyper-parameters for embedding extraction) are only briefly mentioned.

### Questions
1. Since the same test sets are used across training sizes {64, 256, 1024}, how does performance evolve with more training samples? Do the relative advantages of LLM / KG embeddings diminish as data grows?
2. Could the authors release or at least summarize the **raw experimental tables** (AUC/R² per dataset) to enhance reproducibility and allow independent meta-analysis?
3. A valuable extension would be to re-introduce **numerical features** and study how numerical and textual features interact—are their contributions orthogonal or redundant? This would make the findings more applicable to real-world tabular pipelines.

### Soundness
3

### Presentation
3

### Contribution
2
