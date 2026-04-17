# SCRIBES: Web-Scale Script-Based Semi-Structured Data Extraction with Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Semi-structured content in HTML tables, lists, and infoboxes accounts for a substantial share of factual data on the web, yet the formatting complicates usage, and reliably extracting structured information from them remains challenging. Existing methods either lack generalization or are resource-intensive due to per-page LLM inference. In this paper, we introduce SCRIBES (**SCRI**pt-**B**ased Semi-Structured Content **E**xtraction at Web-**S**cale), a novel reinforcement learning framework that leverages layout similarity across webpages within the same site as a reward signal. Instead of processing each page individually, SCRIBES generates reusable extraction scripts that can be applied to groups of structurally similar webpages. Our approach further improves by iteratively training on synthetic annotations from in-the-wild CommonCrawl data. Experiments show that our approach outperforms strong baselines by over 13\% in script quality and boosts downstream question answering accuracy by more than 4\% for GPT-4o, enabling scalable and resource-efficient web information extraction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SCRIBES, a reinforcement learning (RL) framework for scalable information extraction from semi-structured web data (HTML tables, lists, infoboxes). Rather than invoking a large language model (LLM) for each page individually, SCRIBES trains an LLM to generate reusable extraction scripts that can be applied to groups of structurally similar webpages (e.g., pages from the same site).

### Strengths
The script-generation idea bridges wrapper induction and LLM-based extraction, offering a path to web-scale efficiency.

The use of cross-page generalization as a verifiable RL reward is elegant and practical.

The empirical results also show some promising evidence.

### Weaknesses
The system largely combines existing ingredients (RLVR, GRPO, LLM-as-a-judge, fuzzy matching) rather than introducing a fundamentally new algorithm. This is my biggest concern to give a positive score. The “fuzzy F1” and LLM-judged metrics may still be noisy; no human verification of RL reward quality was reported. In addition, although SCRIBES aims for efficiency, it still relies on large proprietary models (e.g., GPT-4o, Qwen-32B) for training and reward calculation, limiting replicability and true scalability.

### Questions
How sensitive is performance to noise in synthetic LLM-based rewards? Any quantitative correlation between F_fuzzy and F_LM?

Does a model trained on finance/legal tables transfer to product or encyclopedia pages?

What is the total compute cost (training + inference) compared to a standard per-page LLM pipeline?

Could you show examples where the generated scripts fail catastrophically (e.g., mismatched tags or wrong data fields)?

### Soundness
2

### Presentation
2

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
In this paper, the authors target the problem of semi-structured data extraction.
They develop a script-based semi-structured content extraction at web-scale framework to improve generalization of this task. The proposed reinforcement learning framework leverages layout similarity across webpages within the same
site as the reward signal. The proposed approach outperforms strong baselines in script quality and boosts downstream question answering accuracy.

### Strengths
1. The paper addresses an important and practical problem:  extracting structured data from the vast amounts of semi‑structured web content.
2. The framework is novel in that it focuses on webpage grouping and employs a semi-supervised-like approach, enhancing the model's ability to generate highly reusable scripts, thereby improving task performance while reducing costs.
3. The experimental results reveal that the designed framework is efficient and effective in both generating robust, reusable scripts and improving performace on downstream tasks.

### Weaknesses
1. The advantages of script-based extraction need to be further elaborated.
2. The domain scalability of the method requires further elaboration.
3. The evaluation baselines seem to be somehow weak.

### Questions
1. A "script" is essentially akin to a type of automatically generable rule, yet the paper does not clearly distinguish it from traditional wrappers, templates, or XPath rules.
There is a potential risk of overstating the method's novelty. It is necessary to more explicitly articulate the advantages of RL-generated scripts—such as generalization ability and automatic learning strategies—highlighting their unique value compared to manually crafted rules.

2. The paper emphasizes black-box data analysis while providing limited discussion on the limitations of the DSL and the handling of anomalous webpages. Its capability to adapt to irregular structures—such as multi-row/multi-column tables, nested lists, or JavaScript-rendered content—remains unclear.

3. The paper claims support for "web-scale" and multi-domain scenarios, yet it does not present the domain differences between the webpages used for training and those used for testing. Including such statistics would make the experimental conclusions more convincing.

4. The selected baselines appear relatively weak. Many other LLM-based methods are mentioned in Section 1; it remains unclear how the proposed method performs in comparison with them.

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
4

### Summary
The paper proposes a web-scale structured information extraction framework called SCRIBES. Instead of calling a large model on every single webpage, the key idea is to first let an LLM generate a reusable extraction program (a Python script) and then apply this script to many structurally similar pages. To make the script generalize, the authors use reinforcement learning on a group of webpages to produce scripts that work across pages. Experiments show that, compared to existing script-generation / agentic baselines, their RL-based training improves group-level reusability, and the extracted triples can also benefit downstream HTML QA.

### Strengths
* Well-motivated problem. Web-scale information extraction is an important and realistic task, and the cost of applying an LLM to every single page is prohibitive; reducing LLM calls is therefore a meaningful objective.
* Novel paradigm. By generating a script for each layout type (i.e., each group of structurally similar webpages), the method reframes the task from page-level IE to program induction and reuse, clearly setting it apart from pure agent-style, one-page-at-a-time extraction.

### Weaknesses
* Absolute performance gap. Despite gains over script-generation baselines at comparable model sizes, the approach still trails the strongest page-wise direct LLM extraction (e.g., GO-120B with flattening).
* Fragile grouping assumption. The heuristic “same domain + similar URL prefix ⇒ similar structure” may not hold on sites with mixed templates, limiting robustness and generalization.
* Bias toward structure-heavy pages. Performance degrades on free-form, text-heavy pages that require deeper semantic understanding.
* Insufficient deployment analysis. A clearer cost–accuracy trade-off (e.g., a curve or table) is needed to show when the script-based approach is preferable to per-page LLM extraction.

### Questions
* Could you include one or two representative examples of the generated Python extraction scripts in the appendix , so that readers can better understand what the model actually produces?
* Is it feasible to integrate a lightweight Open IE (or similar semantic post-processing) component into the generated script to improve its ability to handle less strictly structured content?

### Soundness
3

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
4

### Summary
The paper addresses the task of knowledge extraction, i.e. extraction of triplets (subjects, predicates, and objects), from semi-structured web content such as tables and infoboxes. This paper proposes to extract such knowledge with LLM-generated extraction script that is designed to be applicable to similar pages from the same source website. Furthermore, the authors use reinforcement learning (RL) to train LLMs to produce these scripts. The results show that for the 14B and 32B models, this script generation approach notably outperforms direct LLM-based direct extraction, and the application of RL provides a subsequent performance gain.

### Strengths
1. Script-based knowledge extraction holds promise for significantly improving the speed and efficiency of the extraction process compared to direct, single-instance LLM extraction.
2. The application of RL brings a clear and significant improvement in performance compared to the base models.

### Weaknesses
1. While the extraction of knowledge triplets can ideally benefit downstream tasks such as knowledge-intensive question answering, I am not completely convinced that it is necessary. The knowledge triplet is a relatively narrow format, and may not be sufficient to comprehensively represent the knowledge. A promising alternative could be to directly use the source data through RAG.
2. The reported performance seems inconsistent. For example, the script generation method using the gpt-oss-120b model appears to significantly lag behind the results achieved by direct extraction with the same model.
3. The comparison against established baselines is not comprehensive enough. I would love see a more thorough comparison, including against the numbers previously reported for the SemiBench paper itself, and other related baselines such as specialized relation extraction models, LLM-based relation extraction agents, etc.
4. Could the authors present more details regarding the results in Section 5.1? Where does the QA data come from, and how is it selected?
5. I would also appreciate more implementation details about HTML flattening and cleaning, and whether different strategies were explored. These pre-processing steps can have a dramatic impact on both the performance and the resulting behavior of the extraction system.

### Questions
As in weakness

### Soundness
2

### Presentation
3

### Contribution
2
