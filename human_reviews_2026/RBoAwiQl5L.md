# ZeroGR: A Generalizable and Scalable Framework for Zero-Shot Generative Retrieval

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Generative retrieval (GR) reformulates information retrieval (IR) by framing it as the generation of document identifiers (docids), thereby enabling end-to-end optimization and seamless integration with generative language models (LMs).
Despite notable progress under supervised training, GR still struggles to generalize to zero-shot IR scenarios, which are prevalent in real-world applications. 
To tackle this challenge, we propose ZeroGR, a zero-shot generative retrieval framework that uses natural language instructions to extend GR across a wide range of IR tasks.
Specifically, ZeroGR is composed of three key components: (i) an LM-based docid generator that unifies heterogeneous documents (e.g., text, tables, code) into semantically meaningful docids; (ii) an instruction-tuned query generator that generates diverse types of queries from natural language task descriptions to enhance corpus indexing; and (iii) a reverse annealing decoding strategy to balance precision and recall during docid generation. 
Furthermore, we introduce OpenInstIR, the most diverse open-source instructed retrieval dataset.
We investigate the impact of instruction fine-tuning scale and find that performance consistently improves as the number of IR tasks encountered during training increases.
Extensive experiments on the BEIR and MAIR benchmarks demonstrate that \textsc{ZeroGR} achieves competitive performance across a wide range of retrieval tasks, establishing a new state-of-the-art among GR methods. Our code is available at https://github.com/sunnweiwei/ZeroGR.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents ZeroGR, a framework for zero-shot (in-domain, novel-task) generative retrieval. The framework includes three main components: 

1) a protocol for training a small docid generator model that is finetuned on SOTA LLMs prompted to generate short, keyword-rich sentences from a document d_i that serves as its docid z_i.

2) an indexing protocol that trains the generative retriever model to generate docids z_i described in 1) from synthetic queries q_i,b generated from document d_i by a task-instructed query generator model.

3) A novel reverse-annealed constrained decoding scheme for docid generation at retrieval time that increases decoding temperature at each token of the docid to better balance precision (low temp, earlier tokens in docid) with recall (high temp, later tokens in docid).

They present SOTA generative retrieval performance that is competitive with dense retrievers on the unseen subsets of the BEIR and MAIR zero-shot retrieval benchmarks. They also perform ablation experiments demonstrating and speculating on the benefit of their three main methods.

### Strengths
1. The paper presents 3 advancements to GR semantic docid generation, pseudo-query generation for indexing, and retrieval decoding that are reasonably novel, convincingly better than previous GR methods, and have a high chance of informing future methods, even if their specific instantiations are not kept.

2. The paper’s narrative is well-motivated and clear to follow.

### Weaknesses
1. Several key empirical claims comparing to dense retrievers are overstated or unsupported by the presented results:
- L66-67: “Our best model … surpasses strong dense retrieval … including BEIR”
  - It is beaten on BEIR by even E5 and BGE base models which have >10x fewer params
  - A rough interpolation of the performance frontier defined by DRs in Fig 5 would put InstructGR on or below, not surpassing.
- L338-339 (note that this is in the MAIR eval section 6.1): “Despite having a relatively modest model size of 3B parameters, ZEROGR delivers competitive or even superior results compared to larger instruction-tuned dense retrievers with 7B parameters.” 
  - Beaten on MAIR avg by >5 points by both 7B DRs, no MAIR domains where it is superior. 

2. Evaluations (BEIR and MAIR) omit large-scale (>1M docs) retrieval settings, a known limitation of GR methods, but this scope restriction is not explicitly acknowledged, particularly in direct comparison with DRs. 
- This and the previous issue could be addressed with some remarks comparing the favorable/unfavorable properties of GR vs DR.

3. Some figures and tables could be made clearer with minimal changes:
- Table 2 would benefit from bolding best results for readability, though care should be taken given the incomparability of model sizes
- Table 3 has no column or feature that allows us to distinguish prior methods (e.g. year, model size, docid/indexing/decoding methods). While the changing architectures might make the last option infeasible, anything would help.
- The grey lines vs bolded black lines in Figure 3 middle and right are not explained in the caption or discussion. I assume they are per-task accuracies from the “average” on L449, but this should be clearer.
- Figure 4’s inconsistent y axis scales implicitly equate differences of 3.7 acc@1, 2 nDCG@10, and 8.5 recall@100

4. Typos / Grammar / Formatting:
- L65: sentence ends in two periods.
- L351: “we can see our method achieve best performance among most datasets.” -> should be achieves
- Listing of docids methods in section 6.4 have the numerals separated from the method by a newline - unlucky alignment that could be addressed by a few extra words in the paragraph or removing the numerals entirely.

### Questions
1. Dependence between docid generation scheme and reverse-annealed decoding strategy:
- How does the docid generation scheme affect the trie structure (i.e. avg. # decodable tokens at each step), and does that indicate which docid generation strategies might suit which decoding strategies better?
- Is the docid generation strategy held constant (i.e. ZeroGR’s model) when ablating the decoding strategy in section 6.6 / figure 4, or are they based on prior work? E.g. Tay et al. 2022 uses hierarchical numerical docids with beam search decoding.

2. Task vs domain generalization:
- How do the domain distributions (categories and relative weight/quantity) differ between BEIR, MTEB, and MAIR? 
- When scaling/choosing the training data, which would have the greatest marginal benefit: 
  - An identical task (e.g. counterargument retrieval) in an unseen domain
  - A different task in an already-seen domain

 That is, how to disentangle the claim of quantity & diversity of *tasks* from the naturally better domain match of MAIR-Train (= ZeroGR-Train) which has all the domains in the eval set (by being from the same benchmark) vs BEIR and MTEB which may have sub/supersets of MAIR domains.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ZeroGR, a zero-shot generative retrieval framework that extends traditional generative retrieval (GR) to diverse information retrieval (IR) tasks using natural-language instructions. The framework includes three components: (1) a docid generator that converts heterogeneous documents into unified identifiers; (2) an instruction-tuned query generator that creates diverse pseudo-queries for corpus indexing; and (3) a reverse annealing decoding strategy balancing precision and recall. The model is evaluated on the BEIR and MAIR benchmarks, showing moderate improvements over dense retrieval baselines such as E5 and BGE, and competitive results compared to instruction-tuned retrievers.

### Strengths
- The paper is clearly written and provides a well-structured presentation of the proposed framework.

- It explores a new angle for generative retrieval, combining instruction tuning and docid generation under a zero-shot setting.

- The idea of leveraging task-level instructions for corpus indexing and retrieval is conceptually sound and connects GR with recent trends in instruction-based dense retrieval.

### Weaknesses
- The experimental evaluation is not comprehensive. Most results are based on a single Llama-3B model, and scalability is discussed mainly through small-scale ablations rather than large-scale experiments or efficiency analyses.
- The performance improvements are modest, often within 1–2 points over strong baselines, and sometimes lower on BEIR tasks. The claim of achieving state-of-the-art zero-shot performance is therefore not fully solid.
- Critically, the paper does not directly compare with existing generative retrieval baselines such as [1,2,3]. These methods are closest in formulation, and the lack of head-to-head evaluation makes it difficult to assess ZeroGR’s actual progress

[1] Learning to Tokenize for Generative Retrieval.

[2] Scalable and Effective Generative Information Retrieval. SIGIR 2024

[3] Exploring Training and Inference Scaling Laws in Generative Retrieval. SIGIR 2025

### Questions
- How does ZeroGR compare against other generative retrieval systems when evaluated under similar zero-shot conditions?
- How sensitive is the framework to the quality of instruction-tuned query generation?

Please refer to the weakness part.

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
4

### Summary
The paper presents ZeroGR, a novel framework for zero-shot generative retrieval that unifies document indexing, pseudo-query generation, and decoding under a single generative paradigm. It introduces three coordinated modules: (1) a DocID Generator that converts heterogeneous documents into short, interpretable textual identifiers; (2) an Instruction-Tuned Query Generator that produces diverse pseudo-queries to enrich training coverage; and (3) a Generative Retriever trained to map queries directly to DocIDs through maximum-likelihood learning. During inference, ZeroGR employs a reverse-annealing decoding strategy within a Trie-constrained search space, balancing precision and recall by gradually increasing the sampling temperature. The method is evaluated on MAIR and BEIR benchmarks, demonstrating robust zero-shot generalization and outperforming several state-of-the-art dense and generative baselines across seen and unseen retrieval tasks.

### Strengths
- Introduces a novel approach to extending generative retrieval into the zero-shot setting, demonstrating strong generalization through instruction-tuned pseudo-queries that transfer effectively across tasks.
- Presents a reverse-annealing decoding strategy that balances precision and diversity, leading to consistent improvements in both accuracy and recall.
- Conducts extensive evaluations on MAIR and BEIR, supported by thorough ablation studies that confirm the robustness and reliability of the proposed method.

### Weaknesses
1. Most of baselines originally trained on smaller corpora (e.g., NQ-320k or MS MARCO) rather than on the larger-scale MAIR dataset used for ZeroGR. Given that Section 6.3 itself demonstrates substantial performance gains from scaling training data and task diversity, the comparison may not be fully fair. Similar issues exist for dense retriever baselines (e.g., Contriever trained on MS MARCO). Reporting the precise training configurations and data sources for all baselines would clarify the relative contribution of ZeroGR.

2. The experimental setup for some analyses remains insufficiently described. For instance, Section 6.4 on DocID design does not clearly state which search or decoding configurations were used during evaluation—such as whether reverse-annealing, Trie constraints, and top-k generation parameters were fixed across variants. Likewise, the section 6.6 does not report the beam size used for beam search or the specific hyperparameters applied in the reverse-annealing schedule. These omissions make it difficult to assess the fairness, reproducibility, and stability of the reported comparisons.

Minor presentation issues: figures in appendix and abstracts in openreview still refer to the model as InstructGR, and a few figures appear to have untrimmed borders or inconsistent formatting.

### Questions
Could the authors clarify which datasets and training configurations were used for each generative and dense baseline in Tables 2–3? Would the authors consider retraining or fine-tuning the baselines on comparable data scales to ensure a fair comparison and isolate the methodological gains of ZeroGR?

Could the authors provide more details on decoding and search configurations in Sections 6.4 and 6.6—specifically the beam size, top-k/top-p settings, and reverse-annealing parameters, and confirm whether these were kept consistent across variants?

What is the exact composition of the MAIR dev / unseen-dev subsets used in ablations (e.g., task list and query counts)? This would help improve reproducibility.

Since the DocID length is fixed in the prompt, have the authors tested whether varying this length influences retrieval accuracy or DocID conflict rates?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors proposed a method for zero-shot generative retrieval. First given the document corpus and an instruction that encodes the task, document IDs for each documents are generated. Then a  query generator generates queries for each document for generative retrieval training. The authors showed competitive performance against other dense retrieval or generative retrieval methods for BEIR and MAIR benchmarks.

### Strengths
- Relatively simple method that the authors have shown to be working well
- Highlighted the importance of instruction-based retrieval

### Weaknesses
- There exists simpler methods for unified docID representation that are not compared against, see Questions below.
- Sec 6.2 on BEIR: Seems that more recent dense retrieval methods outperform generative retrieval methods here. Please explain.

### Questions
- According to the prompt in Appendix A, one could envision a much simpler way of deriving the docID: rank the terms in the document and select top-k by IDF value. Would this perform well?
- Please clarify how reverse-annealed generation help with beam search: since it is a temperature-based scaling method, it should not modify the relative ranking of the hypotheses in the beam search -- so the top-K from the beam search should be identical?

### Soundness
3

### Presentation
3

### Contribution
2
