# Dripper: Token-Efficient Main HTML Extraction with a Lightweight LM

- Decision: Reject
- Scores: 2, 8, 6

## Abstract
Accurately and efficiently extracting main content from general web pages is of great significance for obtaining training data for large models. Using well-pre-trained decoder-only generative language models offers excellent document comprehension capabilities, thereby effectively enhancing parsing quality. However, it remains constrained by issues such as context window length, inference cost, and format hallucination. We present Dripper, an efficient HTML main content extraction framework powered by lightweight language models, which addresses these challenges through four key innovations: (1) We design a specialized HTML simplification algorithm that reduces input token count to 22\% compared to raw HTML while preserving critical structural information; (2) We reformulate main content extraction as a semantic block sequence classification task, significantly reducing inference cost; (3) We introduce a controlled decoding mechanism that strictly constrains the output space through logits processors, effectively eliminating hallucination issues common in small-scale models; (4) We propose MainWebBench, an evaluation dataset containing over 7,800 web pages with meticulously human-annotated main content extraction labels. Experimental results demonstrate that using only a 0.6B parameter model, Dripper achieves state-of-the-art performance across all evaluation benchmarks and outperforms all baseline methods, attaining an ROUGE-N F1 score of 81.58\%( 83.13\% with fall-back strategy) on our proposed MainWebBench dataset.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes the Dripper framework, which addresses issues like context window limits, high inference costs, and hallucinations in web main content extraction. Dripper has three core parts: HTML simplification, semantic block classification, and controlled decoding. The authors also build MainWebBench dataset to evaluate model performance. A model fine-tuned on Qwen3-0.6B outperforming traditional methods and generative large models significantly on MainWebBench, with inference costs only 5.18% of the naive approach.

### Strengths
- The pipeline, HTML simplification + block classification + controlled decoding, directly solves efficiency and hallucination issues.
- Comprehensive experiments including 12 baseline methods and 2 datasets ensure the performance of Dripper. and public code, models, and datasets ensure reproducibility.
- MainWebBench offers community value on model evaluation, which is 7x larger than existing datasets, with complete annotations.
- The 0.6B-parameter model suits large-scale processing, meeting real-world needs.

### Weaknesses
- The presentation of figures, equations, and tables in this paper has inconsistencies affecting clarity. For example, the color scheme of Fig. 2 is confusing and lacks a corresponding description. There is a formatting mismatch between the description of Eq. (1) and the equation itself. Additionally, the captions for tables are overly simplistic, making it difficult to understand the tables.
- Lack of analysis on multilingual (e.g., Chinese) and domain generalization; MainWebBench has no domain-specific evaluation.
- No quantitative justification for key HTML simplification thresholds (e.g., 200-character truncation, line 203), with subjective settings affecting reproducibility.
- Lack of specific details for processing and model training.

### Questions
- What are the specific criteria for "splitting tables" in HTML simplification (line 198)? Are there experiments verifying how this rule affects extraction accuracy?
- How does the FSM in controlled decoding ensure no semantic block item_ids are missing (line 232)? Is there a fault-tolerance mechanism for failed state transitions?
- Can you add separate performance data for Chinese web pages in MainWebBench and compare accuracy differences between English and Chinese?
- In the fallback strategy, how do you determine Dripper has failed? What is the basis for choosing the failure threshold?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper tackles extraction of main body content from scraped HTML webpages, which is an important piece of data curation for training and prompting language models. Baselines in this space are usually heuristic tools like trafilatura (used in corpora like RefinedWeb) or newer LM based approaches like ReaderLM v2 (which is the most similar work to this one). These baselines suffer from the usual issues – heuristic based techniques aren’t accurate enough and LM based techniques aren’t efficient enough to use at scale. What this work does to improve on prior work is finetune a small Qwen model on this HTML extraction task and show that it achieves both extraction fidelity and efficiency, best of both worlds.

Of course, finetuning a Qwen model is straightforward, so let’s focus on what this paper has done that’s distinguishing from a simple finetuning. First, they finetune not just a model but also define a system around the model that approaches the HTML content extraction problem as a block sequence classification problem. A preprocessor identifies block regions in the HTML page using HTML tags, turns this into a sequence of blocks, and each block gets classified using the model into categories such as “main” or “other”. To ensure the model adheres to this scheme and produces correctly formed JSON output, they define a constrained decoding module imposed over the model’s inference. 

They evaluate this system over both an established benchmark (MCEB) which they establish a state of the art, as well as introduce their own benchmark using diversely sampled Common Crawl webpages, for which they also establish state of the art results over baselines.

### Strengths
Very solid submission and I’m appreciative of work like this that goes the full effort to build something usable in the real world (not just hill climbing on others’ benchmarks, but curating a good benchmark themselves; not just training a model but designing a system that can use the model, with considerations of deployment efficiency). The results speak for themselves; impressive improvement over past work on an important but under-appreciated problem. This paper should be accepted.

I’ve seen work like this critiqued in the past for being too “engineering-heavy” or critique the modeling aspect for being too straightforward (finetuning Qwen). But preemptively, I want to say that I don’t buy those arguments as this work also shows the underappreciated side of diving into the details of a challenging niche problem and showing it’s not just about finetuning but about also careful approach to the problem to designing a system (e.g. the approach as sequence classification).

### Weaknesses
I think this paper should be accepted. That being said, here are some aspects that bothered me a bit while reading the submission. I would really appreciate if the authors can revise accordingly:

First, I think it’s important for the authors to provide a discussion of relation to prior work in block sequence classification. For clean content extraction of structured, layout-rich documents, this has been done before. For example, “VILA: Improving Structured Content Extraction from Scientific PDFs Using Visual Layout Groups (TACL 2021)”, “Form2Seq : A Framework for Higher-Order Form Structure Extraction (EMNLP 2020)”. I believe on web HTML documents, there is old work that also explicitly looks at HTML web elements as blocks for classification, such as “Web2Text: Deep Structured Boilerplate Removal (ECIR 2018)”. I didn’t do a comprehensive search; just a quick search on “block classification” or “sequence classification” over “layout rich” documents and found quite a few. Of course, none of these are done in the autoregressive language modeling era, so that’s why I still value this current submission, but I would like to see some reference back to this family of work (maybe using the extra space in camera ready to add a paragraph in related work about it.)

Second, I am appreciative of the ablation for the constrained decoding mechanism; it’s quite intuitive that there’s less need for it as we increase the amount of training data. I would like to see some discussion about how this relates to two pockets of work that aren’t referenced here.  First, constrained decoding methods for sequence classification (or any structured prediction) is well studied back in the era of CRFs; would like to see a bit of reference back to that, why they disappeared roughly around the BERT + LLM scaling-up era, and why it makes sense to bring them back today for this HTML content extraction task. I think this will help readers better appreciate why there is this component; otherwise, even if there is an ablation, it feels extraneous as long as one can scale training data. The second is, with any constrained decoding, there is always a penalty to inference speed. For a paper like this, which seems to be highly motivated by the practical applications of the system, it is important to put some numbers down about the inference speed cost (or lack thereof). For example, standard fast inference frameworks like VLLM or SGLANG can run Qwen family models really quickly and also support structured decoding as part of their framework; could one have implemented your FSM natively within those frameworks? Even in those frameworks, there is an inference speed tradeoff.  I don’t think need a major discussion here, but something would be nice.

Besides this, if you’re able to provide, I could use a bit more motivation on why it is important to curate your new benchmark vs rely on MCEB. As a reader, I could benefit from a really straightforward answer to the question: “In what cases would I want to evaluate on MainWebBench? In what cases would I want to evaluate on MCEB? In what cases, both?”  It’s not clear what the tradeoffs between these benchmarks are; whether this work is claiming MainWebBench should replace MCEB because of X, Y, Z properties, or if it is complementary in X, Y, Z ways?

### Questions
Do you happen to have baseline numbers for resilparse? It is actually the more popular tool nowadays compared to trafiliatura.

### Soundness
4

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
5

### Summary
This paper introduces Dripper, a framework for  HTML boilerplate removal; Dripper leverages a Qwen 3 0.6B finetune for efficent extraction. Dripper operates on a simplified version of a web page, and employs constraint decoding to ensure more faithful output. Alongside Dripper, this work introduces MainWebBench, a  benchmark of 7,800 human-annotated pages.

### Strengths
- Efficient approach to improve context extraction from HTML pages.
- The approach is generally well motivated.
- Paper introduces a more comprehensive benchmark for evaluating HTML extraction.

### Weaknesses
- The described constrain decoding mechanism is framed as a contribution; however, constraint decoding via a grammar is a built in feature in major inference, so this aspect of the pipeline, while sensible, it fairly straightforward. 
  - Futher, Seciotn 5.4 shows that the mechanism has limited impact once enough supervision is provided.
- The proposed benchmark is comprised of 90% randomly sampled websites, and 10% head distribution websites. It would have been useful to get more statistics about benchmark composition, including website locale, type of website, statistics of domains in common crawl graph and so on.
  - Given that the task is sequence classification, it would have been more appropiate to measure precision and recall per block rather than ROUGE.
- There are a couple baselines missing; for rule-based toolkits, [resiliparse](https://resiliparse.chatnoir.eu/en/stable/) is often used due to its better recall (e.g., [DCLM](https://arxiv.org/abs/2406.11794), [OLMo 2](https://arxiv.org/abs/2501.00656)); to ground the task, it would have also been useful to see performance of frontier language models on the same input Dipper receives.

### Questions
## Questions: 
- It is unclear what the role of Mapping HTML is. If Simplified HTML is in order, and Dripper does sequence classification, then what's the need to keep a separate representation?

## Minor typos and feedback
- All citations are missing a space. should be "C4 (Raffel et al., 2020)", not "C4(Raffel et al., 2020)"
- `\cite` / `\citet` is used instead of `\citep` in several places (e.g. 044)
- The paper frames the approach as sequence tagging, but, based on the backbone used (Qwen), the task is better described as text generation.

### Soundness
3

### Presentation
3

### Contribution
2
