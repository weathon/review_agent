# InComeS: Integrating Compression and Selection Mechanisms into LLMs for Efficient Model Editing

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6, 4

## Abstract
Although existing model editing methods perform well in recalling exact edit facts, they often struggle in complex scenarios that require deeper semantic understanding rather than mere knowledge regurgitation. Leveraging the strong contextual reasoning abilities of large language models (LLMs), in-context learning (ICL) becomes a promising editing method by comprehending edit information through context encoding. However, this method is constrained by the limited context window of LLMs, leading to degraded performance and efficiency as the number of edits increases. To overcome this limitation, we propose InComeS, a flexible framework that enhances LLMs’ ability to process editing contexts through explicit compression and selection mechanisms. Specifically, InComeS compresses each editing context into the key-value (KV) cache of a special gist token, enabling efficient handling of multiple edits without being restricted by the model’s context window. Furthermore, specialized cross-attention modules are added to dynamically select the most relevant information from the gist pools, enabling adaptive and effective utilization of edit information. We conduct experiments on diverse model editing benchmarks with various editing formats, and the results demonstrate the effectiveness and efficiency of our method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces InComeS, a novel framework for efficient model editing in Large Language Models (LLMs). It addresses a key limitation of In-Context Learning (ICL)-based editing methods, whose performance and efficiency degrade with a large number of edits due to the finite context window. The proposed solution involves compressing each edit context into a special "gist token's" KV cache, bypassing the context length constraint. Furthermore, the authors incorporate cross-attention modules to enable the model to dynamically select the most relevant information from a pool of these compressed gist tokens. Experiments across various complex editing benchmarks (e.g., multi-hop, natural language edits) demonstrate that InComeS outperforms existing methods in effectiveness and efficiency.

### Strengths
The paper clearly identifies a significant and practical challenge in model editing and propose InComeS a flexible framework that enhances LLMs’ ability to process editing contexts through explicit compression and selection mechanism.

The Gist token used in Editing is very interesting. And editing the attention module is also novel.

### Weaknesses
* How can the GIST token be effectively trained? Furthermore, once trained, what metrics should be used to evaluate the generalization capability of the GIST token? 
﻿ 
* The training process requires approximately 11 hours for Llama-3.2-1B and 35 hours for Qwen2.5-7B. Considering the performance gains achieved, how does the efficiency of this approach compare to other model editing techniques, such as In-Context Learning (ICL)? 
﻿ 
* As your experiments indicate, simple fine-tuning (FT) can yield strong results. However, incorporating a more detailed analysis of locality could further strengthen the evaluation, given that locality preservation is a key consideration in model editing methods.

### Questions
* Under what circumstances is the GIST token applied? Is it used for every inference instance, or is it selectively activated based on specific input criteria?
 * Given that model editing typically prioritizes efficiency, what is the rationale behind adopting a teacher-student training framework, especially considering its apparently substantial computational cost?

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
The paper proposes InComeS, an in-context–style editing framework that (i) compresses each edit into a single “gist” token’s KV cache and (ii) equips the base LM with cross-attention modules that, at generation time, select among a pool of cached gists plus a special “zero-gist” option. Training uses token-wise reweighting—based on loss differences with/without edit context from a teacher model—together with a KL term to distill the teacher into the student. Experiments on MQuAKE, DUNE, WikiDataCounterfact and ZsRE-extended show competitive accuracy and efficiency versus ICL and a range of editing baselines.

### Strengths
(1) Compressing edits into re-usable gist KV caches and adding token-level cross-attention to select among them is clean; Zero-gist, serving as a “no-selection” option, reduces interference from the edit context on irrelevant tokens (see ablations) and complements the locality metric.
(2) Evaluates the effectiveness of the method across multiple scenarios, including multi-hop edits (MQuAKE), natural-language edits (DUNE), and ripple/portability settings (WikiDataCounterfact, ZsRE-extended).

### Weaknesses
(1) The paper does not include comparisons with recent strong editors such as memory based RECIPE[1] and ICL retriever based DR-IKE[2]. Without these, the empirical claims lack persuasiveness regarding true advances over contemporary methods.
[1]Lifelong Knowledge Editing for LLMs with Retrieval-Augmented Continuous Prompt Learning
[2]Dynamic Retriever for In-Context Knowledge Editing via Policy Optimization
(2) Despite criticizing ICL’s limitations, results show InComeS often performs on par with or worse than ICL—e.g., On multi-hop, InComeS underperforms ICL on Qwen2.5-7B for single 2- (66.46% vs 69.76%) and 3-hop (71.24% vs 76.91) settings (Table 1); on portability, InComeS is close to or slightly below ICL for Llama-3.2-1B and Qwen2.5-7B on batch editing results, such as WikiDatacounterfact Edit Success (71.44% vs 85.28%) and ZsRE-extended Portability (61.22% vs 64.57%) (Table 3).

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

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
This paper introduces InComeS, a novel and flexible framework for efficient model editing in LLMs. The method is designed to overcome the scalability and efficiency limitations of traditional In-Context Learning (ICL) for editing, where performance degrades as the number of edits increases due to the finite context window. The core contribution of InComeS is a two-stage process of compression and selection. First, each piece of editing information is independently compressed into the KV cache of a special gist token. Second, the model is augmented with specialized cross-attention modules. These modules enable the model to dynamically and selectively attend to the pool of compressed gist tokens at inference time, retrieving the most relevant information for a given query. The authors conduct extensive experiments on a variety of complex model editing benchmarks. The results demonstrate that InComeS consistently outperforms a wide range of existing editing methods, showing marked improvements over the strong ICL baseline.

### Strengths
1. The integration of gist-based context compression with a learnable, dynamic selection mechanism is a novel combination that directly addresses the bottlenecks of ICL for batch editing.

2. By compressing edits in parallel and using a lightweight selection mechanism, it offers substantial speedups over ICL. This makes the approach practical for real-world applications.

3. This paper is clearly written, well organized, and generally easy to understand.

### Weaknesses
1. The method requires a continued pre-training phase to teach the model the compression and selection mechanisms.

2. The results in Table 1 show that the improvement of InComeS on Llama-3.2-1B is much greater than that on Qwen2.5-7B. This may indicate that the effectiveness of the method diminishes as the model scale increases.

### Questions
See weaknesses.

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
InComeS proposes a framework to edit the LLM through compression and selection mechanisms. The authors demonstrate that InComeS offers improved efficiency and accuracy compared to strong baselines and test on several challenging datasets and complex editing scenarios. The method's effectiveness is supported by results on multi-hop, natural language, and editing tasks which need reasoning ability, demonstrating strong scalability and adaptability across diverse editing tasks. And it is further analyzed through extensive ablations and analysis experiments.

### Strengths
1. The paper proposes the use of KV cache to improve efficiency and proposes corresponding training algorithms to improve the performance, and cross-attention modules are added to dynamically select the most relevant information gist.

2. The paper conducts a large number of experiments and thorough analysis.

### Weaknesses
1. There are only two models used in the paper, and the latest model (e.g. Qwen3-8B) is not used. If results about this model are reported, it will be more convincing. If the time is not sufficient, the author could consider only adding a small number of baselines for comparison.

2. The performance of the model in Table 3 is not competitive.

3. The paper uses the method of compressing content into key-value cache of gist tokens to achieve this. However, for different models, different gist token representations need to be saved. This method is similar to RAG, but the contents of RAG are visible. However, it is difficult to trace the contents of gist, which may cause some problems for understanding. And the content searched by RAG can be used by different model without other preprocessing. But the gist is only used for only one model. The design of the InComeS is not complete as RAG. For example, What should be done if the edited content is duplicated, and how to maintain all vectors in pool, like insert new gist and remove? The author did not design a special module to handle this situation.

### Questions
The ICL method in the paper is to fill all the information into the context. If the RAG method is used and only the most relevant information is selected for filling, what would be the result and efficiency?

### Soundness
3

### Presentation
3

### Contribution
2
