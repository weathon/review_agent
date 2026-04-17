# Dataset Protection via Watermarked Canaries in Retrieval-Augmented LLMs

- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Retrieval-Augmented Generation (RAG) has become an effective method for enhancing large language models (LLMs) with up-to-date knowledge. However, it may pose a significant risk of copyright infringement, as IP datasets may be incorporated into the knowledge database by malicious Retrieval-Augmented LLMs (RA-LLMs) without authorization. To protect the rights of the dataset owner, an effective dataset membership inference algorithm for RA-LLMs is needed. In this work, we introduce a novel approach, \textit{CanaryTrace}, to safeguard the ownership of text datasets and effectively detect unauthorized use by the RA-LLMs. Our approach preserves the original data completely unchanged while protecting it by inserting specifically designed canary documents into the IP dataset. These canary documents are created with synthetic content and embedded watermarks to ensure uniqueness, consistency, and statistical provability. During the detection process, unauthorized usage is identified by querying the canary documents and analyzing the responses of RA-LLMs for statistical evidence of the embedded watermark. Our experimental results demonstrate high query efficiency, detectability, and consistency, along with minimal perturbation to the original dataset, all without compromising the performance of the RAG system.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces CanaryTrace, a novel framework for protecting intellectual property datasets used in retrieval-augmented large language models. Instead of altering the original data, the method inserts synthetic, watermarked canary documents designed to be indistinguishable from the authentic dataset. These canaries are generated using a watermarked LLM and maintain consistency with the original dataset. When a potentially infringing RAG system is queried, the presence of the embedded watermark in model outputs serves as statistical evidence of dataset misuse. Empirically, CanaryTrace achieves 100% detection performance with only 12 queries, preserves dataset utility and maintains downstream RAG task performance across multiple datasets.

### Strengths
- The authors propose a novel canary strategy, which combines imperceptible watermarking and synthetic text generation, maintaining the original dataset completely intact.

- Extensive experiments across different datasets and retrievers demonstrate robustness, query efficiency, and minimal dataset distortion.

- The framework operates without access to model logits, making it deployable against closed-source RAG systems like GPT-5 or Gemini.

### Weaknesses
- Although the number of inserted canary documents is small, their presence could still introduce noise in large-scale retrieval systems. Normal user queries may inadvertently retrieve these synthetic canaries, slightly degrading retrieval precision or response quality in downstream applications.

- Despite efforts to ensure attribute-level consistency, there remains an inherent discrepancy between the original IP documents and the synthesized canaries. A sophisticated adversary could exploit subtle stylistic, semantic, or embedding-level differences to identify and filter out canary documents, thereby weakening the protection mechanism.

- The proposed z-test for watermark detection assumes that the queries used are independent. In practice, query correlation (e.g., overlapping semantics or shared retrieval results) could violate this assumption, leading to biased z-scores and inaccurate p-values, potentially affecting the reliability of the detection decision.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces CanaryTrace, a dataset-ownership protection scheme for RAG systems. It inserts a small set of synthetic, watermarked canary documents that (1) match the attributes of the dataset, (2) are fictional to ensure the uniqueness for high retrievability, and (3) are robust enough so their content causes a detectable watermark signal to diffuse into RAG responses when retrieved. Experiments show strong target-retrieval accuracy and detection, while leaving downstream RAG accuracy intact.

### Strengths
1.The DMI-RAG task is well-posed, which keeps the original dataset untouched, avoids quality regressions common to paraphrase-based watermarking, and canaries provide separable evidence

2.The work provides a principled synthesis pipeline. 

3.Detection with statistical guarantees.

4.Extensive experiments and strong empirical results under realistic constraints.

### Weaknesses
1. More broader evaluation of robustness should be done. How does detection performance change if the generator performs an aggressive attack, e.g., paraphrasing, before answering?

2. The experiments show the effectiveness of the watermarked canary on text RAG. Can you discuss its potential application to non-text RAG, e.g., image?

### Questions
Have you considered a standardized disclosure (e.g., a notarized canary list and key escrow) to bolster evidentiary value?

How should practitioners choose $\eta$ to achieve, say, 1% FPR?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses dataset copyright protection in Retrieval-Augmented Generation systems, which risk unauthorized incorporation of intellectual property datasets into Retrieval-Augmented LLMs. The authors propose CanaryTrace, a novel method for detecting dataset misuse. CanaryTrace works by inserting synthetic documents that contain unique, statistically provable embedded watermark into the original dataset without altering it. When unauthorized use occurs, querying these canaries from the RA-LLM reveals statistical evidence of the watermark. Experiments show high query efficiency, detectability, and consistency, with minimal impact on dataset quality or RAG performance.

### Strengths
- The paper introduces a black-box dataset protection framework that preserves the integrity of the original IP dataset while achieving high detection accuracy through LLM-based watermarking in synthetic canary documents

- The authors conduct extensive experiments demonstrating strong quantitative performance, including high retrieval accuracy with minimal queries and negligible impact on downstream RAG tasks

### Weaknesses
- The proposed method lacks clear novelty. Its core idea primarily relies on applying existing watermarking techniques within the RAG framework. While the implementation is well-executed, the approach essentially extends known watermarking methods to a familiar setting without introducing fundamentally new algorithms or theoretical insights.

- The method used to detect the watermark also lacks clear novelty. This paper primarily applies an existing detection algorithm to detect the watermark.

### Questions
What is the retriever model used for the experiments?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel method for dataset copyright protection in retrieval-augmented large language models (RAG-LLMs) via the use of watermarked canaries. The key idea is to embed special, inconspicuous textual fragments (canaries) into the retrieved data with imperceptible yet verifiable watermarks. These canaries enable the dataset owner to later verify unauthorized usage or leakage of their dataset in an LLM system, even when the model or retriever is proprietary.

The authors design semantic-preserving watermarking techniques to inject canaries without degrading retrieval or generation quality.

They introduce an adaptive embedding mechanism that selects insertion points based on retrieval relevance and semantic alignment.

A verification framework detects the presence of watermarked canaries in LLM outputs through probabilistic decoding analysis and semantic matching.

The approach is evaluated on multiple RAG settings (e.g., GPT-4-retrieval hybrid, open-source retrievers).

### Strengths
1. address a timely problem for protecting the intellictual property for external knowledge base used for the RAG system.

2. The results evaluated on various dataset is really good.

3. The presentation is easy to understand.

### Weaknesses
1. There is no thereotical guarantee that this unique  (e.g., "VitalityBoost and ExerciseShield " ) will trigger the retrieval of systhetic paragraph. For example, what happens if these unique content (e.g., "VitalityBoost and ExerciseShield " ) will exist in other  external knowledge bases which would be combined into the protected dataset. 

2. How do you verify the ownership with answers containinng the sythentic content. It is possible that the user claim that their knowledge base contains these contents? Different from other backdoor-based defense, they have malicious behaviour used as verification evidence.  

3. I here challenge the novely of this paper as it seems a variant of previous work [1]. Both of these work use generated / systhetic paragraph / content as evidences for ownership. I would like to see the merits or disadvantages of this work comparing with previous work.

4. There is no evaluation on real-world IP-sensitive database such as Harry Potter series book.
 
5. Lack important baselines, such as membership inference-based approaches and a general framework for data use auditing (CCS). 

6. How can you ensure your question will accurately retrieve the synthetic content aas you expected with varying k and the external datasets ?


[1] Towards copyright protection for knowledge bases of retrieval-augmented language
models via ownership verification with reasoning. arXiv preprint arXiv:2502.10440, 2025

### Questions
See Above.

### Soundness
2

### Presentation
3

### Contribution
2
