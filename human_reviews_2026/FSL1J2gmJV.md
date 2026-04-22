# MergePRAG: Orthogonal Merging of Passage-experts for Multi-hop Parametric RAG

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
Large language models (LLMs) can be enhanced with external knowledge through two dominant approaches: (1) **retrieval-augmented generation (RAG)**, which supplements LLMs with in-context retrieved passages, and (2) **parametric knowledge adaptation (PKA)**, which directly updates model parameters with new domain knowledge. Recently, parametric RAG (PRAG) has emerged as a promising framework, extending RAG by translating retrieved passages into parameter updates, thereby mitigating inefficiency and noise sensitivity inherent to RAG. However, existing PRAG methods remain limited to single-pass retrieval, falling short of the **multi-hop RAG** setting that requires iterative retrieval and reasoning. 
We propose **MergePRAG**(*Orthogonal Merging of Passage-experts for Multi-hop PRAG*), a novel framework that sequentially integrates retrieved passages into LLM parameters through a continual merging mechanism, which is advanced by two key proposals: (1) **orthogonal merging** using the Gram–Schmidt process to minimize conflicts between "passage experts", and (2) **critical-layer parameterization** to efficiently encode in-context passages. Experiments on multi-hop open-domain QA and reasoning-aware knowledge editing show that MergePRAG consistently outperforms both standard and state-of-the-art RAGs as well as existing parametric adaptation methods, achieving superior effectiveness and efficiency. All datasets and code will be released at https://github.com/Liu-Xuebing/MhQA_hypernetwork.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MergePRAG, a novel framework that extends Parametric Retrieval-Augmented Generation (PRAG) to handle multi-hop reasoning, a limitation of previous methods restricted to single-pass retrieval. The proposed approach sequentially integrates knowledge from multiple retrieved passages into the large language model's parameters during inference. This is achieved through two key innovations: (1) orthogonal continual merging, which uses the Gram-Schmidt process to add new "passage experts" without conflicting with previously integrated knowledge, and (2) critical-layer parameterization, which efficiently injects this new knowledge by updating only a select critical layer, reducing computational cost. Experiments on multi-hop question answering and reasoning-aware knowledge editing tasks demonstrate that MergePRAG consistently outperforms standard RAG, state-of-the-art RAGs, and other parametric adaptation methods.

### Strengths
This paper has the following strengths:

- It introduces MergePRAG, the first framework to successfully generalize Parametric Retrieval-Augmented Generation (PRAG) to the more complex multi-hop reasoning setting, addressing a key limitation of prior PRAG methods.
- It proposes a novel "continual merging" technique that uses orthogonal merging to integrate new knowledge without conflicts and critical-layer parameterization to ensure high efficiency, allowing knowledge to accumulate effectively across retrieval steps.
- It performs extensive experiments across multiple benchmarks and LLM backbones, demonstrating that MergePRAG consistently outperforms existing RAG and PRAG baselines in both effectiveness and efficiency.

### Weaknesses
This paper has the following weaknesses:

- The evaluation relies solely on token-level metrics (Exact Match and F1), which may not fully capture the semantic quality or factual accuracy of the generated responses. The study would be strengthened by a more holistic, model-based evaluation, such as using an LLM-as-a-judge.

- The experiments are limited to models up to 8B parameters. It remains an open question whether the proposed method's benefits and efficiency gains will scale effectively to current state-of-the-art, much larger models (e.g., 200B+ parameters).

- The analysis identifying optimal "critical layers" for expert injection (Figure 2) is conducted on only one model (LLaMA3.1-8B). It is unclear if this finding generalizes across different model architectures, which is a key assumption for the "critical-layer parameterization" technique.

### Questions
- Given the limitations of Exact Match and F1 in capturing semantic quality, can the authors provide justification for omitting a model-based evaluation (like LLM-as-a-judge), or discuss how these metrics adequately validate the model's multi-hop reasoning capabilities?

- The experiments are limited to models under 8B parameters. Can the authors provide any analysis or theoretical argument on whether the efficiency and effectiveness benefits of MergePRAG are expected to scale to current, much larger state-of-the-art models?

- The critical-layer parameterization relies on findings from a single model (LLaMA3.1-8B). Have the authors verified that this "early-to-middle layer" sensitivity generalizes to other architectures? If not, how does this potential model-specificity impact the method's general applicability?

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
3

### Summary
This paper introduces MergePRAG, a novel framework that extends Parametric RAG (PRAG) to multi-hop question answering. The key innovation is a continual merging mechanism that sequentially integrates retrieved passages into LLM parameters through orthogonal merging (using Gram-Schmidt process) and critical-layer parameterization. The method shows consistent improvements over existing RAG and PRAG baselines on multiple multi-hop QA datasets.

### Strengths
1. This paper investigates a critical research question: how to inject retrieved knowledge into parameters for multi-hop QA? The motivation is novel, and the proposed methods address the concerns of conflicts during merging and maintaining lightweight.
2. The experiments show advantages on multi-hop QA datasets, comparing with a wide range of methods.
3. Some augmentation designs like only modifying the critical layer, and shared basis low-rank parameters make sense.

### Weaknesses
1. The advantages need further clarification. When compared to other PRAG methods like DyPRAG, how does the pipeline work? Is the sub-question generator used consistently across all PRAG baselines? What are the exact architectural and procedural differences that lead to performance gains?

2. The significance of the method lies in the passage merging that avoids knowledge conflict, compared to other PRAG methods. First, the severity of the knowledge conflict may need illustration. The paper doesn't quantify how often or how severely knowledge conflicts occur when passages from different sub-questions are merged. Second, if the gram-schmidt orthonormalization is ablated, will the performance significantly decrease? There is no empirical evidence showing the actual conflict patterns in the retrieved passages

3. There is no analysis on the significance of critical layer selection, and shared basis low-rank parameters. Missing comparison of different merging strategies. Computational overhead comparison with baselines would strengthen the efficiency claims.

4. The paper would benefit from error analysis showing failure cases.

### Questions
1. Why does lower perplexity indicate higher sensitivity to expert injection?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes MergePRAG, a novel extension of Parametric RAG (PRAG) that enables multi-hop reasoning by orthogonally merging passage-specific experts into a language model’s parameter space. The method introduces an orthogonal merging mechanism, critical-layer parameterization, and a hypernetwork-based low-rank adaptation to efficiently absorb multi-hop retrieved knowledge. The idea is timely and original, addressing an important limitation of current PRAG systems that struggle with multi-hop reasoning. The paper is generally well-written and conceptually interesting, with promising empirical results.

### Strengths
The idea of orthogonal merging of passage experts is innovative and aligns well with current trends in retrieval-augmented and parametric reasoning research.

The approach is conceptually neat, allowing models to accumulate knowledge from multiple retrieval steps while minimizing interference.

The authors evaluate the method on multiple multi-hop QA and knowledge-editing datasets, showing that MergePRAG achieves competitive or even superior performance compared to prior RAG and PRAG variants.

### Weaknesses
1. Inconsistency and Missing Entries in Table 1
Table 1 looks quite confusing. Several entries (e.g., DyPRAG’s EM scores) are missing. For Qwen2.5-7B, there are no PRAG or DyPRAG baselines, which makes it unclear whether the claimed improvement of MergePRAG+ is significant or merely due to the absence of strong comparisons. Moreover, Table 1 reports MergePRAG+ results only—there seems to be no result for the base MergePRAG. It is also unclear whether MultiHopRAG corresponds to MergePRAG or another baseline. Overall, the table lacks a fair and consistent setting for comparison, casting doubt on the validity of the experimental conclusion

2. Effectiveness of Knowledge Injection (Table 4) According to Table 4, the performance of MergePRAG appears to drop substantially under certain configurations. This raises concerns about the effectiveness and stability of the proposed parametric knowledge injection. The authors should discuss why this degradation happens and whether it reflects limitations in the merging mechanism or the hypernetwork parameterization.

3. The paper does not include a natural baseline where all retrieved documents are merged into a single LoRA adaptation (without orthogonal decomposition). Such a comparison would be essential to demonstrate that orthogonal merging is the key factor, rather than simply adding more LoRA parameters. Without this, it is hard to isolate the contribution of the proposed mechanism.

4. The choice of “critical layers” is somewhat arbitrary and lacks theoretical or empirical justification. The method appears to require layer-by-layer scanning to find the best injection point, which seems neither elegant nor scalable. Moreover, it is unclear whether different documents or tasks share the same optimal layer, or whether the critical layer should adapt dynamically. This design choice needs clearer motivation and analysis.

5. The method seems tuned for specific architectures (LLaMA-3.1-8B and Qwen-2.5-7B). There is little evidence that the approach generalizes across other model types, sizes, or training regimes. Since the critical layer and merging behavior may differ per model, the generalization ability of MergePRAG remains uncertain.

### Questions
none

### Soundness
2

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
4

### Summary
The paper proposes extending Parametric RAG (PRAG) to handle multi-hop question answering by decomposing questions into atomic sub-questions and recursively merging a hypernetwork with retrieved passages.

### Strengths
- The problem addressed in this paper is well-motivated: adapting Parametric RAG (PRAG) for multi-hop question answering.
- The paper decomposes a multi-hop question into atomic sub-questions and recursively merges the hypernetwork with each retrieved passage.
- The proposed method extends prior work, enabling PRAG to effectively handle multi-hop reasoning tasks.

### Weaknesses
- Baseline selection and setup: 
  - The baseline selection is weak. Some competitive baselines are missing, such as Adaptive-RAG [1] for (i) RAG, and Search-o1 [2] and Search-R1 [3] for (ii) iterative retrieval.
  - The experimental settings for PRAG should be comparable to those of the proposed method. MergePRAG may gain advantages from sub-question decomposition and sub-answer generation. Even though PRAG was not originally designed for multi-hop retrieval, the sub-question decomposition should still be applied to the plain PRAG for fair comparison.

- Missing ablation study:
  - The paper does not include ablation studies on individual components (e.g., orthogonal merge, critical layer parameterization). Such analyses would help readers understand the necessity and novelty of each component.

- Additional resource considerations:
  - The proposed method requires sub-question decomposition and sub-answer generation, each followed by a hypernetwork merging step.    However, the inference cost (e.g., number of inferences for sub-answer generation or latency) is not compared. Such an analysis is needed to confirm that the performance gains do not come at the expense of computational efficiency.

[1] Jeong et al., "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity", NAACL 2024.               \
[2] Li et al., "Search-o1: Agentic Search-Enhanced Large Reasoning Models", ArXiv 2025.    
[3] Jin et al., "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning", ArXiv 2025.

### Questions
This paper iteratively merges the hypernetwork using retrievals from each step’s sub-question and generates sub-answers, denoted in Eq. (4) as:

$$
s_{a_t} = M_{\theta_0 \oplus F(SP_{1:t})}(sq_t)
$$

However, why not:
1. Generate a single hypernetwork using the passages from all steps at once,  
   $$
   H(SP_{1:T})
   $$
   instead of sequentially?
2. Or generate sub-answers using the fully merged hypernetwork,  
   $$
   s_{a_t} = M_{\theta_0 + F(SP_{1:T})}(sq_t)
   $$

### Soundness
3

### Presentation
2

### Contribution
2
