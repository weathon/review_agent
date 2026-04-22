# Beyond Hard and Soft: Hybrid Context Compression for Balancing Local and Global Information Retention

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Large Language Models (LLMs) encounter significant challenges in long-sequence inference due to computational inefficiency and redundant processing, driving interest in context compression techniques. Existing methods often rely on token importance to perform hard local compression or encode context into latent representations for soft global compression. However, the former struggles to retain global information, while the latter struggles to maintain local details. To address this, we propose Hybrid Context Compression (HyCo2) for LLMs, which integrates both global and local perspectives to guide context compression while retaining both the essential semantics and critical details for task completion. Specifically, we employ a hybrid adapter to refine global semantics with the global view, based on the observation that different adapters excel at different tasks. Then we incorporate a classification layer that assigns a retention probability to each context token based on the local view, determining whether it should be retained or discarded. To foster a balanced integration of global and local compression, we introduce auxiliary paraphrasing and completion pretraining before instruction tuning. This promotes a synergistic integration that emphasizes instruction-relevant information while preserving essential local details, ultimately balancing local and global information retention in context compression. Experiments show that our HyCo2 method significantly enhances long-text reasoning while reducing token usage. It improves the performance of various LLM series by an average of 13.1% across seven knowledge-intensive QA benchmarks. Moreover, HyCo2 matches the performance of uncompressed methods while reducing token consumption by 88.8%. Our code will be available at \url{https://anonymous.4open.science/r/HyCo2}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents HyCo2, a hybrid context-compression plug-in for frozen LLMs that marries hard local token selection with soft global latent representations. It consists of a lightweight adapter—routing between MLP and Q-Former experts—first condenses the whole passage while preserving semantics; a parallel classifier then keeps the top-k% task-relevant tokens for fine-grained details. The author performs three-stage training—paraphrase pre-training, completion pre-training, then instruction tuning to prevent conflicting objectives.

### Strengths
1. The paper has an open-source promise: Code and checkpoints to be released, ensuring reproducibility and community adoption.
2. The paper presents a relatively thorough ablation study.

### Weaknesses
1. Figure 3a is not clear enough. It is quite hard to understand the definitions of "group tokens", how they are computed in the proposed method, and the motivations for applying them. There is no discussion of them in Section 2.2 either.
2. The method discussion section (2.2) is not well organized. Despite the authors providing an overall flow chart in Figure 3a, there is no accompanying text description in Section 2.2 to clarify each step shown in Figure 3a, which is likely to confuse the reader about the overall flow of the proposed method.
3. The encoder and decoder LLM setting of the method seems confusing. Did you use two different LLMs or just two parts of a single LLM? I suggest the author add more details on this.
4. I suggest you add the corresponding mathematical symbols in section 2.2 to your Figure 2a, so it is clearer to read.
5. According to the gating network (line 201 - 202), the equation shows that the output will be the weighted combination of the two adapters. Does this mean every global token feature from the encoder will have one global compressed token? So your compression completely relies on the reduction of the local tokens?
6. For the baseline implementations, it says, "Since the LLM in our method remains frozen, the selected baselines must support plug-and-play functionality without requiring any alteration to the LLM’s parameters". I personally disagree with this; the LLM frozen characteristics of this method cannot be used to exclude other baselines, especially when the author claims this as an advantage of their method.
7. There is no compression rate analysis. In this method, the compression rate is largely determined by neural parameters. Therefore, I think it is necessary to add an analysis to show readers the usual pattern of the compression rate. Also, I think the authors need to compare the compression rate of their method to that of other methods, as it may be the source of the performance increase.
8. The proposed method seems to work only for the encoder-decoder architecture, which makes it less attractive since it is not a mainstream architecture for LLM nowadays.

### Questions
1. About the "learnable tokens" in Figure 3a, are they part of the parameters of your proposed methods? 
2. How do you get the "group tokens" in Figure 3a? Could you give more details?
3. Are the encoder LLM and decoder LLM the same or different? If they are different, does this mean that it only works for the encoder-decoder architecture?
4. Are there any potential variations for the proposed method to work for the LLM decoder architecture nowadays?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes HyCo2, a local-global hybrid context compression model, combining the advantages of both MLP-based (local detail preservation) and QFormer-based (global semantic understanding) compression approaches. The local branch pools a context segment into fixed groups, while the global branch utilizes the learnable query tokens. In addition, the noisy gating module balances the two sources to prevent over-dominance of a single source during training. A separate hard classification layer is added for token-level selection based on the predicted retention probability. Experimental results show superior performance after compression compared to existing methods, such as LLMLingua, EXIT, and xRAG.

### Strengths
* The paper proposes a hybrid local-global design, which has not been explored much.
* The three-stage training strategy (paraphrase-completion-instruction tuning) is novel and carefully designed. The effectiveness of this strategy is empirically verified through ablation studies.

### Weaknesses
* The importance of hard local token selection mechanisms using the classification layer is not sufficiently demonstrated. For example, an ablation study varying Top-k% would be beneficial.
* Overall, there are many unclear and insufficiently justified parts, especially related to model architecture and training.
  * Empirical justification of why “noisy” MoE is essential.
  * The length of G(V), a gating output, does not seem to match the number of output tokens from LocalMLP or QFormer.
  * The definition and role of “Position” Pos(V) is not sufficiently explained.
  * What is the “teacher RAG paradigm”?
  * It is unclear how the hard token selection head is trained, since token selection is a non-differentiable operation. Yet, the paper only mentions end-to-end optimization with an NLL loss.
* Comparison to more recent studies, such as ICAE, COCOM, and PCC, is not included. While PCC is a very recent work, ICAE and COCOM should be included in the result Tables for completeness.

### Questions
* Fixed-size global token may cause information bottlenecks for long or multi-document inputs. It is somewhat interesting that HyCo2 maintains its performance as the number of documents increases.
* The contribution states “minimal parameter updates without relying on additional compressors or embedding models”, but there exist new modules in HyCo2. This claim seems overly broad.
* (minor errors) line 078: Table -> Figure, line 426 Adapte -> Adapter

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
This paper proposes **HyCo²**, a hybrid context compression method for large language models (LLMs) that integrates hard compression (local token selection) with soft compression (global latent encoding). HyCo² efficiently balances local detail preservation with global semantic completeness, significantly reducing computational overhead and context length while maintaining strong performance. Extensive evaluations demonstrate that HyCo² consistently outperforms existing context compression methods across various QA benchmarks. Furthermore, detailed ablation studies provide valuable insights into the effectiveness of each component.

### Strengths
- **Novel Hybrid Compression Approach**: HyCo² effectively integrates hard (explicit token selection) and soft (latent embedding) compression strategies, achieving a balanced retention of both local details and global semantics. This hybrid framework significantly reduces computational costs and context length without substantial performance loss.

- **Strong Empirical Validation**: Comprehensive experiments across multiple QA benchmarks, including LongBench, show HyCo² consistently surpasses existing methods such as LLMLingua2, EXIT, and xRAG, demonstrating its robustness and effectiveness.

- **Detailed Ablation Studies**: The authors conduct thorough ablation analyses to isolate the impact of individual components (e.g., hybrid adapters, alternating training stages, local vs. global modules), clearly illustrating how each design choice improves performance.

- **Clear and Well-Written**: The paper is organized logically, clearly written, and easy to follow, enhancing readability and reproducibility.

### Weaknesses
- **Limited Task Diversity in Evaluation**: The majority of experiments focus on question-answering (QA) tasks. It's unclear how well HyCo² generalizes to other tasks such as summarization, reasoning-heavy tasks, code generation, or dialogue scenarios, where context compression needs might differ significantly.

- **Performance Gap on LongBench**: On the LongBench benchmark under the 2k-token constraint, HyCo²'s performance still substantially lags behind the "vanilla" (uncompressed) setting. This gap raises concerns about HyCo²'s ability to preserve critical details and semantic information required for more complex reasoning tasks or long-context scenarios. The paper does not provide a deep analysis of why these specific tasks are challenging for the method.

### Questions
1. Can you include a soft-compression baseline (e.g., xRAG) in the LongBench evaluation? It would be insightful to compare HyCo² directly with purely latent embedding methods on these complex tasks.

2. Why was LongLLMLingua not included in the LongBench performance comparison? Given its explicit design for long-context compression, how would it perform in comparison to HyCo²?

3. Could you elaborate on why HyCo² substantially underperforms the vanilla model on LongBench tasks? What specific types of information or reasoning capabilities does the compression remove or fail to preserve?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
**Summary**

This work proposes HyCo2, a context compression method capable of preserving both global semantics and local details simultaneously. The global compression module integrates the advantages of MLP structure and Q-Former structure, extracting information related to instructions from local context segments and the overall context. The local compression module is designed with a classifier to select key words, thereby retaining details. The proposed method reduces the context length and improves inference efficiency.

### Strengths
**Strengths**

（1）The paper is clearly written and easy to follow. The motivation is simple yet reasonable, and a direct, effective method is proposed to achieve this motivation.

（2）Thorough ablation experiments are provided, demonstrating the rationale and effectiveness of each design choice.

### Weaknesses
**Weaknesses**

（1）The paper lacks comparisons with some state-of-the-art methods in Soft Compression, such as ICAE and UniICL, which are mentioned in the related work section.

（2）The description of the TRAINING STRATEGY is insufficiently detailed. The paper does not clearly explain why the paraphrase task and completion task allow the compression module to extract high-quality tokens, why these tasks are particularly suited for local and global compression, respectively, and why paraphrase pretraining is performed before completion pretraining. A more detailed explanation of these points would help readers better understand the purpose and role of each stage.

（3）The proposed method involves two hyperparameters: the number of query tokens (NL) for global compression and the keeping ratio (k%) for local compression. The paper does not explain how these hyperparameters were selected, such as the criteria and methods used for choosing them.

### Questions
**Suggestions**

（1）Include comparisons with Soft Compression methods such as ICAE and UniICL.

（2）Provide a more detailed explanation of the TRAINING STRATEGY.

（3）Explain the rationale behind the selection of the number of query tokens (NL) for global compression and the keeping ratio (k%) for local compression. Additionally, it would be beneficial to discuss whether different contexts with varying numbers of tokens were considered, and whether such variations might lead to significant improvements in efficiency.

### Soundness
3

### Presentation
3

### Contribution
3
