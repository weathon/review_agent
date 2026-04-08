## Human Reviewer 1

### Summary
This paper proposes In Context Routing (ICR) for implicit in context learning. Instead of adding demonstration tokens to the prompt or adding shift vectors in the residual stream, the method extracts Principal ICL Directions from multi domain explicit ICL runs by doing PCA at each layer. A small router then maps a new input to layer wise weights and per head gates. During inference the method adds a low rank input conditioned bias to the attention logits. The authors suggest a kernel view interpretation plus low rank reparameterization. Experiments on diverse datasets with several open models show consistent gains over other implicit ICL and few shot ICL and stronger OOD robustness.

### Strengths
- Practical efficiency: No prompt length increase and no weight updates in the base model (compared to vanilla ICL). The added compute is low rank and local to the attention logits, which is friendly to deployment compared with long demonstrations or broad fine tuning.
- Clear design shift in implicit ICL: The key novelty is the move from post hoc residual steering to structural routing at the attention logits. This places the intervention exactly where ICL mechanisms operate and turns implicit ICL into a problem of routing attention paths. This is a fresh axis that is different from LoRA that edits weights and from activation steering that edits residuals.
- Empirical breadth and stability: The method wins against several implicit ICL baselines on both in domain and out of domain sets and shows fewer collapses below zero shot. It sometimes matches or beats few shot prompting while keeping zero shot latency and memory.

### Weaknesses
- Data and supervision needs: Router training uses labeled data from several domains. The limits of generalization to tasks with new label spaces or to settings without labels are not fully explored. It remains unclear how far the train once and reuse promise extends.
- Information usage in PID extraction: Using only the last token Q and K may underuse the rich structure inside demonstrations. The paper argues it is sufficient as an integration point, but alternative choices like pooling across several tokens or using attention rollouts could strengthen the claim.

### Questions
- Why restrict PID extraction to the last token only. Have you tested using several recent tokens or a learned pooling over the demonstration region, and how would that affect out of domain robustness and interpretability?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper introduces In-Context Routing (ICR), a novel approach to improve large language models' in-context learning capabilities without using explicit demonstration examples. ICR extracts generalizable patterns from multi-domain in-context learning by identifying Principal ICL Directions (PIDs) through PCA on attention representations. These patterns are applied via a learnable router that modulates attention logits based on input queries, enabling effective zero-shot inference. Unlike existing vector-based implicit ICL methods that inject task-specific vectors into residual streams, ICR operates at the attention mechanism level, providing better generalization. Experiments on 12 datasets show ICR consistently outperforms baselines, particularly excelling on out-of-domain tasks while maintaining computational efficiency comparable to zero-shot inference.

### Strengths
1.  ICR operates at the attention logits level rather than post-hoc residual stream injection, which is more aligned with how ICL fundamentally works through attention mechanisms
2. The paper provides rigorous theoretical grounding using the Spiked Covariance Model and Davis-Kahan theorem to explain why PCA on multi-domain ICL bases can extract generalizable patterns. 
3. Instead of additive vector interventions, ICR modulates attention through low-rank modifications to query-key interactions. 
4. Novel use of PCA to extract reusable structural directions from cross-domain attention representations

### Weaknesses
1. OOD Design Issues:
 The division into "near-OOD" and "far-OOD" seems subjective. For example, why is MRPC (paraphrase detection) considered "near" while CB (NLI) is "far"? Both involve sentence-pair understanding. The "OOD" tasks are still mostly classification/QA tasks from standard NLP benchmarks. True OOD would include fundamentally different task types (e.g., structured prediction, generation, mathematical reasoning). The paper trains on 5 diverse datasets (AGNews, SST-2, TREC, CSQA, PIQA) which already cover sentiment, QA, and classification. This makes the "generalization" less impressive since the model has seen similar task types during training.
2. The technical contributions are relatively incremental: The core idea of routing attention through PCA-extracted directions is reasonable, but the execution lacks the technical depth and innovation expected for a top-tier venue. A stronger contribution would involve more sophisticated pattern extraction, adaptive routing mechanisms, or novel theoretical insights about ICL.
3. The quality of PIDs heavily depends on the diversity and quality of initial ICL prompts, but no guidelines are provided for this critical step

### Questions
1. The experiments only test on 7B-8B models. How does ICR scale to larger models (70B+) where ICL behavior might be fundamentally different?
2. No analysis of how PID dimensionality (r) should scale with model size or task complexity
3. Computational cost of extracting PIDs grows with the number of domains, but this overhead isn't thoroughly analyzed when compared with few-shot learning which require zero training but may cost at inference.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper introduces ICR, which extracts Principal ICL Directions from attention and adaptively injects them into logits via a lightweight router. Experiments show ICR outperforms prior implicit ICL, remains stable on out-of-distribution tasks, and achieves strong efficiency. It offers a new paradigm with few parameters, zero-shot generalization, and cross-task reusability.

### Strengths
- This paper introduces the new paradigm of attention routing, shifting implicit ICL from residual injection to low-rank bias at the logits level, demonstrating clear novelty.

- It achieves consistent gains on open-source models such as Llama2, Qwen2.5, and Llama3.1, showing strong generality and reusability.

### Weaknesses
1. The evaluation is limited to classification and reasoning tasks, lacking assessment on open-ended QA and long-context reasoning.

2. Experiments are only conducted on 7B/8B models, without validation on larger-scale LLMs.

3. The router relies solely on a fixed MiniLM encoder for query representations, without examining whether alternative encoders could affect routing quality and generalization.

### Questions
see weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper focuses on reconstructing the attention patterns of few-shot inputs on zero-shot settings. Specifically, this paper propose In-context Routing method, which adds a bias term to the attention logits generated in zero-shot inputs to reconstruct the attention patterns observed in few-shot scenarios. Extensive quantitative experiments demonstrate that the proposed method outperforms a wide range of modern baselines, especially on out-of-domain tasks.

### Strengths
1. The authors propose a novel framework called attention routing, which enables automatic additive steering of attention logits, which can be broadly applicable across various scenarios. The attempt to explicitly control LLMs' behavior through a mechanistic understanding is highly revolutionary for the field of interpretability. This is my primary reason for recommending the acceptance of this paper.

2. Based on the above attention routing framework, the authors further introduce the In-context Routing (ICR) method. Through extensive quantitative experiments on sufficiently diverse datasets and model types, the authors demonstrate that ICR outperforms multiple baselines.

3. The analysis section provides insightful details about the proposed method, strengthening their claim that ICR provides generalizable attention shaping. In particular, they find that the reshaped attention scores can capture reasoning-oriented tokens, thereby confirming the soundness of the original motivation to ICR.

### Weaknesses
1. ICR utilizes gradient-based training on relatively large datasets with several tricks to facilitate attention routing. Also, ICR introduces an external text encoder to calculate the two key control gates in the method, and optimizes on a complex loss function. This design may contradict the low-resource spirit of ICL and undermine the overall usability. Furthermore, to my knowledge, the authors did not discuss how the performance of this additional text encoder affects ICR, nor did they provide sufficiently convincing ablation results to confirm the effectiveness of each loss component (e.g., in line 2-4 of Table 4, ablating some loss terms does not harm the accuracy). I consider attention routing to be an elegant framework, but relying on a bulky auxiliary module seems less than ideal.
    
    At the same time, this raises concerns regarding the paper’s main results (Tables 1 and 2): many of the provided baselines (such as TV and FV) involve substantially lower computational costs than ICR, making the comparison somewhat unfair. Although the authors claim that ICR exhibits good generalization and reusability, I would like to see at least a comparison in terms of calculation cost to enhance the credibility of their results.
    
    Moreover, from another perspective, since ICR already uses gradient-based training, it can be reasonable to directly train $\Delta \mathbf{A}$. I hope the authors can include such an experiment to demonstrate that their manual selection of the $\Delta \mathbf{A}$ basis is not redundant.
    
2. Mechanistically, the authors employ an external text encoder ($E(\cdot)$) to predict two key gating units within the ICR framework. These gating units are closely related to the internal structure of the LLM (e.g., selecting the important heads, as the authors mentioned in Section 5.3). Therefore, a crucial question arises: do the $E(x)$ actually contain information about the LLM’s internal structure? Or is $E(x)$ merely irrelevant variables? A simple experiment could address this by ablating $E(x)$ into random vectors. If the former is the case, how is this information then extracted by the two parameters $\theta$? This should be an interesting analysis, yet the authors skipped it.
    
3. There are several writing issues that make the paper somewhat difficult to follow, but I believe that this does not significantly affect my overall judgment of the paper.
    
    1. Line 52. “out-of-domain (OOD)” is ambiguous. You seem to mean that the query lies outside the distribution of the demonstrations, but another possible interpretation is that “the query lies outside the pre-training distribution”. Understanding this is crucial to getting your motivation, so I recommend clarifying it to eliminate ambiguity. Also, the specific experimental setup of Fig. 1 should be described (perhaps in the appendix).
        
    2. Line 115. This paragraph is somewhat unclear. I don’t fully understand the causal link in “Such additive interventions cannot structurally control how information flows, and thus often remain tied to task-specific representations.” I can understand that using task vectors for steering cannot *explicitly* control the information flow (i.e., attention scores, although not absolutely, since injecting certain components into the attention query could indirectly alter the attention scores), but I don’t see how this leads to being “tied to task-specific representations.” If I have missed something, I apologize.
        
    3. I suggest that the authors explain how each introduced symbol is grounded. For example, the symbol $\alpha$ in Equation (3) is confusing, it isn’t clear until Sec. 3.2 introduce it as a parameter to be trained.

### Questions
1. The authors seem to attribute all the benefits of ICL demonstrations to local attention effects within the query’s tokens (i.e., dynamically filtering task-relevant signals through attention scores). However, as far as I know, additional attention behaviors such as induction heads perform global attention operations from the demonstrations to the query. ICR clearly cannot reconstruct such attention patterns, since it is conducted under zero-shot inputs, yet their method still outperforms vanilla few-shot. This might prompt a new perspective on the mechanism of ICL. I would like to ask how the authors interpret this phenomenon, and whether they could expand their discussion of such mechanisms in the paper.
    
2. The analysis of layer/head importance (Fig. 4, left and middle) appears to include only the later layers. Could you release the results for all layers? This seems to suggest that certain specific attention heads induce the 0-shot inference, which can thus be improved by attention routing, therefore, it is interesting to get the detailed distribution of such heads.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4