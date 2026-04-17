# Description-Only Supervision: Contrastive Label–Embedding Alignment for Zero-Shot Text Classification

- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Zero-shot text classification (ZSC) seeks to assign texts to label spaces without relying on task-specific labeled documents. Yet, practical deployments of embedding models for classification often fall back on training task-specific classifiers (e.g., linear probes on frozen embeddings) to recover task-specific performance, reintroducing annotation costs and undermining the zero-shot setting. We introduce \emph{contrastive label-embedding alignment}, a simple, compute-efficient alternative that uses only a handful of natural-language descriptions per label and no labeled documents. We lightly fine-tune a base embedding model so that label verbalizers and their descriptions are aligned in a shared space: a symmetric multi-positive contrastive objective pulls each verbalizer toward its associated descriptions while pushing it away from others, capturing the many-to-one label-description relation. Across four benchmarks (topic, sentiment, intent, emotion) and ten encoders (22M-600M parameters), as few as five descriptions per label yield consistent gains, improving macro-F1 by $+0.09$ on average over zero-shot baselines, corresponding to relative improvements of roughly $5–13$% across models. Compared to a few-shot SetFit baseline with 8 labeled examples per class, our method attains higher mean performance with substantially lower variance across repeated runs, indicating improved stability in low-data regimes. The method uses label descriptions as the sole supervision signal to learn a label-specific embedding geometry for an off-the-shelf dual encoder via a symmetric multi-positive contrastive objective, while retaining efficient pre-encodable dual-encoder inference at test time.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper presents a contrastive learning framework for zero-shot text classification using only label verbalizers and a small set of natural language descriptions per label. The method applies a dual InfoNCE objective to align verbalizers and descriptions, claiming to improve performance without using any labeled documents. Experiments across multiple datasets and embedding models show consistent macro-F1 gains over zero-shot baselines, and improved stability compared to few-shot SetFit.

### Strengths
* Conceptually simple and lightweight method, requiring no labeled training documents.

* Empirical results are broad and convincing, covering 10 encoders and 4 datasets.

* Demonstrates consistent performance improvements and low variance across random runs.

### Weaknesses
* Novelty concern:
The idea of using natural language descriptions for label supervision in zero or few-shot classification has been explored in prior work. This paper applies a contrastive objective to align label verbalizers with their descriptions, but this formulation may not represent a substantial conceptual advance. The core component, which are label descriptions, contrastive learning, and dual encoders, are already widely used.
Furthermore, it is not clear how description-only supervision fundamentally differs from conventional label supervision. Although the authors emphasize that no labeled documents are used, the descriptions are manually written for each label, and thus, still reflect explicit labeling information. In this sense, the line between using label supervision and using label descriptions remains blurred and requires clearer theoretical or empirical justification.

* No ablation studies: The contribution of each component (e.g., dual InfoNCE, verbalizer vs. description) is not isolated. How much gain comes from descriptions alone versus contrastive finetuning?

* Poorly annotated appendix:
Tables and figures in the appendix are presented without explanatory text, reducing clarity and reproducibility.

### Questions
* Novelty and supervision scope:
The proposed method builds on well-known components (verbalizers, natural language label descriptions, and contrastive objectives) all of which have been explored in prior work on zero or few-shot classification. While the paper emphasizes that no labeled documents are used, the manually authored descriptions per label still constitute explicit supervision.
What exactly distinguishes this setup from conventional label supervision?
And what is the substantive novelty beyond recombining existing elements in a contrastive learning framework? A clearer conceptual delineation is needed to justify the claimed contribution.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes description-only supervision for zero-shot text classification by aligning label verbalizers with small sets of human-written label descriptions using a dual-direction contrastive loss (row-wise and column-wise InfoNCE). Experiments on four common classification datasets with ten embedding models show improvements over naive zero-shot baselines, while claiming lower supervision costs than few-shot alternatives.

### Strengths
1. Uses only label descriptions, preserving dual-encoder efficiency (cacheable embeddings; label scaling). Clear deployment upside vs. cross-encoders/ICL. Clean, principled objective: The row-wise InfoNCE + column-wise multi-positive formulation captures one-to-many label–description relationships and adaptively reweights positives. The derivation and gradients are explicit. 

2.  +0.10 macro-F1 on average across 10 encoders × 4 tasks; especially large relative lifts for small models. Comprehensive per-family analysis. 

3. The uniformity criterion is a neat, inexpensive heuristic to prevent collapse in small-data contrastive tuning. 

4. Early stopping, small description sets, and minimal engineering make replication/deployment feasible.

5. The visualized results are visually appealing.

### Weaknesses
1. Banking77 is restricted to six card-related intents, which may understate difficulty relative to the full 77-class benchmark; generality to large label spaces remains partially untested in this paper’s main results.

2. The uniformity-based LR selection samples pairs from the test subset of the target domain, which can blur the line between tuning and evaluation (even though labels are not used). A cleaner protocol (dev split) would avoid potential leakage. 

3. Authors fix five generic descriptions per class and postpone quality optimization; robustness to noisy/misaligned descriptions is not systematically ablated. 

4. While SetFit is a fair few-shot comparison, contemporary description-driven ZSC methods (e.g., NLI-style label entailment or richer definition-based approaches) aren’t exhaustively compared under identical dual-encoder constraints. 

5. The paper states that Figure 1 demonstrates the core idea (Line 131), yet the figure mainly illustrates UMAP embeddings for AGNews rather than providing a conceptual or architectural depiction of the proposed framework.

6. Poor writing, formatting, and referencing quality.   Inconsistent formula numbering: some equations are labeled, while others are not.   Reference formatting is not standardized, and several citations contain extra parentheses (e.g., Lines 45, 50, 64).  Overall layout lacks polish.

### Questions
1. You compute uniformity on pairs sampled from the test subset (labels unused). Could you report results using a separate validation split for uniformity selection to rule out any subtle overfitting and quantify the gap (if any)? 
2. How does performance scale from 1→3→5→10 descriptions per label, and how sensitive is the method to noisy or partially off-topic descriptions? An ablation would help practitioners budget description effort. 
3. Have you tried full Banking77 (77 classes) or other datasets with dozens–hundreds of labels? How does the O(DL) batch construction behave in memory/latency, and does the column-wise term remain stable when K varies widely across labels? 
4. Many zero-shot methods compare documents to labels via entailment or rich label definitions. Could you include a dual-encoderized NLI/definition baseline (not cross-encoder) to isolate the value of the proposed contrastive training? 
5. You argue lower uniformity correlates with better F1. Can you provide per-dataset correlation plots across more models or show cases where the correlation breaks, to bound the reliability of this selection rule? (Some plots are in the appendix; more discussion would help.) 
6. Since descriptions are lightweight to write, have you evaluated cross-domain transfer (e.g., train verbalizer/description alignment on one domain and test on another) or multilingual zero-shot where label descriptions are translated?
7. What clear conceptual advancement distinguishes this method from prior description- or entailment-based ZSC work?

### Soundness
2

### Presentation
1

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
This paper addresses the problem of poor performance of embedding models in zero-shot text classification (ZSC). Existing ZSC methods often have limited performance or require reintroducing annotation costs (e.g., training a linear probe).
To solve this problem, the paper proposes a new method called "Description-Only Supervision".
Experiments across 4 benchmark datasets and 10 different encoders show that this method brings an average improvement of +0.10 in Macro-F1.

### Strengths
Originality:

1.The core contribution of this paper is highly novel and concise: it proposes a method to align "verbalizers" using only "descriptions" as a supervision signal, without relying on any labeled documents.

2.The "multi-positive" InfoNCE loss function, Lcols, is a clever design for handling the "many-to-one" label-to-description relationship.

3.The "label-free uniformity criterion" (Luni) proposed in Section 3.1 is a very smart innovation. It addresses the tricky problem of selecting a learning rate (LR) in the ZSC setting (which lacks a validation set), cleverly avoiding data leakage.

Quality:

1.The experimental quality is very high. The paper conducts validation on 10 different embedding models (with parameters ranging from 22M to 600M) and across different types of tasks (topic, sentiment, intent, emotion). This extensive testing strongly demonstrates the method's generalizability.

2.The qualitative analysis is excellent. The UMAP visualization in Figure 1 very clearly demonstrates the method's mechanism—pulling the "verbalizers" (stars) back to the center of their "document cloud" and "description cloud," which greatly enhances the intuitive understanding of the paper.

3.The Reproducibility statement is very thorough, promising to release all code, data, and models.

Clarity:

1.The paper's writing is (mostly) clear. The elaboration of the methodology in Section 3 is well-executed; the mathematical formulas and "geometric intuition" complement each other, making it easy for readers to grasp the core idea.

Significance:

1. This paper holds extremely high practical value. It provides a method that is computationally efficient (retaining the dual-encoder advantage) and has a very low annotation cost (only requiring a few descriptions to be written), yet significantly improves ZSC performance.

2. The stability comparison in Figure 2(a) is one of the most important findings of this paper. It shows that compared to relying on specific few-shot samples (SetFit) 28, this paper's method (Ours) is far more robust (exhibiting minimal variance). This is crucial for deploying reliable models in the real world.

### Weaknesses
There is a major contradiction regarding the experimental method for LR selection: This is the biggest weakness of this paper. The authors claim in Section 3.1 that they use the "uniformity loss" (Luni) to select the LR, because "lower (uniformity) values correlate with stronger downstream performance".
However, the paper's own data (Appendix D, Figure 3) largely contradicts this core claim.
For example, on the gte-modernbert-base model, the correlation between Luni and Macro-F1 is not significant on all 4 datasets (p-values of 0.722, 0.672, 0.808, 0.0743, respectively). Qwen3-Embedding-0.6B also shows extremely weak correlation (p-values of 0.890, 0.767).
This creates a key contradiction: If the criterion used to select the LR is ineffective on many SOTA models, how were the excellent results for these models in Table 1 achieved? This severely calls into question the rigor of the experiments.

Lack of key ablation study: The paper proposes a framework composed of multiple novel components ("verbalizer + description", "row-wise + column-wise" loss), but provides no ablation studies to demonstrate the necessity of these design choices. We cannot know if Lrows and Lcols are both indispensable.
We also cannot know if using the "Verbalizer" as the inference anchor is truly superior to other (potentially simpler) alternatives.
Insufficient sensitivity analysis on "description quality": The paper states they wrote 5 descriptions for each class and "did not tune them".

While this simplifies the experiment, it also evades an important question: To what extent does the method's performance depend on the quality, quantity, and diversity of these descriptions? How would performance change if the descriptions were poorly written, or if only 1-2 descriptions were provided? Although the authors mention this in "future work", it is a clear limitation of the current study.

Typesetting Issues: The submitted PDF manuscript has severe typesetting problems. Many pages have large vertical blank spaces, which seriously affect the reading experience and does not meet the conference's formatting standards.

### Questions
Here are the key questions I hope the authors will clarify during the Rebuttal phase:

(Most important question) 

Regarding the contradiction in the LR selection criterion: 

My biggest concern is the apparent contradiction between Section 3.1 and Appendix Figure 3. You claim to use the "uniformity loss" (Luni) to select the LR, and claim a correlation exists between the two.
However, the data in Figure 3 shows that for many of the stronger models (such as gte-modernbert-base and Qwen3-Embedding-0.6B), this correlation is not statistically significant (p-values are very high).

Please clarify:

For these models where the correlation was not significant, how exactly did you select the final LR for Table 1? Did you still use this (ineffective) criterion, or did you pick the best-performing LR on the test set (which is not allowed in a ZSC setting)? This must be clarified.

Regarding the ablation of the loss function: 

Your symmetric loss L = 1/2Lrows + 1/2Lcols is core to the method. Can you provide an ablation study showing the performance when using only Lrows and only Lcols, respectively? This is crucial for understanding the individual contributions of these two components.
Regarding the ablation of the inference anchor:
You use the "label verbalizer" vy as the anchor during inferenc. What would the performance be if you instead used the mean embedding vector of the set of "label descriptions" Dy as the inference anchor? Providing this comparison would help justify the necessity of vy as an "intermediate anchor".

Regarding the ablation of "Verbalizer" vs. "Label Word":
 
Why did you choose to use a full "label verbalizer" (vy, e.g., "This...is about sports.") as the alignment target, instead of directly using a simpler "label word" (e.g., "Sports") to align with the "descriptions" Dy? Can you provide an experiment comparing the effectiveness of these two anchor choices?

Regarding the typesetting issues:

The submitted PDF manuscript contains a large amount (on almost every page) of vertical whitespace. Will this be corrected in the final version?

### Soundness
3

### Presentation
3

### Contribution
4
