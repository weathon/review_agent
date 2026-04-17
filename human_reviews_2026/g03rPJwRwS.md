# Delta-MIA: Measuring Membership Inference Attacks in Large Language Models via self-Contrast Framework

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Membership inference attack (MIA) underpins privacy risk assessment, provenance, and compliance for large language models (LLMs). 
Observational evaluations confound membership with distribution shift, hide sample-level behavior, and assume access to proprietary corpora. 
We present Delta-MIA, an interventional self contrast framework that isolates genuine membership signals by comparing a model before and after controlled exposure to the same dataset. 
The pipeline records pre exposure responses on verifiably unseen data, performs full-parameter fine tuning on that data followed by stabilization, and computes sample level deltas. 
We introduce three diagnostics: explained variance ratio (EVR), mean vertical distance (MVD), and above diagonal ratio (ADR), which quantify noise, separation, and baseline detectability. 
Re-evaluating $9$ MIA methods, several remain robust once shift is removed, while others such as DC-PDD and Con-ReCaLL decline markedly; 
Min K\%++ shows strong separation with high MVD. 
Delta-MIA enables bias-free, interpretable, and transferable evaluation for MIA in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces Delta-MIA (Δ-MIA), an interventional self-contrast framework for evaluating membership inference attacks in large language models. The work argues that existing MIA benchmarks are fundamentally limited by data distribution shift, coarse-grained analysis, and dependence on proprietary corpora, which confound evaluation and inflate performance metrics.

The key contribution is the interventional paradigm: instead of constructing separate member/non-member datasets, Δ-MIA compares a model’s responses before and after exposure to the same dataset. This self-contrast design eliminates distributional confounders and enables sample-level analysis through newly proposed diagnostics—ADR, MVD, EVR, and n-MVD—which quantify detection accuracy, discriminative strength, and noise sensitivity.

### Strengths
The research problem is clearly defined, and the paper's writing is good, making it easy for the reader to follow. 
The paper introduces four well-motivated diagnostic metrics (ADR, MVD, EVR, n-MVD) that provide fine-grained insights into model behavior. 
Also, the paper presents a solid empirical result from an experiment on their proposed evaluation method.
The authors commit to open-sourcing all code and evaluation data, and clearly articulate ethical safeguards to prevent misuse.

### Weaknesses
While the paper raises an important issue—the risk that current MIA benchmarks overestimate attack effectiveness due to distributional shortcuts—the motivation and the validation of Δ-MIA are not fully aligned.
The central claim is that Δ-MIA provides a faithful, bias-free framework for assessing membership inference in LLMs. However, the experiments mainly show that some existing MIA methods still yield non-trivial results under the Δ-MIA setting. This empirical observation alone does not demonstrate that Δ-MIA truly measures the same construct as “real-world” MIA evaluation.

Specifically, the paper does not theoretically or empirically establish equivalence between the Δ-MIA setting (before/after fine-tuning on a held-out dataset) and the conventional MIA problem (evaluating a pretrained model on candidate data with no distribution shift). Without such validation, it remains unclear whether the deltas measured by Δ-MIA genuinely reflect membership signals rather than overfitting signals of fine-tuning or optimization.

A stronger demonstration—either a theoretical argument showing that Δ-MIA preserves the same membership decision boundary as the standard setting, or an empirical study comparing Δ-MIA scores with ground-truth membership probabilities in a controlled environment—would significantly strengthen the paper’s central claim.

### Questions
1. What if the MIA method relies on the fine-tuning process? Can the D-MIA still be a good evaluation framework for MIA?  For example, the technique proposed in the paper "FINE-TUNING CAN HELP DETECT PRETRAINING DATA FROM LARGE LANGUAGE MODELS"

2. As I mentioned in the weakness part, can you show some evidence that if the MIA works under the Delta-MIA, which means it can evaluate a pretrained model on candidate data with no distribution shift. Because I still think fine-tuning cannot approximate pretraining, even if you add extra samples during fine-tuning to avoid overfitting. If you can persuade me that the MIAs working under Delta-MIA indicate a clear membership signal, I will raise the score.

3. Which subset did you use in MIMIR?  7-gram? 13-gram? It is essential because the MIA methods you showed in the paper all fail (close to random guessing) on 13-gram subsets of MIMIR (Wiki, arXiv, PubMed).

### Soundness
2

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
This paper introduces Delta-MIA, a comprehensive framework for measuring membership inference attacks (MIAs) on fine-tuned large language models (LLMs). The authors propose a set of novel metrics (Above-Diagonal Ratio, Mean Vertical Distance, Explained Variance Ratio, and Noise-Normalized MVD) that provide a more nuanced evaluation of MIA effectiveness beyond traditional dataset-level metrics like AUC. The framework validates the core assumption that fine-tuning doesn't significantly change model behavior by comparing pre- and post-fine-tuning model outputs. The paper evaluates nine representative MIA methods across three domains (Pile-CC, PubMed Abstracts, Wikipedia) using the proposed metrics, providing insights into the relative effectiveness of different approaches.

### Strengths
S1: The paper introduces a comprehensive evaluation framework that goes beyond standard metrics (AUC, TPR@FPR) to provide deeper insights into MIA effectiveness through multiple complementary metrics.

S2: The proposed metrics (ADR, MVD, EVR, n-MVD) are theoretically well-motivated and address key limitations of existing evaluation approaches, particularly the lack of nuance in current evaluation practices.

S3: The framework successfully validates the core assumption that fine-tuning doesn't significantly alter model behavior, which is crucial for the validity of the Delta-MIA evaluation framework.

S4: The paper provides a clear benchmark for evaluating MIA methods across different domains, which will be valuable for future research in this area.

S5: The empirical results (Table 4) clearly demonstrate the utility of the proposed metrics, showing that methods like ReCall and Min-K%++ consistently perform well across different domains.

### Weaknesses
W1: The paper lacks sufficient comparison with existing MIA evaluation frameworks, making it difficult to fully appreciate the novelty of the proposed metrics.

W2: The evaluation is limited to Pythia models across only three domains, which limits the generalizability of the findings to other LLM architectures and datasets.

W3: The paper doesn't adequately address the practical implications of the proposed metrics for real-world privacy risk assessment and defense mechanisms.

W4: The theoretical justification for the proposed metrics could be strengthened with more detailed mathematical analysis and comparison to related work.

W5: The paper doesn't explore the relationship between the proposed metrics and the actual privacy risk in fine-tuned LLMs, which is the ultimate concern for the field.

### Questions
Q1: Could you provide a more detailed comparison between your proposed metrics and existing evaluation metrics (AUC, TPR@FPR) to better demonstrate the added value of your framework?

Q2: How would the proposed metrics perform when evaluated against different LLM architectures (e.g., GPT, LLaMA, Qwen) beyond the Pythia models used in your experiments?

Q3: Could you explore the relationship between your proposed metrics (especially n-MVD) and actual privacy risk, potentially by comparing with real-world privacy leakage measurements?

Q4: How would the Delta-MIA framework be adapted to evaluate MIAs on different types of fine-tuned models (e.g., instruction-tuned, domain-specific, or reinforcement learning fine-tuned models)?

Q5: Could you investigate the potential for using your metrics to guide the development of better privacy defenses, rather than just evaluating existing attacks?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes $\delta$-MIA, an interventional, self-contrast framework that evaluates membership inference by comparing a model’s behavior on the same data before and after controlled exposure, thereby removing cross-dataset distribution shift and enabling sample-level diagnostics (ADR, MVD, EVR, n-MVD). The pipeline logs pre-exposure responses, fine-tunes then stabilizes the model, and computes per-sample deltas; it is instantiated on the Pythia family with The Pile and used to benchmark nine representative MIA methods. Empirically, methods like DC-PDD and Con-ReCaLL decline markedly, while Min-K%++ (and Ref when available) remain strong, with trends strengthening for larger models.

### Strengths
1. The paper defines a clear evaluation framework that isolates membership effects by comparing pre- and post-fine-tuning behavior on the same data.


2. The experiments are comprehensive including various MIA probing methods.


3. The introduction of new sample-level metrics (ADR, MVD, EVR, n-MVD) provides additional diagnostic views of membership signals, even if the conceptual novelty is moderate.


4. The paper is generally well organized and readable, with clear visualizations that support its main claims.

### Weaknesses
1. The approach requires access to both pre- and post-fine-tuning models (and control the training data to get rid of cross-dataset distribution shift), which may limit practical applicability in real-world privacy audits.
2. The methodological novelty appears limited. The four metrics (ADR, MVD, EVR, n-MVD) appear to be designed as heuristic diagnostics to visualize per-sample score changes before and after exposure.
3. The fine-tuning process isolates the 1.5k target instances, rather than mixing them with non-target samples. Evaluating under more realistic mixed-batch settings would strengthen the validity of the conclusions.

### Questions
See weaknesses.

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
This paper proposes a new evaluation framework, Delta-MIA, for evaluating MIA performance on LLMs to tackle three major limitations of typical observational evaluation framework: 1) unintended distribution shift due to challenges with partitioning member/non-member data (e.g., temporal shifts) resulting in inflated performance, 2) lack of coarse-grained analysis, obscuring sample-level behavior, and 3) poor transferability due to reliance on training corpora access. The authors instead propose an alternative perspective using an interventional paradigm. The main idea is to measure pure membership signals by performing a self-comparison between a model before and after exposure to some sample data using scores from a candidate MIA. The authors also introduce new sample-level metrics (e.g., ADR, MVD, EVR, and n-MVD) that allow for more granular analysis over captured membership signals. They then conduct a broad evaluation over nine modern MIAs on subsets of Pile data over the Pythia model family. Most MIA methods remain robust, but certain methods such as DC-PDD and Con-ReCaLL are shown to not capture membership signals as well as previously claimed.

### Strengths
- Delta-MIA tackles several important challenges in MI evaluation that are crucial for validating the effectiveness of current and future MIAs. The paper is well-written, easy to follow, and the problem is clearly motivated.
- The proposed framework is conceptually-straightforward and the introduced metrics are also intuitive. The authors also take care to clearly interpret visualizations to help readers understand nuanced performance differences between the different MIAs. 
- The evaluation is conducted over a broad range of models, MIAs, and data domains.

### Weaknesses
- This framework is a useful, necessary check to ensure that a candidate MIA is not capturing spurious signals (e.g. from unintended distribution shifts between benchmark members/non-members) and can actually detect true membership signals. However, I’m unconvinced this framework is sufficient to determine the effectiveness (i.e., vulnerability risk) of an MIA. Unless I have a misunderstanding, without comparing MIA signals between members and non-members under the same model, the discriminatory power of the MIA isn’t interpretable. For example, what if the MVD (or n-MVD) is also high for a set of non-members (to both the pre-/post-exposure models) from the same distribution as the target data? Then the model may not truly be that vulnerable to the MIA and the signal being captured may be something more than just membership. Perhaps it would be stronger to more clearly frame it as a complementary method (e.g., sanity check) to standard observational evaluation? If this is not the case, then it is still a little unclear to me how delta-MIA would be sufficient on its own.
- This framework also seems to be heavily dependent on the tuning process. I feel that there could be more discussion about the impact of the tuning phase, such as the choice of target tuning data (e.g., what domains), how much target data is used, and other design choices. Ablations in these and similar directions would be appreciated.
- Closely related to the above comment, it’s not clear to me how this framework bypasses the issue of choosing non-member samples. For example, target tuning data still needs to be verified as non-members to the pre-exposure model, which remains difficult for modern frontier models. 

I include more specific questions related to these points in the question section.

### Questions
- Could the scores presented still be inflated due to the recency of the target data relative to the entire training lifecycle? What if the injected data was instead inserted, for example, halfway through training. Currently, the scope seems more like “evaluating MIAs on finetuned LLMs”.
- Similar to the subexperiment in Appendix A, could the authors show how the tuning impacts performance on data from roughly the same distribution (e.g., another sub-sample from Pile test for Pile-CC, Wikipedia, etc.)? 
- Using the post-exposure model as the target model, could the authors also conduct a standard observational evaluation (using the target tuning data as members and another random sample of Pile test data selected the same way as non-members). It’d be interesting to concretely see if performance trends in this observational setting align with those under delta-MIA (e.g., maybe some attacks still seem performant in this observational setting, but under delta-MIA aren’t).
- What is the reason for having two-stage finetuning? For example, why not batch/randomize the selected 1500 samples with the 100000 samples, training in one stage?
- Do the authors have any results on different model families (e.g., Llama) to demonstrate the transferability of their approach?

Minor comments:
In Figure 1, target is misspelled as “traget”

### Soundness
2

### Presentation
3

### Contribution
3
