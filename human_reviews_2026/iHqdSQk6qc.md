# Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Effectively processing long contexts is a critical challenge for language models. While standard Transformers are limited by quadratic complexity and poor length extrapolation, alternative architectures like sliding window attention and state space models sacrifice the ability to effectively utilize the full context due to their fixed-size memory. Chunk-based sparse attention has emerged as a promising paradigm for extreme length generalization, yet the key architectural principles underpinning its success are not yet fully understood. In this work, we present a systematic dissection of these models to identify the core components driving their performance. Through a unified framework and comprehensive ablation studies, we demonstrate that a combination of three design principles is critical: (1) an expressive, non-linear Chunk Encoder with a dedicated CLS token to produce   representations for retrieval; (2) a Bypassing Residual Path to stably integrate retrieved global information without it being overridden by the local residual stream; and (3) enforced selection sparsity during pre-training to bridge the train-test distribution gap. We provide a theoretical motivation for intra-chunk information processing and landmark generation. By combining these principles, we establish a new state-of-the-art for training-free length extrapolation, successfully generalizing models trained on a 4K context to 32 million tokens on RULER and BABILong. Our findings provide a clear and empirically-grounded set of design principles for developing future, highly-capable long-context language models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explores a method combining chunk-wise attention and retrieval, enabling models trained on short texts to extrapolate to ultra-long contexts. The framework, an enhancement of DRT, investigates improvements like adding encoder for global KV representation, a CLS token for predicting landmark, and bypass residual connection. Experiments on RULER and babilong demonstrate that these incremental modifications yield significantly superior performance.

### Strengths
1. The proposed strategies—such as adding an encoder for global KV representations, using a CLS token as a landmark, and incorporating a bypass residual connection—are simple, yet deliver significant and immediate gains on downstream tasks. This makes the approach highly practical.


2. The paper provides detailed ablation studies that convincingly demonstrate the necessity of each component and the synergistic benefits of their combination.


3. The model is evaluated on well-known long-text benchmarks (RULER, babilong). The choice of benchmarks is reasonable and effectively characterizes the model's long-context capabilities.

4. The authors attempt to provide a rationale for their modifications, such as explaining how the encoder-processed CLS token landmark can better approximate the highly non-linear attention denominator.

### Weaknesses
Despite the strong empirical results, the paper's primary weakness lies in its insufficient theoretical justification for why the proposed modifications are effective. The framework is fundamentally similar to DRT, yet the analysis of the new components is cursory.

1. Ambiguous Role of the Encoder: The distinction in performance with and without the encoder is not substantial. It is unclear if the limited improvement stems from its purported "decoupling" function or simply from increased model capacity via extra parameters. The paper fails to clarify if this addition removes a specific bottleneck, or if adding the same parameter count elsewhere (e.g., in the base model) would yield similar gains.

2. Unclear Mechanism of the Bypass Connection: The large performance boost from the simple bypass residual connection is surprising. The authors' explanation—that "the Standard Sequential Path may disrupt the information flow"—is vague and unconvincing. The paper lacks a sufficient explanation for the mechanism by which this simple "re-wiring" dramatically improves retrieval performance.

Overall, the theoretical arguments for these modifications feel brief and tenuous, which undermines the reliability of the empirical findings.

### Questions
1. The authors state that sliding-window attention cannot access information outside its window. Given this limitation, why is its performance on the babilong benchmark at 8M context nearly identical to its performance at 4K? Does this imply the task itself does not require genuine long-range dependencies?

2. Do these long-context modifications harm the model's short-text capabilities? It would be beneficial to evaluate the model on standard short-text benchmarks (e.g., MMLU, Wikitext-2) to verify that the proposed changes do not cause performance regression in non-long-context scenarios.

### Soundness
2

### Presentation
3

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
This paper analyzes chunk-based hierarchical sparse attention models for extreme length extrapolation, specifically calls out three design choices: a non-linear chunk encoder, a bypassing residual path, and high selection sparsity. By combining these elements, the authors achieve state-of-the-art training-free length extrapolation, generalizing models trained on a 4K window to 32M tokens on the RULER and BABILong benchmarks. The work also provides theoretical motivation for the Chunk Encoder and diagnostic analysis correlating retrieval accuracy with task performance.

### Strengths
1. The paper has a clear presentation and nicely unifies different sparse attention approaches under a single framework. 
2. This method achieves significant training-free length extrapolation gains and retains them consistently on RULER and BABILong, marking strong empirical evidence.
3. The justification for the chunk encoder as approximating full attention scores is compelling.
4. The result in Appendix C showing that this method generalizes to hybrid SSMs is exciting and empirically sound.

### Weaknesses
1. The theoretical analysis is informal, and is not derived — there is no formal analysis of when and why this approximation is reasonable to make, although it is intuitive and stated as such. 
2. It would be valuable to have an analysis of computational costs, and any impact on inference speed (e.g. TPS and TTFT statistics) through this method.
3. Evaluation primarily focuses on mostly retrieval tasks and synthetic reasoning, some real-world tasks that are a bit more diverse (e.g. summarization over long-documents) would be valuable to show general utility. 
4. The model scale is limited, and it is unclear whether this extends to larger models; I am willing to overlook this given the results are quite compelling, but for practical utility, this is important to study.

### Questions
1. How do you initialize the CLS token, and does it matter? Have you experimented with multiple CLS tokens per chunk? 
2. Are there tasks where you find that your method underperforms full attention at shorter contexts, which would demonstrate a fundamental limitation of chunk-based retrieval?
3. Is your method sensitive to the chunk size hyperparameter, and does this vary with context length or task? An ablation on this would be valuable.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Paper summary

The paper studies why hierarchical, chunk-based sparse attention models can extrapolate to extreme context lengths and how to make them more reliable. Using a unified view of “dynamic chunkwise sparse attention,” it decomposes these systems into (i) how chunks are encoded and landmarks are formed for retrieval, (ii) how retrieved global information is fused back into the model, and (iii) how sparsity is enforced during training. A theoretical argument shows landmarks must approximate a chunk’s total attention mass, which requires a nonlinear chunk encoder rather than simple pooling. Empirically, models trained on 4K tokens generalize, without further tuning, to contexts up to 32M on RULER and to 8M on BabiLong. Figure 1 (page 3) diagrams the architecture; Figure 3 (page 6) shows BabiLong accuracy remaining high as length grows; Table 2 (page 7) reports strong RULER results at 32M; Table 3 (page 9) details sparsity ablations. 

Key contributions

* Unified framework + ablations: Formalizes chunk processing as functions over hidden states and compares weighting schemes (NSA, softmax, stick‑breaking) and encoder designs (with/without CLS), isolating which parts drive extrapolation. 
* Theory for landmark design: Shows that effective landmarks must approximate a chunk’s total attention mass, motivating a nonlinear Chunk Encoder and a dedicated CLS token for learnable summarization instead of mean pooling. 
* Cross‑layer fusion mechanism: Introduces a Bypassing Residual Path so retrieved global context is integrated via the MLP rather than overwritten by the residual stream; diagnostics separate retrieval recall from integration precision. 
* State‑of‑the‑art training‑free length extrapolation: 4K‑trained models reach 32M tokens on RULER with graceful degradation (e.g., ~80% average at 32M in Table 2) and maintain high BabiLong accuracy up to 8M tokens (Figure 3). 
* Principled sparsity insights: Finds that selection sparsity during pre‑training (small Top‑K like 8), sufficient training length (≥4K), and per‑token retrieval are critical for extreme generalization (Table 3). 
* Generality to hybrids: Applying the same principles to RAMba (Mamba + HSA) yields consistent gains across long lengths (Appendix C, Table 5).

### Strengths
Originality.

* Introduces a clean formalization of *Random Context Access* as the capability a long‑context model actually needs, which sharpens the objective beyond “long perplexity is stable” and motivates the architectural choices thereafter (Section 3.1). 
* Provides a unified lens for *dynamic chunkwise sparse attention* that decomposes chunk selection, weighting, and intra‑chunk processing; the side‑by‑side comparison of weighting rules (NSA, softmax, stick‑breaking) in Eq. 1a–1c and the f/g functions for landmark vs KV formation is a helpful synthesis that organizes a messy literature (Section 3.3; Table 1 on page 5). 
* Offers a simple but convincing theoretical target for landmark design: landmarks should approximate a chunk’s total attention mass, which in turn argues for a nonlinear chunk encoder and a CLS‑based summarizer rather than linear pooling (Eq. 4a–4b on page 5). 
* Proposes a *Bypassing Residual Path* that changes how retrieved global information is fused, with a concrete formulation (Eq. 6b on page 5) and a diagnostic that separates retrieval recall from integration precision (Figure 4 on page 8). 
* Demonstrates transfer of the principles to an SSM hybrid (RAMba) by inserting an explicit MLP to test the bypass mechanism and encoder/CLS design in a different backbone (Appendix C; Table 5 on page 18). 

Quality.

* Careful experimental design: all DRT variants hold parameter count (~240M) fixed and standardize key knobs (window 512, chunk size 64, Top‑K 8) so ablations isolate the proposed components rather than incidental capacity changes (Section 5.1). 
* Strong and transparent baselines: full‑attention with RoPE scaling (Llama‑YaRN), Landmark Attention, sliding‑window, stick‑breaking, and Mamba2 with State Passing are included; training FLOPs or receptive fields are matched where relevant (page 6). 
* Evaluations cover both retrieval‑only skills (RULER: S‑N, MQ‑N, VT) and retrieval‑plus‑reasoning (BabiLong), reducing the chance of overfitting to a single metric. The curves on BabiLong reach up to 8M tokens (Figure 3 on page 6), and RULER tables extend to 32M (Table 2 on page 7). 
* Diagnostics move beyond end accuracy: retrieval recall@Top‑K and *marginal integration precision* clarify that the bypass path mostly helps utilization while the encoder/CLS improves ranking prominence (Figure 4 on page 8). This is rare and genuinely informative. 
* Thoughtful sparsity study: clean ablations over training length, Top‑K, and tokens‑per‑retrieval show why selection sparsity during pre‑training and per‑token retrieval matter at extreme lengths (Table 3 on page 9). 

Clarity.

* The architectural flow is easy to follow: the DRT diagram highlights lower vs upper decoders and the chunking layer that produces landmarks and encoded chunks (Figure 1 on page 3). The two encoder variants are visually contrasted (Figure 2 on page 5). 
* Mathematical definitions are compact and targeted: Eq. 2 stitches chunk weights into standard attention; Eq. 4b states the landmark learning objective in one line; Eq. 6a–6b precisely defines the bypass change. The paper ties each equation to an experiment that tests it, which keeps the narrative crisp. 
* Results are presented with the right granularity: a single multi‑length RULER table makes degradation profiles legible (Table 2 on page 7), BabiLong’s averaged curve makes plateau/failure modes obvious (Figure 3 on page 6), and Appendix B lays out hyperparameters and training schedules in a reproducible table (Table 4 on page 17). 

Significance.

* Establishes training‑free extrapolation far beyond common practice: models trained at 4K hold up to 32M on RULER with graceful decay and maintain strong BabiLong accuracy to 8M, while RoPE‑style baselines collapse shortly after the training length (Table 2 on page 7; Figure 3 on page 6). This pushes the empirical frontier and sets a new reference point for “extreme” long‑context capability. 
* Distills actionable design rules that others can adopt: use a nonlinear chunk encoder with a CLS landmark, fuse via a bypass that lets the MLP mediate cross‑layer information, enforce selection sparsity during pre‑training, and retrieve per token. These prescriptions are validated again in a distinct hybrid (RAMba), which broadens the paper’s impact beyond a single architecture (Table 5 on page 18). 
* Clarifies what it means to “use the whole context” by centering Random Context Access and measuring retrieval prominence vs integration. That conceptual shift is likely to influence how future long‑context work frames objectives and ablations, not just how it reports numbers (Sections 3–5). 

Overall, the paper combines a tidy theoretical target, disciplined ablations, and record‑scale evidence to produce design guidance that is both novel and broadly useful to the long‑context community.

### Weaknesses
1. Positioning and novelty are under‑specified.
   The paper’s core design elements (chunkwise retrieval with landmarks, per‑token retrieval, and hierarchical sparse attention) are largely inherited from DRT/RAMba/NSA; the main additions are the specific *bypass* residual and the non‑linear chunk encoder with a CLS landmark. The text acknowledges these antecedents but doesn’t clearly delimit what is genuinely new vs. what is a careful re‑combination. Tighten the novelty claim by: (i) explicitly contrasting your equations and learning signals with DRT’s GCA and RAMba’s stick‑breaking weighting; (ii) running a head‑to‑head where you *retrofit* prior baselines with your two contributions (encoder+CLS and bypass) without other recipe changes, isolating marginal gains. Table 5 partially does this for RAMba‑MLP; extend the same “surgery” to Landmark and DRT to make the incremental contribution undeniable. 

2. Theory is insightful but stops short of guarantees.
   The “landmark should approximate a chunk’s total attention mass” target (Eq. 4a–4b) is a compelling heuristic, but it is not accompanied by identifiability or approximation bounds. As written, it is unclear when a finite‑depth encoder can satisfy Eq. 4b under distribution shift or adversarial distractors. Strengthen the section by: (i) providing an upper bound on the KL or total‑variation gap between α and α̂ (Eq. 3a vs 3b) under assumptions on within‑chunk key dispersion; (ii) proving that mean pooling fails in a controlled setting (e.g., mixture of von Mises–Fisher keys) and that a 2‑layer encoder with CLS can represent the log‑sum‑exp proxy; (iii) stress‑testing Eq. 4b with synthetic counter‑examples where per‑chunk keys are multi‑modal. Figure 2 and the argument around Eq. 4b are the right hooks; add formal statements and failure cases. 

3. Evaluation is narrow; external validity is uncertain.
   The study leans heavily on RULER S‑N/MQ‑N/VT and BabiLong. These are excellent for retrieval diagnostics, but they are synthetic and cue‑rich; they do not stress discourse coherence, paraphrase invariance, or multi‑hop compositionality across heterogeneous documents. Broaden the evidence by adding: (i) long‑document QA with paraphrase and entity aliasing; (ii) multi‑file code navigation where identifiers are obfuscated; (iii) book‑length summarization with contradiction traps; and (iv) “lost‑in‑the‑middle”‑style probes that deliberately place relevant spans at low prior positions. Figure 3’s averaged BabiLong curve and Table 2’s RULER tables are strong; add at least one real‑world, non‑templated task to bolster claims of “utilize the entire context.” 

4. Comparative fairness has confounds.
   Several baseline choices can be contested: Landmark is trained with shorter sequences to match FLOPs, which handicaps its retrieval policy; LLaMA‑YaRN is known to degrade beyond train length, so a single positional‑scaling baseline is a soft target. Since Section 5.1 already modifies DRT to use stick‑breaking for per‑token retrieval, some of the gains vs “DRT 0” could be due to weighting, not just encoder/CLS/bypass. Make the comparison harder to dispute by: (i) giving baselines the same retrieval frequency (per‑token) and Top‑K; (ii) matching *training sequence length* rather than FLOPs for Landmark; and (iii) adding a second strong positional baseline (e.g., an ALiBi/PI variant) trained with the same 4k context and SFT recipe. See the setup on page 6 and Table 2 on page 7. 

5. Compute and efficiency claims are underspecified.
   The paper emphasizes accuracy at 1M–32M tokens but gives no end‑to‑end latency, memory, or throughput numbers for these lengths, and no complexity constants for chunk selection. Appendix Table 4 signals substantial training/SFT budgets (≈8.05T pre‑training tokens; ≈1T tokens per SFT), which can swamp architectural effects. Add: (i) wall‑clock and peak‑memory curves vs. sequence length for Top‑K∈{4,8,16} and chunk size 64 on a standard accelerator; (ii) ablate retrieval index cost (score computation on landmarks) separately from token‑level attention; (iii) a compute‑normalized comparison (accuracy vs. TFLOPs) at long lengths. This will make the “practical” story credible. 

6. Scale and backbone generality remain open.
   All primary models are ~240M parameters. The transfer to RAMba is helpful, but results are still at small scale and with a custom RAMba‑MLP that inserts an MLP not present in the original block. Demonstrate robustness by: (i) repeating key ablations at ≥1B parameters; (ii) applying the encoder+CLS and bypass to a pure Transformer baseline with different local windows; and (iii) showing the trend holds when the chunking boundary is moved earlier/later in depth (Figure 1 shows a fixed mid‑point). Table 5 provides a good template; broaden it to other backbones and sizes.

### Questions
1. Precisely delineate what is new.
   The paper combines chunk encoders, a CLS landmark, and the bypass residual within a unified HSA view, but it is not crystal-clear what is truly novel relative to DRT, RAMba, and NSA beyond the specific encoder+CLS objective and the bypass wiring. Please provide a one‑page “delta table” contrasting your equations and learning signals with DRT’s retrieval and RAMba’s stick‑breaking setup, and quantify the marginal gains when retrofitting only your two advertised contributions (encoder+CLS and bypass) into prior baselines without other recipe changes. A tight novelty delineation would materially strengthen the paper. See the unified comparison in Table 1 (page 5), Eq. 6a/6b (page 5), and the baseline list in Section 5.1 (page 6). 

2. Landmark theory: can you give a formal guarantee or falsifiable test?
   Eq. 4a/4b (page 5) proposes that the landmark score should track a chunk’s total attention mass. Could you provide either: (i) a bound on how well a finite encoder can approximate this mass under simple distributions (e.g., mixture of von Mises–Fisher keys), or (ii) a calibration plot showing correlation between your selection score and the true within‑chunk mass across lengths? A short, synthetic study or correlation analysis would make the claim more than heuristic. 

3. Bypassing residual: mechanism and gradients.
   The bypass changes the residual addition point (Eq. 6a vs 6b, page 5), and Figure 4 suggests improved utilization. Can you show gradient‑flow or activation‑scale diagnostics demonstrating that the bypass prevents residual stream “overwrites”? A simple ablation with a learned gate or scalar on the bypass path and a rank‑1 sensitivity analysis would help attribute the gain to integration rather than retrieval side effects. Cite Figure 4 (page 8) and Eq. 6a/6b (page 5). 

4. Baseline fairness and apples‑to‑apples choices.
   Landmark Attention is trained on shorter sequences to match FLOPs, whereas your models train at 4k; retrieval frequency and Top‑K also differ across methods. Please report a “length‑matched” Landmark baseline at 4k and harmonize retrieval frequency (per‑token) and Top‑K across baselines. Adding one strong positional baseline beyond YaRN (e.g., ALiBi/PI) at 4k would reduce the chance that your positional baseline is a soft target. The request is small at 32k/128k (not 32M). See Section 5.1 (page 6) and Table 2 (page 7). 

5. Compute and efficiency at long lengths.
   The results emphasize accuracy up to 32M tokens, but the paper lacks wall‑clock, memory, and throughput numbers for chunk selection vs token attention. Please add a figure with end‑to‑end latency and peak memory versus length for Top‑K in {4, 8, 16}, chunk size 64, and the hardware used. Even a 32k→128k→1M sweep would make the practical story credible. Table 4 (page 17) shows large training/SFT budgets; efficiency data would help separate algorithmic benefits from sheer scale. 

6. Scale and backbone generality.
   Most main results use ~240M parameters. Appendix C shows encouraging RAMba results, but still at small scale and with a RAMba‑MLP variant. Can you share any existing runs or partial results at ≥1B parameters, or at least trends when moving the chunking boundary earlier/later in depth in the Transformer backbone? Figure 1 (page 3) and Table 5 (page 18) are the relevant anchors. Even partial curves at 32k/128k would be informative.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper try to interpret why chunk-based sparse attention models, for examples DRT and RAMbe, tend to generalize well to longer contexts than they were trained on. The authors argue that these models implicitly approximate a property they call Random Context Access (RCA)—the ability to flexibly retrieve any relevant part of the past without full attention. They identify three key design choices that make this work. 1. A non-linear chunk encoder with a CLS token, which learns better chunk summaries than simple mean pooling. 2. Bypassing residual path, which helps integrate retrieved global information without it being drowned out by local residuals. 3. Training with small Top-K retrieval, which avoids train–test mismatch.
They build on a 240M-parameter DRT backbone trained at 4k tokens and show strong length extrapolation—up to 32M tokens—on RULER and BabiLong, outperforming full attention and other long-context baselines.

### Strengths
1.The paper offers a clear conceptual story. They provide a nice way to unify many previous chunk-based designs with the proposed concept, and the proposed concept is simple and effective.
2. The empirical results are strong: models trained on 4k contexts can extrapolate to tens of millions of tokens. 
3.The ablation study is especially thorough—it covers encoder depth, presence of CLS, bypass design, Top-K size, and training length, giving a convincing picture of what really matters.

### Weaknesses
1.The theory part is more intuition-driven than rigorous; the RCA argument isn’t formally analyzed.
2. The experiments more rely on synthetic or controlled tasks. The experiment on realistic long-context applications is limited.
3. The detailed efficiency analysis of the method is not discussed.

### Questions
see weakness section

### Soundness
3

### Presentation
3

### Contribution
3
