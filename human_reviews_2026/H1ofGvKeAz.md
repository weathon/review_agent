# Grounding the Ungrounded: A Spectral-Graph Framework for Quantifying Hallucinations in multimodal LLMs

- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Hallucinations in LLMs—especially in multimodal settings—undermine reliability. We present a rigorous information-geometric framework, grounded in diffusion dynamics, to quantify hallucinations in MLLMs where model outputs are embedded via spectral decompositions of multimodal graph Laplacians, and their gaps to a truth manifold define a semantic distortion metric. We derive Courant–Fischer bounds on a temperature-dependent hallucination profile and use RKHS eigenmodes to obtain modality-aware, interpretable measures that track evolution over prompts and time. This reframes hallucination as quantifiable and bounded, providing a principled basis for evaluation and mitigation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
The paper proposes a rigorous theoretical framework that reframes hallucination as a measurable and bounded phenomenon. By embedding model outputs on multimodal graph Laplacians and modeling their divergence from a truth manifold as a spectral energy, the authors derive Courant–Fischer bounds and temperature-dependent decay dynamics that quantify how hallucination evolves over time. Using a KL-smoothed, reference-free semantic distortion score and energy-based modeling within RKHS, the framework unifies information geometry, diffusion processes, and graph theory. Experiments across COCO, VQAv2, and AudioCaps with multiple model stacks show consistent improvements over entropy- and margin-based baselines. While the method is mathematically elegant and modality-interpretable, it is complex, computationally heavy, and tested only on modular pipelines, leaving questions about scalability and practical integration into real-world MLLMs.

### Strengths
Modality-Aware and Interpretable. The method handles text, vision, and audio jointly using a product kernel representation. 

Reference-Free Metric. Unlike dataset-based hallucination detection, this method does not require external ground truth, making it suitable for open-domain or partially verifiable tasks.

### Weaknesses
Strong Hyperparameter Dependence. Requires tuning of several sensitive parameters.

Lack of Comparison with Practical Mitigation Methods. No comparison against RLHF, Contrastive decoding, attention fix methods.

### Questions
See weakness.

### Soundness
3

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
This paper tackles the problem of hallucination in multimodal large language models (MLLMs) by proposing a principled, information-geometric framework for quantitative measurement. Unlike prior heuristic or annotation-based methods, the approach embeds model outputs on multimodal graph Laplacians and measures their deviation from a truth manifold as a semantic-distortion metric.
Using diffusion dynamics and RKHS eigenmodes, the authors derive Courant–Fischer bounds on a temperature-dependent “hallucination energy,” providing interpretable, modality-aware measures that evolve over prompts and time. Experiments across multiple MLLM configurations (e.g., CLIP, Whisper, T5, BLIP) validate the theoretical results, showing consistent spectral behavior under varying temperatures.

### Strengths
1. The study of hallucination in LLMs and MLLMs is an important and timely research topic.

2. This work provides a theoretical grounding for hallucination quantification, offering valuable insights and potential guidance for future research in this area.

3. The open-sourced codebase enhances reproducibility and supports further validation by the community.

### Weaknesses
As I am not familiar with the theoretical aspects, my assessment focuses mainly on the experimental design and empirical evaluation:

1. The experimental setup does not fully align with the paper’s claim. Although the authors aim to address hallucination in LLMs/MLLMs, the experiments only involve traditional multimodal models such as BLIP, CLIP, Whisper, and T5, rather than modern autoregressive LLMs like Qwen-Audio/VL/Omni or the GPT-4/5 series. This limitation significantly constrains the paper’s potential impact and generalizability.

2. The effectiveness of the proposed method remains uncertain. As shown in Table 1, the improvement is relatively modest, with around 3% on simpler tasks such as captioning. Its applicability and scalability to more complex reasoning tasks or stronger LLM backbones remain unclear.

3. I'm also curious about the efficiency. For both autoregressive and non-autoregressive models, does the proposed framework introduce additional inference overhead or reduce generation speed compared to the original model?

### Questions
Please see the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a theoretical and quantitative framework for measuring hallucinations in multimodal large language models (MLLMs).
The authors model hallucination as a measurable deviation from a “truth manifold” within a multimodal RKHS, where model outputs are represented as nodes on a multimodal graph. Specifically, the paper presented 1. a KL-calibrated semantic distortion score that serves as a reference-free metric for hallucination; 2. a multimodal energy-based formalism; 3. a spectral decomposition of hallucination energy through eigenmodes of a multimodal Laplacian.

### Strengths
1. The paper aims to quantify hallucination via information metrics, which is a novel perspective.

2. The paper aims to define measure from mathematical formulations, which paves the way for rigorous evaluations and analysis.

### Weaknesses
1. There is a general lack of motivation for the development of formulations in the paper. While $K_g$ appears in early set-ups, all subsequent developments are built on $K$. The paper neither discuss how $K$ and $K_g$ are different in practical LLM usages, nor provide clues on how $K$ is obtained in experiments (is it dependent on the specific LLM, training data or evaluation data?), making the definition of hallucination idealized and not connected to practice.

Furthermore, the paper does not exhibit the need for introducing information metrics to quantify hallucination at the cost of computation complexity. Namely, why cannot hallucination be simply quantified as a mean Hilbert distance between MLLM generations and $K$? There is a general lack of discussion for the properties and necessities for the measure it proposed.

2. In presenting the theorems, there is a lack of logic flow and therefore the reader are prone to get confused. For both Theorem 1 and theorem 2, there are hidden assumptions are not explicitly presented before the theorem, (for instance line 977 in the proof of Theorem 1; for theorem 2 how the energy form in (8) is linked to (10) as a polynomial form of the embeddings), which makes the theorems very confusion. Upon reading the theorem, the reader cannot understand what is needed to be proved in the theorem, making the statements lacking in mathematical rigor.

3.  The measure in Theorem 1 is not a mathematically natural measure as it incorporates the truncation operator ([]+) in its calculation, making the measure not continuous in space. While the paper claims that the measure is $=0$ on $K$ and $>0$ outside $K$, the proof can only show that the untruncated measure is $<0$ on $K$, therefore given the continuous nature of the function, the claim is highly likely unreliable and there must also be $x\not\in K$ for which $d(x)=0$.

4. For Section 5, despite the paper decomposes the energy into eigenspaces and shows bounds in terms of the spectrum, there is a lack of analysis on the actual scale and shape of spectrums for real MLLMs. Therefore the reader cannot get useful messages, conclusion or insights from the establish of formulations and derivations. First, the Courant–Fischer bounds are given to the hallucination energy, and there is still a large gap between energy and observables like hallucination rates or sematic distances. Second, some unmeasurable coefficients (m(t) , M(t) for instance) are introduced to present the result, whose shape we cannot know in practice. Third, the time is a confusion factor in the analysis as it does not correspond to the real time in MLLM applications, and the read cannot know what assumptions are made w.r.t. time.

### Questions
Please refer to the weakness part for questions.

In section 5: why is there "time" in the analysis? Is it the annealing process of an MLLM whose parameters are left unchanged? How does it relates to practical MLLM behaviors?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes a theory-backed, modality-aware framework to quantify hallucination as a continuous quantity rather than a binary label. The core statistic is a KL-smoothed semantic distortion $d^{(\varepsilon,h)}_{\text{sem}}$ that is $0$ on an admissible set $K$ of “grounded” outputs for a prompt and $>0$ off $K$, obtained by contrasting a $K$-restricted smoother with an unconditional smoother. On top of this, the authors build a multimodal hypergraph over outputs and define a hallucination energy via a spectral graph-Laplacian formulation, with Courant–Fischer bounds providing theoretical envelopes for the energy. They evaluate across three datasets (COCO Captions, VQAv2, AudioCaps) and three inference stacks, reporting AUROC/AUPRC against uncertainty baselines (entropy, max-prob, margin) and visualizing energy surfaces versus $\varepsilon$ and temperature. Results indicate consistent gains over baselines and empirical consistency with the spectral bounds, while practical details (how $K$ is constructed/normalized per dataset, tighter bound diagnostics, and artifact completeness) remain under-specified in the main text.

### Strengths
•	Clear positioning: gap = no theory-backed, modality-aware framework that quantifies hallucinations.

•	Proposes a KL-smoothed semantic distortion $d^{(\varepsilon,h)}_{\text{sem}}$ that is $0$ on an admissible set $K$ and $>0$ off $K$, via a $K$-restricted vs. unconditional smoother (Eq. 6).

•	Hypergraph construction is naturally multimodal, yields a hallucination energy, and admits CF bounds.

•	Empirically validates theory (three datasets × three stacks) and outperforms baselines (AUROC/AUPRC).

•	Mathematical development is careful: assumptions explicit, derivations clear, measurability addressed.

### Weaknesses
1.	**Primary weakness:** Dense formalism buries the practical message—what problems does this framework actually solve, and under what conditions should practitioners prefer it over standard uncertainty baselines? Also, please explain how the mathematical assumptions for the theorems (beyond “g-free” which is adequately explained) are translated to real world scenarios.
2.	“Reference-free” vs. operational $K$: Reconcile the “independent-of-$g$” claim with the use of a finite admissible set $K$ and selector $\Pi_K$. Add a short subsection on what is observable, assumed, and estimated **and how this connects to plausible practical scenarios.** Also, please explain the advantage of a continuous (not binary) hallucination quantity.
3.	$K$ construction & labeling unclear: Specify $K(p)$ per dataset (COCO captions; VQAv2 normalized unique answers; AudioCaps references) and provide the exact normalization/tokenization for membership. 
4.	Baselines: Describe each competitor in the main text, and why this choice of competitors suffices.
5.	Empirical Bounds:  In Fig. 3 the CF planes appear loose and the explored $\varepsilon$ range seems narrow. Justify ranges and report quantitative gap-to-bound stats (median/percentiles) and their relation to errors.
6.	Unclear mathematical roadmap: Add a one-paragraph roadmap (“first…, then…, finally…”). A symbol table is adviseable too.
7.	Equation numbering issues: In Sec. 5.3 numbering appears misaligned (mirrors appendix). Please fix.
8.	Conclusion takeaways. Distill actionable guidance, e.g. where this fits; when energy vs. score is most predictive. Add practical takeaways to method: default choices for $\varepsilon, h, T_t$, etc.
9.	Please perform literature check. For instance, I could not verify “Spectral characterization of hallucination in large language models.” This may even undermine authenticity of writing. Please clarify.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
1

### Contribution
3
