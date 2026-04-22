# When Symbols Speak: Understanding Logo Triggered Texts in Vision-Language Models

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Vision Language Models (VLMs) have achieved impressive progress in multimodal reasoning, yet they remain vulnerable to hallucinations where outputs are not grounded in visual evidence. In this paper, we investigate a previously overlooked setting: logo hallucination, where models generate brand names or textual content despite logos containing no visible words. Using curated splits of pure symbols, hybrids, and text-bearing logos, as well as the challenging Hard-60 subset, we systematically measure hallucination across leading VLMs. We further probe robustness through nine structured perturbations and show that hallucinations persist even under strong distortions, with occlusion exposing the sharpest weakness. Embedding-level analysis with open-weight LLaVA demonstrates that hallucination is tied to a small subset of projector dimensions, and targeted ablation substantially reduces errors while preserving OCR accuracy. Together, these findings reveal that VLMs often rely on symbolic priors rather than genuine glyph perception, particularly for iconic circular logos, and that projector subspaces play a decisive role in this failure mode. Our work contributes both a novel diagnostic lens and actionable mitigation insights, highlighting projector disentanglement and OCR-guided decoding as promising directions for building more trustworthy multimodal systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces logo hallucination, the phenomenon where Vision-Language Models (VLMs) incorrectly generate brand names or textual outputs from logos that contain no actual text. They apply controlled perturbations and embedding-level diagnostics to demonstrate that hallucination persists under image distortions and originates from specific projector subspaces within the VLM architecture. Targeted ablation of a small number of projector dimensions reduces hallucination by ~30% with minimal OCR accuracy loss.

### Strengths
1. Combines taxonomy (text/symbol/hybrid logos), controlled perturbations, and embedding diagnostics in a reproducible and well-defined experimental pipeline.

2. Clarity and presentation: The figures, tables, and structure (bias, perturbation, projector) make the argument cohesive and empirically grounded.

### Weaknesses
1. The study is constrained to logo datasets. While logos are a clean diagnostic, it’s unclear whether findings generalize to natural images or scene text. This restriction limits the paper’s general significance.

2. The work is primarily diagnostic rather than methodological. It identifies a failure mode (logo hallucination) and analyzes it well, but does not propose or validate a concrete new training or architectural solution. The projector ablation experiment is an insightful analysis tool, not a deployable method.

3. The link between projector dimensions and hallucination is correlational; ablation reduces the symptom but does not establish a principled mechanism or causal model.

4. The discussion on emotional/value hallucination (luxury, elegance) is anecdotal—no quantification or modeling of this axis.

### Questions
Does logo hallucination persist in other symbolic domains (flags, icons, traffic signs)? Could this be a broader symbolic-text entanglement issue?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies "logo hallucination" in VLMs — where models generate brand names from purely symbolic logos. Through careful experiments across logo types, image perturbations, and projector analysis, the authors show this issue is widespread, robust to input changes, and linked to specific embedding directions. The work offers both a new diagnostic perspective and a practical mitigation path.

### Strengths
Logo hallucination is a new and relevant failure mode that hasn't been systematically studied before.

The three-stage framework provides a rigorous, systematic, and multi-faceted investigation into the phenomenon.

### Weaknesses
- The paper positions "logo hallucination" as a new and overlooked phenomenon. However, the tendency for models to hallucinate highly correlated labels for salient visual concepts is a well-known issue (e.g., "object hallucination"). The paper would be strengthened by a more explicit discussion of how the demonstrated "logo hallucination" constitutes a meaningfully distinct failure mode, rather than simply being a specific manifestation of the broader object hallucination problem.

- Insufficient Detail on the Hard-60 Subset: More detailed statistics and examples in the main text would help readers understand its composition and the nature of its challenge.

- The current separation of the comprehensive methodology (Section 3) from the experimental results (Section 4) makes the paper occasionally hard to follow. Perhaps restructuring the sections to present each experimental phase alongside its immediate results would improve the narrative flow and readability.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
2

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
This paper investigates the phenomenon of logo hallucination in Vision-Language Models (VLMs). The authors discover that VLMs generate brand names even when logos contain no visible text, suggesting that models rely on symbolic priors rather than genuine visual recognition. The study systematically examines this phenomenon, revealing logo hallucination behavior across different models, perturbations, and scenarios.

### Strengths
1. Logo hallucination is an overlooked yet significant security risk in VLMs. This problem has practical implications for applications such as brand impersonation detection and fraud prevention.

2. Experiments are conducted across multiple mainstream VLMs to validate the generalizability of the findings.

### Weaknesses
1. Insufficient evidence for causal attribution.
- The paper claims that hallucinations stem from "symbolic priors in token embeddings" rather than visual feature extraction problems, but the evidence is insufficient to support this causal inference.
- The paper's logic: Ablating k=32 key dimensions → reduced hallucination rate → concludes these dimensions contain "symbolic priors" → proves it's a token embedding problem. However, this finding can equally be explained as a visual feature representation issue.

2. Lack of mechanistic explanation. While the paper states that logic hallucination stem from "symbolic priors rather than genuine glyph perception," it does not explain how these priors form during training

3. Limited experimental scale. Hard-60 contains only 60 logo samples, potentially insufficient to support strong generalization claims.

### Questions
1. Experiments primarily use English brand names. Do logos with text in other languages (e.g., Chinese, Japanese signage) produce similar hallucinations?

2. How was k=32 determined? Was grid search or cross-validation performed? How would different k values (e.g., k=16 or k=64) affect the results?

### Soundness
2

### Presentation
3

### Contribution
2
