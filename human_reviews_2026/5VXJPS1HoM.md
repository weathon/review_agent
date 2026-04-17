# Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning

- Decision: Accept (Oral)
- Scores: 4, 8, 8, 6

## Abstract
Deepfake detection remains a formidable challenge due to the evolving nature of fake content in real-world scenarios. However, existing benchmarks suffer from severe discrepancies from industrial practice, typically featuring homogeneous training sources and low-quality testing images, which hinder the practical usage of current detectors. To mitigate this gap, we introduce **HydraFake**, a dataset that contains diversified deepfake techniques and in-the-wild forgeries, along with rigorous training and evaluation protocol, covering unseen model architectures, emerging forgery techniques and novel data domains. Building on this resource, we propose **Veritas**, a multi-modal large language model (MLLM) based deepfake detector. Different from vanilla chain-of-thought (CoT), we introduce *pattern-aware reasoning* that involves critical patterns such as "planning" and "self-reflection" to emulate human forensic process. We further propose a two-stage training pipeline to seamlessly internalize such deepfake reasoning capacities into current MLLMs. Experiments on HydraFake dataset reveal that although previous detectors show great generalization on cross-model scenarios, they fall short on unseen forgeries and data domains. Our Veritas achieves significant gains across different out-of-domain (OOD) scenarios, and is capable of delivering transparent and faithful detection outputs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces the HydraFake dataset and the VERITAS model.
The HydraFake dataset encompasses a diverse range of deepfake generation techniques and real-world forgery cases. It provides a rigorous training and evaluation protocol that covers unseen model architectures, emerging forgery techniques, and novel data domains.The VERITAS detector, built upon a Multimodal Large Language Model (MLLM), differs from conventional Chain-of-Thought (CoT) based reasoning frameworks. It employs a two-stage training pipeline — pattern-guided cold start followed by pattern-aware exploration — with a particular emphasis on key reasoning modes such as “planning” and “self-reflection”.By simulating the human forensic reasoning process, VERITAS integrates the cognitive capabilities of large language models into deepfake detection (DFD), thereby achieving enhanced cross-manipulation and cross-domain generalization performance.

### Strengths
The proposed VERITAS framework demonstrates strong innovation in the field of Deepfake detection. Its core contribution lies in transforming the traditional feature-memorization-based detection paradigm into a reasoning-based framework grounded in pattern-aware reasoning.Unlike previous models that rely on specific forgery artifacts, VERITAS explicitly models forgery patterns as transferable reasoning units, enabling the understanding of common structural characteristics shared across different forgery types. In addition, the authors introduce a Chain-of-Thought (CoT) reasoning mechanism, allowing the model to progressively identify manipulated regions through multi-step logical inference, thereby enhancing interpretability and robustness.
The paper further proposes a Mixed Preference Optimization (MiPO) strategy, which balances the learning preferences among various forgery patterns, effectively improving the model’s cross-manipulation and cross-domain generalization capabilities.Overall, this study presents substantial innovation and academic value across three dimensions — detection paradigm, reasoning mechanism, and optimization strategy.

### Weaknesses
1. Lack of fair comparison with recent large-scale multimodal frameworks.

Although VERITAS demonstrates impressive performance, the experimental comparison is incomplete and lacks fairness.
The paper does not include evaluations against recent large-scale multimodal or reasoning-based deepfake detection frameworks, such as FakeShield (ICLR 2025), M2F2-Det (CVPR 2025), and SIDA (CVPR 2025).

These methods share similar reasoning or multimodal fusion paradigms and therefore represent the most relevant baselines for comparison.

In contrast, most of the reported baselines are lightweight detectors (with only tens or hundreds of millions of parameters), while VERITAS is built upon an 8B-scale MLLM.


2. Limited domain coverage.

The current work focuses exclusively on face-oriented deepfake detection, whereas a growing number of studies have begun exploring cross-domain and generalized forgery detection, such as AIGC-generated content, IMDL (image manipulation localization), and general multimedia forensics frameworks (e.g., Effort, FakeShield, ForensicHub(NeurIPS 2025) ).

Expanding the scope of VERITAS to handle non-facial or multi-domain forgery types would not only enhance its practical value but also align it with the emerging trend toward universal content authenticity verification.

### Questions
Q1: Could the authors provide additional comparisons or discussions with recent large multimodal reasoning-based detectors (e.g., FakeShield, M2F2-Det, SIDA) to better contextualize VERITAS?

If this question is properly addressed, it would increase my score, as a fair comparison with large models is the most critical issue.

Q2: Does the proposed VERITAS framework have the potential to generalize beyond face forgery detection — for example, to AIGC-generated or other cross-domain manipulation tasks?

If this aspect is explored or discussed, it would further raise my score, though it is secondary to the first point.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper makes two primary contributions to the field of deepfake detection. First, it introduces HydraFake dataset, which extends the evaluation into a rigorous and hierarchical setting, providing a more realistic and challenging benchmark to the field. Second, the paper proposes Veritas, a novel MLLM-based detector. Experiments show both qualitatively and quantitatively the effectiveness of the model.

### Strengths
- The proposed dataset is well-motivated. The division of four evaluation levels (i.e., in-domain, cross-model, cross-forgery and cross-domain) is reasonable. A fine-grained evaluation protocol is critical and reasonable at the moment, and the constructed dataset is of high quality, providing a challenging evaluation suite for the community.
- The proposed pattern-aware reasoning is effective and insightful compared to previous explainable methods. Experiments clearly show its superiority compared to previous pipelines.
- The proposed MiPO and P-GRPO is coherent with the proposed reasoning framework. Thorough ablations are conducted to validate their effectiveness.

### Weaknesses
- The authors should provide some failure cases to understand the model’s limitations.
- More fine-grained ablations on the reasoning patterns could be done, e.g., what if removing the “reflection”/“planning” pattern?
- How "reflection" improves model's generalization capability to unseen forgeries? The author should provide more explanations to it.
- The human has very good reasoning capabilities. Why even human cannot accurately detect some (realistic) deepfakes? Is semantic-level reasoning capability truly crucial for deepfake detection?

### Questions
1.	In Figure 1, the proposed Veritas can perceive the textual anomalies. Is such type of analysis contained in the training set? If not, can the base model conduct similar analysis?

2.	In the MiPO stage, how is the image selected? What if removing the non-preference $s_l^{\phi}$ in the training data? It would be great to provide some case studies.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces HydraFake, a deepfake detection dataset with hierarchical OOD evaluation (in-domain, cross-model, cross-forgery, cross-domain), and VERITAS, an MLLM-based detector using pattern-aware reasoning. VERITAS employs five thinking patterns (fast judgment, planning, reasoning, self-reflection, conclusion) trained via a two-stage pipeline: pattern-guided cold-start with MiPO and pattern-aware exploration with P-GRPO. The approach shows significant improvements in cross-forgery and cross-domain scenarios, which are important in real-life scenarios.

### Strengths
- The proposed HydraFake dataset addresses a crucial gap between academic benchmarks and industrial deployment scenarios, making it valuable for real-world applications.
- The authors have conducted comprehensive experiments, including comparisons with SOTA methods and detailed ablation studies, verifying the effectiveness of the proposed VERITAS model.
- The pattern-aware reasoning approach is reasonable, drawing inspiration from human cognitive processes to create more interpretable and robust detection systems.

### Weaknesses
- The two-stage training pipeline with MiPO and P-GRPO, though modified for forgery detection tasks, still seems to be a direct application of vanilla DPO and GRPO methods, which may undercut its novelty.
- The paper evaluates VERITAS exclusively on the proposed HydraFake dataset, raising concerns about overfitting to their specific evaluation protocol. It would strengthen claims about VERITAS's superiority and provide more convincing evidence of its effectiveness if authors could perform evaluations on other benchmarks such as LOKI [cite 1] and Forensics-bench [cite 2].

[cite 1] LOKI: A COMPREHENSIVE SYNTHETIC DATA DETECTION BENCHMARK USING LARGE MULTIMODAL MODELS. In ICLR 2025.

[cite 2] Forensics-Bench: A Comprehensive Forgery Detection Benchmark Suite for Large Vision Language Models. In CVPR 2025.

### Questions
None

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
This paper first introduces the HydraFake dataset, which aggregates real and fake images from existing sources and by reimplementing and crawling 10K deepfake samples produced by 10 advanced generators, resulting in 50K real and 50K fake images. It also proposes a two-stage training pipeline for MLLM-based deepfake detection. In the first (SFT) stage, MiPO is introduced to internalize reasoning patterns; in the second stage, the pattern-aware GPRO promotes comprehensive reasoning and enables potential self-reflection. Extensive experiments on in-domain, cross-model, cross-forgery, and cross-domain evaluation sets demonstrate the effectiveness of the proposed method.

### Strengths
a.The proposed dataset spans diverse domains and sources of real and manipulated images, including generative face‑swapping, visual autoregressive models, and deepfakes collected from social media.
b.The two‑stage training pipeline substantially outperforms existing deepfake detection methods, as demonstrated in Table 1.

### Weaknesses
a.The proposed method employs SFT and GPRO within an MLLM‑based deepfake detection framework—an established post‑training strategy.
b.The difference between the proposed pattern "<fast><planning><reasoning><conclusion>" and the commonly used "<think>... </think>" paradigm has not been analyzed.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
