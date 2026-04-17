# Easier Painting Than Thinking: Can Text-to-Image Models Set the Stage, but Not Direct the Play?

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Text-to-image (T2I) generation aims to synthesize images from textual prompts, which jointly specify what must be shown and imply what can be inferred, which thus correspond to two core capabilities: ***composition*** and ***reasoning***. Despite recent advances of T2I models in both composition and reasoning, existing benchmarks remain limited in evaluation. They not only fail to provide comprehensive coverage across and within both capabilities, but also largely restrict evaluation to low scene density and simple one-to-one reasoning. To address these limitations, we propose **T2I-CoReBench**, a comprehensive and complex benchmark that evaluates both composition and reasoning capabilities of T2I models. To ensure comprehensiveness, we structure composition around scene graph elements (*instance*, *attribute*, and *relation*) and reasoning around the philosophical framework of inference (*deductive*, *inductive*, and *abductive*), formulating a 12-dimensional evaluation taxonomy. To increase complexity, driven by the inherent real-world complexities, we curate each prompt with higher compositional density for composition and greater reasoning intensity for reasoning. To facilitate fine-grained and reliable evaluation, we also pair each evaluation prompt with a checklist that specifies individual *yes/no* questions to assess each intended element independently. In statistics, our benchmark comprises 1,080 challenging prompts and around 13,500 checklist questions. Experiments across 38 current T2I models reveal that their composition capability still remains limited in high compositional scenarios, while the reasoning capability lags even further behind as a critical bottleneck, with all models struggling to infer implicit elements from prompts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce T2I-CoReBench, a benchmark for evaluating both composition and reasoning capabilities of text-to-image (T2I) generation models with 12 dimensions. The benchmark comprises 1,080 prompts and ~13,500 checklist visual questions, enabling fine-grained evaluation with an MLLM-based yes/no answerer. Experiments on 28 T2I models (diffusion, autoregressive, and unified architectures) reveal that while composition is steadily improving, reasoning remains the main performance bottleneck. Further, prompt rewriting shows some effectiveness in bridging this gap but remains limited in scenarios requiring deeper visual reasoning.

### Strengths
1. The authors provides the first benchmark that evaluate both composition and reasoning capabilities of T2I models.

2. The main findings (reasoning lags behind composition & prompt rewriting helps but still there’s gap) is well supported with large experiments (e.g., table 3) and valuable for future research direction.

### Weaknesses
1. **Evaluation method.** In L143-145, the authors mentioned `We propose an automatic checklist-based evaluation protocol ... individual yes/no questions ... allows fine-grained and reliable assessment' as a second contribution. However, such automatic checklist-based evaluation protocol (i.e., question generation followed by question answering) was already extensively studied in previous works [e.g., A, B, C, D], but none of them have been cited or discussed. I don't think authors can claim evaluation methodological novelty, which is separate from benchmark construction. The authors should cite these works and clarify their contributions.

- [A] Hu et al., TIFA: Accurate and Interpretable Text‑to‑Image Faithfulness Evaluation with Question Answering. ICCV 2023
- [B] Yarom et al., What You See is What You Read? Improving Text‑Image Alignment Evaluation. NeurIPS 2023.
- [C] Cho et al., Visual Programming for Text‑to‑Image Generation and Evaluation. NeurIPS 2023
- [D] Cho et al. Davidsonian Scene Graph: Improving Reliability in Fine‑grained Evaluation for Text‑to‑Image Generation. ICLR 2024.

### Questions
1. This is minor point, but I feel the title is hard to understand. I didn't understand what title meant before reading the paper, and didn't expect to read text-to-image evaluation benchmark paper at all. Now I guess that authors meant composition and reasoning by 'set the stage' and 'direct the play', but still, such connection wouldn't be clear for many new readers.

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
This paper introduces T2I-CoReBench, a Composition and Reasoning Benchmark for systematic evaluation of T2I models. T2I-CoReBench addresses research gaps in comprehensiveness and complexity, evaluating both composition and reasoning within a broader taxonomy. The dataset is constructed by an automatic pipeline, using LRMs to generate data points given instructions, and further examined by humans. Each data point is accompanied by a checklist, utilizing Google Gemini Flash 2.5 as the evaluation protocol to assess checklist accomplishment and provide a quality evaluation score. Comprehensive experiments evaluate 28 T2I models with different architectures and find that most models fall short on overall performance.

### Strengths
1. Solid contribution in constructing a comprehensive and challenging benchmark on T2I. T2I-CoReBench comprehensively covers composition and reasoning tasks in a detailed taxonomy. 
2. Broad evaluation of different models offers great insights: The paper benchmarks 28 T2I models, covering open-source and closed-source models, as well as different architectures (Diffusion, Autoregressive, Unified). This makes the results comprehensive and convincing. Insights on "Prompt Rewriting" in Section 4.3 are intriguing.
3. Good Presentation Quality.

### Weaknesses
1. Evaluation relies on a single indicator, which cannot comprehensively measure the quality of T2I: There is only one indicator, the achievement rate of checklist questions, which cannot measure such as the overall rationality and stylization of the pictures. Given the examples, it also mainly examines item appearance in images, regardless of model hallucinations, etc.
2. Both the composition and reasoning tasks have their own previous benchmarks. Merely merging the two tasks into a more comprehensive benchmark limits their necessity.

### Questions
See weaknesses above.

### Soundness
2

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
The paper presents T2I-CoReBench, a benchmark for evaluating the compositional and reasoning abilities of text-to-image (T2I) models. The benchmark dissects model performance across multiple reasoning dimensions—logical, behavioral, hypothetical, and generalization reasoning—along with compositional skills such as multi-instance, multi-attribute, and text rendering. The authors argue that current T2I models excel in simple composition but struggle with structured reasoning that connects textual causality and visual outcomes. Comprehensive experiments compare state-of-the-art open- and closed-source models, uncovering systematic reasoning failures.

### Strengths
1. The authors categorize reasoning into fine-grained dimensions (e.g., logical, behavioral, hypothetical, abductive), offering a structured framework that is more interpretable than single-score benchmarks.

2. The inclusion of over a dozen models, spanning proprietary and open-source variants, makes the findings broadly representative and informative.

### Weaknesses
1. Some reasoning categories overlap (e.g., behavioral vs. hypothetical), raising questions about independence and orthogonality among evaluation axes.

### Questions
1. Is there any bias in the prompt generated synthetically? 
2. Is there any distinction between the generated images of open and closed source models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces T2I-COREBENCH, a high-density, checklist-based benchmark to jointly test composition (multi-instance, multi-attribute, multi-relation, text rendering) and reasoning (eight subtypes derived from deductive/inductive/abductive patterns) for text-to-image models. Evaluating 28 recent models, the authors find a clear gap: models can “paint” the specified elements but struggle to “think” through multi-step or implicit requirements, especially in dense scenes. Reasoning, not composition, is the current bottleneck.

### Strengths
- Well-motivated split between composition vs. reasoning, with 12 concrete dimensions.
- Dense prompts + yes/no checklists enable fine-grained, scalable evaluation.
- Broad study on 28 models gives useful community signals about where T2I still fails.

### Weaknesses
- Evaluator dependence and possible bias. The pipeline crucially depends on an MLLM to judge success. The paper would be stronger with a multi-evaluator or human-heavy subset to rule out evaluator-specific artifacts .

### Questions
- How did you de-correlate prompt/checklist style from the evaluator to avoid agreement-by-style?
- Did you experiment with soft scores or multiple MLLM votes, and if not, why stick to strict yes/no, which can make dense prompts disproportionately harsh?

### Soundness
3

### Presentation
3

### Contribution
3
