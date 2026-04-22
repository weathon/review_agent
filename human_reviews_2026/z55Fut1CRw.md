# UniFork: Exploring Modality Alignment for Unified Multimodal Understanding and Generation

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Unified image understanding and generation has emerged as a promising paradigm in multimodal artificial intelligence. Despite recent progress, the optimal architectural design for such unified models remains an open challenge. In this work, we start by analyzing the modality alignment behaviors of task-specific expert models for understanding and generation, as well as current unified models. Our analysis reveals a crucial observation: understanding tasks benefit from a progressively increasing modality alignment across network depth, which helps build up semantic information for better comprehension; In contrast, generation tasks follow a different trend—modality alignment increases in the early layers but decreases in the deep layers to recover spatial details. These divergent alignment patterns create a fundamental conflict in fully shared Transformer backbones, where a uniform representational flow often leads to performance compromises across two tasks. Motivated by this finding, we introduce UniFork, a novel Y-shaped architecture that shares the shallow layers for cross-task representation learning, while employing task-specific branches in deeper layers to avoid task interference. This design effectively balances shared learning and task specialization. Through extensive ablation experiments, we demonstrate that Unifork consistently outperforms conventional fully shared Transformer architectures, and achieves performance on par with or better than task-specific models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Empirical characterization of divergent modality-alignment patterns for understanding vs generation and evidence of representational compromise in fully shared NTP models.

UniFork Y-architecture that shares early layers and decouples late layers into task-specific branches while keeping a simple AR/NTP interface.

Simple, effective training recipe (three stages, no loss reweighting) that allows independent branch fine-tuning without delicate data balancing.

### Strengths
The paper’s exploration of modality alignment patterns in understanding versus generation tasks fills a gap in prior research by investigating how these tasks differ at a deep level. The observation that understanding benefits from progressively increasing alignment, while generation requires an initial rise and later drop, is a novel and insightful contribution to the multimodal field.

The presentation of the Y-shaped Transformer architecture is clear, and the distinction between the shared early layers and task-specific branches is well-articulated. The decision to share the first half of the layers and split the latter half for task-specific learning is explained in an understandable manner, making the technical concept more accessible without oversimplifying it.

### Weaknesses
The paper correctly notes that when M=0, UniFork is structurally similar to Mixture-of-Transformers designs like BAGEL, and its dual-pathway approach is conceptually related to the Janus series. However, the results tables lack a direct, quantitative comparison with these specific models under similar training budgets. This omission makes it difficult for the reader to gauge whether the Y-shaped architecture provides a tangible advantage over other state-of-the-art parameter-efficient unification strategies.

Actionable Improvement: Include direct comparisons with BAGEL and/or a Janus-series model in Tables 2, 3, and 4. This would precisely delineate UniFork's contribution within the existing landscape of unified model architectures.

The performance jump from the 0.57B to 0.76B ablation variant is promising, but it's a single data point. A critical unanswered question is how the UniFork architecture behaves at different scales (e.g., with a 3B or 7B backbone). Does the optimal ratio of shared-to-task-specific layers (M and N) change with model size?

Actionable Improvement: Conduct a basic scaling study. For instance, implement UniFork on two different backbone sizes (e.g., 0.5B and 2B) and show that the architecture's benefits are consistent. This would significantly bolster the claim that it is a general-purpose design.


The generative performance is assessed primarily on GenEval (text-image alignment) and FID (general quality). This misses important dimensions like compositional reasoning, complex scene rendering, and aesthetic quality.

Actionable Improvement: Supplement the evaluation with benchmarks like T2I-CompBench， WISE for compositionality and, ideally, a human evaluation study to assess image realism and prompt adherence on more complex, creative prompts.

The "Where to Fork" Ablation is Missing: The analysis reveals that the alignment trends diverge, but the choice to decouple specifically in the later layers is a design decision. The paper lacks an ablation study that justifies this specific choice. Is forking in the final third of the network truly optimal? What is the performance impact of forking earlier (e.g., after the first third) or later?

Actionable Improvement: Introduce a critical ablation experiment that varies the forking point M within the Transformer stack while holding total depth constant. A plot showing task performance versus forking depth would provide direct, empirical evidence that the chosen configuration is optimal, transforming a design choice into a data-driven conclusion.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces UniFork, a novel Y-shaped architecture that shares the shallow layers for cross-task representation learning for unified model. The authors diagnose that image understanding and image generation tasks demand different cross-modal feature alignment behaviours in transformer layers. Experiments demonstrate performance gains relative to fully shared architectures, achieving similar or better results compared to task-specific expert models.

### Strengths
1. Insightful analysis of modality alignment patterns: The paper provides a systematic investigation into the alignment dynamics between text and image representations across different architectures and tasks (understanding vs. generation). By empirically identifying distinct alignment patterns and connecting them to architectural design choices, the work offers valuable conceptual insights for building more principled and efficient unified multimodal models.
2. Well-documented training and implementation details: The paper reports detailed information on training strategy, data composition ratios, and optimization settings, which facilitates reproducibility.

### Weaknesses
1. Outdated and weak baselines: The compared baselines (e.g., MobileVLM, Emu, LaVIT, LDM, LWM) are mostly early-generation unified or multimodal models whose performance and architecture are now considerably behind state-of-the-art models such as Emu3, Bagel, or Janus-Pro. Even when compared to these relatively weak baselines, UniFork shows only marginal or inconsistent improvements across several benchmarks. This weakens the empirical strength of the claimed performance gains and raises questions about whether the proposed method would still hold advantages under stronger baselines.
2. Lack of ablation or sensitivity analysis on branch duplication and split ratio: The proposed UniFork architecture duplicates the latter half of the Transformer backbone (based on Qwen2.5) to construct two independent branches for understanding and generation. However, there is no thorough analysis or comparison regarding this specific design choice—e.g., why the fork point is placed in the latter half, and how sensitive the results are to the split ratio between shared and task-specific layers.
As described in §3.2 (“Given a Transformer of (M + N) total layers”), the determination of M and N (the numbers of shared and branch-specific layers, respectively) seems largely heuristic. A systematic study or empirical justification of these hyperparameters would substantially strengthen the paper’s credibility and generality.

### Questions
1. Choice of fork depth (M, N): In §3.2 the authors mention that the Transformer backbone is divided into M shared and N task-specific layers (“Given a Transformer of (M + N) total layers”). How was the specific fork point determined in practice?
  - Was it based on alignment measurements, empirical tuning, or a fixed heuristic?
  - Have you explored sensitivity analysis on different M/N splits (e.g., forking earlier vs. later), and how does this affect performance or modality alignment?
2. Scalability and future experiments: The current experiments are conducted on relatively moderate-scale models. Do the authors plan to extend UniFork to larger-scale settings to further validate the scalability and general effectiveness of the proposed architecture?
It would be valuable to know whether the observed alignment patterns and performance trends remain consistent at higher scales.

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
4

### Summary
The paper analyzes the variation curves of modality alignment scores in the intermediate layers of understanding models and generative models from the perspective of modality alignment. Based on the differences in the score variation curves between understanding and generative models, the paper proposes UniFork, a novel Y-shaped architecture that shares shallow layers for cross-task representation learning while employing task-specific branches in deeper layers to avoid task interference. The proposed model shows a significant improvement compared to baseline models.

### Strengths
1. This paper innovatively analyzes unified generative large models from a modality alignment perspective. Based on this analysis, the proposed Y-shape structure is experimentally tested and demonstrates good performance
2. The paper's presentation is good; the figures clearly convey the conclusions and experimental results of the paper.

### Weaknesses
1. The scale of the experiment is insufficient. The experimental model in the paper is only 0.5B~0.76B in size, which is too small compared to other existing unified understanding-generation models. I personally believe that it is necessary to further expand the model scale and verify the effectiveness of the method with a larger number of parameters.
2. The other methods compared in the paper are somewhat outdated. For example, widely accepted and published unified large model papers such as show-o2, tokenflow, and unitok were not compared or discussed
3. The paper did not conduct experiments on the number of layers for shared and unshared layers.

### Questions
Please see weakness
1.  Regarding the curve of intermediate layer alignment scores in generative models, it shows a distribution of first rising and then falling. However, the author only conducted comparative experiments on generative models using a pixel tokenizer and neglected the unified tokenizer. If semantic information is introduced in the tokenizer, would such an alignment score curve still hold?

### Soundness
3

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
2

### Summary
This paper introduces UniFork, a Y-shaped Transformer model designed to handle both image understanding and image generation in one model. It works by sharing early layers for general learning and splitting into task-specific branches later. UniFork achieves strong results on both understanding and generation tasks, outperforming several understanding-only and unified models, despite of being much smaller in size.

### Strengths
- The paper is mostly well written and structured. The core idea—the conflict in alignment patterns and the Y-shaped solution—is easy to follow.
- The alignment pattern analysis is insightful and seems new. 
- On the chosen evaluation benchmarks, UniFork has shown stronger performance than the ablated instances and prior models.

### Weaknesses
- Limited Understanding Benchmarks. The understanding benchmarks tested in this paper is quite outdated (e.g., VQAv2 and GQA). For visual perception, consider including benchmarks like MMBench, BLINK, CVBench, MM-VET, MMVP. 
- Limited baseline comparison, although Janus Pro and Bagel are cited, they are being compared against UniFork.
- The paper fixes the split point between shared and task-specific layers, but doesn’t show what happens when you change how many layers are shared (e.g., early split vs. late split). This would test how sensitive UniFork is to the choice of fork depth and whether the proposed setting is optimal.

Minor Presentation Issues
- Figure 7 caption: clouseup -> closeup
- the GenEval benchmark is spelled “GenEval” in some places and “Geneval” in others

### Questions
In addition to the weakness above, please find additional questions below.

- Is symmetric branch really necessary? As it is task-specific head anyways, would asymmetric branches (e.g., even more branches for generation) work just as well or better?

### Soundness
2

### Presentation
2

### Contribution
2
