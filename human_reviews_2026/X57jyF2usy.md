# SPG-SAM: Semantic Prompt Graph Learning for Multi-class Medical Image Segmentation

- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
Existing visual foundation model-based methods (e.g., SAM) for multi-class medical image segmentation typically face a trade-off between insufficient semantic information and spatial prompt interference, while extending SAM with fully automated semantic segmentation compromises its inherent interactive prompting capabilities. To bridge the semantic specificity gap, we propose SPG-SAM (Semantic Prompt Graph learning for SAM), a novel framework that seamlessly integrates spatial and semantic prompting for efficient and accurate multi-class medical image segmentation. SPG-SAM introduces dedicated semantic prompts to complement SAM’s spatial prompts, establishing an explicit mapping between object locations and semantic categories. Furthermore, we introduce a semantic prompt graph learning module that employs a graph attention network to explicitly model anatomical priors and structural relationships among medical objects. This design enables cross-category feature interaction, mitigates prompt interference, and facilitates accurate and efficient multi-class segmentation within the SAM-based paradigm. Experimental results demonstrate that SPG-SAM achieves average Dice coefficients of 94.27\% and 91.83\% on the abdominal multi-organ segmentation (BTCV) and pelvic target segmentation (PelvicRT) tasks, respectively, outperforming the second-best state-of-the-art baselines by 2.1\% and 3.65\%. The code will be available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose SPG-SAM, a novel framework that extends the Segment Anything Model (SAM) for multi-class medical image segmentation. SPG-SAM introduces semantic prompts alongside traditional spatial prompts (e.g., points or bounding boxes) to endow SAM with class-specific awareness. The key idea is to bridge SAM’s lack of semantic understanding by coupling spatial cues with categorical information and by modeling inter-class anatomical relationships using a Semantic Prompt Graph (SPG).

### Strengths
- The paper presents a technically novel idea of augmenting SAM with semantic category prompts and a prompt graph. By introducing class-specific token embeddings and coupling them with SAM’s spatial prompts, the method creates an explicit mapping between object locations and their labels. This dual prompting strategy is innovative and addresses a key limitation of vanilla SAM (which lacked intrinsic semantic understanding). The addition of a Graph Attention Network to model inter-class anatomical relationships is a creative architectural contribution, enabling cross-category feature interactions and encoding prior knowledge of organ topology (e.g., which organs are adjacent) in the segmentation process.

- The method still allows user input (e.g., bounding boxes or points for each target) but enhances the results with semantic context.

- Overall, the presentation is logical and easy to follow. Experiments, support important claims, and even potential failure cases (like slight drops on certain classes) are noted, reflecting a high level of transparency.

### Weaknesses
- A notable concern is that the evaluation is conducted only on 2D slice-based segmentation, ignoring the 3D nature of CT scans. The method processes each axial slice independently (treating BTCV’s volumetric CT data as 2D images), and all baselines compared (UNet, TransUnet, SwinUnet, etc., as well as SAM and its variants) are 2D models. This choice leaves out state-of-the-art 3D segmentation approaches (for example, a 3D U-Net or nnUNet), which often achieve superior accuracy by leveraging volumetric context. Not including any 3D baseline makes the comparisons narrower. It is possible that a strong 3D model could close the gap or outperform SPG-SAM on these tasks, especially since organs span multiple slices.

- The experiments are limited to two relatively small datasets. BTCV (a MICCAI 2015 challenge dataset) has on the order of only 30 patients (the paper uses 2,178 axial slices for training/validation/testing combined, and PelvicRT is a private dataset of similarly limited scale (7 target structures). While the results on these are strong, it is unclear if the method would generalize to larger, more diverse benchmarks. 

- The way the method is evaluated involves using ground-truth bounding boxes as spatial prompts for each target organ. This is a reasonable proxy to benchmark the model’s best-case performance, but it assumes an oracle provides perfect prompts. In practice, a user might not know the exact tight bounding box of an organ without substantial effort (which partly defeats the purpose of an “interactive” assistive tool). 

- SPG-SAM, as presented, has a fixed set of semantic prompts corresponding to known target classes (13 organs for BTCV, 7 for PelvicRT). The graph’s nodes and the prompt tokens are specialized to those categories. A natural question is how the method would handle an unforeseen class or a different set of organs.

### Questions
- Could the authors please consider extending or applying SPG-SAM in a 3D context? Given that medical scans are 3D volumes, would it be feasible to utilize 3D patches or slices in multiple orientations as input to incorporate more contextual information? Additionally, how do you anticipate SPG-SAM to compare with a robust 3D segmentation model (e.g., nnUNet) on these tasks? Could the current approach be adapted to leverage depth information, or is it fundamentally constrained by SAM’s 2D image encoder?

-  In the experiments, the authors use ground-truth bounding boxes as prompts for each class. In a real use case, a clinician might provide less precise prompts (or fewer prompts). Have you evaluated how the accuracy degrades with imperfect prompts or fewer prompts? 

- How are the semantic class embeddings (the “category embeddings”) obtained or initialized? Are they learned parameters associated with each class label during training, or something like word embeddings (e.g., using the class name)?

-  The graph attention network adds computational overhead. Could the authors provide some insight into how this affects inference speed or memory usage in practice?

- The concept of using a graph to encode anatomical priors is quite intriguing. Is the graph attention mechanism solely learning relationships from data, or do you introduce any prior knowledge? For instance, do you ever explicitly encode that certain nodes shouldn’t connect because those organs don’t co-occur?

### Soundness
3

### Presentation
2

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
This paper addresses the ​​trade-off between insufficient semantic information and spatial prompt interference​​ in SAM-based multi-class medical image segmentation by proposing the ​​SPG-SAM framework​​. The core innovations include: 1) A coordinated encoding scheme that integrates semantic prompts with spatial prompts to establish explicit mapping between object locations and semantic categories; 2) A semantic prompt graph learning module using graph attention networks to explicitly model anatomical priors and structural relationships. Experimental results demonstrate average Dice coefficients of 94.27% and 91.83% on BTCV and PelvicRT datasets respectively, outperforming the second-best baselines by 2.1% and 3.65%. This approach effectively resolves semantic ambiguity in multi-class segmentation while preserving SAM's interactive capabilities.

### Strengths
- Originality:​​ The pseudo-prompt filling strategy for absent categories and learnable semantic embeddings establish a novel category-to-spatial mapping paradigm, overcoming SAM's single-class limitation.
- Technical Quality:​​ Extensive experiments cover 13 abdominal organs (BTCV) and 7 pelvic targets (PelvicRT). Ablation studies validate the impact of graph attention insertion positions (α vs β in Table 4), while t-SNE visualizations demonstrate enhanced feature clustering.
- ​​Clarity:​​ The three-stage pipeline (prompt encoding → graph learning → decoding) is well-described with complete mathematical formulations (Equations 5-8). Code availability promotes reproducibility.
- ​​Significance:​​ Provides a new framework combining anatomical priors with interactive prompting, with potential applications in surgical navigation and radiation therapy planning.

### Weaknesses
- ​​Computational Efficiency:​​ Although the paper reports only 4.9ms latency overhead for the graph module (Table 6), the O(N^2) complexity of GAT may become problematic with larger category sets - scalability analysis is lacking.
- ​​Generalization Validation:​​ Experiments focus exclusively on CT modality without cross-modal validation on MRI or ultrasound, limiting conclusions about broader applicability.
- ​​Interactive Prompt Analysis:​​ Table 7 compares point/box prompts but doesn't evaluate robustness to prompt quality variations (e.g., annotation errors), which is crucial for clinical deployment.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses a key limitation of SAM-based approaches for multi-class medical image segmentation: the tension between insufficient semantic information and spatial prompt interference, and the loss of interactive prompting when fully automating SAM. The authors propose SPG-SAM, which augments spatial prompts with dedicated semantic prompts and introduces a Semantic Prompt Graph Learning module using a graph attention network to encode anatomical priors and inter-structure relationships. Experiments on BTCV and PelvicRT show strong gains, with average Dice improvements of 2.10% and 3.65% over the SOTA.

### Strengths
1. Assigning semantics to spatial prompts to bridge SAM's foreground/background nature and enable true semantic segmentation is elegant and well-motivated. Retaining interaction while enabling multi-class output addresses a core gap between fully-automatic fine-tuning and prompt-driven SAM usage.

2. The graph-based fusion of spatial and semantic prompts to model anatomical priors and cross-category interactions is reasonable and technically sound.

### Weaknesses
1. A central advantage of SAM is its strong zero-shot generalization to unseen object categories given a prompt. By injecting learned semantic prompts and category-specific graph structure, SPG-SAM risks category lock-in. If the model is trained on a subset of target organs, how does it perform on new organs with only spatial prompts available? Can the framework accept a new “semantic prompt token” on-the-fly without re-training? How sensitive is performance to ontology changes (merged/split labels, institution-specific definitions)?

2. The ablations indicate large drops without SPGL, but the paper should articulate the failure modes it resolves. What specific inter-class confusions or spatial conflicts does SPGL mitigate?
It's better to provide qualitative failure cases without SPGL and show how graph attention corrects them (e.g., improved boundary consistency? reduced overlap conflicts? better small-organ recall?).

3. How SPG-SAM resolves overlapping prompts and adjacent organ conflicts compared to running SAM separately per class. Is the final output a single mutually exclusive multi-class mask or multiple per-class masks post-processed with conflict resolution? If the latter, what is the arbitration policy?
In Figure 2 row 1, for the case where vanilla SAM mis-segments the red class but SPG-SAM succeeds, analyze the causal mechanism. Is it due to cross-category context, graph-induced attention to adjacent structures, or disambiguation from semantic prompts?

### Questions
See weaknesses

### Soundness
2

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
3

### Summary
SPG-SAM is a framework that extends the SAM for multi-class medical image segmentation by introducing semantic prompts alongside SAM’s original spatial prompts. The method augments SAM’s point or box prompts with learned semantic embeddings for each anatomical class, establishing an explicit mapping between object locations and their semantic.A GAT module is incorporated to model the relationships among different organs/tissues, enabling cross-category feature interactions and mitigating interference between multiple prompts in complex multi-organ scenes.

### Strengths
The paper is generally well-written and organized, with extensive experiments and ablation studies that validate the contributions of the semantic prompts and the graph module.

### Weaknesses
While the method is designed to preserve SAM’s interactivity, the evaluation assumes one prompt per target class (using ground-truth annotations to simulate prompts) for multi-organ segmentation. The work does not explicitly demonstrate how SPG-SAM performs in a truly interactive setting with a limited number of user inputs or how it balances user effort versus automation.

### Questions
It would be valuable to include experiments simulating real user interactions. For instance, the authors could evaluate SPG-SAM under varying numbers of prompts per image (e.g., one, few, or many), or measure segmentation accuracy as a function of user input effort.

### Soundness
3

### Presentation
3

### Contribution
3
