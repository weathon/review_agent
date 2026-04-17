# AutoMat: Enabling Automated Crystal Structure Reconstruction from Microscopy via Agentic Tool Use

- Decision: Reject
- Scores: 4, 4, 2, 8, 4

## Abstract
Machine learning models for interatomic potentials and force fields require high-quality structural data, yet experimental crystal structures remain limited, creating a critical gap between computational simulations and real structures. While atomic-resolution electron microscopy can provide valuable images, converting these into simulation-ready structures is time-consuming and error-prone. We present **AutoMat**, an agent-based framework that directly converts Scanning Transmission Electron Microscopy (STEM) images to atomistic crystal structures for property prediction. The framework adaptively calls tools, including pattern-adaptive denoising, physics-guided template retrieval, symmetry-aware atomic reconstruction, and property prediction, enabling closed-loop reasoning from "image → structure → property." For systematic evaluation, we introduce **STEM2Mat-Bench**, a benchmark dataset containing 450+ annotated samples. Performance is assessed using lattice root-mean-square deviation (RMSD), formation energy mean absolute error (MAE), and structure matching accuracy.  Results demonstrate that AutoMat outperforms existing approaches including SOTA models, specialized domain tools, and closed-source multimodal large models. This work establishes a direct pathway from microscopic characterization to atomic-scale modeling, addressing a fundamental challenge in materials science.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents AutoMat, an agent-based framework that directly converts Scanning Transmission Electron Microscopy (STEM) images into Crystallographic Information Files (CIFs), aiming to bridge the gap toward end-to-end structural analysis. The main contribution lies in the agent framework, which consists of four components: MOE-DIVAESR, Template Matching, STEM2CIF, and MatterSim (a sota tool that is not part of the paper’s original contribution). The idea is reasonable; however, several important issues need to be addressed.

### Strengths
The overall storyline is reasonable. The end-to-end structural analysis is decomposed into four key steps, and the agent is capable of orchestrating the tools to implement the entire workflow. The validation is multi-faceted and relatively comprehensive.

### Weaknesses
The current dataset is quite limited, containing only 2,143 structures, with 450 used for validation. This significantly affects the generalizability of the proposed workflow. Moreover, AI-assisted characterization is indeed a highly practical strategy; however, relying solely on simulation-based validation poses a high risk under a purely theoretical framework. Therefore, validation using experimental real-world data should be incorporated.

### Questions
+ As I emphasized in the limitations, please provide experimental validation, at least the case studies, to demonstrate the framework’s effectiveness on real data.

+ It is unclear whether the structures retrieved from the database are used as ground truth in the validation process. If that is the case, then after Template Matching, the system essentially matches a noisy synthetic STEM image to the already recorded crystal structure, which is considered the “ground truth.” In that scenario, the STEM2CIF step seems unnecessary, as it only introduces additional noise and produces a structure that deviates from the retrieved “ground truth.”
This also raises another concern: if the retrieved structures are directly used as reference, then the composition correctness (C.C.) metric will trivially reach 100%, thus losing its discriminative meaning.
	
+ The validation using formation energy MAE does not provide additional insight, since MatterSim is relatively robust to small structural mismatches, making the results insensitive to moderate reconstruction errors.
 
+ The novelty of each component is rather limited. The current agent mainly relies on rule-based and prompt-driven tool scheduling, lacking any learnable policy optimization such as reinforcement learning-based tool selection, which would significantly strengthen the framework’s adaptability.

### Soundness
3

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
3

### Summary
The work presented proposes AutoMat, an LLM agentic pipeline that converts a single STEM image into a simulation‑ready crystal structure (CIF) and then predicts formation energy via an ML interatomic potential. The agent coordinates four tools: (1) MOE‑DIVAESR for pattern‑adaptive denoising, (2) template retrieval from a library of simulated STEM projections, (3) STEM2CIF for symmetry‑aware reconstruction, and (4) MatterSim for energy relaxation/prediction.

### Strengths
* The image → structure → property loop is valuable and timely for AI-for-science, integrating denoising, retrieval, symmetry-aware reconstruction, and MLIP relaxation in a single agent is a meaningful system contribution.

* The presented benchmark shows that AutoMat outperforms domain toolkits and VLMs on structural accuracy

### Weaknesses
* Internal inconsistency with dataset and tiering count, its stated that there is  450-case test split, yet in section 3.3 its reported that tier sizes 35, 456, 79.
* The pipeline is motivated by experimental workflows, yet all quantitative results use abTEM simulations (fixed pixel size, dose ranges, aberration/noise models). No results are shown on real STEM micrographs
* AtomAI is a segmentation/localization library, not a CIF reconstructor; holding it to CIF-level metrics seems mismatched.
* The upper bound using the ground-truth CIF and MLIP shows an MAE of 48–57 meV/atom, substantial residual versus DFT even for perfect structures. Reported AutoMat MAEs (~320–330 meV/atom) thus reflect a mix of (i) structural error, (ii) MLIP bias vs. DFT, and (iii) sensitivity to starting geometries.
* The appendix reports agent/backbone swaps with ±3% variation and the retry mechanism, but there is no controlled ablation showing the marginal benefit of MOE-DIVAESR vs. a strong single denoiser, nor of symmetry constraints, nor of retrieval vs. no-retrieval. Given the four modules, their individual contributions should be quantified.

### Questions
* Considering the 450 test cases vs. 35/456/79 tier sizes (sum 570). Which is correct, and what numbers were used in Tables 1–2? If 570 is correct, where do the extra cases come from?
* How does AutoMat perform if the ground-truth structure is not in the retrieval library? Can you report structural metrics and energy MAE in that setting? 
* Can you add real-image evaluation on well-characterized monolayers with known CIFs? 
* Is it possible to provide module-wise ablations?
* How sensitive is retrieval to microscope parameter mismatch (defocus, aberrations, pixel scale)? Any simple calibration step before retrieval?

### Soundness
2

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
4

### Summary
The paper introduces AutoMat, an agent-based system that claims to reconstruct atomic crystal structures directly from Scanning Transmission Electron Microscopy (STEM) images and predict material properties. The proposed workflow integrates, denoising, a template-matching retrieval module, STEM2CIF for lattice reconstruction, and MatterSim for energy prediction—managed by an LLM “agent” for orchestration. Additionally, the authors present STEM2Mat-Bench, a synthetic benchmark dataset of 450 annotated samples, used to evaluate reconstruction accuracy and formation-energy errors. Results show AutoMat outperforming other baselines methods in structural RMSD and energy MAE metrics.

### Strengths
The problem from STEM images to downstream atom position and properties is very useful in connecting Experiments with simulation's.

### Weaknesses
a) Overstated novelty. Many ideas(agenticEM2CIF, agentic orchestration) are incremental extensions of existing frameworks such as SciLink, MicroscopyGPT and Mdcrow.

b) Benchmark validity. The comparison between agentic and non-agentic systems is not apples-to-apples. LLM-VLM baselines are not designed for such end-to-end tasks, making the reported “order of magnitude” improvement misleading. Also AtomAI is like python module with Neural nets like “unet” to segment images, how is it fair to compare it with agentic systems? Rather a study over different llm's orchestrating the same "tools" would be more useful. Basically answering questions like what models perform better or worse, do domain models beat bigger non domain models?

c) Lack of experimental(data collected on Microscope) validation, severely limits scientific impact.

 d) Unclear originality of released data. Since the dataset is built on existing repositories (C2DB, OpenCrystal), the “data release” claim is more of a repackaging effort.




references :
- AtomAI - https://github.com/pycroscopy/atomai, 
- SciLink - https://github.com/ziatdinovmax/SciLink
- Mdcrow - https://github.com/ur-whitelab/MDCrow
- MicroscopyGPT - notebook present at  : https://github.com/atomgptlab/atomgpt

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces AutoMat, an LLM Agent-based framework to automate the conversion of raw STEM images into simulation-ready crystal structures (CIFs) for property prediction. The agent orchestrates a pool of specialized tools (including a Mixture-of-Experts denoiser, template retrieval, a symmetry-aware reconstruction module, and an ML potential for property prediction) to create an end-to-end "image -> structure -> property" pipeline. On a new benchmark, STEM2Mat-Bench, AutoMat outperforms all baselines by an order of magnitude.

### Strengths
- Solves a Critical Problem: Provides the first end-to-end automated solution for the high-value bottleneck of converting experimental microscopy images into usable computational models.

- Specialized Toolset: The framework's power lies in its domain-specific tools which possess knowledge that general-purpose VLMs lack .

- Overwhelming SOTA Performance: AutoMat achieves order-of-magnitude improvements in both structural (RMSD) and energy (MAE) metrics over all baselines, including domain-specific and general models .

- Benchmark Contribution: The paper introduces STEM2Mat-Bench, a high-quality, tiered benchmark (Tier 1-3) that is essential for evaluating robustness and driving future research in this area .

### Weaknesses
- Agent Role Unclear: The workflow appears to be a mostly fixed, sequential pipeline (Denoise -> Match -> Reconstruct -> Predict). The "agent's" role seems more like a controller than a dynamic planner, making the "agentic" claim feel overstated.

- 2D Limitation: The entire framework and benchmark are restricted to 2D monolayer materials, which is a significant simplification of the general materials science problem.

- Template Dependency: The error analysis shows that 39.3% of failures are due to "Template retrieval failure", indicating a critical dependency on a comprehensive template database

### Questions
- Necessity of the Agent: How much "intelligent" reasoning does the agent perform? How much would performance drop if AutoMat were implemented as a hard-coded pipeline (Denoise -> Match -> Reconstruct -> Predict) with a simple retry loop?

- Novel Structure Performance: How does AutoMat handle a truly de novo crystal structure that is not in its template database? Can the STEM2CIF module reconstruct a lattice from scratch without a template?

- Path to 3D: What are the primary blockers to extending this framework to 3D bulk materials? Is it fundamentally impossible to reconstruct a 3D structure from a single 2D STEM projection as used here?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a framework that takes raw STEM images and textual information as inputs. It leverages an agent system to call four tools, enabling image preprocessing, atomic structure analysis, and property prediction. The authors also construct a comprehensive benchmark for this task, and the proposed method achieves SOTA performance compared with several baseline VLMs.

### Strengths
The paper tackles a meaningful problem: converting STEM images (and related textual context) into atomic structures and properties usually requires manual expert intervention. Using an agent framework with an LLM as the controller to call different domain tools is an interesting and valuable direction for the field.
1. The authors construct a benchmark covering unary, binary, and ternary materials to test the feasibility of this idea, and show that the proposed system achieves strong performance and outperforms several general-purpose vision-language models.
2. The tool pool is clearly defined and mostly self-contained: a self-trained denoising module (MOE-DIVAESR), a custom template matching algorithm, a custom STEM2CIF reconstruction module, and MatterSim for property prediction.
3. The paper shows that the LLM agent can perform multi-step reasoning, including rollback and re-calling tools when errors occur, suggesting genuine decision-making ability rather than fixed scripting.

### Weaknesses
1. The contribution is mainly twofold: (1) the benchmark, and (2) the agent framework. The benchmark construction, although based on simulated data, is fairly comprehensive and valuable. However, the agent framework feels more like an engineering implementation rather than a methodological innovation.
2. The paper’s description may give the impression that the agent itself directly interprets visual information from STEM images. In practice, however, the LLM component is text-only (DeepSeek V3), and the visual understanding is handled by separate image-processing modules. Moreover, the comparison with other VLM baselines does not clearly demonstrate that incorporating visual modalities provides any tangible benefit for this task. Clarifying this distinction—and discussing whether visual reasoning is necessary or helpful in this setting—would make the paper’s claims and design choices easier to understand.
3. To strengthen the benchmark, adding a human-expert baseline would make the comparison more meaningful, since the goal is partly to evaluate how close the agent can come to replacing manual human work. Moreover, if the dataset could include real STEM images in addition to simulated ones, it would significantly increase its impact and realism.

Minor issues:
1. Citation formatting errors, e.g., “in predicting atomic energies and forcesYang et al. (2024)” → “in predicting atomic energies and forces (Yang et al., 2024)”.
2. In Fig. 1, the text “Output: Optimized AttomicStructure” appears to contain a typo.
3. In Sec. 3.2, “To simulate realistic large-field STEM imaging conditions, We” — the “We” should be lowercase.

### Questions
- Beyond this specific system, what do the authors see as the key missing capabilities for a mature scientific agent? Is the bottleneck in tool diversity, model multimodality, or domain adaptation from natural to experimental images? (This is not a criticism, just an open question.)
- How dependent is the end-to-end reconstruction accuracy on the template retrieval step?
- In Sec. 4.3, are the branch decisions (‘repeat denoising / proceed to STEM2CIF / fallback to template matching’) implemented via hard programmatic thresholds on tool outputs, or are they purely prompt-driven?
- The workflow in Sec. 4.3 seems to follow a fairly linear sequence of predefined steps, guided by quality metrics and failure logs. It would be helpful if the authors could clarify to what extent the agent’s decisions rely on such programmed criteria versus genuine reasoning by the LLM, and what unique advantages the agent provides compared with a scripted automation pipeline.

### Soundness
3

### Presentation
2

### Contribution
3
