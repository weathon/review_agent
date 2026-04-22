# SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Deep learning holds immense promise for spectroscopy, yet research and evaluation in this emerging field often lack standardized formulations. To address this issue, we introduce SpectrumLab, a pioneering unified platform designed to systematize and accelerate deep learning research in spectroscopy. SpectrumLab integrates three core components: a comprehensive Python library featuring essential data processing and evaluation tools, along with leaderboards; an innovative SpectrumAnnotator module that generates high-quality benchmarks from limited seed data; and SpectrumBench, a multi-layered benchmark suite covering 14 spectroscopic tasks and over 10 spectrum types, featuring spectra curated from over 1.2 million distinct chemical substances. Thorough empirical studies on SpectrumBench with 23 cutting-edge multimodal LLMs reveal critical limitations of current approaches. We hope SpectrumLab will serve as a crucial foundation for future advancements in deep learning-driven spectroscopy. The anonymous code and experimental records are available at https://ai4s-chem.github.io/SpectrumWorld

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SpectrumLab, an LLM-centric platform for creating benchmarks across spectroscopic and spectrometric data modalities and for evaluating LLMs on these benchmarks. The authors test a wide range of state-of-the-art LLMs and release a well-documented codebase.

### Strengths
- Identifying molecules from spectroscopic measurements is crucial across many scientific domains, yet existing benchmarks are limited. This work proposes an extensible approach to benchmarking in these fields.  
- The paper unifies datasets from multiple spectrometric modalities into a single resource, including some in-house, previously unpublished data.  
- A new taxonomy for spectrometric annotation tasks is proposed.

### Weaknesses
- Although the paper proposes a benchmark, it contains little exploratory data analysis (EDA) of the seed datasets. For instance, Appendix G.1 does not list all data sources—a fundamental requirement for any new benchmark. Without EDA, it is difficult to assess the benchmark’s quality and significance. By EDA I mean, for example, what molecules are contained in the dataset, how noisy the spectra are, what instruments were used to generate the spectra, etc.
- The emphasis on multimodality is insufficiently justified:  
  - The text states, “As illustrated in Figure 1, this comprehensive data foundation accurately reflects the diverse and complex multi-modal spectroscopic scenarios encountered in real-world applications.” Please provide a concrete example where multiple spectroscopy types must be annotated simultaneously. In natural products research, for example, high-throughput LC-MS/MS is often followed by NMR of selected isolated molecules; these steps are typically separate, and the overall pipeline would likely benefit more from stronger single-modal models for LC-MS/MS and for NMR than from a multimodal approach.  
  - Strong multimodal models usually require overlapping training examples across modalities (e.g., different spectra for the same molecules). Can the authors demonstrate that the aggregated dataset offers meaningful intersections between modalities? Without such overlap, the aggregation is hard to justify.  
- The platform is heavily LLM-driven and as a result many LLM-generated tasks do not reflect practical needs. Several tasks look like like textbook exercises where machine learning is unnecessary, while others are so difficult that general-purpose LLMs are unlikely to help. For example, “Spectrum Type Classification” or “Basic Feature Extraction” seem of limited real-world relevance; lines 986–994 describe a task that appears solvable without ML; and the task on lines 976–986 is unclear even for a human. Conversely, “Molecular Structure Elucidation” is extremely challenging even for domain- and modality-specific SOTA models.  
- The manuscript contains numerous typos (e.g., lines 367–368 or end of line 298).

### Questions
- Can the authors better demonstrate the practical utility of the proposed resource? How would spectroscopic machine learning community and practitioners benefit from the proposed platform?

### Soundness
2

### Presentation
2

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
The authors introduce a "platform" for AI research in spectroscopy. The platform has multiple compoments, including Python libraries and a benchmark. The paper benchmarks many recent multimodal LLMs on the tasks.

### Strengths
The benchmark library and other Python libraries introduced here seem useful for a specific community.

### Weaknesses
The manuscript is extremely dense, very hard to read, also due to the introduction of more than 20 abbreviations, and the use of a lot of domain jargon. The relationship between different components (SpectrumBench, SpectumLab) is hard to understand. 

The modalities that are actually being used are often unclear. Figure 1 mainly talks about images and text, while other parts discuss model inputs and outputs without specifying the modality or data representation (e.g. molecules, spectra, ...). 

The SpectrumLab introduced here seems to be only addressing multimodal language models, and not other, more task-specific models. It seems like a lot of the work of the past years regarding (single- or multi-modal) representation learning of spectra and molecular structures seems to be incompatible with this work, and certainly not benchmarked here.

Overall, the benefit for the general ML community seems limited. The paper is mainly speaking to a very specific, application-oriented community focusing on spectral data. If the authors' target audience are computer scientists, then parts of the manuscript should be rephrased and improved, to make it more generally understandbale, introduce or remove community jargon, and make sure a more mathematically correct and consistent nomenclature regarding data types is used, and potentially also a more abstract task specification should be introduced.

### Questions
Figure 1: Why is the word "image" used so often? The modality is not an image if you have smiles codes or spectra. Smiles codes are strings that can also be represented as graphs. Spectra are lists of peaks or lists of intensities, so essentially vectors. Neither of them are images, but of course they can be visualized in images. Or do you call those image modalities, because you are actually using them as images for the multi-modal foundation models in your benchmark?

Figure 2 introduces many architectures, such as GNN, CNN, MLPs. Why are those discarded in your benchmark? Which of the tasks in your benchmark could also be solved without the text modality, or with a hybrid model that does not depend on an image representation of molecules and spectra?

How is the "Text-to-Any (De novo Generation)" task related to spectra? How is it different from what the entire community of generative models for molecules and materials is doing?

What does GLM4.5V in Figure 5 mean?
How can there be a unique ground truth output in the bottom right box in Figure 5? This is a generation task that can only sample but not predict "correct" answers.

Minor: Most of the fonts in Figure 1 are too small. Please improve Figure 1 for readability.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents **SpectrumWorld**, an AI infrastructure designed for spectroscopy, consisting of two main components: the **SpectrumLab** toolkit/platform and the large-scale benchmark **SpectrumBench**. SpectrumBench organizes evaluations along a four-level task taxonomy—**Signal → Perception → Semantic → Generation**—and provides a unified definition of tasks and evaluation protocols across multiple modalities (spectra, molecular graphs/SMILES, and natural language). Its goal is to systematically assess the capabilities of multimodal large language models (MLLMs) in spectroscopy.

The authors further design an automated benchmark generation and annotation module, **SpectrumAnnotator**, complemented by **SpectrumVerifier** and human-in-the-loop validation to establish a closed-loop quality control pipeline. The entire framework covers data collection, cleaning and standardization, automated and manual annotation, evaluation, and leaderboard publication. Experiments benchmark over 20 mainstream MLLMs under a unified setting, reporting both overall strengths and weaknesses, along with a detailed analysis of resource usage and computational cost.

### Strengths
- Clear infrastructural positioning and broad coverage:

The system systematically abstracts spectroscopy tasks into a four-level hierarchy and provides a modular evaluation ecosystem (tasks, evaluators, leaderboards), offering an extensible common platform for the research community.

- Reproducible data and annotation workflow:

The proposed SpectrumAnnotator automatically constructs multimodal QA/generation samples, and together with SpectrumVerifier and expert review forms a closed-loop quality control process that is transparent and reproducible.

- Extensive model evaluation with unified protocol:

Low- and mid-level tasks are unified as four-choice questions, while generation tasks are scored by GPT-4o, producing a single accuracy metric for easy comparison across 23 open- and closed-source MLLMs.

- Well-motivated problem statement:

The paper clearly identifies key pain points in spectroscopy AI—such as the domain gap between experimental and theoretical spectra, modality heterogeneity, and the lack of standardized benchmarks—and designs SpectrumWorld to address them.

- Openness and community potential:

Anonymous release of code, datasets, and data schemas encourages long-term community extension and reuse.

### Weaknesses
- Lack of quantitative evidence for annotation quality:

Although the paper describes the quality control process, it lacks systematic statistics on annotation quality (e.g., inter-annotator agreement, error rate, or comparison between automatic and manual annotations). Providing such quantitative analysis or visualization would strengthen the reliability of the benchmark.

- Single-dimensional evaluation metrics:

For generative tasks, using GPT-4o as the sole evaluator may introduce evaluation bias. In addition, the benchmark mainly relies on accuracy as the metric, without sufficient task-specific measures.

### Questions
- Evaluation bias and robustness:

For generative tasks evaluated using GPT-4o, could there be systematic bias favoring certain model families? Have you considered using multiple evaluators or calibration methods (e.g., open-source LLM judges or rule-based metrics) to verify the consistency of evaluation results?

- Task-specific metrics:

Beyond overall accuracy, do you plan to introduce domain-specific metrics for tasks such as de novo peptide sequencing (peptide precision) ?

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
3

### Summary
This paper introduces SpectrumLab, a modular toolkit and leaderboard for AI in spectroscopy, and SpectrumBench, a hierarchical benchmark spanning four levels (signal, perception, semantic, generation), 14 tasks, and 10+ spectrum types. The dataset is built from seed data encompassing >1.2M distinct chemical compounds, with a mixed auto+human curation pipeline. The authors evaluate 23 MLLMs and report large performance gaps between low-level recognition and higher-level reasoning/generation. Code and evaluation pipelines are packaged for reproducibility.

### Strengths
* The benchmark is hierarchically structured to mirror real spectroscopy workflows (signal → perception → semantic → generation) across roughly 14 task types and more than 10 modalities, which makes failure modes and capability gaps observable.
* Coverage of 20+ multimodal LLMs under a single evaluation protocol yields stable empirical patterns, strong performance on low-level recognition but weak generation/reasoning.
* The curation pipeline combines automated item generation with human-in-the-loop verification for better quality assurance.

### Weaknesses
* The scope is limited to MLLMs, with no baselines from domain-specific models (e.g., neural network trained on molecular strings and spectra data). Including such baselines and perhaps a hybrid setting where an LLM orchestrates specialized models for reasoning would yield a more informative comparison.
* Due to the focus on MLLMs, the task design and metrics are narrow. Reliance on multiple choice and plain accuracy underutilizes domain-specific measures (e.g., spectral similarity, shift MAE).
* Open-ended generation is graded by an LLM judge (GPT-4o), enabling scale but risking bias and score inconsistency, and may lead to metric validity concerns. How can we be sure of the quality of LLM judge-based evaluations?

Minor:
* Presentation can be improved. For example, the texts in Figure 1 and Figure 3 are very hard to read when printed on papers.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
