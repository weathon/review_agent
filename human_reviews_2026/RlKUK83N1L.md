# Better STEP, a format and dataset for boundary representation

- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Boundary representation (B-rep) serves as the primary format for 3D geometry in computer-aided design (CAD), integrating parametric geometry with explicit topology to model complex components and assemblies. Despite its omnipresence, research in machine learning rarely leverages B-reps directly; instead, STEP files are typically parsed with (proprietary) kernels and reduced to meshes, point clouds, or basic face-edge-vertex graphs. These simplifications discard important details (e.g., parametric patches or topology) present in B-reps and introduce additional challenges related to licensing, compatibility, and scalability.

We introduce Better STEP, an open source format that preserves the full fidelity of B-reps while enabling direct, efficient access in standard ML frameworks, removing dependence on proprietary software. In addition, we introduce an open-source Python library for querying and processing B-rep data. The Python package provides standard functionalities for querying geometry (e.g., surface sampling, normal estimation, curvature computation), as well as topological structure,
thereby facilitating integration into existing pipelines.

To demonstrate the effectiveness of our format, we converted the Fusion 360 and ABC datasets, comprising over one million CAD models. We further showcase the universality of our Python package by generating test data for four representative downstream tasks; these experiments did not require fine-tuning or modifying the original models, underscoring the ease with which our data can be integrated into existing machine learning workflows.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces Better STEP, an ML-friendly HDF5 schema that preserves exact B-rep geometry/topology with optional per-face meshes. It releases STEPTOHDF5 (conversion) and ABS (sampling/labeling), converts >1M CAD models, and demonstrates plug-and-play data generation for normals, denoising, reconstruction, and primitive/degree segmentation using existing models. The core contribution is practical infrastructure that standardizes CAD access and reduces reliance on proprietary kernels, enabling scalable, reproducible ML on B-reps.

### Strengths
1. This work provides highly practical infrastructure by offering an ML-ready representation that avoids lossy mesh/point conversions and preserves parametric detail and topology.
2. It integrates easily into pipelines through simple Python APIs for sampling, labeling, and per-face meshes, which reduces data-preparation friction.

### Weaknesses
1. Although the paper presents a new format and dataset to facilitate machine learning in the B-rep domain, it includes no experiments on “learning this new representation.” It is unclear to me whether this paper falls within ICLR’s scope.
2. Lines 72–77 mention that prior datasets require hand-picking and are small. But I cannot understand why this new dataset and format can bridge the gap, why they do not need hand-picking, and how they ensure high-quality data.
3. Lines 62–68 show that STEP requires CAD kernels for processing, and version incompatibility is an issue. Why not just convert to the newest version so it is general and does not need the Better STEP format?
4. The paper claims the proposed format and dataset “bridge the gap,” enabling LLMs to directly generate B-rep models, but it provides no supporting experiments.

### Questions
See weakness

### Soundness
2

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
4

### Summary
This paper proposes "Better STEP"—an open format equivalent to B-rep and its accompanying Python libraries. This aims to bypass the dependency of proprietary CAD kernels on STEP, preserving fully parameterized geometry and topology for direct access and sampling by ML. The authors convert large datasets such as Fusion 360 and ABC to this format and demonstrate data generation and evaluation pipelines for downstream tasks such as normal estimation, denoising, reconstruction, and segmentation. They reproduce close to published results without fine-tuning, demonstrating the versatility and practicality of the format and toolchain.

### Strengths
Bypassing proprietary kernels and version incompatibilities, it directly provides B-rep equivalent representations that can be consumed by ML frameworks; 

Provides a clear hierarchical structure (geometry/topology/mesh), standardized APIs (sampling, normals, curvature, topology traversal, etc.), and reports statistics such as conversion and failure rates; 

The same interface can generate data for multiple types of downstream tasks, with reasonable example coverage, showing plug-and-play support for existing methods.

### Weaknesses
The main problem is the insufficient demonstration of reproducibility and usability: there is currently no external demo, sample data or minimum runnable script, and reproduction must wait for formal acceptance and release, which has a high threshold; there is a lack of online browsing/interactive examples to demonstrate "ease of use" (such as visualization of typical B-rep, one-step sampling/export of point cloud); insufficient display of generated scenes - although the paper discusses the potential of LLM/CAD generation, it lacks specific case studies or quality inspection indicators from point cloud to B-rep, or from text/code to B-rep via GPT; the legal and licensing aspects are not detailed enough (the redistribution terms after Fusion 360/ABC conversion, the scope of subset disclosure, commercial use restrictions, etc. need to be clarified).

### Questions
Is better to provide a "minimal working subset" (e.g., HDF5 for 50–100 models, corresponding visualization and sampling scripts, and pre-generated downstream task examples) during the review period to verify usability? Or provide an online demo.

Regarding the promise of "generating B-reps from point clouds/from LLMs," can you provide at least a few end-to-end or overfitting examples to demonstrate how the data/interface actually facilitates the generation tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose BetterSTEP, a dictionary-like, half-edge-based, open-source format for representing B-Reps (Boundary Representation) of 3D CAD models, which enables easier integration into standard machine learning pipelines. They provide an OpenCascade-based conversion library that can convert STEP files into the BetterSTEP format, as well as another library called 'abs' for processing and querying B-rep data in the BetterSTEP format. The 'abs' library provides utilities for querying and processing both the geometry and the topological structure.  They provide a dataset formed by combining the ABC dataset and the Fusion 360 dataset and converting to the BetterSTEP format using the proposed conversion library. The authors sample point clouds from the dataset and utilize these point clouds as inputs to existing deep learning models for four different tasks: normal estimation, denoising, reconstruction, and segmentation. On these tasks, they provide quantitative and qualitative results, finding equivalent or slightly reduced performance compared to the original reported results (without fine-tuning or training the models). Overall, the paper proposes a new representation for B-Reps and CAD models that is open-source and equivalent to the original B-Reps while allowing easier integration with SoTA deep learning pipelines.

### Strengths
- A standard format for representing B-Reps from different sources in a consistent manner that can be easily utilized in Python, and a dataset in this format, is beneficial for the research and development of new approaches, simultaneously allowing for better and more consistent benchmarks and evaluation of these approaches.
- The provided format is independent of the original (commonly proprietary) CAD file format, which allows combining different datasets. In addition, the proposed 'abs' library allows querying and processing the inputs using the proposed format, reducing the requirement for different data processing pipelines.

### Weaknesses
- Code listings are not clear. Listings 4 and 5 are used to replace the compute_labels function from Listing 2; however, inconsistent return formats (1/0 vs 1/None) are used between different examples. Additionally, Listing 3 does not provide any meaningful/helpful information. Pseudocode detailing how read_meshes/get_mesh worked would be more useful than the current provided code. In general, I think the provided code could be clearer, and more details could be provided in addition to the very high-level usage examples. Moreover, the examples could be made more compact; for instance, by removing the multiple empty lines from Listing 2, and utilizing abs.function_name instead of having similar imports in different Listings.
- The contributions are not clear. The main contribution regarding the dataset is conversion, which is also repeated as part of the libraries as a conversion library. The provided dataset is a combined version of the ABC and Fusion 360 datasets, converted to the proposed format by parsing the STEP files with OpenCascade, extracting the geometric and topological information, and saving them into the proposed dictionary-like format. Based on this, it is unclear what additional improvements are provided beyond the geometric and topological information provided by OpenCascade. Moreover, failure percentages are reported for the meshing algorithm of OpenCascade, along with some failure case examples; however, the reasons for these failures are not discussed beyond the reported percentage failure rates. In addition, it is not clear whether StepToHDF5 should be considered a library here instead of a function handling inputs as part of the provided 'abs' library, which handles reading and navigating the HDF5 output files. Overall, the paper would benefit from improved structuring and clearer explanations regarding its contributions.
- Even though the format is independent of the conversion process, only OpenCascade is utilized in the paper to convert STEP files to the proposed format. However, OpenCascade also has its own BREP format. Currently, the paper does not discuss the advantages/strengths of the proposed format over this format, which would strengthen the claims.
- Limited comparisons and modality in experiments. The paper claims that the proposed format for B-reps would make them easier to use in standard machine learning frameworks. However, only point-cloud input models are considered. Moreover, only performances on existing models are reported, using point cloud inputs sampled from the proposed format and library, with no comparison to inputs sampled from B-rep or meshes directly. In these cases, the original performances are not reported, making it hard to evaluate the provided experiments. Additionally, there are no examples of training or fine-tuning. In my opinion, these provide very limited support to the claims. The paper would be significantly improved with the incorporation of geometric/topological deep learning models and/or mesh-based models.
- The paper claims that the proposed dataset and format will "bridge the gap, enabling LLM to generate complex B-rep models fully"; however, no LLM-based experiments are provided to support this claim.

### Questions
My primary concerns are the unclear claims and contributions, as well as the limited experimental results. Could the paper clarify the contributions regarding the dataset, libraries, and the format? Additionally, I am not certain how the numerical results in the provided experiments were evaluated or what the expected numbers were, as there are no proper comparisons to the base work/other works; could you provide a detailed explanation regarding these?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Paper prosed better step, an open source cad brep storage format to support further research in cad ML community. CAD models from abc and fusion360 are converted to hdf5 format. Authors provide python conversion script and other functions (e.g sampling normals).

### Strengths
HDF5 is a well supported and common format. Authors demonstrate some builtin functions using their python library. Converting different dataset into this version might be beneficial to open source? Although I have some difficulties understsanding what open source mean in this context. My understanding is that this python script is still built upon opencascade.

### Weaknesses
Paper lacks contribution in dataset and benchmark. No new dataset is introduced. Authors merely converted abc and fusion360 datasets into their "better step" format. Also there is no new data structure or more ML-friendly data representation for training. BRep is still represented by parametric faces, shells, edges, and vertices but with their parameters stored in hdf5 format. The topology is also still a linked list (top-down now). To me this doesn't really make the data any more "ML-friendly" than the standard STEP format.

### Questions
How is the format "better" besides stored as hdf5 and has some python library functions. The rebuild seems to be slightly better but not signifiantly different from opencasde. 

STEP is very efficient and the most common data format for CAD model sharing. How does hdf5 compared to STEP in terms of storage size or reading / loading speed. 

Would abc data converted to "better step" format help improve CAD generation?

### Soundness
2

### Presentation
2

### Contribution
1
