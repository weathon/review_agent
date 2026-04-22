# A Novel Benchmark Framework for Neural Embeddings in Earth Observation

- Avg Score: 5.33
- Decision: Reject
- Scores: 8, 6, 2

## Abstract
We introduce a novel benchmark framework for evaluating (lossy) neural compression and representation learning in the context of Earth Observation (EO). Our approach builds on fixed-size embeddings that act as compact, task-agnostic representations applicable to a broad range of downstream tasks. Our benchmark comprises three core components: (i) an evaluation pipeline built around reusable embeddings, (ii) a new challenge mode with a hidden-task leaderboard designed to mitigate pretraining bias, and (iii) a scoring system that balances accuracy and stability. To support reproducibility, we release a curated multispectral, multitemporal EO dataset. We present initial results from a public challenge at a workshop and conduct ablations with state-of-the-art foundation models. Our benchmark provides a first step towards community-driven, standardized evaluation of neural embeddings for EO and beyond.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
SUMMARY: This paper introduces a new benchmark Benchmarks specifically aimed at EO data embeddings. The benchmark is simple by design and includes both regression and classification tasks that have to be solved given an embedding input. The authors outline how they test their benchmark in a challenge, report results and learning and outline efforts for expanding the benchmarking and community building efforts; details here are omitted, presumably to maintain the anonymity of the submission. The authors nonetheless test a few baseline methods on their benchmarks and discuss outcomes.

### Strengths
This is a timely and relevant paper. I especially appreciate the following aspects:

- Benchmarks specifically aimed at EO data embeddings are desperately needed. That basically every new geo embedding model / paper evaluates on different tasks (Alpha Earth most recently) emphasizes this. Geo embedding benchmarks are also not directly comparable to vision benchmarks in the remote sensing domain, which often focus on fine tuning and specialized architectures, rather than linear probing.
 
- The paper is very simple - and that is a strength. It is easy to follow and understand. Same goes for the benchmark itself. Very simple, no big compute needed; extremely accessible.

- The projects focus on community building and the clear outlines of expanding this work are great to see!

### Weaknesses
- Given that it is such a simple benchmark, it would have been cool to see comparisons to more GeoFM based embeddings, e.g. Clay, Prithvi or AEF. 

- Lack of implicit neural representations as a way to obtain geo embeddings; they are initialized by location, not image, but could still be tested using this framework (I appreciate that this might be beyond the scope of the paper) and should definitely be present in the related work section.

- Spatial coverage is, as unfortunately with pretty much all EO benchmarks, favoring Western countries.

- I understand that given that this is an anonymous submission not more details can be revealed about the challenge, but what are the authors plans for changing this in a potential camera-ready version? How would the challenge outcomes be presented then?

- Why is it necessary to constrain embeddings to exactly 1024 dimensions? Wouldn't relaxing this constraint allow for a more direct comparison of different methods?

### Questions
Overall this is a super relevant paper. The review is quite short, but that's because I am mostly happy with this paper. I'd ask the authors to consider my questions and concerns in the "weaknesses" section, but overall, this is already a clear accept to me. We need more benchmarks on this specific topic and conducted in such a community and access-centric way at ICLR.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a comprehensive benchmarking framework for earth observation embeddings. The framework specifically aims at evaluating embeddings, foregoing end-to-end finetuning and requiring no access to models used to produce embeddings. The benchmarking procedure also explicitly introduces embedding-dimensionality as a dimension for comparison. The datasets used consist of 5 kinds of tasks with moderate sample sizes (1,000<N<10,000), split between binary classification and regression. The benchmarking framework has been tested in a real-world setting and integrates with at least one established submission platform. 
Overall this work presents a solid foundation for EO embedding evaluation but should be expanded to include more evaluation tasks and dataset, and, less urgently, more diverse evaluation settings.

### Strengths
- The fundamentals of this submission are sound. The included tasks and scoring procedure are well-motivated, as is the overall work, with current model comparison often not considering embedding dimensionality and how that affects task performance and viability of downstream implementation of the models in any real data pipelines. 
- The fact that this framework has already been successfully explored in a practical setting, including integration with an established submission platform, is encouraging and definitely a strength of this work. 
- The results on MLP evaluation versus linear evaluation at least somewhat justify the minimal evaluation models. 
- The experiments assessing the framework are overall well done and support the framework in its current state.

### Weaknesses
- The authors claim in the Introduction the benchmark tasks include “novel EO downstream tasks” but it is unclear what this refers to? Are the datasets novel? Certainly the tasks are based on foundational EO problems. 
- The authors repeatedly claim that an issue with embedding models that goes underexplored is the fact that the embeddings they produce exceed the dimensionality of the actual input data, causing issues of data transfer and efficient pipelining. While this is technically true, in practice for ViTs, often only the CLS token is used for downstream tasks while convolutional embeddings are usually pooled into just one spatial dimension. There is still merit to this work and encouraging low embedding dimensionality, but it is unclear how much of a real limitation this is in practice. 
- The tasks evaluated seem lacking. It should be straightforward to include at least the subset of the copernicus bench (arXiv:2503.11849) datasets that have the required CC-BY 4.0 license. The authors note this requirement in the Future Work section but without any such effort the benchmark is severely lacking in maturity and ultimately disconnected from central pieces of the EO literature. 
- The description of the procedure for equalizing embedding size between the models seems somewhat arbitrary and is hard to understand. Maybe this can be justified further or at least described more clearly. 
- Other work shows that evaluating with shallow MLPs can yield different performance estimates from simple linear models. The justification for not including this based on the fact that the MLPs might “compensate for weak embeddings” is inadequate as they have no additional information beyond what is extracted by the embedding model. Especially as more datasets and tasks get included, broadening the evaluation to include at least some non-linear models is critical.

### Questions
The scoring is fine but seems sensitive in early stages of the benchmark, when std estimates can fluctuate widely from the impact of a single submission. Have the authors explored any alternative scores to supplement the early phases of releasing this benchmark or simply adding more common embedding methods (even just ImageNet pretrained models or general vision foundation models) to solidify the std estimates. Also, will old scores be updated as new submissions are received and scored?

### Soundness
2

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
5

### Summary
The paper presents a benchmark for remote sensing embeddings — i.e., the output features of remote sensing foundation models. It appears to be largely a write-up of challenge results from a competition run at the CVPR EarthVision workshop.

The benchmark covers five tasks and six comparison models, which is narrower in scope than recent related efforts such as Pangea or GeoBench. Section 3 provides a clear description of how to set up a dataset challenge and evaluation framework. However, the task selection reveals a strong geographic bias towards the US and Europe (Fig. 2), and the taxonomy of tasks is not clearly motivated in terms of coverage of the remote sensing problem space. For example, there is no marine/water domain task (e.g., marine litter like MADOS), and tasks such as “Landcover Agriculture” and “Crops” seem thematically overlapping. Task difficulty is also not discussed: e.g., crop type classification can be a simple binary setting (“soy vs corn”) or a highly fine-grained setting with hundreds of visually similar classes (e.g. EuroCrops).

The results are also not particularly conclusive. The analysis of embedding size vs. performance is potentially interesting, but the underlying mechanisms are not investigated. For example: how correlated are the embedding dimensions (e.g., how many principal components capture most of the variance)? How does the curse of dimensionality play into these observations? What semantic / spatial / temporal patterns do these embeddings capture?

This paper clearly captures a large amount of work, but, I believe, for ICLR, a stronger contribution would require going beyond ranking models on a challenge, and instead probing and explaining the structure of these embeddings. As written, this reads more like a data challenge report suited to a domain workshop than a venue focused on advancing ML understanding.

### Strengths
* The underlying challenge presents extensive work summarizing the results of a benchmarking challenge.
* The description of the benchmarking framework provides a good example how to implement a challenge and outline its main take-aways.

### Weaknesses
* Narrow scope compared to existing benchmarks (only 5 tasks / 6 models).
* Strong geographic bias in dataset tasks (mostly US / Europe).
* Task taxonomy and coverage not well justified; missing important domains (e.g. marine) and overlapping task definitions.

### Questions
* How relevant is the "compression" aspect to the models and results of this benchmark? How would an embedding from a classic compression algorithm Discrete Cosine Transform (DCT) to create non-deep features for comparison?
* One of the main conclusions was that larger embedding sizes degrade performance, but what are here the underlying factors? The curse of dimensionality, the underlying low dimensional manifold? What are the scaling laws here? It still seems that Terramind is improving with larger embedding sizes?

### Soundness
2

### Presentation
1

### Contribution
1
