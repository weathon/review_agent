# KaLM-Embedding-V2: Superior Training Techniques and Data Inspire A Versatile Embedding Model

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Recent advancements in Large Language Models (LLMs)-based text embedding models primarily focus on data scaling or synthesis, yet limited exploration of training techniques and data quality, thereby constraining performance. In this work, we propose KaLM-Embedding-V2 from the Lychee-KaLM team, a series of versatile and compact embedding models, systematically incentivizing advanced embedding capability in LLMs by superior training techniques and high-quality data. For model architecture, we implement the models on a 0.5B compact size with simple mean-pooling to produce fixed-length embeddings and remove the causal attention mask to enable fully bidirectional representation learning. For training techniques, we propose a progressive multi-stage training pipeline: pre-training on weakly supervised large-scale datasets, fine-tuning with supervised high-quality datasets, and contrastive distillation with fine-grained soft signals, integrated with focal-style reweighting and online hard-negative mixing to emphasize difficult samples and enrich hard negatives, respectively. For training data, we curate over 20 categories for pre-training and 100 categories for fine-tuning and contrastive distillation, to improve both performance and generalization, leveraging task-specific instructions, hard-negative mining, and example-based multi-class labeling to ensure high quality. Combining these techniques, our KaLM-Embedding-V2 series achieves state-of-the-art performance on the Massive Text Embedding Benchmark, outperforming models of comparable size and rivaling models 3-26x larger, setting a new standard for versatile and compact embedding models under 1B parameters. The code, data, and models are available at https://kalm-embedding.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents KaLM-Embedding-V2, a compact (0.5B parameters) text embedding model trained with improved data curation and training techniques.

### Strengths
1. The model achieves state-of-the-art results among models under 1B parameters and performs competitively with much larger models.
2. The paper includes detailed ablation studies, comparisons, and analyses across multiple benchmarks, which make the results convincing.
3. Open-source release

### Weaknesses
The contribution is mainly the integration of multiple known techniques, rather than a fundamentally novel algorithm.

### Questions
1. The teacher model is Qwen3-Embedding-8B. Why did the authors choose to initialize KaLM-Embedding-V2 from Qwen2-0.5B instead of using an existing Qwen-Embedding model (e.g., Qwen-Embedding-0.6B)? Wouldn’t that potentially eliminate the need for the expensive pre-training stage? Have the authors compared starting from Qwen2-base versus Qwen-Embedding models to verify whether the pre-training stage is truly necessary or beneficial?
2. Whether the proposed distillation strategy generalizes across different teacher families or if it only works within the Qwen series.

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
This paper presents KaLM-Embedding-V2, an embedding model that leverages Qwen2-0.5B and different training tricks to improve the embedding representation and accuracy on MTEB tasks. The model surpasses most of the selected small models and even selected larger models on MTEB English and Chinese v1.

### Strengths
1 - New embedding models that surpass some SoTA LLM-based and encoder-based models with very few parameters.  

2 - Authors give extensive details about the training recipe and the datasets used for it.  

3 - The model is compared to multiple existing models and evaluated on both English and Chinese.  

4 - Good open-source contribution if the training code is released, as it goes beyond most recent models that are open-weights only.

### Weaknesses
1 - The originality of this work is not very clear, as it combines existing training recipes that were used for training embedding models (see NV-Embed models and Qwen3 embedding models).   

2 - Table 2 is clearly missing SoTA models on MTEB(eng, v1). Checking the MTEB leaderboard (on MTEB(eng, v2)), I see for example that Qwen3-4B and 8B achieve good performance on the benchmark but are not listed in the >1B models list.  They outperform KaLM-v2 but they are larger, it would be great to list them.  

3 - Table 5 is very dense and hard to read, probably a plot of the average score on MTEB with the ablations could be better.  

4 - The model is fine-tuned on many training sets of the datasets that are used for MTEB evaluation. MTEB uses the test sets, but there is a distribution similarity between the sets that may help the model get better on the benchmark. Evaluating on MTEB(eng, v2) could have fixed some of these issues, as for example it removes MS MARCO that is used for this model finetuning. Also, evaluating on AIRBench could give more insights on how the model performs on out-of-domain and out-of-distribution data (see NV-embed paper)  

5 - The model is not evaluated in a multilingual setting (2 high resource languages are not enough) but is compared to multilingual models. It would be great to add a multilingual evaluation.

### Questions
1 - How does the model perform on other languages than English and Chinese?

2 - Why not evaluate on MTEB(eng, v2) [1]?


[1] Enevoldsen, Kenneth, et al. "Mmteb: Massive multilingual text embedding benchmark." arXiv preprint arXiv:2502.13595 (2025).

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
4

### Summary
This paper proposes KaLM-Embedding-V2, a 0.5B LLM-based text embedding model. The approach modifies a decoder-style architecture by removing the causal attention mask and applying mean pooling, enabling more effective bidirectional representation learning. Training follows a three-stage progressive pipeline. As a result, the model achieves SOTA on English and Chinese MTEB.

### Strengths
- Methodical system design. Detailed explanation of architecture, data, training all contribute
- Ablations support most claims (focal loss, hard-negative mixing, distillation components)
- Description of training data and pipeline for it’s creation

### Weaknesses
- The distillation stage is not fully justified, as the paper does not explore whether applying distillation earlier or interleaving phases could be equally or more effective.
- Instruction dependence remains unclear, since the paper does not evaluate performance without instructions or under instruction mismatch.
- Causal-vs-bidirectional attention choice insufficiently validated, lacking direct comparison with strong causal-mask baselines from the same model family.
- No comparison of performance to teacher (Qwen3-Embedding-8B) and no comparison to smaller (to teacher) Qwen3-4b model.

### Questions
Suggestions:
- it is not clear what represents different colors on figure 1
- Some citations are duplicated (MTEB, Sentence BERT, NV-Embed)
- While MTEB (v1) is larger than  MTEB (v2), MTEB v2 implements multiple fixes on existing MTEB tasks reducing estimation noise. It also removed a few faulty tasks, either due to frequent overfitting/leakage or outright bugs, e.g., SummEval has a known bug that makes its estimates unreliable (wrong). This mteb software will raise a warning about the fact. I would at least remove SummEval if not entirely replace MTEB(v1) with MTEB(v2)

### Soundness
3

### Presentation
2

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
This paper presents techniques behind the KaLM text embedding model. It applies various training techniques such as focal-style re-weighing to upweigh loss on harder examples, online hard negative mixing, contrastive distillation from a bigger model, and multi-stage training.

### Strengths
* The final model performs well on benchmarks with fewer parameters.
* Focal-style reweighing, online hard-negative mixing and contrastive distillation are novel in the text embedding space.
* Very detailed and sounds experimental results leave few questions unanswered.

### Weaknesses
* The most impactful technique of focal-style reweighing is commonly used in machine learning.

### Questions
Where does the high performance with the small 0.5B model come from exactly? Is it the:
1) base model 2) training data 3) contrastive distillation 4) the two techniques of focal style re-weighing, hard-negative mixing.
I ask this because based on Table 5, removing both the techniques (4) would still create a strong text embedding model for 0.5B standards.

### Soundness
4

### Presentation
4

### Contribution
3
