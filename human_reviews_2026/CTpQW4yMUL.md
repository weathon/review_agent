# CSRv2: Unlocking Ultra-Sparse Embeddings

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 8, 2, 4

## Abstract
In the era of large foundation models, the quality of embeddings has become a central determinant of downstream task performance and overall system capability. 
Yet widely used dense embeddings are often extremely high-dimensional (e.g., 4096), incurring substantial costs in storage, memory, and inference latency. 
To address these, Contrastive Sparse Representation (CSR) is recently proposed as a promising direction, mapping dense embeddings into high-dimensional but $k$-sparse vectors, in contrast to compact dense embeddings such as Matryoshka Representation Learning (MRL). 
Despite its promise, CSR suffers severe degradation in the ultra-sparse regime (e.g., $k \leq 4$), where over 80\% of neurons remain inactive, leaving much of its efficiency potential unrealized.
In this paper, we introduce CSRv2, a principled training approach designed to make ultra-sparse embeddings viable. 
CSRv2 stabilizes sparsity learning through progressive $k$-annealing, enhances representational quality via supervised contrastive objectives, and ensures end-to-end adaptability with full backbone finetuning. 
CSRv2 reduces dead neurons from 80\% to 20\% and delivers a 14\% accuracy gain at $k=2$, bringing ultra-sparse embeddings on par with CSR at $k=8$ and MRL at 32 dimensions, all with only two active features. 
While maintaining comparable performance, CSRv2 delivers a 7$\times$ speedup over MRL, and yields up to 300$\times$ improvements in compute and memory efficiency relative to dense embeddings in e5-mistral-7b-instruct-based text representation.
Extensive experiments across text (MTEB, multiple state-of-the-art LLM embeddings (Qwen and e5-Mistral-7B), SPLADEv3, GraphRAG) and vision (ImageNet-1k) demonstrate that CSRv2 makes ultra-sparse embeddings practical without compromising performance, where CSRv2 achieves 7\%/4\% improvement over CSR when $k=4$ and further increases this gap to 14\%/6\% when $k=2$ in text/vision representation.
By making extreme sparsity viable, CSRv2 broadens the design space for large-scale, real-time, and edge-deployable AI systems where both embedding quality and efficiency are critical. 
Code is available at https://github.com/Y-Research-SBU/CSRv2.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
(I tend to write shorter reviews and the length of the review does not reflect the quality of paper or the time spend on reviewing it -- especially in this paper, I understand the background and technical details well enought to be able to write a to the point review). 

This paper proposes an extension (fixing) of CSR (contrastive sparse representation) idea that was introduced to enable adaptive on-the-fly sparse vectors or desired complexity to enable faster retreival. This is an improvement on top of dense compact representations which order information implcitily like in MRL (matryoshka representation learning) where the shorter embeddings are obtained by truncation of dimensions. 

This work investigates the case of ultra-sparsity (which is with active dims less than k=16) and find several underlying causes and fix the CSR training to be more supervised and matryoshka style to enable better ultra sparse retreival. The findings are supported by strong experimental results and discussion.

### Strengths
Jus a strong and useful paper overall. The study of each of the issues and natural fixes are commendable. 

I really like the k=1 experiments.

### Weaknesses
The Table 1 adds little value especially with calibrated use of stars -- it does not give reader any quantifiable takeaway and would recommend removing the use of stars. 

Re CSR and CSRv2: For a new reader it is not clear as to how one can vary "k" in CSR style models (unlike mrl where the truncation is defined). After chosing different "k" do we re-normalize. These are good things to go into the paper. 

While completly complementary, it would be good to show results of a fixed cost quanitzed embedding of same size. ie., if k=4 with bf16 you should have a result for Quanitzed embedding os size 64 bits (be it binary quant or PQ). It will give more completeness to the paper.

### Questions
Please make the 3 changes suggested above. the paper is strong and will have a lot of impact. all the best.

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
This work proposes CSRv2, a training paradigm for learning contrastive sparse representations via progressive k-annealing and address the dead neuron problem of prior work. The authors compare to other popular adaptive representations like Matryoshka Embeddings and CSR and show promising empirical compute-accuracy tradeoff compared to these baselines on text (MTEB) and visual (ImageNet-1K) embedding tasks.

### Strengths
* This paper addresses an important problem of learning sparse embeddings that are compute optimal for their memory and inference time, showing strong empirical performance on real-world embedding benchmarks like MTEB and ImageNet-1K.
* The paper is written and presented very well, and is easy to follow. The figures and tables are clean, interpretable, and provide supporting evidence to the claims in the paper
* Design choices made by the authors are empirically validated to show the benefits of these choices (Tab 4). For example, adding supervised training on top of CSR (Fig 3), using k-annealing SAE and its affect on dead neurons (Fig 2). The authors also provide details on the experimental setup (App. C, D), limitations (App. F), and additional ablations and qualitative analysis on sparsity (App. E) which all add to the overall quality of the work.
* CSRv2 shows strong empirical accuracy and sparsity on both text and visual embedding tasks compared to strong baselines like MRL and CSR (Tab 2- 4, Fig 2, Fig 5)

### Weaknesses
I do not have any major concerns with this work. I will highlight several minor weaknesses that could be addressed to further improve its quality:

* I suggest rearranging the abstract content to be more specific about empirical performance on each dataset/task:
    * L22 - L26: *"delivers a 14% accuracy gain at k = 2 ... a 7× speedup over MRL... 300× improvements in compute and memory efficiency"* - gain /speedup with respect to what and on what data?
    * For L27 - 31, how much (metric value) did CSRv2 beat the strongest baseline on MTEB (Tab 2, 3, 4) and ImageNet-1K (Fig 2, 5) respectively?

* For MTEB experiments, please clarify these details explicitly in the main paper:
    * you select "six types of tasks" (Fig 1b, Tab 2- 4). Why were these specific task types selected (Sec 4.1)?
    * you mention in Tab 2 that you train all methods on same backbone and data (this is great!). Is this also the case for Fig 1b? 
    * Does the "backbone" line in Fig 1b and 5a represent zero-shot accuracy with base e5-mistral-7b and Resnet50 FF2048? 
    * In Tab 5, are all methods trained on MTEB from the base Qwen3-Embed-4B? If so, is the MRL number there zero-shot or is it finetuned on MTEB similar to what you do with e5 (L365)?

* For Figure 1b, consider adding MR line for the same active dims as a baseline for better contextualization of CSR annealing

Overall, I really like this work. If these minor weaknesses are addressed in the revision, I am willing to increase my score to a strong accept.

### Questions
I have a minor nitpick that can contextualize the future impact of this work, which I suggest adding in the next revision. You briefly touched on **vector quantization** in the concluding remarks (L484), which is extremely important for real-world applications (vector search). There are several works focusing on the compute-optimality of embeddings when popular quantization techniques [1 - 3] are involved, especially in context of the recent adaptive representations like MRL [4]. In the next revision, I suggest adding a small note (in the Appendix) commenting on the role of CSRv2 with quantization.

### References
[1] Jegou, Herve, Matthijs Douze, and Cordelia Schmid. "Product quantization for nearest neighbor search." IEEE transactions on pattern analysis and machine intelligence 33.1 (2010): 117-128

[2] Guo, Ruiqi, et al. "Accelerating large-scale inference with anisotropic vector quantization." International Conference on Machine Learning. PMLR, 2020.

[3] Jayaram Subramanya, Suhas, et al. "Diskann: Fast accurate billion-point nearest neighbor search on a single node." Advances in neural information processing Systems 32 (2019).

[3] Rege, Aniket, et al. "Adanns: A framework for adaptive semantic search." Advances in Neural Information Processing Systems 36 (2023): 76311-76335.

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
Contrastive Sparse Representation is a fairly recent paradigm that aims to improve the efficiency beyond Matryoshka Representation Learning. However, at ultrahigh sparse levels, the performance metrics drop quite a bit. In this work they show that it is due to dead neuron problem at that level, as well as usecase of SSL in CSR. CSRv2 applies a curriculum over the sparsity and replace SSL with SupCL, and show improvement over standard representation learning benchmarks along with timing analysis.

### Strengths
In general, improving any algorithm is great, and this work improves CSR, a fairly recent paradigm for ultrasparse representations, as well as providing improvements over the equivalent dimensional MRL representations.

### Weaknesses
I feel this work has several weaknesses, from novelty to an architectural point of view, leading me to reject this work. 

1. On curriculum: Do we always want a backbone that is only producing a sparse representation of specific sparsity? I believe that a more practical setup is where you've one single backbone, which is trained on multiple top-k values, from the largest desirable sparsity to the smallest sparsity. This current setup seems to be impractical, as I believe if there is a backbone that works on multiple sparsity all at once, it would not suffer from the dead neuron problems. 
2. For MRL, I believe one can use a different weightage for smaller representation sizes. Can authors point to where such a study has been done?
3. There are more recent SAEs, called Matryoshka Sparse Autoencoders, and I think there can be a similar CSR that uses MRL-SAE. 
4. Supervised Contrastive learning seems quite a bit important for good performance at ultrahigh sparsity. However, for representation learning, we want the representations to be as general as possible (if they're to be used as vector databases), which may or may not have a supervised contrastive learning amenable dataset. Therefore, this requirement inherently limits the use case as a general-purpose representation for edge devices, say.
5. While curriculum helps a little bit, it doesn't seem to solve the dead neuron problem, with a major fraction still dead, leading to a waste of resources. Therefore, are such ultrahigh sparsities really desirable, given that the performance dip is quite a bit, and may not be something a user is willing to sacrifice over? 
6. What are the ultrahigh sparse representations learning? Are they learning to separate hyperplanes for some superclasses? An analysis of this would be good!

### Questions
Refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes CSRv2 to create ultra sparse embeddings to address the challenges in Contrastive Sparse Representation of dead neurons, mismatch between the pretraining and downstream semantics and limited capacity when training a linear head. The authors improve the training recipe with progressive k-annealing, supervised sparse contrastive learning on the sparse codes, and full backbone fine-tuning instead of fine-tuning a linear head only. This creates ultra sparse embeddings for retrieval and downstream tasks as an alternative to matryoshka representation learning. Experimental results on MTEB with Mistral-7B and Qwen3-4B, and vision dataset ImageNet-1k with ResNet-50 show reduced dead neurons, accuracy gain at small k’s and matches the performance of CSR and MRL at relatively moderate k’s.

### Strengths
1. The analysis of failure modes of CSR in the ultra sparse regime exposing dead neurons with and without annealing is interesting. 

2. Evaluation setup is exhaustive and the accuracy improvements are compelling. The efficiency results on retrieval are also strong and indicate that the method should scale well to practical settings.

### Weaknesses
1. Novelty of the method is fairly limited. K-annealing and supervised contrastive objectives have been discussed (and experimented with a lot) before in prior work, as has been acknowledged by the authors. 


2. The results from annealing itself are relatively small compared to adding supervision and finetuning, which raises the concern to me whether the improvements are coming from the sparse learning principle, or mostly engineering training recipe. 


3. The efficiency cost focuses on retrieval but there are no end to end latency measurements, encoder latency, cost of index construction which are all relevant for any practical use.

### Questions
Q. How sensitive are the results to the k-schedule shape, the length and initialization? 

Also see weaknesses section above

### Soundness
2

### Presentation
3

### Contribution
2
