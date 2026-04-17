# Smoothie: Smoothing Diffusion on Token Embeddings for Text Generation

- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Diffusion models have achieved state-of-the-art performance in generating images, audio, and video, but their adaptation to text remains challenging due to its discrete nature. Prior approaches either apply Gaussian diffusion in continuous latent spaces, which inherits semantic structure but struggles with token decoding, or operate in categorical simplex space, which respect discreteness but disregard semantic relation between tokens. In this paper, we propose Smoothing Diffusion on Token Embeddings (Smoothie), a novel diffusion method that combines the strengths of both approaches by progressively smoothing token embeddings based on semantic similarity. This technique enables gradual information removal while maintaining a natural decoding process. Experimental results on several sequence-to-sequence generation tasks demonstrate that Smoothie outperforms existing diffusion-based models in generation quality. Furthermore, ablation studies show that our proposed diffusion space yields better performance than both the standard embedding space and the categorical simplex.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces **SMOOTHIE**, a diffusion model that operates on token embeddings by constructing a *semantic distance tensor* for each token. Instead of performing diffusion in the discrete simplex (as in D3PM or SSD-LM) or the embedding space (as in Diffusion-LM), SMOOTHIE models the evolution of the *pairwise distances* between each token and all vocabulary embeddings. This design allows the diffusion process to preserve both discrete structure and semantic smoothness: at each timestep, Gaussian noise is added to perturb these distances, generating soft distributions that reflect evolving semantic relationships.

The authors prove that simplex diffusion (D3PM-style) is a special case of SMOOTHIE when using a trivial distance metric, thus theoretically generalizing prior discrete diffusion frameworks. Experiments on summarization (XSum), paraphrasing (QQP), detoxification (ParaDetox), story generation (ROCStories), and question answering (Quasar-T) show consistent improvements over both discrete and continuous diffusion baselines. While the paper includes a basic step-count and self-conditioning analysis, more systematic ablations (e.g., noise schedule, δ̃ magnitude) are needed to isolate where the gains truly come from.

### Strengths
1. **Unified formulation:** The paper provides a clear mathematical unification of discrete simplex diffusion and continuous embedding diffusion via a distance-based representation. Theorem 4.1 elegantly connects distance regression with embedding regression.
2. **Semantic-aware diffusion:** By perturbing the distance tensor with Gaussian noise, the model embeds semantic structure directly into the diffusion process, enabling smooth transitions between semantically related tokens.
3. **Comprehensive evaluation:** Experiments cover diverse tasks (XSum, QQP, ParaDetox, ROCStories, Quasar-T) and compare fairly with both discrete (TESS, SSD-LM, D3PM) and continuous (DiffuSeq, Diffusion-LM) baselines.
4. **Fair comparison setup:** Model sizes (~100M), datasets, and decoding strategies are kept consistent; pretrained advantages of baselines (e.g., TESS) are removed.
5. **Strong performance:** SMOOTHIE outperforms all diffusion baselines and achieves results comparable to autoregressive models like FLAN-T5.

### Weaknesses
1. **Computation cost not analyzed:** Although vectorization and mean-embedding compression are mentioned, there is no report on runtime, GPU memory, or FLOPs. Given that every token interacts with all vocabulary embeddings, the training cost may be substantial.
2. **Limited ablations on general enhancements:** While the authors tested self-conditioning and found minimal gains (and thus did not adopt it), other general mechanisms such as the tanh-style noise schedule or δ̃ magnitude lack comprehensive ablations. More systematic sensitivity studies would clarify whether improvements stem primarily from the distance-based diffusion rather than secondary hyperparameters.
3. **Embedding dependency untested:** The model fixes the BERT embedding matrix during training but does not evaluate alternative embeddings (e.g., random, GloVe, or fine-tuned). It remains unclear how sensitive performance is to the embedding quality or domain.
4. **No convergence or efficiency analysis:** Beyond theoretical equivalence, convergence behavior and stability relative to token-level diffusion are unreported; adding epoch-wise loss curves would clarify training efficiency.

### Questions
1. **Computation cost:** Can you report training time, GPU memory, or FLOPs compared to TESS or DiffuSeq? How is the full distance tensor computation optimized, and how does it scale with vocabulary size?
2. **Embedding dependence:** Have you evaluated SMOOTHIE using other embeddings (e.g., random or domain-specific)? How robust is the model to embedding quality and distribution shifts?
3. **Noise schedule and δ̃ ablations:** Why was the tanh-style schedule chosen? Have you tested alternative schedules or δ̃ values to confirm robustness across datasets?
4. **Self-conditioning and fairness:** Since self-conditioning yields limited improvement and was excluded, can you confirm that no other general enhancements influenced the main gains? Could similar gains be achieved by applying self-conditioning to baselines?
5. **Convergence and stability:** Please include training curves or epoch-wise loss comparisons with baseline diffusion models to demonstrate convergence efficiency and numerical stability.
6. **Scalability:** How does SMOOTHIE perform with larger vocabularies (e.g., >50K tokens)? Are there feasible strategies (e.g., top-k distance pruning or clustering) to reduce complexity while maintaining accuracy?

### Soundness
4

### Presentation
4

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
This paper introduces SMOOTHIE (Smoothing Diffusion on Token Embeddings), a novel diffusion model framework for text generation that aims to bridge the gap between continuous (Gaussian) and discrete (Simplex/Categorical) text diffusion methods. The core innovation lies in defining a new latent space where each token is represented by a vector of negative squared Euclidean distances between its embedding and the embeddings of all vocabulary tokens.
The authors demonstrate that SMOOTHIE consistently outperforms prior diffusion-based approaches across several sequence-to-sequence tasks, achieving generation quality comparable to strong autoregressive baselines.

### Strengths
The core contribution of defining the diffusion space using semantic distances (Euclidean proximity) in the embedding space is highly intuitive and well-justified. It elegantly addresses the major trade-off in existing work: retaining semantic structure (like Gaussian diffusion) while enabling natural decoding from discrete representations (like Simplex diffusion).

### Weaknesses
SMOOTHIE (like most text diffusion models) runs over fixed-length sequences. In practice, they set a dataset-specific max length and pad shorter sequences with a special padding token that the model learns to predict. The generation process is bounded by the preset max. It can emit different effective lengths up to a cap, but it doesn’t truly sample variable length the way an autoregressive model does.

### Questions
In Figure 1 (a), why is the color of the arbitrary i-th embedding always the same? What is the meaning of the structured shape (a flag-shaped pattern with an oval in the bottom) that the dots form?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
SMOOTHIE proposes diffusion over distance-based token-embedding logits, smoothing semantics while preserving discreteness, outperforming prior text diffusion on seq2seq tasks; analyses highlight noise, steps, and self-conditioning.

### Strengths
1.	Unifies prior lines: maps each token to a vector of negative squared distances to all vocab embeddings, then diffuses and feeds softmax(D_t) to the model; enables natural argmax decoding while preserving semantics and discreteness. Clear training/sampling pseudocode. 
2.	The distance-based latent generalizes simplex diffusion (simplex emerges under a trivial metric), giving a clean conceptual frame. 
3.	Practical guidance on schedules/self-conditioning; moderate steps (~100–200) are sufficient, with analysis of step count and reverse-noise. 
4.	Consistent gains over diffusion baselines on multiple seq2seq tasks.

### Weaknesses
1.	Fixed pre-trained embeddings (E) cap expressivity; authors acknowledge end-to-end training would likely help but leave it to future work. 
2.	Fixed sequence length forces substantial padding; variable length is emulated by truncating after EOS, which is inefficient; prior early-truncation is ad hoc. 
3.	Every step computes softmax over the full vocabulary V (and final argmax), which scales poorly for large V and long m; no top-k/approximation is provided. 
4.	Relies on the Euclidean semantic space hypothesis; authors note other domains may need different distances—raises concerns for polysemy/anisotropy. 
5.	Little on throughput/memory vs. competing diffusion methods under equal steps; limited evaluation breadth (small/medium datasets, few human evals).

### Questions
1. Please fine-tune embeddings or learn a task-adaptive metric (e.g., Mahalanobis) on at least one dataset; report lifts vs. fixed E. 
2. Efficiency. Provide tokens/sec, GPU memory, and wall-clock vs. SSD-LM/TESS/embedding-diffusion at matched steps; include step–quality curves. 
3. Try top-k candidate sets (ANN/FAISS) or hierarchical/adaptive softmax; quantify quality vs. speed trade-offs for long sequences and large vocabularies. 
4. Beyond EOS truncation, test a general dynamic-length denoising strategy (e.g., entropy/energy-based early stop) and compare to SeqDiffuSeq’s early truncation. 
5. Add longer-form generation, dialogue, or factual QA, and report mean±σ over multiple seeds; analyze sensitivity to δ ̃and steps across tasks.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces SMOOTHIE, a diffusion model framework designed for text generation. The authors identify a key challenge in adapting diffusion models to discrete data like text: existing methods either operate in a continuous latent space (e.g., Gaussian diffusion on embeddings), which struggles with accurate token decoding, or in a discrete/categorical space, which ignores the semantic relationships between tokens. SMOOTHIE perturbs distance-based representations of tokens, dissolving semantic structure over time. The authors claim this technique is superior to both standard embedding space diffusion and categorical diffusion. They provide empirical evidence on several sequence-to-sequence generation tasks.

### Strengths
- The proposed diffusion space, which perturbs representations based on semantic similarity, is a contribution to the field.
- The authors provide empirical validation across multiple text generation tasks. The reported results suggest that the proposed method may offer a performance improvement over other diffusion-based baselines, and the inclusion of ablation studies helps to substantiate the specific design choices made in the SMOOTHIE framework.

### Weaknesses
- The proposed method's reliance on a pre-trained word embedding model (in this case, BERT) may limit its scalability and applicability. This dependency raises questions about the framework's potential to scale effectively with larger models or different architectures, as it is tied to the properties and constraints of the initial embedding space.
- The experimental evaluation is missing a common and important conditional text generation task: machine translation. Including results from machine translation would provide a more comprehensive assessment of the method's capabilities and generalizability.
- The paper lacks an analysis of the method's sensitivity to the choice of the pre-trained word embedding. It would be beneficial to investigate whether the approach is viable with other types of embeddings, such as those from GPT-based models, to better understand the robustness and flexibility of the proposed framework.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
1
