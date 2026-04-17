# Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Video recognition models are typically trained on fixed taxonomies which are often too coarse, collapsing distinctions in object, manner or outcome under a single label. As tasks and definitions evolve, such models cannot accommodate emerging distinctions and collecting new annotations and retraining to accommodate such changes is costly. To address these challenges, we introduce category splitting, a new task where an existing classifier is edited to refine a coarse category into finer subcategories, while preserving accuracy elsewhere. We propose a zero-shot editing method that leverages the latent compositional structure of video classifiers to expose fine-grained distinctions without additional data. We further show that low-shot fine-tuning, while simple, is highly effective and benefits from our zero-shot initialization. Experiments on our new video benchmarks for category splitting demonstrate that our method substantially outperforms vision-language baselines, improving accuracy on the newly split categories without sacrificing performance on the rest. Project page: https://kaitingliu.github.io/Category-Splitting/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new task named "category splitting", which aims to refine a coarse category into more detailed subcategories through classifier edits. They propose a zero-shot editing method that leverages the latent compositional structure within video models to create these new distinctions without requiring additional data. Besides, the study shows that low-shot fine-tuning is highly effective, and its performance is further enhanced when initialized with the edited weights. Experimental results on new benchmarks demonstrate that this approach significantly outperforms baselines, improving performance on the newly split categories while preserving performance on the others.

### Strengths
* This task introduces a new setting called "category splitting" and creates new benchmarks by reorganizing existing ones (SSv2, FineGym).
* The proposed method is reasonable and performs well on new splits of category, outperforming existing prevalent Vision-Language Models.
* The analysis section provides a sound analysis of the method's effectiveness, offering readers a deeper understanding of the method.

### Weaknesses
* The task setting proposed in this paper is akin to zero-shot classification or continual learning, but it is less challenging since the distributions of the fine-grained subcategories and the original coarse category are relatively close. Besides, a major limitation of this task setting is the requirement that the coarse and fine-grained categories must share the same base name, limiting the task's significance and practicality.
* The method section is hard to follow. There exist undefined or unclaimed symbols, and some statements are confusing.
* As stated in line 139, why use the mean of the associated fine-grained weight vectors as the pseudo vector of the coarse category? What if using the real text vector of coarse categories?
* According to Table 2, despite achieving sound results on the new categories, it still causes a drop on the other categories, whereas VLMs (e.g. CLIP) can avoid the problem.

### Questions
* Why not choose the image encoder from a VLM as the base model, as it may possess better latent compositional structure during the pre-training.

### Soundness
2

### Presentation
1

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
The paper proposes a method for zero-shot fine-grained classification by editing classifier weight matrices without retraining backbones. They extract "modifier vectors" by averaging coarse category weights from existing fine-grained examples and subtracting to isolate semantic differences, then use these to generate new subcategory weights via w_subcategory = w_coarse + v_modifier. The method is evaluated on SSV2-Split and FineGym-Split benchmarks, expanding classifier matrices from ~100 to ~150 categories while demonstrating improved performance over standard CLIP and VideoCLIP baselines on fine-grained classification tasks.

### Strengths
-- Practical & Elegant Solution: Addresses real problem of fine-grained classification without expensive retraining - just intelligent matrix manipulation

== Compositional Approach: The weight arithmetic (w_subcategory = w_coarse + v_modifier) is intuitive and enables systematic fine-grained category generation

### Weaknesses
-- Scalability Questions: Method requires existing fine-grained examples to extract modifiers, and matrix growth (100→150 categories) may not scale to truly large taxonomies

-- Clarity and Organization Issues: Paper was slightly difficult to follow on the main contribution and method - would benefit from restructuring for better readability (more intuitive figures perhaps?)

-- Insufficient Baseline Comparisons: Only compares against basic vision-language models (CLIP, VideoCLIP) rather than established fine-grained classification methods (e.g., other compositional approaches)

### Questions
I wonder, isn't it worth testing your model against fine-grained approaches (e.g., compositional approaches or methods that improve CLIP with additional losses for fine-grained evaluations)? There are many such methods, and while I'm not sure which one would be the best fit, I still raise this question. The current evaluation seems limited to basic vision-language baselines.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces the task of category splitting for video recognition: starting from an already trained video classifier with a fixed label set, you later realize a coarse label like “poking something” or “dropping something” actually needs to be broken into several fine-grained subactions, but you don’t want to retrain the whole model or disturb performance on the other, unrelated classes. The paper’s key insight is that existing video models already contain reusable, compositional structure in their classification head: across related actions, the difference between labels often looks like a consistent “modifier” (e.g. direction, manner, target), so they build a modifier dictionary from existing labels and then synthesize new subcategory weights by adding an appropriate modifier vector to the original coarse class weight, producing new labels in a fully zero-shot way; to go beyond seen modifiers, they train a small alignment module that maps modifier text to modifier vectors, and if a few labeled clips for the new sublabels are available, they fine-tune only those new head weights to improve accuracy while keeping the rest of the model fixed, thus preserving locality. Evaluated on split versions of video datasets (like SSv2-Split and FineGym-Split), the method improves recognition of the new, finer labels without degrading performance on untouched classes, showing that you can “edit” a video model’s label space after training by exploiting the structured differences already present in its classifier.

### Strengths
- Proposes and explores a novel problem of clear practical relevance
- The work explores different forms of class splitting, both where the modifiers are seen or unseen
- The proposed solution is tidy and lightweight: they edit only the classifier head, reuse structure already present in the model by building a modifier dictionary from existing labels
- The authors construct 2 datasets for this problem from SSv2 and FineGym, which pose challenging test cases

### Weaknesses
- Method depends significantly on the assumption that the original label space already contains enough compositional variation to learn good modifier vectors

- Because the edit happens at the classifier head, it also assumes the backbone already captures the visual distinctions the new sublabels require; if the new split introduces visual novelty rather than just semantic refinement, a head-only edit will struggle, and the paper doesn’t really explore that failure mode

- There is some amount of over-claiming going on in this paper, e.g. the title claims "YOUR VIDEO MODEL CAN BE EDITED", but this method is fairly limited to a very specific set of situations.

### Questions
- You show that adding a retrieved/aligned modifier vector to the coarse class weight works, but how often is the best modifier actually coming from the same base action vs. being borrowed from a semantically different action? I would like to see some statistical analysis of this.

- How robust is the text encoder you use to messy, real-world label names (typos, multi-sentence definitions, multilingual labels etc) ? This choice should ideally be ablated

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a method for zero-shot adaptation of video-language models to perform fine-grained classification. This is tackled by "splitting" existing categories to sub-categories. The method improves over baselines. The paradigm is also extended to a few-shot setting where similar gains are shown.

### Strengths
The method shows good improvement over baselines. The authors also provide many ablations such as choice of encoder, pretraining among others. The dataset generated has also been provided fully aiding in transparency. The concept of zero shot adaptation to finer granularity levels is interesting and warrants more attention.

### Weaknesses
The paper uses the term "video model" very generally to refer specifically to a kind of video-language model. This is misleading as "video model" can refer to other concepts such as video generation models.

The method is very hard to follow as the authors do not provide any preliminary information of the architecture that they are based upon. Eg. it is hard to follow which are the weight vectors that are being referred to as additive.

Subjective: The title of the paper does not convey the problem being tackled.

### Questions
None

### Soundness
2

### Presentation
2

### Contribution
3
