## Human Reviewer 1

### Summary
This paper investigates the categorisation behaviour of LLMs for colour, comparing it with human generated color categories in many languages. It adopts an information bottleneck analysis, testing whether generated colour categories optimally trade off accuracy and complexity. The paper shows that LLMs can differ dramatically in their alignment to English, and the complexity of their categories, with larger models tending to perform better. Second, the paper investigates an iterated learning scheme in which LLMs receive an initially random colour categorisation and over several rounds of in-context classification, converge to a new categorisation. This categorisation can then be assessed for its position on the IB tradeoff plane and compared to human languages. Results show that iterated in context learning progressively improves IB efficiency over iterations, and the strongest models show a similar range of solutions as in humans. Finally, the paper shows that similar ideas can apply beyond the domain of colour, by investigating the categorisation of a class of simple shapes that differ in orientation and size.

### Strengths
This paper presents a thorough investigation of colour categorisation behaviour in LLMs. The extent of the experiments, range of models considered, and variety of training hyper parameters investigated (such as training epoch, instruction tuned or not, text vs image prompted, etc) paint a particularly complete picture of the full space of models' abilities in this area. 

The results show that, surprisingly, only the most capable models show colour categorisation behaviour matching English. Further, only these models can develop diverse pseudo labels that span the IB tradeoff curve across complexities, similarly to humans. 

The basic observations are extended to 'Shepard circles', a particular set of visual stimuli, to show that the proposed ideas are not specific to colour naming.

The paper contains an excellent review of prior work, and motivation for studying colour behaviour as a highly studied instance of human categorisation behaviour.

The experiments appear to be done to a high standard, and the plots are clear and easy to follow.

### Weaknesses
There were several points that I struggled to understand which may reflect my own limited knowledge of the area:

-The clusterings seem to more or less subdivide the colour space into a set of contiguous regions with approximately equal size, as one might expect by running k-means on the feature vectors (sRGB values) or any other clustering technique. If this is so, then it may be quite easy for a system to have an implicit bias that would place it near the IB curve--it just needs to approximate k-means or similar. If this is right, then statements like "LLMs can acquire a human-like inductive bias toward optimally-compressed semantic representations, without being trained for this objective" become less remarkable. It is easy to see how in context learning might approximate clustering, and this implicit bias has no clear human-specific content. 

-The observation that Gemini 2.0 can recover English alignment from rounds of iterated learning could be consistent with the model inferring that the feature values are sRGB values, and aligning pseudo labels with real colour names. That is, in-context learning may be operating in a ‘task inference’ mode in which it guesses that the features correspond to colours, and then assigns pseudo labels real colour names accordingly. This mechanism would mean that iterated in-context learning is just another measurement of whether an LLM is aligned to English colour categories, not that something fundamental about how its inductive bias aligns with humans.

-The experiments present colours as sRGB triplets, but many other colour spaces are possible. Does the choice of colour space features affect the results?

### Questions
How does k-means or another clustering method on the same features compare to the behaviour of the LLMs?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
The authors investigate large language models’ linguistic capabilities to partition semantically the colour domain and study the efficiency of the resulting systems in terms of the IB framework. Further, they investigate their capabilities to evolve inefficient semantic systems towards more regular categories ((again, in the IB framework sense), both in the colour domain and in the less humanly intuitive Shepard circles one. They show that while many models struggle to recover an efficient colour system, many times exhibiting trivial, low complexity categories, the ones that show efficient, more human-like systems also show the capabilities to evolve a non-efficient artificial system towards more efficient ones via in-context iterated language learning

### Strengths
The paper’s biggest contribution is to study an important question in cognitive science, namely the emergence of IB-efficient linguistic systems, in the context of large language models.  The paper is sound, well-written and very easy to follow. The experiments are well designed and multiple models, as well as multiple modalities, are systematically tested. The author’s extension of the in-context iterated learning paradigm of Zhu and Griffiths (2024) is a natural and effective way of studying efficient communication questions in the context of large language models, while the author’s replication of the human experiments from Xu et al. (2013) with large language models offers us an additional perspective on iterated learning as a paradigm to evolve artificial linguistic systems. The authors’ results are both well documented and well presented visually, and are also in line with efficient communication theory. 

While previous literature has analysed large language models’ capabilities of recovering human-like semantic systems in the colour domain, or the role of iterated learning in large language models, to my knowledge, this is the first comprehensive study of large language models from an efficient communication point of view.

### Weaknesses
While the authors analyse a large number of different families of models and, inside those,  models of many different sizes, they restrict their iterated learning analysis to only the best-performing models at the previous task. It would have been interesting to see if iterated learning might have improved their ability to evolve degenerate systems towards more IB-efficient ones. Similarly, the preliminary work on Shepard Circles is lacking a more quantitative analysis similar to the results in Section 4.2, while a figure showing the evolution of the LLM systems like Figure 3 would have helped quantify the evolution of these systems.

A more general feedback is that the authors do not analyse the possible role for large language models of language use via communication in evolving inefficient linguistic systems toward more human-like ones. While Imel et al. (2025) showed how learnability might be the principal pressure that leads towards efficient systems in the colour and Shepard circles domain, an analysis on the role of both cultural transmission via iterated learning and language use via communication would be a very interesting avenue for future work. More specifically, the framework of Kouwenhoven et al. (2024) might be of interest for the authors. 

Smaller points to address would be to fix the citations in the appendices, while some figures (cf. Figure 3) are grainy if zoomed in.

### Questions
What do you think might be the role of communication in evolving these systems towards more efficient ones? Do you think that an innate preference for informative, non-trivial systems (which previous efficient communication literature argues is a consequence of language use via communication) is present in the evolution of these artificial languages because of the data these models are trained on?

How do you think discrete domains (e.g. kinship) would influence these results? One could argue that continuous domains like colours and Shepard circles might be simpler to efficiently partition for both humans and large language models than more discrete ones.

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
The present paper investigates whether LLMs are capable of evolving efficient human-aligned semantic systems in the domain of color categorizatzion. The authors replicate two previous human studies on LLMs and find that they vary in their complexity and English-alignment. Larger instruction-tuned models achieved better alignment and IB-efficiency. Finally, they simulate LLMs on cultural evolution of pseudo color-naming systems and find that LLMs iteratively restructure initially random systems towards greater IB-efficiency.

### Strengths
Overall, the paper was well written and easy to follow. It builds on prior work with sound methodology, but extends and adapts it for LLMs. The authors consider a large set of models, including models with different properties, different checkpoints, as well as open-source and frontier models.

### Weaknesses
The main weakness from my perspective is that novelty and contribution is rather limited. The authors take an existing task and evaluation method and simply run LLMs on it. That might have been enough a few years ago, but it is not  anymore (at least in my opinion). The extension to ICCLLL, which would be the main novelty of the paper, is also fairly straight-forward.

While the authors show some results on how their findings would generalize beyond the color naming domain, more could be done to establish a robust and general pattern. 

Even if one would find that LLMs are IB-efficient in their naming systems, would would that mean? What implications would come with that? Or, vice versa, if it were not the case, what would the implications be? From my perspective, these questions deserve some discussion because at present it is a bit unclear to me why I should care.

Minor: p2 rarely -> readily?

### Questions
It is interesting that instruction-tuned models achieve better alignment and IB-efficiency. Is that merely an reflection that these models are better at the task? Could one take, for instance, an instruction-tuned and a base model that achieve the same accuracy and look at whether they differ in terms of alignment and IB-efficiency?

To what extend is the observeation that (some) LLMs are IB-efficient a reflection of the fact that they are simply trained on human language?

The authors write "IB-alignment measures the similarity between a system and the nearest optimal IB system. WCS-alignment measures the average alignment between a system and the WCS languages, and English-alignment measures the alignment between a system and English." Are these measures defined somewhere?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4