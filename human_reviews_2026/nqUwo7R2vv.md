# mGRADE: Minimal Recurrent Gating Meets Delay Convolutions for Lightweight Sequence Modeling

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Processing temporal data directly at the sensor source demands models that capture both short- and long-range dynamics under tight memory constraints. While State-of-the-Art (SotA) sequence models such as Transformers excel at these tasks, their quadratic memory scaling with sequence length makes them impractical for edge settings. Recurrent Neural Networks (RNNs) offer constant memory scaling, but train sequentially and slowly, and Temporal Convolutional Networks (TCNs), though efficiently trainable, also scale memory with kernel length. For more memory-efficient sequence modeling, we propose mGRADE (minimally Gated Recurrent Architecture with Delay Embedding), a hybrid-memory system that integrates a temporal convolution with learnable spacings with a gated recurrent component equivalent to a minimal Gated Recurrent Unit (minGRU). The convolution with learnable spacings can express a flexible delay embedding that captures rapid temporal variations, while the recurrent component efficiently maintains global context with minimal memory overhead. We theoretically ground and empirically validate our approach on two types of synthetic tasks, demonstrating that mGRADE effectively separates and preserves temporal features across multiple timescales. Furthermore, on the challenging Long-Range Arena (LRA) benchmark, mGRADE reduces the memory footprint by up to a factor of 8, while maintaining competitive performance compared to SotA models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents mGRADE, an examination of a hybrid architecture combining recurrent models with convolutions.  The objective is to use the convolutions as a short-term processing mechanism, and then use a lighter recurrent network to process long-range dependencies.  The core mGRADE layer is a 1-d convolution (in time), a minGRU recurrent layer (providing gating), and a position-wise FFN.  A stacked architecture is then built out of this.  Some simple theoretical exploration of fundamental tasks is presented so explore the core properties.  The model is then benchmarked on LRA and achieves very competitive results despite using fewer parameters, smaller memory footprints and being strictly unidirectional.

### Strengths
1. Reducing (or even remarking on) memory consumption and memory complexity is often underserved in the SSM literature.  A lot of SSM architectures eulogize constant time-complexity per step, but assume access to massive computational hardware.  This is a nice examination of that angle. 
2. I like how quickly the paper gets into the meat of the topic.  
    (I will note, mind you, it could use another paragraph of signposting on what you will show and why:  some combination of a concrete problem specification and notation section;  a list of key elements to prove or introduce;  expanded discussion of the shortcomings in other methods you will overcome;  etc.  Just to prime the reader for what is coming in a bit more detail.  It is a bit of a "smash-cut" into technical detail.)
3. The banner figure is a good for introducing the architecture.  
4. There are a good number of experiments and ablations presented.

### Weaknesses
# Summary
I find this paper hard to place.  Memory and edge applications are super important to consider, and _are_ underserved in the SSM literature.  However, this paper finds itself in an awkward in-between space where: (1) it is comparing to SSMs not optimized for memory consumption; and (2) doesn't compare to memory-efficient architectures; and (3) doesn't actually evaluate the edge suitability of any of the models.  Instead of being in the overlapping section of the three circles in a Venn diagram, it is in a region where no regions overlap.  

I therefore struggle to convince myself that this paper fully and accurately encapsulates the current state of the field(s).  Ultimately, the paper (as presented) states:  "here is _a_ more memory efficient SSM," without fully fleshing out the bounds or expectations on practical performance.  Therefore, I unfortunately cannot endorse this paper for publication in its _current_ form.  

With that said, I think the core of a _very_ good paper is here, and I strongly encourage the authors to continue working on it, expanding the tie-ins and evaluation, because it could be a very compelling study.  Good luck.  

# Major Weaknesses
1. **Comparing with performance-optimized SSMs**:  A lot of the results quoted didn't consider parameter complexity (in fact, most of the architectures were developed with roughly matching parameter complexity).  It is therefore a bit of an unfair comparison to claim superiority in terms of parameter complexity, when it is not clear how well, e.g., LRU would do with 40k parameters.  It is impressive that you get so close with 20% of the parameters, but the comparison is, in my opinion, incomplete.  Equally, although this isn't part of the initial pitch of your paper, I would be interested to know how parameter-matched mGRADE compares in terms of performance.  To answer this, I'd need to see numerous experiments sweeping across parameter counts, across different hyper parameter and architectural configurations.  
2. **Absence of efficient architectures**:  I am not an expert in highly compressed neural architectures, but I know there are things like MobileNet, neuromorphic computation, spiking neural networks, low-rank factorized models etc that are specifically designed for low-resource environments.  I therefore don't know if I'm really convinced by the breadth of the evaluation of your claims of parameter and memory efficiency.  Yes, you have made a possibly more parameter-efficient SSM, but I feel that this is a very narrowly scoped claim that maybe doesn't rise to the level of significance required for ICLR. 
3. **Absence of edge-evaluation**: Despite discussing efficient computation throughout, none of the architectures are actually deployed on edge hardware.  I'd _love_ to see an evaluation of these architectures, for instance, running on a Raspberry Pi, and recording maximum sequence length, latency, failure rate etc.  Simply claiming improved memory efficiency, while executing all experiments on a 24GB GPU feels like missing an opportunity.  If you could show your network deployed on these architectures where others can't, then that would be huge strengthener to your claims.  
4. Absence of Path-X:  I respect the authors may not have access to huge computational hardware, but unfortunately Path-X is where differences in architectures is really observed:  many modern architectures/hyperparameters that perform comparably on the other LRA tasks fail critically on Path-X.
5. Long convolutions?:  I notice that the convolutions in Image and Pathfinder are very long -- specifically 1/4 of the input length.  I query therefore how much of the lifting this convolution is doing, and how much the recurrence is even needed.  I would like to see a baseline of how a purely-convolutional model (using the $\Gamma$ you use) with a suitable pooling operator (mean/max) performs.  

# Minor Weaknesses/Typographical Comments
1. The minimal gating is just a direct application of minGRU, this should be advertised more clearly and earlier.  I don't love the way some of the "minimal gating" is advertised earlier, making it feel like it is a novel contribution.  
2. Table A6 is very important, I think.  It shows that the actualized gains are far lower than advertised (if at all).  This table should be brought up to the main paper.
3. Why are HGRN and S5 omitted from Table A6? 
4. Mamba is also missing from all your evaluations, why? 
5. Are you able to run all the baselines in Table 2?  It is a bit of a shame to have missing entries. 

# Missing Literature
1. MEGA (and MEGA-Chunk) [Ma+, 2023] should be (at least qualitatively) compared to, as they use a lightweight 1-D convolution layer before inputting into a chunked transformer.  This would achieve a similar effect.

### Questions
I invite the authors to respond to my comments in "weaknesses".

Q1. There is a known equivalence between convolutions and SSMs (this is the basis of S4!).  Is it just that directly parameterizing the convolutions is more parameter efficient?  Why can't you achieve this fully recurrent?  I think at least qualitative discussion of this is absent.

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
The paper introduces mGRADE which is a minimal linear GRU with convolutions that support parallel training using scan algorithm and RNN-style inference of State Space Models and Linear Transformers. The main motivation behind design is to build a device for embedded systems and resource constraint environments. The experiments are done in LRA benchmark and Lorenz attractor.  moreover, the paper highlights that mGRADE enjoys selective gating which helps for long-context modeling (which is shown in LRA benchmark).

### Strengths
The major strengths of paper are:

- **Direction of the paper** Paper build upon a current important direction of efficient sequence modeling with sub-quadratic models such as linear transformers and SSMs.

- **Background analysis** Paper nicely present a very sufficient background material on SSMs and progress of literature specially in its related work section and introduction.

- **Loranz and LRA Experiments** The design of these experiments are indeed relevant and sound.

- **Presentation and writing** The paper is well-written and clearly presented its approach.

### Weaknesses
However, while the paper is generally well written and the overall design appears sound, there are several important gray areas regarding comparisons with SSMs—particularly Mamba, Mamba2, and GLA [2], as well as some outdated training strategies.

* **Model differences with Mamba2 and Gated RFA:**
  The main architectural components appear very similar to existing SSMs. In particular, the sequence mixing mechanism based on the minimal GRU closely resembles the recurrence structure of the Gated RFA. Moreover, the role and necessity of short-range causal Conv1D layers have already been explored in modern Linear Transformers such as Gated DeltaNet [1] and in prior analyses (see [this post](https://kexue.fm/archives/11320)). This raises the question: what is the key difference and contribution of mGRADE compared to existing models, given that both causal Conv1D and GRU-style recurrence have been well studied in SSMs?

* **Lack of time-series experiments:**
  Although the paper claims to target sensory data and signal processing, it does not include any experiments on time-series tasks. The only experimental setup is the LRA benchmark, which is both outdated and largely synthetic. For a model motivated by real-world signals, it would be important to demonstrate performance on datasets such as *Speech Commands* (as used in S4 models), or other representative time-series benchmarks.

* **Training efficiency and scan algorithm:**
  The paper relies on the parallel scan algorithm for training, which has been shown to be significantly slower and less efficient than chunkwise parallel training strategies such as those used in SSD and GLA (see Figure 6 of the paper). It would be useful to clarify how the training time of mGRADE compares to Mamba2 and GLA, and whether mGRADE also supports chunkwise parallel training.

* **Selectivity and language modeling:**
  If the model is designed to be selective, how does it perform on language modeling tasks? Selective SSMs like Mamba were specifically developed for language tasks and are known to perform poorly on LRA benchmarks while excelling on linguistic datasets (as highlighted by the authors [here](https://github.com/state-spaces/mamba/issues/282#issuecomment-2221135197)). It would be important to see how mGRADE performs in comparison, and whether its selectivity mechanism aligns with those used in language-oriented SSMs.




-------


### References

[1] Gated Delta Networks: Improving Mamba2 with Delta Rule: Songlin Yang, Jan Kautz, Ali Hatamizadeh


[2] Gated Linear Attention Transformers with Hardware-Efficient Training: Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim

### Questions
1) What is the training time of mGRADE compared to Mamba2 specifically since it is using SSD?

2) How does the model perform in time-seires tasks such as **Table 13** of S4 paper [1]?

3) How does mGRADE works without causal convoulution on LRA to see how performant is the sequence mixing of the mGRADE?

4) What is the difference of mGRADE's minimal GRU compared to GRFA's or Mamba2's sequence mixing?

------

### References

[1] Efficiently Modeling Long Sequences with Structured State Spaces: Albert Gu, Karan Goel, and Christopher R´e

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper propose mGRADE, a new neural network architecture that combine a temporal convolution learned spacings with a gated recurrent component. The advantages of the method are demonstrated on two synthetic tasks, as well as on the Long Range Arena benchmark, where the number of parameters/parameter buffers is reduced by a factor of up to 8.

### Strengths
The idea of combining a recurrent gating structure with temporal convolutions is interesting to capture temporal patterns both over short and long time-scales.

The figures in the paper are overall good, for instance figure 1 gives great intuition about the method.

Several of the benchmarks are interesting (Lorentz, Flip-Flop), and are connected to the theory the authors develop.

### Weaknesses
The authors are focusing in the introduction about the need to create a memory efficient architecture. With this in mind, it would have been more interesting in the LRA experiments to compare the models with respect to runtime memory requirements of the task rather than the number of parameters as the difference between loading 1 million parameters and 1k parameters is tiny in terms of memory footprint. Hence saying that the memory footprint is reduced by a factor of 8 is misleading.

In your table 2, you report results for H3 on selective copying, but I am not seeing these results reported in the H3 paper. The H3 paper focus on induction head and associative recall which appears related but not the same. Can you please clarify where these numbers are found?

It would be more clear if the parameterization of the conv kernel is given in the main text rather than in the appendix
It would have been good if the authors clarified in the main paper how the model achieves a lower parameter count (i.e. is it by having fewer layers, fewer parameters per layer, fewer parameters in the convolutions etc.). 
The flip-flop task is not described in sufficient detail

### Questions
In the appendix (line 778) you say that a temporal convolution needs a buffer to store past inputs with size proportional to the kernel length. Why this is the case when the convolution is computed with the FFT?

Why is Path-X (part of LRA) not included in the analysis? Did your model not work on this task?

### Soundness
3

### Presentation
3

### Contribution
2
