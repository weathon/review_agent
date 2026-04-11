# ENERGY-EFFICIENT RANDOM VARIATE GENERATION
## VIA COMPRESSED LOOKUP TABLES


**Johann Ukrow** _[∗]_ **, Anna Kazachkova** _[∗]_ **, Nicolas Alder, Sven K¨ohler,**
**Rainer Schlosser & Ralf Herbrich.**
Hasso Plattner Institute
University of Potsdam
Potsdam, Germany
_{_ johann.ukrow,anna.kazachkova,nicolas.alder,sven.koehler,
rainer.schlosser,ralf.herbrich _}_ @hpi.de


ABSTRACT


Generating (pseudo-)random variates lies at the core of probabilistic machine
learning and prediction algorithms and yet remains a major bottleneck due to its
high computational and energy cost. In this paper, we introduce a general and
scalable sampling strategy that enables fast and energy-efficient random variate
generation from arbitrary distributions. Our approach is based on compressed
lookup tables (cLUT) combined with a fast index sampling scheme. Using only
a handful of fast and energy-efficient compute operations on simple array structures, we achieve superior speed, energy efficiency, and precision at near-optimal
entropy cost compared to state-of-the-art techniques. Microbenchmarking our approach with a C implementation shows up to 40% savings in time and 50% in energy compared to state-of-the-art approaches. Compared to commonly employed
Python samplers, we achieve a 100 _×_ time improvement.


1 INTRODUCTION


Sampling from probability distributions is a fundamental yet computationally expensive operation in
machine learning. In representation learning and in broader machine learning, sampling underpins
core methods such as variational autoencoders (Kingma and Welling, 2022), contrastive learning
with negative sampling (Chen et al., 2020), diffusion-based generative models (Ho et al., 2020b),
and probabilistic inference techniques such as Bayesian deep learning (Sommer et al., 2025). While
the quality and efficiency of sampled variables directly shape the expressiveness and scalability of
learned representations, sampling costs often remain a primary barrier to scalability and widespread
deployment. In this paper, we address this bottleneck by introducing a novel, efficient sampling
approach for arbitrary distributions. Our method achieves 10-100 _×_ speedups and up to 60% reduction in energy consumption compared to commonly employed approaches, significantly reducing
the resource-intensity of many machine learning tasks.


The continued deployment of machine learning methods in data centers, cloud devices, and user
appliances alike is accompanied by increased concerns about the growing energy demand of the
field (International Energy Agency, 2025; Gadepally, 2025). Countermeasures include reducing the
carbon-intensity of the electricity supply or shifting training and inference to times or physical locations with a higher share of renewable energy sources throughout the day (Yang et al., 2023; Wiesner
et al., 2023). However, we argue that reducing the energy demand of the operations themselves is
worthwhile, with emphasis on frequently performed actions like sampling. Motivating us to perform extensive energy measurements in addition to speed measurements, comparing our method to
classical, widely used approaches and recent state-of-the-art advances.


On digital computers, sampling from arbitrary probability distributions is reduced to sampling from
finite discrete distributions due to fundamental constraints of finite precision and memory. All probability distributions, whether continuous or infinite discrete, must be discretized for computational


_∗_ Equal contribution.


1


implementation (see Appendix B for discretization techniques). Standard sampling algorithms in
widely used libraries (NumPy, PyTorch, JAX) assume infinite precision arithmetic, where computation can be performed with arbitrarily precise real numbers (Shamos 1978; Devroye 1986, Chapter
2, p.1, Assumption 1). Additionally, they assume the ability to generate infinitely precise samples
from the real unit interval (Devroye 1986, Chapter 2, p.1, Assumption 2). However, actual implementations rely on finite floating-point representations and don’t have access to exact samples from
the real unit interval, causing generated distributions to deviate from their intended target distributions in uncontrolled ways. These deviations are often intractable to quantify, precluding theoretical
guarantees about sampling accuracy. To address this issue, our proposed sampling method features
controllable precision and exactly represents target distributions with clear theoretical guarantees.


**Problem formulation** In what follows, we will describe a novel method to generate random variates from any finite discrete distribution, represented by _n_ probabilities _p_ 1 _, . . ., pn_ _∈_ [0 _,_ 1] of the
corresponding _n_ outcomes _x_ 1 _, . . ., xn_ _∈X_ . As we denote no constraints on the structure of _X_ the
outcomes can be of arbitrary types, such as real numbers, strings, pointers to more complex data
structures, or any mixture of those. Our objective is to make this generation process fast, energy
efficient, and adjustable to an arbitrary and controllable precision.


**Our** **contributions** This work introduces a new sampling approach for arbitrary distributions
based on operations with lookup tables. Besides being a generic method for efficient and arbitrary
precise sampling, our approach is especially suitable for situations where floating-point operations
are either unavailable or too error prone, and situations with a low power supply. We summarize our
contributions as follows:


1. We propose a novel a random variate generator based on compressed lookup tables (cLUT),
optimized for highly efficient sampling. We introduce a lossless compression strategy for
compact representations of distributions achieving an exponential compression ratio.

2. We compare cLUT against state-of-the-art approaches in terms of speed, energy efficiency,
memory usage, and entropy efficiency. It runs 30-40% faster and saves 25-50% energy in
a diverse set of distributions. For larger distribution sizes, it performs particularly well.

3. We benchmark cLUT against standard sampling routines from widely used Python machine learning libraries. cLUT achieves up to 10-100 _×_ acceleration in speed. Furthermore,
we illustrate the impact of our approach in real-world machine learning applications by
showcasing that cLUT substantially reduces the execution time and energy consumption of
the exemplary TrueSkill application.


2 RELATED WORK


Sampling methods are classically divided into two categories: _exact methods_, which produce samples from the target distribution **p** as specified, and _approximate methods_, which generate samples
from a distribution **p** ˜ that only approximately matches the desired distribution, i.e., **p** ˜ _≈_ **p** . Note
that our approach is exact.


**Exact** **methods** Knuth and Yao (1976) established the theoretical foundation for exact discrete
sampling using discrete distribution generating trees. Their seminal result shows that any optimal sampling algorithm requires between _H_ ( **p** ) and _H_ ( **p** ) + 2 bits per sample, where _H_ ( **p** ) =

- _i_ _[−][p][i]_ [ log] 2 [(] _[p][i]_ [)] [is] [the] [Shannon] [entropy.] [While] [entropy-optimal,] [discrete] [distribution] [generating]

trees typically require exponential memory in the distribution precision. Lumbroso (2013) overcame this limitation for uniform and Bernoulli distributions with a linear-memory implementation,
but the approach does not generalize to arbitrary distributions. The generic interval algorithm (Hao
and Hoshi, 2006) achieves linear memory usage while consuming at most _H_ ( **p** )+3 bits per sample.
However, implementations require expensive binary searches at each sampling step, limiting practical efficiency (Devroye and Gravel, 2020; Uyematsu and Li, 2003). Saad et al. (2020) presented the
FLDR algorithm that combines entropy-optimal sampling with rejection sampling, achieving an upper bound of _H_ ( **p** )+6 bits per sample. **?** improved this to _H_ ( **p** )+2 bits with faster sampling speed
for the ALDR algorithm, though at a higher memory cost. Building on Marsaglia (1963), Marsaglia
et al. (2004) proposed compressed lookup tables for discrete sampling. However, their compression scheme requires conditional branching and searches across multiple tables during sampling,


2


reducing efficiency. In contrast, our approach uses a single compressed table with direct indexing,
eliminating conditional overhead.


**Approximate** **methods** Most samplers for discrete and continuous distributions used in practice
are so-called approximate samplers (for an introduction, see Schwarz, 2011). These methods typically rely on the assumptions of the _real Random Access Machine_ (RAM) model (Shamos, 1978,
computations can be performed with arbitrarily precise real numbers), and the assumption of having infinitely precise uniform random generators (Devroye, 1986), which cannot be fully realized
on digital computers. For a comprehensive overview, Devroye (1986) presents the mathematical
foundations of random sampling and details numerous approximate samplers built on the real RAM
model. As noted by Draper and Saad (2025), implementations consequently suffer from multiple sources of approximation error and are often inefficient in their use of bits, since generating
a single uniform random variable typically already consumes 32 or 64 bits. A widely employed
general approximate sampler is the Alias method (Walker, 1974), which preprocesses distributions
into probability and alias arrays, enabling sampling via one uniform random variable and a single
coin flip (see Schwarz (2011) for a detailed explanation). While being fast, it produces approximate
samples and lacks controllable error bounds. Similarly, the Index method (Chen and Asau, 1974)
uses preprocessed index tables to guide inversion-based approximate sampling, but still requires
expensive search operations.


In contrast, our proposed method does not rely on the real RAM assumption or on access to arbitrarly precise random samples from the unit interval. It achieves exact sampling while remaining
highly entropy-efficient, using close to the minimum number of bits required to represent the target
distribution (see Figure 4).


3 APPROACH


Our approach is based on the idea of lookup tables, reusing precomputed results, while conserving
memory requirements and memory accesses as detailed in this section. A schematic of the sampling
pipeline is given in Figure 1a.


**Naive approach** We will describe a method to generate random variates from any finite discrete
distribution, represented by _n_ probabilities _p_ 1 _, . . ., pn_ _∈_ [0 _,_ 1] of the _n_ outcomes _x_ 1 _, . . ., xn_ _∈X_ .
Ideally, we would construct a lookup table containing duplicates of each outcome proportional to its
probability:
occurrences of _xi_ in table

= _pi._ (1)
table size


Sampling would then reduce to uniformly selecting a random table index _I_ _∼_ Uniform _{_ 1 _, . . ., N_ _}_
and returning _S_ = Table[ _I_ ], where _N_ is the table size. See Figure 1b for an examplary ‘naive‘
lookup table.


**Memory** **constraints** In practice, memory constraints bound the table size _N_, limiting representable probabilities to multiples of 1 _/N_ . Approximating probabilities to precision _b_ bits requires
quantizing each _pi_ to _fi_ = round( _pi_ _·_ 2 _[b]_ ), yielding frequencies **f** =( _f_ 1 _, . . ., fn_ ) and a table of size
_N_ = [�] _i_ _[f][i]_ [=] [2] _[b]_ [(see] [the] [Appendix] [C] [for] [rounding] [schemes] [and] [error] [analysis).] [While] [approx-]

imation error decreases logarithmically with _b_, memory requirements grow exponentially, making
high-precision sampling prohibitive. Our approach also handles continuous and infinite discrete distributions through discretization techniques detailed in the Appendix, Section B. The basis of our
main approach is a lossless compression strategy for the lookup tables and the following sampling
scheme.


**Compression** **scheme** To tackle the prohibitive memory requirements of lookup tables, we propose to use the following compression scheme. Intuitively, the compressed lookup table can be
viewed as a two-dimensional array consisting of _r_ + 1 rows and 2 _[c]_ columns, with _r, c ∈{_ 0 _, . . ., b}_
satisfying 2 _[r]_ [+] _[c]_ = 2 _[b]_ = _N_ . Each row _i_ of the first _r_ rows corresponds to a frequency of 2 _[r][−][i]_, where
row indices run from 1 to _r_ . The _r_ +1-th row corresponds to the same frequency as the _r_ -th row,
namely 2 _[r][−][r]_ = 1. For an exemplary compression, see Figure 1b. This lossless compression scheme


3


compressed
lookup table


corresponding
frequencies


  **x** =(a, b, c, d)


=( 16 _[,]_ 16 _[,]_ 16 _[,]_ 16 [)]

  **f** =(7 _,_ 5 _,_ 3 _,_ 1)


16 [7] _[,]_ 16 [5]


16 [5] _[,]_ [3]


  **p** =( [7] _[,]_ [5]


16 [3] _[,]_ [1]


naive
lookup table


|a|a|a|a|
|---|---|---|---|
|a|a|a|b|
|b|b|b|b|
|c|c|c|d|


|a|b|
|---|---|
|a|c|
|a|b|
|c|d|


|4|4|
|---|---|
|2|2|
|1|1|
|1|1|


(a) Schematic of sampling.


(b) Example (compressed) lookup table.


Figure 1: **a.** Schematic of generating a single sample using our approach: i.i.d. Ber(0 _._ 5) bits are
drawn from an entropy source to compute a row and a column index, yielding in a single lookup on
the precomputed table. **b.** Illustration of the precomputation step: A naive and compressed lookup
table for an example distribution given by **x** =(a, b, c, d) and **p** =( 16 [7] _[,]_ 16 [5] _[,]_ 16 [3] _[,]_ 16 [1] [)][.] [The naive lookup]

table (left table) contains each value according to its frequency **f** =(7 _,_ 5 _,_ 3 _,_ 1) at a precision of _b_ = 4
bits. The compressed lookup table (middle table) stores the same distribution when considering
the geometric frequency scheme (right table). For example, the frequency of “a“ is given by the
compressed lookup table as 4 + 2 + 1 = 7 and thus equals the frequency of “a“ in the naive lookup
table.


[7] [5]

16 _[,]_


[5] [3]

16 _[,]_


[3] [1]

16 _[,]_


preserves the total frequencies


_r_

   - 2 _[i][−]_ [1] _·_ 2 _[c]_ + 2 _[c]_ = (2 _[r]_ _−_ 1 + 1) _·_ 2 _[c]_ = 2 _[r]_ [+] _[c]_ = _N,_


_i_ =1


while drastically decreasing the size of the lookup table. Compressing a naive lookup table with
_N_ = 2 _[r]_ [+] _[c]_ entries to a compressed lookup table with ( _r_ + 1) _·_ 2 _[c]_ entries (organized in _r_ + 1 rows
and 2 _[c]_ columns) yields a compression ratio of _ρ_ = 2 _[r]_ _/_ ( _r_ + 1).

In the example of Figure 1b, the compression ratio would be _ρ_ = 2 [3] _/_ (3 + 1) = 2, which yields a
compressed table half the size of the naive table. The compression ratio _ρ_ improves exponentially
with _r_, up to a linear correction factor of _r_ + 1, with the concrete values of _r_, and therefore of _ρ_,
depending on the frequencies **f** . Intuitively, better compression corresponds to an increase in the
number of rows (larger _r_ ) accompanied by an exponential decrease in the number of columns, resulting in a “taller” and much “narrower” lookup table. This compression scheme is always possible
for lookup tables of size 2 _[b]_ . To see that, note that the compressed and the naive lookup table coincide
for the choice of _r_ =0 and _c_ = _b_ .


**Sampling** **step** To generate a sample _S_ _∈X_ using a compressed lookup table, we generate two
indices independently: a row index _I_ _∈{_ 1 _, . . ., r_ +1 _}_ and a column index _J_ _∈{_ 1 _, . . .,_ 2 _[c]_ _}_ . We
sample the index _I_ according to a truncated geometric distribution, and the column index _J_ uniformly:

P( _I_ = _i_ ) = max(2 _[−][i]_ _,_ 2 _[−][r]_ ) for _i_ = 1 _, . . ., r_ +1 _,_ and P( _J_ = _j_ ) = 2 _[−][c]_ for _j_ = 1 _, . . .,_ 2 _[c]_ _._


Therefore, we sample a table-index ( _I, J_ ) = ( _i, j_ ) with probability 2 _[−]_ [min(] _[i,r]_ [)] _[−][c]_ . The column index
_J_ can be efficiently sampled using any uniform sampler. The row index _I_ can also be sampled
extremely efficient using the entropy optimal procedure detailed in Algorithm 1 in lines 2-8. A
sample is then generated by returning the value stored in the compressed lookup table at that index:


_S_ = compressedTable[ _I, J_ ] _._


**Preprocessing** **step** Before sampling, we must construct the compressed table. Conveniently,
we do not have to construct the uncompressed lookup table, which could induce severe memory
issues. We rather construct the compressed lookup table directly from the binary expansion of the
frequencies _fi_ . A value _xi_ appears in row _j_ if and only if the _j_ -th bit _fi_ [(] _[j]_ [)] of _fi_ is one. The
frequencies **f** can be adjusted to sum to exactly 2 _[b]_ by using a sum-preserving rounding scheme,
making our sampling procedure rejection-free. Although the total probability mass and relative
ratios are preserved in the compressed lookup table, the number of active bits across the binary
representations of the _fi_ may differ, which results in rows of unequal width in the initial construction
of the compressed lookup table. To improve the sampling speed, we ensure that all rows have
uniform width as detailed in Algorithm 2 and Figure 2.


4


**Algorithm 1** Sampling from compressed lookup tables

**Require:** number of samples _K_, compressed lookup table compressedTable of size ( _r_ +1) _×_ 2 _[c]_
**Ensure:** array of samples _S_

1: **for** _k_ = 1 **to** _K_ **do**
2: _I_ _←_ 1
3: **while** _I_ _< r_ + 1 **do** _// Sample row index geometrically:_
4: **if** randomBit() = 1 **then**
5: **break**
6: **end if**
7: _I_ _←_ _I_ + 1
8: **end while**
9: _J_ _←_ Uniform _{_ 1 _, . . .,_ 2 _[c]_ _}_ _// Sample column index uniformly._
10: _S_ [ _k_ ] _←_ compressedTable[ _I, J_ ] _// Generate a sample from the distribution._
11: **end for**
12: **return** _S_


distribution

  **x** = (a, b, c, d, e)


Frq. rectified
Frq. compressed
Frq.—lookup table–

4 a b c a
2 b c d e
1 c d a a
1 a a a a


   **p** = ( [14] _[,]_ [6] _[,]_


[7] [3]

32 _[,]_


[14] [6]

32 _[,]_


[6] [7]

32 _[,]_


[3] [2]

32 _[,]_


32 _[,]_ 32 _[,]_ 32 _[,]_ 32 _[,]_ 32 [)]

  **f** = (14 _,_ 6 _,_ 7 _,_ 3 _,_ 2)


binary expansion

  ( _f_ 1)2 = 1110
( _f_ 2)2 = 0110
( _f_ 3)2 = 0111
( _f_ 4)2 = 0011
( _f_ 5)2 = 0010

  


Frq. initial compressed
Frq.——lookup table—
8 a
4 a b c
2 b c d e a
1 c d
1


Figure 2: The initial and final compressed lookup table for an example distribution given by
**x** =(a, b, c, d, e) and **p** =( [14] 32 _[,]_ 32 [6] _[,]_ 32 [7] _[,]_ 32 [3] _[,]_ 32 [2] [)][.] [In a first step, the table is filled according to the binary]

expansion of the frequencies (left table). Then, the table is rectified by moving entries from higher
to lower rows while doubling (right table). For example, the “a” in the top row corresponding to a
frequency of 8 (blue), is replaced by one “a” in the second row, which corresponds to a frequency of
4, and 4 “a”’s in the bottom rows, which correspond to a frequency of 1, while the “a” in the third
row ( frequency of 2) is replaced by further two “a” in the bottom rows. The total frequency of each
value is preserved, and the rows have equal length. In this case, _b_ = 5 _, r_ = 3, and _c_ = 2.


[14] [6]

32 _[,]_


[6] [7]

32 _[,]_


[7] [3]

32 _[,]_


[3] [2]

32 _[,]_


4 EVALUATION


To demonstrate the advantage of our sampling method, we compared it to state-of-the-art sampling
methods in five experiments.


**Evaluation Setup** All measurements were taken on a standard laptop equipped with an Intel i71255U CPU and 16 GiB DDR4 memory running Ubuntu Linux.


Modern CPUs provide hardware counters that monitor the current power and energy demand. On
the x86 64 platform, _Running Average and Power Limit_ (RAPL; David et al. 2010) counters provide
energy readings at a 1 ms resolution. RAPL is organized into different power domains, representing
different parts of the system. For this work, we focus on the CPU domains _cores_ and _package_ (pkg).
The latter includes the former and additionally other parts of the CPU socket, such as caches and
the memory controller. There are several factors that make energy measurements noisy, apart from
default hardware noise. They include background activities, battery charging, artificial noise against
side-channel attacks for security reasons (Lipp et al., 2021), etc. We limit these influences by disabling CPU security features, keeping the laptop charged, providing additional warm-up rounds, and
measuring multiple iterations. Additionally, we set a constant CPU frequency and CPU core to get
meaningful energy readings, as detailed in the Appendix, Section J. We evaluate all methods (both
in Python and in C) on a fixed set of synthetically generated distributions of sizes _n_ _∈_ [10 [1] _,_ 10 [7] ]
drawn from exponential distributions with varying parameters to span a broad range of entropy values, with zero probabilities removed. The bit precision is set to _b_ = 16 for _n_ _∈_ [10 [1] _,_ 10 [4] ), _b_ = 20
for _n_ _∈_ [10 [4] _,_ 10 [6] ) and _b_ = 23 for _n_ _∈_ [10 [6] _,_ 10 [7] ]. Evaluations on further distributions are in the
Appendix, Section G.


5


cLUT NumPy JAX PyTorch


10 [1] 10 [2] 10 [3] 10 [4] 10 [5] 10 [6] 10 [7]

Number of outcomes n


10 [2] 10 [4] 10 [6]

n


10 20
Entropy H(p)


Figure 3: Comparison of our cLUT approach with standard sampling methods from popular machine
learning libraries (NumPy, JAX, PyTorch). Shown are (1) the average wall time (in seconds) to
generate 10 [7] samples from distributions of varying sizes _n ∈_ [10 [1] _,_ 10 [7] ], (2) the preprocessing time,
and (3) the compression ratios _ρ_ of the cLUT algorithm. Distributions are discretized Exponential
distributions with varying parameters to cover a broad range of entropies, with zero probabilites
excluded. The plots are shown on a log-log scale. Each measurement was repeated ten times and
averaged.


Table 1: Average wall time (in seconds, mean _±_ std) for generating 10 [7] samples and the preprocessing step in Python. Evaluated in two subsets of the distributions from Figure 3, split by size.


**# Outcomes:** _n ∈_ [10 [1] _,_ 10 [5] ) _n ∈_ [10 [6] _,_ 10 [7] )


**Method** **Sampling time (s)** **Preprocessing time (s)** **Sampling time (s)** **Preprocessing time (s)**


NumPy 0 _._ 6680 _±_ 0 _._ 2650 0 _._ 0001 _±_ 0 _._ 0001 9 _._ 6248 _±_ 3 _._ 5823 0 _._ 0308 _±_ 0 _._ 0202
PyTorch 0 _._ 6073 _±_ 0 _._ 2436 0 _._ 0003 _±_ 0 _._ 0006 3 _._ 3768 _±_ 1 _._ 0615 0 _._ 1028 _±_ 0 _._ 0655
JAX 0 _._ 3647 _±_ 0 _._ 1277 0 _._ 2898 _±_ 0 _._ 0894 0 _._ 7982 _±_ 0 _._ 1815 0 _._ 6528 _±_ 0 _._ 0948
cLUT 0 _._ 0374 _±_ 0 _._ 0051 0 _._ 0006 _±_ 0 _._ 0011 0 _._ 1016 _±_ 0 _._ 0129 0 _._ 3925 _±_ 0 _._ 2219


**Sampling speed in Python** We benchmarked our method against standard discrete sampling routines from widely used Python machine learning libraries: RandomGenerator.choice() from
NumPy, multinomial() from PyTorch, and random.choice() from JAX (Harris et al.,
2020; Paszke et al., 2019; Bradbury et al., 2018). As shown in Figure 3, our method achieves a
10–100 _×_ speedup across a wide range of distributions. The performance advantage is most pronounced for distributions with a large number of outcomes: Table 1 reports a 10 _×_ improvement
for distributions with 10 [4] to 10 [5] entries, with the speedup growing to over 100 _×_ already for distributions with 10 [6] to 10 [7] entries. This is particularly relevant when targeting high-precision and
high-diversity random variate generation: a 16-bit data type can already represent 65 _,_ 536 distinct
values, whereas 32-bit and 64-bit types can represent vastly more (over 10 [9] and 10 [19], respectively),
making conventional sampling increasingly inefficient. Furthermore, one should note that the samplers in NumPy, PyTorch, and JAX build on the Inversion method (Devroye, 2006) and produce
distributions that are only approximately similar to the desired distribution, whereas our proposed
method produces exactly the specified distribution.


For evaluations including graphics processing units see the Appendix, Section F, and Figure 7.


**Sampling** **speed** **compared** **to** **SOTA** **implementations** **in** **C** We compare our cLUT sampler
to the following state-of-the-art sampling methods: the Alias method (Walker, 1974), ALDR, and
FLDR (Draper and Saad, 2025). All samplers are implemented in C, as it provides a lower execution overhead compared to, e.g., Python. This allows for proper assessment of the actual costs of the
algorithm. To ensure comparability, we apply the same degree of fair but not over-engineered opti

6


cLUT Alias method ALDR FLDR


2 [4]


2 [3]


2 [2]


2 [1]


10 [2] 10 [4] 10 [6] 10 [8]

Number of outcomes n


30


25


20


15


10


5


10 20
Entropy H(p)


2 [−][24]


2 [−][25]


2 [−][26]


2 [−][27]


10 [2] 10 [4] 10 [6] 10 [8]


Number of outcomes n


2 [3]


2 [−][1]


2 [−][5]


2 [−][9]


2 [−][13]


2 [−][17]


2 [−][21]


10 [2] 10 [4] 10 [6] 10 [8]

Number of outcomes n


Figure 4: Comparison of our cLUT approach with existing state-of-the-art sampling methods in C.
Shown are (1) the wall time required for generating a single sample (averaged over 10 [7] repetitions)
and (2) preprocessing (averaged over 10 repetitions), as well as (3) the cumulative energy demand
of the CPU socket for generating 10 [7] samples. Time and energy are shown on a log-log scale. The
fourth subfigure shows the average consumed bits per sample from the entropy source.


Table 2: Average energy demand, wall time, and power draw of a single sampling operation in C.
The power draw series is computed by dividing the energy series by wall time. It averages over all
CPU instructions of a sampling iteration. High variance in entropy and distribution size results in
the high standard deviation here. Shown are distributions from Figure 4, split by size.


**# Outcomes:** _n ∈_ [10 [3] _,_ 10 [4] ) _n ∈_ [10 [7] _,_ 10 [8] )


**Method** **Energy (nJ)** **Time (ns)** **Power (W)** **Energy (nJ)** **Time (ns)** **Power (W)**


ALDR 263 _._ 451 _±_ 50 _._ 309 22 _._ 431 _±_ 1 _._ 250 11 _._ 836 _±_ 2 _._ 694 1223 _._ 520 _±_ 194 _._ 457 102 _._ 792 _±_ 12 _._ 066 12 _._ 168 _±_ 2 _._ 899
Alias method 319 _._ 804 _±_ 72 _._ 045 26 _._ 803 _±_ 2 _._ 934 12 _._ 156 _±_ 3 _._ 549 887 _._ 653 _±_ 185 _._ 627 55 _._ 946 _±_ 13 _._ 709 16 _._ 502 _±_ 3 _._ 897
FLDR 290 _._ 223 _±_ 47 _._ 274 21 _._ 268 _±_ 2 _._ 373 14 _._ 031 _±_ 3 _._ 770 1214 _._ 382 _±_ 177 _._ 125 101 _._ 404 _±_ 9 _._ 702 12 _._ 195 _±_ 2 _._ 753
cLUT 199 _._ 233 _±_ 38 _._ 579 15 _._ 475 _±_ 1 _._ 689 13 _._ 271 _±_ 4 _._ 091 450 _._ 155 _±_ 74 _._ 604 33 _._ 026 _±_ 4 _._ 880 14 _._ 188 _±_ 4 _._ 188


mization across these methods. We avoid multi-threading or (auto) vectorized code and use identical
compiler flags. All methods use the identical entropy source.


For our experiment, we distinguish between the preprocessing phase (ten repetitions) and the actual
sampling operation (ten million repetitions). The latter can be performed quickly and repeatedly
after the higher, one-time upfront cost. Figure 4 shows our results, indicating that our cLUT method
samples consistently faster for all distributions than our competitors in terms of sampling time.
Table 2 shows mean and standard deviation of the sampling time on two representative subsets of
the distributions ( _n ∈_ [10 [3] _,_ 10 [4] ) and _n ∈_ [10 [7] _,_ 10 [8] )).


**Energy** **consumption** **compared** **to** **SOTA** **implementations** **in** **C** To demonstrate the energysaving potential of our approach, we compare the energy demand of all implementations. Due to
space restrictions, Figure 4 only shows the energy demand of the sampling operation across the
entire RAPL package domain (CPU socket and memory controller), which is representative of the
other measurements. Again, our cLUT approach works best across all sizes.


Although the energy demand roughly follows the same trend as the required time, the scale is not
linear. This is because time is not an accurate indicator of the energy required by complex real-life
computing systems. Rather, energy is the integral of the dynamic power demand over time.


Our cLUT’s single index-based memory lookup requires fewer switching transistors in the memory
subsystem, compared to, e.g., ALDR with multiple memory accesses to a flattened search tree.


7


cLUT ALDR Entropy (H) Time (C) Energy (C)
Alias method FLDR 10 20 Time (Python)


ALDR
FLDR


Entropy (H) Time (C)
10 20
Time (Python)


10 [4] 10 [5] 10 [6] 10 [7] 10 [8]

Number of outcomes n


10 [4] 10 [5] 10 [6] 10 [7] 10 [8]

Number of outcomes n


10 [4] 10 [5] 10 [6] 10 [7] 10 [8]

Number of outcomes n


Figure 5: Comparison of cLUT with state-of-the-art sampling methods in terms of memory usage
and break-even analysis: (1) peak memory usage for all methods (including preprocessing), (2)
memory usage of compressed cLUT table, and (3) break-even analysis against the Alias method. The
break-even point _n_ _[∗]_ is the minimum number of samples needed for cLUT to offset its preprocessing
overhead relative to the Alias method (in terms of sampling time or energy consumption).


**Memory usage and preprocessing overhead** We measure peak memory usage for all approaches
and perform a break-even analysis for sampling time and energy consumption compared to the commonly used Alias method. _Peak memory usage_ refers to the maximum amount of memory utilized
by a program during execution. As shown in Figure 5, the cLUT approach consumes slightly more
memory at peak times than other state-of-the-art algorithms. However, the constructed compressed
lookup tables and therefore the memory usage after preprocessing is relatively small, especially for
low-entropy distributions due to high compression ratios (see Figure 5, middle Figure; and compare
Figure 3 for compression ratios). A break-even analysis against the Alias method shows that this
overhead is offset after a reasonable number of sampling iterations.


As shown in Figure 4, the preprocessing phase (look-up table creation) scales log-linear with the
distribution size across all investigated methods. Our cLUT method shows the highest time demand
for the preprocessing phase and the Alias method the lowest one. Thus, our approach requires more
sampling operations to offset its higher initial costs, but is then more time efficient, especially for
larger distributions. As shown in Figure 5, the break-even point _n_ _[∗]_ for sampling time compared to
the Alias method is approximately linear in the distribution size. For energy efficiency, it ranges from
1 to below 10 [3], indicating that the energy efficiency gains of our algorithm outweigh the increased
preprocessing overhead already for small sampling sizes, even for large distributions.


**Bit** **efficiency** **compared** **to** **SOTA** **algorithms** Sampling algorithms are commonly evaluated
based on the average number of independent fair coin flips (i.e., i.i.d. Bernoulli(0 _._ 5) bits) required to
generate a single sample. Generating a single sample with our cLUT method requires _c_ random bits
to generate the column index _J_ _∈{_ 1 _, . . .,_ 2 _[c]_ _}_ (i.e., uniformly sampling one of the 2 _[c]_ entries in a row,
cmp. Line 9 in Algorithm 1) and between 1 and _r_ bits to generate the row index _I_ _∈{_ 1 _, ..., r_ + 1 _}_
(cmp. Lines 2-8 in Algorithm 1). Since _I_ follows a truncated geometric distribution, the expected
number of random bits required to generate the row index _I_ is [�] _i_ _[r]_ =1 _[i]_ _[·]_ [2] _[−][i]_ [ +] _[r][ ·]_ [2] _[−][r]_ [= 2] _[−]_ [2] _[−]_ [(] _[r][−]_ [1)][.]
Hence, the expected number of required random bits to produce a single sample is _b−r_ +2 _−_ 2 _[−]_ [(] _[r][−]_ [1)] .


Furthermore, 50% of the generated samples consume as few as _b −_ _r_ + 1 random bits, as in these
cases only a single bit is needed to generate the row index. Empirical evaluations indicate that our
method is close to the information-theoretic minimal cost of sampling ( _−_ [�] _i_ _[p][i]_ _[·]_ [log] 2 [(] _[p][i]_ [)][, see Knuth]

and Yao (1976)) and approaches the minimum for high-entropy distributions (see Figure 4).


**Typical values** Discretizing (using the _finite tail extension_ as detailed in the Appendix, Section B)
a standard gaussian distribution to the values of the 16-bit floating point format at a precision of
_b_ = 20 bits (removing values with probability less than 2 _[−]_ [20] ) yields _n_ = 20136 values with nonzero probability, covering 99 _._ 66% of the total probability mass. Applying cLUT yields a compressed
table with _r_ = 6, _c_ = 14, and 114688 entries (229 _._ 38 kB, _ρ_ = 9 _._ 14 _×_ smaller then the uncompressed


8


Table 3: Average energy consumption and wall time for TrueSkill with different sampling methods.


**Method** **mcp (J)** **rapl:cores (J)** **rapl:pkg (J)** **Sampling time (s)**


NumPy’s discrete sampler 201 _._ 05 _±_ 2 _._ 45 91 _._ 19 _±_ 1 _._ 71 116 _._ 31 _±_ 1 _._ 91 1 _._ 65 _±_ 0 _._ 01
NumPy’s continuous sampler 160 _._ 26 _±_ 1 _._ 89 72 _._ 65 _±_ 1 _._ 34 93 _._ 36 _±_ 1 _._ 71 0 _._ 88 _±_ 0 _._ 03
cLUT (ours) _._ **69** _±_ **1** _._ **13** **60** _._ **82** _±_ **0** _._ **72** **77** _._ **99** _±_ **0** _._ **91** **0** _._ **46** _±_ **0** _._ **01**


table). Discretizing a Gamma distribution with parameter _k_ = 2 to the 16-bit format at a precision
of _b_ = 24 bits yields _n_ = 11058 values with non-zero probability, covering 99 _._ 99% of the total
probability mass. Applying cLUT yields a compressed table with _r_ = 11, _c_ = 13, and 98304 entries
(196 _._ 61 kB, _ρ_ = 170 _._ 67 _×_ smaller then the uncompressed table). Values for other precisions and
distributions are shown in Figure 6 in the Appendix, Section B.


4.1 SAMPLING OF UNIFORM FLOATING-POINTS


Besides the proposed cLUT method, our index-based sampling scheme is ideally suited for generating uniformly distributed floating-point numbers over fixed intervals, such as the unit interval

[0 _,_ 1]. Specifically, by considering their binary expansions, we can interpret the row and column
indices generated by our method as the exponent and mantissa of the floating-point representation,
respectively. Using this approach, we achieve truly uniform sampling with maximal coverage of
representable values. In contrast, classic approaches for generating random numbers in fixed intervals cover only a small fraction of all representable numbers in the intervals and oftentimes fail
statistical tests on uniformity (see Appendix K).


4.2 EXEMPLARY APPLICATIONS


In addition to evaluating the algorithm, we aim to show the potential impact of our approach on
real machine learning applications. To reduce overhead and avoid confounding factors, we select a
task in which sampling accounts for a significant share of total energy consumption. One example
of such a task is sampling Bayesian posteriors with non-conjugate priors, and TrueSkill (Herbrich
et al., 2006) system serves as an illustrative case.


The purpose of TrueSkill is to infer posterior skill distributions of players from match outcomes; this
probabilistic machine learning systems currently in use on a large scale. Although the original algorithm is limited to closed-form solutions for Gaussian priors, we extend its applicability to arbitrary
prior distributions through an importance sampling scheme, as detailed in the Appendix, Section H.
This extension enables more flexible modeling of assumptions about the skill distributions, allowing
for non-conjugate priors.


First, we conduct experiments against a fair discrete competitor (RandomGenerator.choice
from NumPy). We then highlight the broader applicability of the approach by testing it against a fast
distribution-specific sampler for a Gaussian mixture (RandomGenerator.normal with mixture
logic from NumPy), showing that our method is effective not only for discrete unparameterized distributions but also for parametrized distributions that are (slightly) more complex than standard ones.
We measure the end-to-end energy demand in the setup detailed in the Evaluation section, recording
core and pkg RAPL domains. As an additional ground truth and better electricity bill proxy, we
include the laptop’s wall socket energy consumption using a Microchip MCP39F511N device. As
a result, our method reduces the total execution time of TrueSkill by 72% and decreases the total
energy consumption by 34% compared to the discrete sampler. Even against the specialized mixture
sampler, cLUT demonstrates competitive performance with a 48% reduction in sampling time and a
17% decrease in overall energy consumption, as shown in Table 4. At the same time, cLUT outputs
near identical posterior distribution as the two NumPy-based methods (see the Appendix H).


Additionally, an exemplary application of cLUT to the training and inference of a diffusion model
is given in the Appendix, Section I.


9


5 CONCLUSION


We present cLUT, a new fast and energy-efficient sampling method for sampling from arbitrary
distributions, based on operations with compressed lookup tables. Time to sample a distribution
speeds up 10-100 _×_ compared to commonly used machine learning Python libraries. It saves up to
50% in energy compared to state-of-the-art methods. We further showcase the value of our sampler
in real-world applications by reducing up to 34% energy consumption and 72% execution time in
the TrueSkill example.


We provide a fairly optimized, robust, and understandable reference implementation of our algorithm in C, as well as a wrapper library that can be used with other programming languages, such
as Python. We have not vectorized or parallelized our implementation to improve understandability
and facilitate comparison with other methods. However, our sampling method only requires a single index-based memory lookup and some arithmetic and bit-shift operations. This makes it better
suited than competing approaches for single instruction, multiple data devices (Flynn, 1972), such
as modern vector and graphics processing units, given a compatible streaming source of entropy.


ACKNOWLEDGMENTS


This research has been conducted through the funding of a research scholarship by the Hasso Plattner
Foundation and was partially funded by the German Research Foundation (DFG) - 502228341
(“Memento”).


10


REFERENCES


L. A. Barroso and U. H¨olzle. The case for energy-proportional computing. _Computer_, 40(12):33–37,
2007. doi: 10.1109/MC.2007.443.


J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke,
J. VanderPlas, S. Wanderman-Milne, and Q. Zhang. JAX: composable transformations of
Python+NumPy programs, 2018. [URL http://github.com/jax-ml/jax.](http://github.com/jax-ml/jax)


H.-C. Chen and Y. Asau. On Generating Random Variates from an Empirical Distribution. _A_ _I_ _I_
_E Transactions_, 6(2):163–166, June 1974. ISSN 0569-5554. doi: 10.1080/05695557408974949.
[URL http://www.tandfonline.com/doi/abs/10.1080/05695557408974949.](http://www.tandfonline.com/doi/abs/10.1080/05695557408974949)


T. Chen, S. Kornblith, M. Norouzi, and G. Hinton. A Simple Framework for Contrastive Learning of Visual Representations, July 2020. URL [http://arxiv.org/abs/2002.05709.](http://arxiv.org/abs/2002.05709)
arXiv:2002.05709 [cs].


H. David, E. Gorbatov, U. R. Hanebutte, R. Khanna, and C. Le. Rapl: memory power estimation
and capping. In _Proceedings_ _of_ _the_ _16th_ _ACM/IEEE_ _International_ _Symposium_ _on_ _Low_ _Power_
_Electronics and Design_, ISLPED ’10, page 189–194, New York, NY, USA, 2010. Association for
Computing Machinery. ISBN 9781450301466. doi: 10.1145/1840845.1840883.


L. Devroye. _Non-Uniform Random Variate Generation_ . Springer New York, New York, NY, 1986.
ISBN 978-1-4613-8645-2 978-1-4613-8643-8. doi: 10.1007/978-1-4613-8643-8. [URL http:](http://link.springer.com/10.1007/978-1-4613-8643-8)
[//link.springer.com/10.1007/978-1-4613-8643-8.](http://link.springer.com/10.1007/978-1-4613-8643-8)


L. Devroye. Nonuniform random variate generation. _Handbooks in operations research and man-_
_agement science_, 13:83–121, 2006.


L. Devroye and C. Gravel. Random variate generation using only finitely many unbiased, independently and identically distributed random bits, Nov. 2020. [URL http://arxiv.org/abs/](http://arxiv.org/abs/1502.02539)
[1502.02539.](http://arxiv.org/abs/1502.02539) arXiv:1502.02539 [cs].


T. L. Draper and F. A. Saad. Efficient rejection sampling in the entropy-optimal range, Apr. 2025.
[URL http://arxiv.org/abs/2504.04267.](http://arxiv.org/abs/2504.04267) arXiv:2504.04267 [cs].


M. J. Flynn. Some computer organizations and their effectiveness. _IEEE Transactions on Comput-_
_ers_, C-21(9):948–960, 1972. doi: 10.1109/TC.1972.5009071.


V. Gadepally. Ai has high data center energy costs - but there are solutions,
2025. URL [https://mitsloan.mit.edu/ideas-made-to-matter/](https://mitsloan.mit.edu/ideas-made-to-matter/ai-has-high-data-center-energy-costs-there-are-solutions)
[ai-has-high-data-center-energy-costs-there-are-solutions.](https://mitsloan.mit.edu/ideas-made-to-matter/ai-has-high-data-center-energy-costs-there-are-solutions)


T. S. Hao and M. Hoshi. Interval algorithm for random number generation. _IEEE Trans. Inf. Theor._,
43(2):599–611, Sept. 2006. ISSN 0018-9448. doi: 10.1109/18.556116. [URL https://doi.](https://doi.org/10.1109/18.556116)
[org/10.1109/18.556116.](https://doi.org/10.1109/18.556116)


C. R. Harris, K. J. Millman, S. J. van der Walt, R. Gommers, P. Virtanen, D. Cournapeau, E. Wieser,
J. Taylor, S. Berg, N. J. Smith, R. Kern, M. Picus, S. Hoyer, M. H. van Kerkwijk, M. Brett,
A. Haldane, J. F. del R´ıo, M. Wiebe, P. Peterson, P. G´erard-Marchant, K. Sheppard, T. Reddy,
W. Weckesser, H. Abbasi, C. Gohlke, and T. E. Oliphant. Array programming with NumPy.
_Nature_, 585(7825):357–362, Sept. 2020. doi: 10.1038/s41586-020-2649-2. URL [https://](https://doi.org/10.1038/s41586-020-2649-2)
[doi.org/10.1038/s41586-020-2649-2.](https://doi.org/10.1038/s41586-020-2649-2)


R. Herbrich, T. Minka, and T. Graepel. Trueskill™: a bayesian skill rating system. _Advances_ _in_
_neural information processing systems_, 19, 2006.


J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. _Advances in neural infor-_
_mation processing systems_, 33:6840–6851, 2020a.


J. Ho, A. Jain, and P. Abbeel. Denoising Diffusion Probabilistic Models, Dec. 2020b. [URL http:](http://arxiv.org/abs/2006.11239)
[//arxiv.org/abs/2006.11239.](http://arxiv.org/abs/2006.11239) arXiv:2006.11239 [cs].


11


M. Horowitz. 1.1 computing’s energy problem (and what we can do about it). In _2014_ _IEEE_
_international_ _solid-state_ _circuits_ _conference_ _digest_ _of_ _technical_ _papers_ _(ISSCC)_, pages 10–14.
IEEE, 2014.


International Energy Agency. Energy demand from ai, 2025. URL [https://www.iea.org/](https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai)
[reports/energy-and-ai/energy-demand-from-ai.](https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai)


D. P. Kingma and M. Welling. Auto-Encoding Variational Bayes, Dec. 2022. URL [http://](http://arxiv.org/abs/1312.6114)
[arxiv.org/abs/1312.6114.](http://arxiv.org/abs/1312.6114) arXiv:1312.6114 [stat].


D. Knuth and A. Yao. Algorithms and complexity: New directions and recent results. Academic
Press, 1976. Section: The complexity of nonuniform random number generation.


E. Le Sueur and G. Heiser. Dynamic voltage and frequency scaling: the laws of diminishing returns.
In _Proceedings_ _of_ _the_ _2010_ _International_ _Conference_ _on_ _Power_ _Aware_ _Computing_ _and_ _Systems_,
HotPower’10, page 1–8, USA, 2010. USENIX Association.


S.-g. Lee, H. Kim, C. Shin, X. Tan, C. Liu, Q. Meng, T. Qin, W. Chen, S. Yoon, and T.-Y. Liu.
Priorgrad: Improving conditional denoising diffusion models with data-dependent adaptive prior.
_arXiv preprint arXiv:2106.06406_, 2021.


M. Lipp, A. Kogler, D. Oswald, M. Schwarz, C. Easdon, C. Canella, and D. Gruss. PLATYPUS:
Software-based Power Side-Channel Attacks on x86. In _2021 IEEE Symposium on Security and_
_Privacy (SP)_ . IEEE, 2021.


J. Lumbroso. Optimal Discrete Uniform Generation from Coin Flips, and Applications, Apr. 2013.
[URL http://arxiv.org/abs/1304.1916.](http://arxiv.org/abs/1304.1916) arXiv:1304.1916 [cs].


G. Marsaglia. Generating discrete random variables in a computer. _Communications of the ACM_, 6
[(1):37–38, Jan. 1963. ISSN 0001-0782, 1557-7317. doi: 10.1145/366193.366228. URL https:](https://dl.acm.org/doi/10.1145/366193.366228)
[//dl.acm.org/doi/10.1145/366193.366228.](https://dl.acm.org/doi/10.1145/366193.366228)


G. Marsaglia, W. W. Tsang, and J. Wang. Fast generation of discrete random variables. _Journal of_
_Statistical Software_, 11:1–11, 2004.


A. J. Martin, M. Nystr¨om, and P. I. P´enzes. Et2: A metric for time and energy efficiency of computation. In _Power aware computing_, pages 293–315. Springer, 2002.


E. Nachmani, R. S. Roman, and L. Wolf. Non gaussian denoising diffusion models. _arXiv preprint_
_arXiv:2106.07582_, 2021.


A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein,
L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy,
B. Steiner, L. Fang, J. Bai, and S. Chintala. PyTorch: An Imperative Style, High-Performance
Deep Learning Library. In _Advances in Neural Information Processing Systems_, volume 32. Cur[ran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper_files/](https://proceedings.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html)
[paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html.](https://proceedings.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html)


F. Saad, C. Freer, M. Rinard, and V. Mansinghka. The Fast Loaded Dice Roller: A Near-Optimal
Exact Sampler for Discrete Probability Distributions. In _Proceedings of the Twenty Third Interna-_
_tional Conference on Artificial Intelligence and Statistics_, pages 1036–1046. PMLR, June 2020.
[URL https://proceedings.mlr.press/v108/saad20a.html.](https://proceedings.mlr.press/v108/saad20a.html) ISSN: 2640-3498.


K. Schwarz. Darts, Dice, and Coins, Dec. 2011. [URL https://www.keithschwarz.com/](https://www.keithschwarz.com/darts-dice-coins/)
[darts-dice-coins/.](https://www.keithschwarz.com/darts-dice-coins/)


M. I. Shamos. _Computational Geometry_ . Yale University., 1978. Ph.D. dissertation.


E. Sommer, J. Robnik, G. Nozadze, U. Seljak, and D. R¨ugamer. Microcanonical langevin ensembles:
Advancing the sampling of bayesian neural networks. _arXiv preprint arXiv:2502.06335_, 2025.


T. Uyematsu and Y. Li. Two algorithms for random number generation implemented by using arithmetic of limited precision. _IEICE Transactions on Fundamentals of Electronics, Communications_
_and Computer Sciences_, E86A:2542–2551, Oct. 2003.


12


A. J. Walker. New fast method for generating discrete random numbers with arbitrary frequency
distributions. _Electronics Letters_, 10(8):127–128, 1974.


P. Wiesner, R. Khalili, D. Grinwald, P. Agrawal, L. Thamsen, and O. Kao. Fedzero: Leveraging
renewable excess energy in federated learning. _arXiv_ _preprint_ _arXiv:2305.15092_, 2023. URL
[https://arxiv.org/abs/2305.15092.](https://arxiv.org/abs/2305.15092)


Z. Yang, L. Meng, J.-W. Chung, and M. Chowdhury. Chasing low-carbon electricity for practical
and sustainable dnn training. _arXiv preprint arXiv:2303.02508_, 2023. [URL https://arxiv.](https://arxiv.org/abs/2303.02508)
[org/abs/2303.02508.](https://arxiv.org/abs/2303.02508)


13


A NOTATION


Table 4: Notation.


**Symbol** **Description**


_b_ Precision of frequencies **f** in bits (e.g., minimal probability is 2 _[−][b]_ )
2 _[c]_ Number of columns in the compressed lookup table
_f_ A vector of frequencies **f** = ( _f_ 1 _, . . ., fn_ ) _∈_ N _[n]_ _≥_ 0 [corresponding to] **[ p]** [ and] _[ b]_
_I_ Row index in cLUT sampling
_J_ Column index in cLUT sampling
_N_ Size of naive lookup table
_n_ Distribution size (number of outcomes)
**p** A vector of probabilities **p** = ( _p_ 1 _, . . ., pn_ ) _∈_ [0 _,_ 1] _[n]_ specifying the target distribution
_H_ ( **p** ) Shannon entropy of **p** specifying the target distribution
_r_ + 1 Number of rows in the compressed lookup table
_ρ_ Compression ratio (size of compressed lookup table divided by size of naive table)
_X_ Domain of sampled values, e.g. the set of representable floating point numbers
**x** A vector of values **x** = ( _x_ 1 _, . . ., xn_ ) _∈X_ _[n]_ specifying the target distribution


14


B DETAILS ON NON-FINITE DISTRIBUTIONS


Many distributions relevant to machine learning belong to the class of continuous, real-valued, univariate distributions, with the Gaussian distribution as a prominent example. These distributions are
discretized in a computational setting, as hardware can only represent a finite set of values.


A natural discretization proceeds as follows. Let a distribution on R be specified via its cumulative density function _F_ . To discretize it on a finite support _X_ _⊂_ R ( _|X|_ _<_ _∞_ ), e.g., the set of
representable numbers in the IEEE 754 16-bit floating-point format, we define the probability mass
function _p_ : _X_ _→_ [0 _,_ 1] of the discretized distribution by


_p_ ( _x_ ) := [1]

_c_


- - _x_ + _x_ +
_F_
2


- - _x_ + _x−_

_−_ _F_
2


��
_,_ _∀x ∈X_ _,_


where _x_ + := min _{y_ _∈X_ : _y_ _>_ _x}_ is the next number to the right of _x_ in _X_, and _x−_ := max _{y_ _∈_
_X_ : _y_ _< x}_ is the next number to the left.


Special care is required for the extrema of _X_ . Let _x_ [max] := max _X_ and _x_ [min] := min _X_ . The next
numbers beyond these limits can be defined in two ways, depending on how you would like to
attribute the probability mass of the tails:


1. _Finite tail extension:_

_x_ [max] + := _x_ [max] + _[x]_ [max] _[−]_ 2 _[x]_ [max] _−_ _,_

+ _[−]_ _[x]_ [min]
_x_ [min] _−_ [:=] _[ x]_ [min] _[ −]_ _[x]_ [min] _,_

2

which requires a normalization constant _c_ = 1 _−_ _F_ ( _x_ [max] + [) +] _[ F]_ [(] _[x]_ _−_ [min][)] [to] [ensure] [that] [the]
discretized probability mass function sums to one.

2. _Infinite tail extension:_ _x_ [max] + := + _∞_ and _x_ [min] _−_ [:=] _[ −∞][,]_ [ in which case] _[ c]_ [ = 1][ suffices.]


Discrete distributions with infinite support, such as the Poisson distribution over N _≥_ 0, also require
truncation to be represented in finite precision. A common approach is to apply a cutoff.


15


Normal(0,1) Normal(0,5) Gamma(k=2) Exponential(λ=1)


16 20 24


16 20 24


16 20 24


16 20 24
Precision b


16 20 24


16 20 24


Figure 6: Typical values for classic continuous distributions when discretized to the 16-bit floating
point range _X_ with a precision of _b_ _∈{_ 15 _, . . .,_ 24 _}_, as described in section 3. Shown are (1) the
number of outcomes _n_, i.e., the number of values with non-zero probability, and (2) the covered
probability mass (sum of all probabilities before normalizing) after rounding to precision _b_ . (3)
Memory consumption, (4) row parameter _r_, (5) column parameter _c_, and (6) achieved compression
ratio _ρ_ of the compressed lookup table.


16


C DETAILS ON APPROXIMATED DISTRIBUTIONS


Since memory constrains impose a boundary on the size _N_ of any lookup table, a lookup table
might suffer from the inability to represent certain probabilities, such as very small or irrational
probabilities, e.g., _pi_ = ~~�~~ 1 _/_ 2. In these cases we fill the table according to the frequencies


_fi_ := round( _pi ·_ 2 _[b]_ ) _∈_ N _≥_ 0 _,_ _i ∈{_ 1 _, . . ., n},_


where round( _·_ ) is an arbitrary sum-preserving rounding scheme. The approximation error of a
distribution stored in a lookup table with probabilities _fi/_ 2 _[b]_ directly depends on the precision _b_, as an
upper bound on the KL divergence can be expressed as a function of min1 _≤i≤n fi_ (see Theorem 1).

**Theorem** **1** (KL-Divergence of approximated distribution) **.** _The_ _KL-Divergence_ _between_ _a_ _distri-_
_bution on_ **x** =( _x_ 1 _. . ., xn_ ) _given by the associated probabilities_ **p** =( _p_ 1 _, . . ., pn_ ) _and the distribution_
_approximated to a precision of b ∈_ N _>_ 0 _bits given by the frequencies_ **f** =( _f_ 1 _, . . ., fn_ ) _is bounded by_


_DKL_ - **p** _||_ **f** - _≤_ log �1 + 2 [1] _κ_


_,_


_where κ_ := min1 _≤i≤n fi._


_Proof._ Write
_pi_ = _fi ·_ 2 _[−][b]_ + _δi,_

with _δi_ _∈_ [ _−_ 2 _[−][b][−]_ [1] _,_ 2 _[−][b][−]_ [1] ]. Then, the KL-Divergence is given by


_D_ KL� **p** _||_ **f** - =


=


=


_≤_


_n_

- _pi_ log _pi_

_i_ =1 _fi ·_ 2 _[−][b]_


_n_

- _pi_ log

_i_ =1


min _i fi_


_n_

- _pi_ log _[f][i][ ·]_ [ 2] _[−][b]_ [ +] _[ δ][i]_

_i_ =1 _fi ·_ 2 _[−][b]_


_n_


_fi ·_ 2 _[−][b]_


_n_

- _pi_ log

_i_ =1


1 + _[δ][i][ ·]_ [ 2] _[b]_


_fi_


1 + [2] _[−][b][−]_ [1] _[ ·]_ [ 2] _[b]_


_n_

- _pi_


_i_ =1


= log


1
1 +
2 min _i fi_


_·_


  = log 1 + [1]

2 _κ_


  = log 1 + [1]


_,_


where _κ_ := min1 _≤i≤n fi_ . In the third step, we used that _δi_ _≤_ 2 _[−][b][−]_ [1] and _fi_ _>_ min _i fi_ for all _i_ .


Note that _κ_ = min1 _≤i≤n fi_ = min1 _≤i≤n_ round( _pi_ _·_ 2 _[b]_ ) and therefore _D_ KL _∈O_ (log(1 + 2 _[−][b]_ )).
Clearly, _D_ KL _→_ 0 for _b_ _→∞_ . However, while approximation error decreases logarithmically with
precision _b_, the lookup table size _N_ required to store all values **x** with their respective frequencies
**f** =( _f_ 1 _, . . ., fn_ ) grows exponentially in _b_ :


_N_ :=


_n_

- _fi_ = 2 _[b]_ _._


_i_ =1


17


D PREPROCESSING DETAILS


A pseudo code of the cLUT preprocessing algorithm that constructs the compressed lookup table is
shown in Algorithm 2. Algorithm 2 calls a the function distribute() in line 4, which is detailed
in pseudo code in Algorithm 3.


**Algorithm 2** Constructing a compressed lookup table

**Require:** probability distribution given by **x** = ( _x_ 1 _, x_ 2 _, . . ., xn_ ) and **f** = ( _f_ 1 _, f_ 2 _, . . ., fn_ ) _∈_ N _[n]_ _≥_ 0
**Ensure:** compressed lookup table compressedTable of size ( _r_ + 1) _×_ 2 _[c]_

_▷_ _Compute optimal r and c:_
1: _b ←_ log2( [�] _[n]_ _i_ =1 _[f][i]_ [)]

2: _r_ _←_ max _{v_ _∈_ [0 _, b_ ] : [�] _j_ _[w]_ =0 - _ni_ =1 _[f]_ _i_ [ (] _[j]_ [)] _·_ 2 _[v][−][b][−]_ [1] _≤_ 1 _∀w_ _∈{_ 0 _, . . ., b}}_
3: _c ←_ _b −_ _r_

_▷_ _Compute counts per row for each value:_
4: _D_ _←_ distribute( **f** _, r, c_ )

_▷_ _Fill compressed lookup table:_
5: compressedTable _←_ [ ]
6: **for** _i_ = 1 to _r_ +1 **do**
7: **for** _j_ = 1 to _n_ **do**
8: **for** _k_ = 1 to _Dji_ **do**
9: compressedTable.append( _xj_ )
10: **end for**
11: **end for**
12: **end for**
13: **return** compressedTable


**Algorithm 3** Distribute counts across bit levels with distribute()

**Require:** frequencies **f** = ( _f_ 1 _, f_ 2 _, . . ., fn_ ) _∈_ N _[n]_ _≥_ 0 [,] _[ r]_ _[∈]_ [N] _[n]_ _≥_ 0 [,] _[ c][ ∈]_ [N] _≥_ _[n]_ 0
**Ensure:** bit levels _D_ _∈_ N _[n]_ _≥_ _[×]_ 0 _[r]_ [+1]

_▷_ _Expand counts into bit-level representation_
1: **for** _i_ = 1 to _n_ **do**
2: **for** _j_ = 1 to _b_ **do**
3: _Dij_ _←_ _fi_ [(] _[j]_ [)]
4: **end for**
5: **end for**

_▷_ _Redistribute bits above level r_
6: **for** _k_ = _b_ downto _r_ **do**
7: **for** _i_ = 1 to _n_ **do**
8: _Dir_ _←_ _Dir_ + 2 _[k][−][r]_ [+1] _· Dik_
9: **end for**
10: **end for**

_▷_ _Adjust lower levels if cumulative sum exceeds_ 2 _[c]_

11: **for** _k_ = _r −_ 1 downto 1 **do**
12: _a ←_ 0
13: **for** _i_ = 1 to _n_ **do**
14: _a ←_ _a_ + _Dik_
15: **if** _a >_ 2 _[c]_ **then**
16: _δ_ _←_ _a −_ 2 _[c]_

17: _Dik_ _←_ _Dik −_ _δ_
18: _Di_ ( _k−_ 1) _←_ _Di_ ( _k−_ 1) + 2 _· δ_
19: **end if**
20: **end for**
21: **end for**
22: **return** ( _Dij_ ) _j≤r_ +1


18


E IMPLEMENTATION DETAILS


We implemented the preprocessing and sampling methods in C and reused the computed data structures in Python. To do so, we created a foreign function library that conveniently interfaces between
C and other languages. This library is used in our evaluation.


Like the reference implementation of ALDR and FLDR (Draper and Saad, 2025), we used bit operations, compiler intrinsics and linearized arrays where possible to ensure fast computation. We
extended the existing SOTA implementations to also work with 64-bit input values to make them
comparable with our test distributions.


Our implementation, wrapper library and changes to existing SOTA implementations are publicly
available on GitHub under (omitted for blind review).


19


F JAX INTEGRATION AND GPU EVALUATION


To demonstrate the integratability of cLUT as well as potential performance gains from SIMD implementations, we have integrated our cLUT approach into the JAX library, as shown in Listing 1,
and compared with the default sampling method from JAX, see Figure 7. This experiment was run
on a single A100 GPU, using JAXs internal GPU mechanisms.


Listing 1: Integration of cLUT into the JAX library.


1 # jax._src.random.py

2 from jax._src import numpy as jnp

3 from jax._src import prng

4 ...


5

6 def choice(key: ArrayLike,

7 a: int | ArrayLike,

8 shape: Shape = (),

9 replace: bool = True,

10 p: RealArray | None = None,

11 # ---- CHANGES ---
12 b = -1,

13 c = -1,

14 # ---- END OF CHANGES ---
15 axis: int = 0,

16 mode: str | None = None) -> Array:

17 ...

18 if replace:

19 # ---- CHANGES ---
20 return _choice(arr, key, c, b, shape, dtype)

21 # ---- END OF CHANGES ---
22 else:

23 ...


24

25 # ---- CHANGES ---
26 @partial(jax.jit, static_argnames=[’b’, ’shape’, ’dtype’])

27 def _choice(arr, key, c, b, shape, dtype):

28 mask = (1 << c) - 1

29 B = prng.random_bits(key, b, shape)

30 return jnp.take(arr, (clz(B | mask) << c) | (B & mask), 0)

31 # ---- END OF CHANGES ---

JAX with cLUT JAX (default)


10 [2] 10 [4] 10 [6] 10 [8]

Number of outcomes n


Figure 7: Comparison of our cLUT approach integrated into the JAX library with the default sampling method from JAX on GPU. Shown is the average wall time (in seconds) to generate 10 [7]
samples from distributions of varying sizes _n_ _∈_ [10 [1] _,_ 10 [8] ]. Distributions were extracted from exponential distributions with varying parameters (and shuffled) to cover a broad range of entropies,
using variable precisions _b_ _∈_ [4 _,_ 30]. The plots are shown on a log-log scale. Each measurement
was repeated ten times and averaged.


20


G ADDITIONAL EVALUATIONS


cLUT NumPy JAX PyTorch


10 [4] 10 [5] 10 [6]

Number of outcomes n


10 [4] 10 [5] 10 [6]

n


0 10
Entropy H(p)


Figure 8: Comparison of our cLUT approach with standard sampling methods from popular machine
learning libraries (NumPy, JAX, PyTorch). Similar to Figure 3, but evaluated on sparse distributions.
Shown are (1) the average wall time (in seconds) to generate 10 [7] samples from distributions of
varying sizes _n_ _∈_ [10 [4] _,_ 10 [7] ], (2) the preprocessing time, and (3) the compression ratios _ρ_ of the
cLUT algorithm. Distributions were sampled from Dirichlet priors with varying parameters to cover
a broad range of entropies, using a fixed precision of _b_ = 16. The plots are shown on a log-log scale.
Each measurement was repeated five times and averaged.


Table 5: Average wall time (in seconds, mean _±_ std) for generating 10 [7] samples and the preprocessing step. Evaluated in two subsets of the distributions from Figure 8, split by size.


**# Outcomes:** _n ∈_ [10 [4] _,_ 10 [5] ] _n ∈_ [10 [6] _,_ 10 [7] ]


**Method** **Sampling time (s)** **Preprocessing time (s)** **Sampling time (s)** **Preprocessing time (s)**


NumPy 0 _._ 847 _±_ 0 _._ 288 **0** _._ _±_ **0** _._ 7 _._ 230 _±_ 4 _._ 138 **0** _._ _±_ **0** _._ PyTorch 0 _._ 720 _±_ 0 _._ 253 0 _._ 001 _±_ 0 _._ 001 2 _._ 400 _±_ 1 _._ 138 0 _._ 070 _±_ 0 _._ 039
JAX 0 _._ 407 _±_ 0 _._ 132 0 _._ 335 _±_ 0 _._ 087 0 _._ 616 _±_ 0 _._ 250 0 _._ 572 _±_ 0 _._ 105
cLUT (ours) **0** _._ _±_ **0** _._ 0 _._ 001 _±_ 0 _._ 001 **0** _._ _±_ **0** _._ 0 _._ 177 _±_ 0 _._ 090


Table 6: Average energy demand, wall time, and power draw of a single sampling operation. The
power draw series is computed by dividing the energy series by wall time. It averages over all CPU
instructions of a sampling iteration. High variance in entropy and distribution size results in the high
standard deviation here. Shown are distributions from Figure 9, split by size.


**# Outcomes:** _n ∈_ [10 [4] _,_ 10 [5] ] _n ∈_ [10 [6] _,_ 10 [8] ]


**Method** **Energy (nJ)** **Time (ns)** **Power (W)** **Energy (nJ)** **Time (ns)** **Power (W)**


ALDR 240 _._ 65 _±_ 73 _._ 91 18 _._ 83 _±_ 4 _._ 75 13 _._ 21 _±_ 4 _._ 28 225 _._ 38 _±_ 53 _._ 64 19 _._ 56 _±_ 4 _._ 56 12 _._ 29 _±_ 4 _._ 61
FLDR 221 _._ 25 _±_ 62 _._ 66 20 _._ 08 _±_ 4 _._ 99 11 _._ 45 _±_ 3 _._ 75 204 _._ 54 _±_ 47 _._ 07 20 _._ 83 _±_ 4 _._ 14 10 _._ 24 _±_ 3 _._ 56
Alias 180 _._ 58 _±_ 73 _._ 82 17 _._ 94 _±_ 7 _._ 82 **10** _._ **21** _±_ **0** _._ **87** 315 _._ 04 _±_ 93 _._ 36 33 _._ 89 _±_ 9 _._ 94 **9** _._ **32** _±_ **0** _._ **48**
cLUT (ours) _._ **09** _±_ **18** _._ **34** **14** _._ **15** _±_ **1** _._ **58** 10 _._ 22 _±_ 1 _._ 06 _._ **23** _±_ **17** _._ **05** **13** _._ **41** _±_ **1** _._ **75** 9 _._ 60 _±_ 0 _._ 94


21


cLUT FLDR ALDR Alias method


2 [−][24]


2 [−][25]


2 [−][26]


2 [−][27]

10 [4] 10 [6] 10 [8]

Number of outcomes n


2 [2]


2 [−][1]


2 [−][4]


2 [−][7]


2 [−][10]


2 [−][13]


10 [4] 10 [6] 10 [8]

Number of outcomes n


0 10
Entropy H(p)


2 [−][1]


2 [−][2]


2 [−][3]


10 [4] 10 [6] 10 [8]

Number of outcomes n


25


20


15


10


5


0


Figure 9: Comparison of our cLUT approach with existing state-of-the-art sampling methods. Similar to Figure 4, but evaluated on sparse distributions described in Figure 8. Shown are (1) the wall
time required for generating a single sample (averaged over 10 [6] repetitions) and (2) preprocessing
(averaged over 10 repetitions), as well as (3) the cumulative energy demand of the CPU socket for
generating 10 [6] samples. Time and energy are shown on a log-log scale. The fourth subfigure shows
the average consumed bits per sample from the entropy source.


cLUT ALDR Entropy (H) Time (C) Energy (C)
FLDR Alias method 5 10 15 Time (Python)


ALDR
Alias method


Entropy (H) Time (C)
5 10 15
Time (Python)


10 [4] 10 [6] 10 [8]

Number of outcomes n


10 [4] 10 [6] 10 [8]

Number of outcomes n


10 [4] 10 [6] 10 [8]

Number of outcomes n


Figure 10: Comparison of cLUT with state-of-the-art sampling methods in terms of memory usage
and break-even analysis. Similar to Figure 5, but evaluated on sparse distributions described in
Figure 8. (1) peak memory usage for all methods (including preprocessing), (2) memory usage of
compressed cLUT table, and (3) break-even analysis against the Alias method. The break-even point
_n_ _[∗]_ is the minimum number of samples needed for cLUT to offset its preprocessing overhead relative
to the Alias method (in terms of sampling time or energy consumption).


H DETAILS ON TRUESKILL


Our TrueSkill extension uses importance sampling as follows: (1) independently sample skills _si_
and performances _yi_ from their respective priors, (2) compute importance weights as the product of
prior densities and match likelihood, and (3) use these weights to estimate posterior distributions.
Independent sampling of correlated variables enables parallelization while maintaining correctness
through importance weighting (Algorithm 4). We discretize the continuous bimodal prior over the
range [ _−_ 10 _,_ 10] with resolution 10 _[−]_ [3] and construct cLUT tables with _b_ = 32 bit precision.


To evaluate the precision of the posterior distribution sampled by cLUT, we ran the TrueSkill algorithm 50 times using both the NumPy-based continuous sampler and the cLUT sampler. Considering


22


**Algorithm 4** TrueSkill with importance sampling for two players


**Require:** prior skills distributions _π_ 1( _θ_ 1) and _π_ 2( _θ_ 2), performance standard deviation _β_, match
outcome data _R_
**Ensure:** posterior skills distributions _π_ 1( _θ_ 1 _|R_ ) and _π_ 2( _θ_ 2 _|R_ )

1: **for** _i_ = 1 **to** _N_ **do**
2: _s_ 1 _←_ _π_ 1( _θ_ 1) _, s_ 2 _←_ _π_ 2( _θ_ 2)
3: _y_ 1 _←G_ (1 _, β_ ) _, y_ 2 _←G_ (1 _, β_ )

_▷_ _Compute match outcome:_
4: _r_ = I _y_ 1 _>y_ 2

_▷_ _Compute importance sampling weights:_
5: _w_ 1 = _pπ_ 1 _|θ_ 1( _s_ 1) _, w_ 2 = _pπ_ 2 _|θ_ 2( _s_ 2)
6: _w_ 3 = _pG_ ( _s_ 1 _,β_ )( _y_ 1) _, w_ 4 = _pG_ ( _s_ 2 _,β_ )( _y_ 2)

7: _w_ = _r ·_ [�] _i_ [4] =1 _[w][i]_

_▷_ _Write down the results to arrays S_ 1 _, S_ 2 _, W_ _:_
8: _S_ 1[ _i_ ] = _s_ 1 _, S_ 2[ _i_ ] = _s_ 2 _, W_ [ _i_ ] = _w_
9: **end for**

_▷_ _Assign new posterior distribution as probability mass function:_
10: _π_ 1( _θ_ 1) _|R_ ) := _{_ ( _S_ 1[ _i_ ] _, W_ [ _i_ ]) _}_ _[N]_ _i_ =1
11: _π_ 2( _θ_ 2 _|R_ ) := _{_ ( _S_ 2[ _i_ ] _, W_ [ _i_ ]) _}_ _[N]_ _i_ =1
12: **return** _π_ 1( _θ_ 1 _|R_ ) _, π_ 2( _θ_ 2 _|R_ )


that these two samples operate on different domains, we cannot employ test that compare density
functions. For this reason, we evaluate sampled results by comparing first and second moments. For
each iteration, we computed the mean and variance of a player’s skill posterior distribution. We then
applied a t-test to assess statistically significant differences in means and variances between the two
samplers, obtaining p-values greater than 0.2 in both cases, meaning that the moments of sampled
distributions do not have meaningful differences.


23


I APPLICATION TO DIFFUSION MODELS


To demonstrate cLUT’s impact in a core ML problem, we apply cLUT to a small-scale generative
model. We train and validate a toy diffusion model Ho et al. (2020a) designed to learn a noise
distribution from corrupted data. In our experiments, we generate the training data from a bimodal
distribution (green line in the Figure 11) and introduce corruption through another bimodal distribution. While the original algorithm assumes training and inference with Gaussian noise, previous
work has shown that reducing the difference between the data and noise distributions can improve
the precision of a model Lee et al. (2021). Additionally, using a Gaussian mixture can be a beneficial
replacement for certain tasks Nachmani et al. (2021). Our additional experiments are consistent with
these findings: when training on bimodal data, using Gaussian noise results in a substantially larger
Wasserstein distance between generated samples and the training distribution (greater than 1), while
using bimodal noise reduces this distance to below 0.07.


Sampling is employed to simulate noise during both training and inference. We incorporate cLUT
in both stages and compare its performance and energy consumption against the default sampling in
JAX, as JAX was the most efficient library in our main evaluation. For the CPU evaluation, we use
the same hardware setup as described before. We define a shallow neural network with two linear
layers and train on small batches of 8 samples for 3 _×_ 10 [5] steps. For the inference stage, we run
the trained model for 2 _×_ 10 [3] iterations with the same batch size. To sample noise with the cLUT
algorithm, we construct a table with a fixed precision of _b_ = 8. This preprocessing cost is included
in the evaluation of the overall application’s time and energy consumption.


Table 7 shows that incorporating cLUT can save energy by 37% in the training stage and by 65% in
the inference stage compared to the default sampler of JAX. Additionally, to validate the quality of
generated samples, we compare the output of the inference stages using the two different sampling
algorithms, utilizing a model trained with JAX’s default sampler. As shown in Figure 11, the two
samplers return nearly identical distributions for the generated data, with a Wasserstein distance
from training data to samples of 0.069 for JAX’s default and 0.054 for cLUT, respectively.


Table 7: Comparison of our cLUT approach with JAX incorporated into training and inference
processes of a denoising diffusion model.


**Training** **Inference**


**Method** **rapl:cores (J)** **rapl:pkg (J)** **Time (s)** **rapl:cores (J)** **rapl:pkg (J)** **Time (s)**


JAX 3811.42 _±_ 102.38 4702.91 _±_ 86.61 265.77 _±_ 5.29 331.63 _±_ 3.67 403.29 _±_ 6.53 19.89 _±_ 0.22
cLUT 2392.14 _±_ 111.05 2962.17 _±_ 102.96 172.62 _±_ 5.11 116.11 _±_ 1.97 143.08 _±_ 3.06 7.28 _±_ 0.08
Reduction with cLUT 37.2% 37.0% 35.0% 65.0% 64.5% 63.4%


0.75


0.50


0.25


0.00


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|Generated s<br>Generated s<br>Training da|amples (cLUT)<br> amples (JAX)<br> ta|||
|||||
|||||
|||||


−5 0 5
Value


Figure 11: Generated data by the denoising diffusion model with cLUT and JAX sampling algorithms and comparison to the target data.


24


J DETAILS ON ENERGY EFFICIENCY


It is crucial to understand different metrics and their relation to assess the efficiency of modern
(electrical) computing systems and design experiments. While power is the rate at which electricity
is consumed at a given point in time, energy is the amount of electricity required to perform an
operation (power’s integral over time). Electric energy translates to battery life, electricity bills or
emitted carbon dioxide, making it the most reasonable metric to optimize for when seeking _energy_
_efficiency_ .


An exception would be if the computer system has actively changing clock frequencies. Apart from
the number of active switching transistors, the CPU’s clocking frequency and supply voltage play
into the dynamic power demand at a given point in time (Le Sueur and Heiser, 2010). In this case,
the energy-delay-squared product (Martin et al., 2002) would be a more suitable metric, combining
execution time and energy demand.


Even at fixed clock rates, switching between CPU architectures can significantly alter power demand
but not necessarily energy demand. A low-power device (a micro-controller or efficiency CPU
core) can run for a longer time than a more power-intense one, resulting in comparable energy
integrals—or not, depending on the static power demand and thus energy proportionality of the
system (Barroso and H¨olzle, 2007). For a fixed problem size, the latter device can switch to idle
mode after completion or process more elements for a given unit of energy. Consequently, to obtain
more representative measurements, we fixed the CPU frequency and micro-architecture (cores) in
our experiments. As our particular Intel _Hybrid_ CPU architecture comprises of larger **p** erformance
cores and limited **e** fficiency cores, we opted for the P-cores for consistent measurements.


There is a direct connection between the memory access behavior of modern computer systems
and their electricity consumption (Horowitz, 2014). Memory subsystems and CPU caches have
long been overlooked in comparison to computational cores but constitute a large portion of active
transistors in today’s chip designs, leading to higher dynamic power demands. This means that,
for general-purpose computers, algorithms that trade computation for memory lookups may have
slightly worse energy efficiency than plain recomputation. This effect is more pronounced with
multiple, nested lookups (also known as _pointer chasing_ ) because it involves more active transistors,
which increases power demand. It also breaks CPU cache locality and access prediction, resulting in
prolonged CPU stalls (increased time demand) and thus non-linear increase in energy demand. This
motivates our idea to create a compression strategy for a lookup table that preserves all the statistical
properties of sampling with simple lookup tables but reduces energy consumption.


25


K DETAILS ON SAMPLING OF UNIFORM FLOATING-POINTS


In the IEEE 754 floating-point format, numbers are organized into dyadic intervals of exponentially
increasing size, each containing a fixed number of equally spaced values. This structure makes
our index-based sampling scheme ideally suited for generating uniformly distributed floating-point
numbers over fixed intervals, such as the unit interval [0 _,_ 1]. Specifically, by considering their binary
expansions, we can interpret the row and column indices generated by our method as the exponent
and mantissa of the floating-point representation, respectively. Using this approach, we achieve truly
uniform sampling with maximal coverage of representable values.


In contrast, the classic approach of generating uniformly random mantissa bits to obtain a float in

[1 _,_ 2), and then subtracting 1 covers only a small fraction of all representable numbers, approximately 13%. PyTorch’s common method for generating random variates uniformly on the interval

[0 _,_ 1] is torch.rand(). When generating values directly in 16-bit floating-point format, this
method covers only 13 _._ 3% of all representable values in [0 _,_ 1]. A Pearson’s _χ_ [2] test for uniformity
fails significantly, yielding _χ_ [2] = 1 _,_ 277 _,_ 749 _,_ 854 _._ 249 with _p_ _<_ 10 _[−]_ [10] . Alternatively, generating
values in 32-bit floating-point format and converting them to a 16-bit representation results in 100%
coverage of 16-bit floating-point values in the unit interval. However, this approach also fails the
Pearson’s _χ_ [2] test, with _χ_ [2] = 21 _,_ 425 _._ 2924 and _p <_ 10 _[−]_ [10] .


26