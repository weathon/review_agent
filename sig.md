Models via Avian Message Passing Interfaces
Jared Fernandez*
Red Hawk Institute, Carnegie Mellon University
Pittsburgh, PA, United States
Amanda “Birdtsch” Bertsch*
Red Hawk Institute, Carnegie Mellon University
Pittsburgh, PA, United States
Index Terms—avian, distributed systems, fault tolerance, concurrency, biological neural networks
I. INTRODUCTION
Modern artificial intelligence applications have been enabled by the development of large artificial neural networks
that benefit from empirical scaling laws which incentivize
models of ever increasing size. However, as the size of models
have grown, it has become necessary to leverage growing
numbers of parallel processors. The extent of model growth
has made it such that training can no longer take place within
a single facility. As we approach artificial general intelligence,
it will become the case that the models are so large that
computation exceeds that of what is available in a single
data center – requiring innovations in algorithmic methods
and system design [?]. At this scale, novel challenges emerge
in order to (1) coordinate computation, (2) minimize interfacility communication overhead, and (3) ensure that both the
computation and communication systems are fault-tolerant.
Unfortunately, due to the complexity of the underlying
parallel computing hardware and hierarchical network topology, developing infrastructure for distributed systems and
training infrastructure at-scale remains an open challenge for
the research community. We hypothesize that the scale and
inefficiency of modern deep learning systems has grown so
large that we can instead leverage biological neural networks
as drop-in replacements for key components of state-of-the-art
machine learning distributed training systems.
Previous works have sought to estimate costs via models of
kernel execution time; however these are idealized settings. In
this work, we develop a model based on an ensemble of biological neural networks based on existing agent architectures.
As is common in modern distributed training of neural networks, we consider a computing topology in which
large model training is coordinated across multiple nodes of
GPU accelerators. In particular, we examine the setting in
which model size is sufficiently large such that accelerators
are distributed across multiple computing facilities (i.e. data
centers); and communication of data is required over large
real-world physical distances, a setting known to exhibit
* Equal Contribution. Author order determined by highest score in Flappy
Bird.
Fig. 1: A common misconception is that birds draw power
from the electric grid via wire perching. In fact, birds run on
power derived from a biological process of digestion.
packet loss. In this regime, it is not possible to utilize the
high-bandwidth, high-speed networking infrastructure highperformance computing clusters (e.g. NVLink, Infiniband,
ROCE); instead data must be transferred utilizing standard
communication protocols over the internet. However, these
methods for data transmission are known to be financially
costly, power-intensive, and lossy. Can we do better?
To address these concerns, we develop an organic, lowpower approach for data transmission in large-scale distributed
systems utilizing avian-based communication protocols, which
we refer to as Avian Message Parsing (AMP; See Figure 2).
Birds are notorious for requiring minimal electricity relative
to their electrical and optical telecommunication-based counterparts (Fig 1); and are capable of supporting the weight of
physical memory hardware (e.g. a raven is capable of carrying
a hard disks; as described in Section ??).
II. PROPOSED METHOD: AVIAN MESSAGE PARSING
We introduce Avian Message Parsing (AMP), a biologically
inspired replacement for electrical and optical networking
infrastructure. In a standard distributed system, we replace
electrical and optical inter-facility interconnects with avian
messengers (i.e. birds) which travel between computing units
and facilities via flight; bypassing bottlenecks and sub-optimal
routing required by land-based infrastructure. In our proposed
architecture, data center operators and ornithologists attach
hard disks with model predictions and intermediate activations
8
49
(a) Traditional GPU topology for distributed computation which leverages costly
electrical infrastructure for data transmission.
(b) Proposed Avian Message Parsing which utilizes organic, low-power data
transmission for increased efficiency deep learning training.
Fig. 2: Comparison of GPU-GPU communication with standard electrical and improved bird-based infrastructure.
to trained birds at data centers; which then fly to sister
data centers carrying other components of the model parameters and computation; where receiving operators integrate
the received information into their compute infrastructure.
Furthermore, we direct our avian messengers to use existing
convection patterns in the Earth’s atmosphere to accelerate
transmission speeds. Under this configuration, data centers no
longer need to be densely co-located in a single region (e.g.
Northern Virginia), a practice known to induce strain on local
electrical grid infrastructure; potentially increasing the electricity costs to rate-payers. Instead, data centers can be placed
freely along wide swaths of regions along naturally occuring
thermal cells and take advantage of renewable resources and
wind patterns.
A. Transfer to persistent storage
Unfortunately, there are no power outlets in the sky so it
necessary to offload data to persistent storage such as solidstate and hard disks as it is impossible for GPUs to retain
the values of weights and activations without power. While it
would be ideal to use cloud storage to write and retrieve data,
such technology does not exist as clouds are made of water
and are incapable of holding the weight of physical hard disks.
As a result, it is necessary for technicians to offload data
from VRAM to persistent memory storage devices to birds at
computing facilities.
Fig. 3: The ring-billed gull, widely considered to be the first to
implement ring attention in the wild. Its effective data transfer
speed is approximately faster than a snail.
B. Bandwidth and data weight
Birds are limited by their small little wings and thus cannot
carry all of the weights required.
Luckily, people have spent a lot of time thinking about what
could reasonably be strapped onto birds for a different type of
science. Biologists regularly affix transponders onto wild birds
to track nesting locations, migration patterns, survival rates,
and other community statistics.
A common choice is to limit transponder weight to ≤ 5%
of the bird’s body weight (or the mean body weight of that
species). [?]
However, transponder weights are designed to be minimal
so that birds are not affected by the device’s placement and
with the understanding that birds may carry other loads (such
as nesting materials or prey) while the transponder is carried.
Since the data transfers are relatively short (often only a few
hours to days) and transient, we can allocate much heavier
loads.
To find a reasonable load value, we consider homing pigeons. Homing pigeons are frequently cited as carrying up to
75g with training, when they only weigh 315-425g This is a
load-to-weight ratio of 0.1875, for a 400g bird.
C. Directionality of data transfer
In contrast to traditional networking communications that
assume bandwidth speeds are constant regardless of directionality, AMP encounters real-world physical limitations in
regards to the transmission topology of data.
1) Ring Attention: Unfortunately, avian message transmitters are known to exhibit a condition known as “bird-brain”
in which their spatial senses are limited; and it is a commonly
observed phenomena in which birds are only able to home in
a single direction effectively.
2) Thermals: In our proposed architecture, we leverage
naturally occurring thermals, air pathways of high velocity,
which enable faster flight by our transmitters. However, these
3) Existing migratory pathways: We can limit the amount
of training necessary by working along existing migration
paths.
50
(a) Red-tailed hawk (b) Airbus A320
(c) Recently cleaned window (d) Gray rat snake
Fig. 4: Packet loss in AMP can be caused by a number of
environmental factors.
D. Data transfer speed
Bandwidth is measured in FLAPs (flown logs by aviation
post, per second). To measure bandwidth, we must look at the
flying speed of birds and their distance per day covered.
E. Packet loss
All data transfers incur some risk of data loss. AMP is
robust to power grid failure, civil unrest, road closures, and
most minor solar flares;1 However, AMP also introduces new
potential sources of packet loss. Long-haul bird flight carries a
number of inherent risks. Figure 4 demonstrates four common
environmental hazards for small birds.
As a proxy for expected packet loss rates on AMP, we
consider a more well-studied bird transit pattern: seasonal
migration. Seasonal migration is not a perfect proxy– migrations are much longer trips than datacenter transfers, typically
performed in weather that is marginal for that bird’s flight, and
without safe overnight nesting locations. Despite this, many
species have quite low migration loss rates [?].
We take as an example the Greater Snow Goose, which
has a particularly punishing autumn migration from the North
Arctic, often featuring freezing temperatures. While the exact
staging locations during this part of the snow goose migration
are not well understood, this is a migration of no less than
3,000km from Bylot Island to a St. Lawrence River estuary,
undertaken in approximately 5 weeks [?]. goosepath state a
monthly survival rate of adult Greater Snow Geese during the
migration period of 98.9%.2 Given these numbers, we compute
an approximate per-km expected loss of 4.58 × 10−6 birds.
Note that because of the particularly difficult conditions of
1A sufficiently strong solar flare, however, can cause packet redirection [?].
2The juvenile monthly survival rate, on the other hand, is an astonishingly
low 66.2%. For the purposes of this work, we assume that AMP is not using
child bird labor.
this migration and the likelihood that the geese take an indirect
path much longer than 3,000 km, this should be considered a
very weak lower bound on packet loss.
How does this compare to existing systems? In some
situations (e.g. voiceover), packet loss up to 5-10% may
be acceptable [?]. An acceptable Ethernet packet loss rate
has elsewhere been listed as 1-2% Using the lower bound
calculated above, we can estimate that goose AMP over
distances less than 21,818km will have a less than 1% packet
loss rate; this means than goose data transfer over half the
circumference of the Earth has an acceptably low packet loss
rate.
1) Bird vs. flock failure rates: Of course, we care not just
about total packet loss rate but the grouping of packets lost.
Because we are considering each bird to be a “packet,” the size
of birds chosen impacts the amount of data lost if a single bird
is lost.
2) Reduction of packet loss via cross-bird data redundancy:
The above analysis assumes that each bird carries completely
unique data. Of course, in practice, some redundancy in
the system is advised. We recommend sharing the desired
data to transfer across all birds in the flock by maintaining
two copies of each data point, distributed to two different
birds. Rather than making duplicates of each physical storage
device’s contents, we share sub-packets of smaller data units,
e.g. weights for an individual matrix. Each bird’s data storage
device must also maintain a lightweight registry of IDs for
weight matrices in the total data transfer.
This has two functions: first, it reduces the likelihood of any
catastrophic data loss event. Second, it also increases FLAPs
over the total data transfer: because the data is shared across
all birds and we maintain a listing of the total data expected
within each bird’s storage, it is trivial to verify when all data
has arrived, even if not all birds have yet arrived. This means
that, on average, the data transfer can be considered complete
when half the flock has arrived, reducing the odds of slowdown
because of a lost, delayed, or simply lazy bird.
III. LIMITATIONS
We believe our proposed system provides a strong model
for estimating the cost and efficiency of machine learning systems. Unfortunately, its feasibility suffers because of several
technological limitations of our time.
Most data storage devices, strangely, are not well-suited
for transport in cold, potentially wet conditions, despite being
regularly used for “cloud storage.” This discrepancy is not
well-understood; at least, we emailed Sandisk and they didn’t
have any answers for us.
Additionally, our work relies on a strong assumption that
birds are real, which has been recently called into question in
the scientific community [1].
REFERENCES
[1] Shoemaker, Lauren. ”Birds Aren’t Real.” Avian Aesthetics in Literature
and Culture: Birds and Humans in the Popular Imagination (2022): 21
