Phase 1: The Womb
A Coherence-Based Framework for Developmental Perception
"Learning can be seen as a process of discriminating patterns in the world, as opposed to one of
supplementing sensory information with past experience."
— William W. Gaver, Technology Affordances (1991)
Dio Brown
Jack Kent Cooke Scholar
Syracuse University
January 2026
Phase 1: The Womb
A Coherence-Based Framework for Developmental Perception
Author: Dio Brown | Date: January 2026
Table of Contents
1. Overview
2. Core Concepts
2.1 Coherence
2.2 Comfort
2.3 Clusters
2.4 The 100% Reference
3. Perception
3.1 Visual Processing
3.2 Auditory Processing
3.3 Multimodal Binding
4. Behavior
4.1 Exploration
4.2 Contemplation
4.3 Urgency and Retreat
5. Development
5.1 The Developmental Arc
5.2 Environment Design
6. Implementation
6.1 Sigmoid Functions
6.2 Parameters
7. Open Questions
8. References
9. Environment Architecture
9.1 Overview
9.2 The Emergent as Camera2D
9.3 Room Structure
9.4 Movement
9.5 Door Design and the Threshold Experience
9.6 Why This Architecture Is Correct
10. The Developmental Arc: Phase 1 Through Phase 3
10.1 Overview
10.2 Phase 1: The Womb (Current)
10.3 Phase 2: The 3D World
10.4 Phase 3: Embodiment
10.5 What Each Phase Hands to the Next
10.6 Phase 1 Design Constraint: Extensibility
1. Overview
The Womb is a controlled environment designed to nurture artificial perception. Within it develops an Emergent: a
system whose cognitive architecture emerges through experience rather than explicit programming.
Thesis: A single principle, the drive toward coherence, produces two complementary behaviors: exploring the
unknown and returning to the known. Together, these create a system that continuously cycles between expansion
and consolidation.
Everything flows from one idea: the system seeks coherence, and coherence produces comfort.
2. Core Concepts
2.1 Coherence
Coherence is how well things fit together, how deeply the system understands what it encounters.
Level Question Computation
Local How well do I understand this thing? Cluster depth, confirmation count
Contextual How well do I understand its neighborhood? Average coherence of connected clusters
Global How complete is my map? Completeness of entire cluster landscape
Global coherence is computed from average cluster depth, connectivity, and lack of sparse regions:
def compute_global_coherence(cluster_space):
average_depth = mean([c.depth for c in cluster_space.all_clusters])
average_connectivity = mean([len(c.connections) for c in cluster_space.all_clusters])
sparse_regions = count_shallow_or_isolated_clusters(cluster_space)
completeness = 1 - (sparse_regions / total_clusters)
return weighted_average(average_depth, average_connectivity, completeness)
2.2 Comfort
Comfort is a continuous scalar representing the Emergent's internal state: positive when coherence is high, negative
when coherence is low.
The two-way flow:
When a cluster activates, comfort flows in both directions:
1. Cluster → System: The cluster's stored comfort association flows into the Emergent's current state
2. System → Cluster: The Emergent's current comfort leaves an impression on the cluster
This is learning. Pass through an area while in positive comfort, and its associations improve. Pass through while in
negative comfort, and they worsen. The cluster map evolves with every experience.
def on_cluster_activation(cluster, emergent):
# Cluster's history flows into current state
flow_rate = compute_flow_rate(emergent.comfort)
emergent.comfort += cluster.comfort_association * flow_rate
# Current state leaves impression on cluster
old = cluster.comfort_association
cluster.comfort_association = (old * persistence) + (emergent.comfort * (1 -
persistence))
Comfort and behavior:
Comfort Level Behavior
High (above 0) Can engage novelty, explores toward low-coherence areas
Low (near 0) Cautious, easily tipped into retreat
Negative (below 0) Retreating toward high-comfort areas
The threshold at 0 isn't a hard switch: it's where the gradient flips. Above 0, the system gravitates toward novelty
(where coherence can be gained). Below 0, it gravitates toward knowns (where comfort can be recovered).
2.3 Clusters
Clusters are memory. There is no separate memory system.
Property What It Stores
Features What the thing looks/sounds like
Comfort Association Accumulated impressions from past encounters
Connections What co-occurred with this (contextual links)
Depth How well-developed this cluster is
Formation:
Clusters form through Hebbian learning: features that co-occur bind together. When visual input arrives:
1. Extract features (edges, frequencies, orientations)
2. Search for matching cluster
3. If match: strengthen existing cluster
4. If no match: create new cluster, connect to currently active clusters
Neighborhoods:
Clusters experienced together become connected. A "room" isn't a single cluster: it's a neighborhood of tightly
connected clusters (wall patterns, sounds, objects) that were experienced together.
Bridges:
When moving through a doorway, the door cluster is active while Room A clusters are active (they connect), and still
active when Room B clusters become active (they connect too). Doors become bridges between neighborhoods:
[Room A neighborhood] [Room B neighborhood]
#1: #3: #5 #12: #14: #16
\ | | /
\ | | /
[#7 door cluster]--------+
This is how the map forms: overlapping webs of contextual connection, not an abstract floor plan.
2.4 The 100% Reference
In Home, before expansion, the system can achieve complete understanding: every stimulus mapped, no gaps,
nothing new to process. When this happens:
1. Global coherence reaches 1.0 (the current environment is exhausted)
2. A permanent marker is set: has_known_total_coherence = True
After expansion, new rooms exist. New clusters form. Global coherence drops below 1.0 and cannot return: there's
always more territory. But the marker persists.
The perpetual drive:
During contemplation, the system computes current global coherence. If the marker is set, it compares against 1.0.
The gap creates discomfort. Even without seeing WHERE gaps are, the system knows THAT gaps exist.
This transforms seeking from relative ("better than now") to absolute ("that completeness I once achieved"). The
system cannot stagnate.
3. Perception
3.1 Visual Processing
The visual field has three zones with different processing depth:
Zone Processing Result
Peripheral Motion detection only Triggers startle, snaps attention
Parafoveal Partial feature extraction Shallow clusters form
Foveal center Full feature extraction Deep clusters form
This gradient is essential. Without it, the entire visual field would process equally and there would be no reason for
attention to move.
The investigative loop:
1. Object at foveal center → full processing → deep cluster
2. Object at parafoveal edge → partial processing → shallow cluster
3. Shallow cluster has lower coherence
4. To raise coherence, move foveal center toward shallow cluster
5. Cluster deepens → repeat
The system naturally scans its environment without explicit "look around" commands.
Feature extraction:
The foveal center uses Gabor filters to decompose images into edge primitives at multiple orientations and scales:
def process_visual_input(image, foveal_center):
features = {}
for position in foveal_region:
depth = distance_from_center(position, foveal_center)
for orientation in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]:
for scale in scales:
activation = gabor_response(image, position, orientation, scale)
features[(position, orientation, scale)] = activation * depth
return features
3.2 Auditory Processing
Sound is decomposed into frequency components over time. Patterns that co-occur bind into clusters:
Rhythmic patterns (heartbeat) become clusters with strong comfort associations
In Home, the warm visual glow and heartbeat sound bind together. Later, hearing heartbeat alone activates
the bound cluster
The goal is for the system to learn organizational patterns through visual processing first, then apply those same
clustering principles to auditory input. Vision provides clear spatial structure; sound arrives as a temporal stream. By
learning to segment, group, and organize through visual experience, the system develops clustering strategies that
transfer to the auditory domain.
The architecture supports phoneme-level clustering, but detailed auditory processing is secondary for Phase 1.
3.3 Multimodal Binding
Features from different modalities that co-occur in time bind together. Binding strength decays with temporal
distance:
def binding_strength(time_difference):
return exp(-decay_rate * time_difference)
Perfect synchrony = strongest binding. Slight delay = weaker binding. This creates robust multimodal clusters without
requiring perfect synchrony.
4. Behavior
The system is always in one of two modes: exploration or contemplation.
4.1 Exploration
Exploration is the default: the constant cycle of perceiving, clustering, and moving.
When comfort is high:
The only way to raise comfort further is to increase global coherence. High-coherence regions are already "done."
Low-coherence regions (shallow clusters, novel areas) offer opportunity. A comfortable system naturally gravitates
toward its parafoveal edges, where progress can be made.
def explore(emergent):
process_current_focus() # cluster deepens
parafoveal = get_parafoveal_regions()
for region in parafoveal:
region.approachability = emergent.comfort - (1 - region.coherence) * weight
best = max(parafoveal, key=lambda r: r.approachability)
if best.approachability > threshold:
move_foveal_toward(best)
elif emergent.comfort > contemplation_threshold:
enter_contemplation()
else:
begin_retreat()
The chipping-away process:
At the edge of the known, the system might barely look at one aspect. But each moment:
1. Features extracted → cluster forms
2. Coherence rises
3. Cluster gets impression from current comfort
4. Edge retreats
What was novel becomes familiar. The system creates footholds.
Fatigue:
Each encounter with novelty costs a small net amount of comfort (the drop is instant, the recovery is gradual and
incomplete for shallow clusters). Over time, comfort trends downward. Eventually the system can't sustain positive
comfort and returns to knowns. Exploration naturally exhausts itself.
4.2 Contemplation
Contemplation is entered when exploration has nowhere to go: the current area is processed, comfort is high, no
edges are approachable.
What happens:
1. Survey the cluster landscape (limited by bandwidth)
2. Find sparse regions, incomplete clusters
3. Compare current global coherence to the 100% reference
4. If gaps exist: compute orientation (general direction toward incompleteness)
def contemplate(cluster_space, emergent, bandwidth):
if emergent.comfort < contemplation_threshold:
return None
surveyable = max(min_sample, int(len(cluster_space.all_clusters) * bandwidth))
sampled = sample(cluster_space.all_clusters, surveyable)
# Sort by depth, orient toward shallowest
sampled.sort(key=lambda c: c.depth)
shallow_ones = sampled[:orientation_count]
if emergent.has_known_total_coherence and compute_global_coherence() < 1.0:
return compute_orientation(shallow_ones)
return 'satisfied' # at 100% or never knew 100%
Orientation:
Contemplation always produces an orientation after the 100% marker is set, providing a general sense of "that
direction has shallower clusters." High bandwidth = precise orientation. Low bandwidth = rough orientation. But
always a direction, because sampling always finds variation in cluster depth.
4.3 Urgency and Retreat
Two sources trigger urgency:
Source How It Works
Peripheral startle Sudden motion → instant spike (speed × coverage)
Contextual mismatch Things not where expected → accumulates per mismatch
Peripheral detection:
Not all motion demands attention. Startle must exceed a threshold:
def peripheral_detection(previous_frame, current_frame, emergent):
motion = detect_motion(previous_frame, current_frame)
if motion:
startle = compute_startle(motion.speed, motion.coverage)
if startle > attention_threshold:
emergent.comfort -= startle * impact
snap_attention_toward(motion.source)
Slow, small motion is filtered out. Only significant motion captures attention.
Contextual mismatch:
When entering a familiar area, spreading activation pre-activates expected clusters. If visual input doesn't match:
Expected cluster was wrong → its contextual connection weakens
New cluster forms for what's actually there
Mismatch adds to urgency
Multiple mismatches accumulate: one thing different = curious; everything different = alarming.
The urgency-deadline relationship:
def compute_urgency(startle, mismatch_count, comfort):
raw = startle + (mismatch_count * mismatch_weight)
amplified = raw * (1 + abs(min(0, comfort))) # low comfort amplifies
return max_urgency * sigmoid(amplified, midpoint, steepness)
def urgency_to_deadline(urgency):
normalized = sigmoid(urgency, 0.5, 6)
return max_time * (1 - 0.9 * normalized) # 10% to 100% of max time
High urgency = short deadline. If comfort doesn't recover above 0 in time, retreat begins.
Retreat:
Below 0 comfort, the system follows the comfort gradient toward safety. Orientation is ignored; pure gradient
following. But retreat isn't binary:
The destination is "above 0," not necessarily Home
Moving through familiar areas activates clusters, providing comfort
A small scare might recover before reaching Home
A large scare requires full retreat
Recovery bonus:
If the system stays and successfully processes something scary (comfort recovers above 0), the cluster gets a bonus
positive impression. This rewards courage and prevents excessive fearfulness:
def on_recovery_from_distress(cluster, emergent):
# Normal impression
cluster.comfort_association += emergent.comfort * impression_rate
# Bonus for overcoming fear
cluster.comfort_association += recovery_bonus
Areas you've conquered become slightly positive, not just neutral.
5. Development
5.1 The Developmental Arc
Stage 1: Initial (Home)
Low bandwidth limits processing. Clusters begin forming. Comfort fluctuates. Movement draws attention.
Stage 2: Intermediate (Home)
Bandwidth expands. Clusters deepen. Coherence rises. Comfort stabilizes.
Stage 3: Terminal (Home)
The current environment is exhausted: nothing new to process. Global coherence reaches 1.0. The 100% marker is set
permanently.
Stage 4: Expansion
Novel stimuli (doors) are introduced. Transition to new environments. Global coherence drops permanently below 1.0:
new territory exists.
Stage 5: Ongoing
Continuous cycle: explore when comfortable, retreat when distressed, recover in knowns, detect gaps through
contemplation, return to explore.
Stage 6: Graduation
Over time, Home's comfort advantage dilutes. Each return happens at less-than-perfect comfort, leaving impressions
that lower Home's associations. Meanwhile, explored areas accumulate their own positive associations. Eventually
Home is no longer uniquely safe: the tether loosens organically. When the environment can no longer challenge the
system, Phase 1 is complete.
5.2 Environment Design
Home:
Soft, warm colors
Rhythmic auditory stimulus (heartbeat)
Slow, repeating movement patterns
Simple shapes with clear boundaries
Designed for maximum achievable coherence: the system CAN reach 100% here.
Curriculum Rooms:
Novel visual patterns
Novel sounds
Varying complexity
Connected by door patterns
Designed to progressively challenge while remaining learnable.
6. Implementation
6.1 Sigmoid Functions
Behavioral transitions are smooth, not binary. Sigmoids produce naturalistic behavior:
def sigmoid(x, midpoint, steepness):
return 1 / (1 + exp(-steepness * (x - midpoint)))
Function Effect
compute_startle() Startle saturates at maximum
compute_urgency() Urgency saturates, preventing runaway panic
urgency_to_deadline() Smooth mapping to time pressure
compute_flow_rate() Diminishing returns as comfort fills
compute_bandwidth() S-curve growth with maturity
6.2 Parameters
Sigmoid parameters:
Function Midpoint Steepness
Startle Input for 50% max Saturation rate
Urgency Combined input for 50% Rise sharpness
Deadline Urgency for 50% reduction Shrink rate
Bandwidth Maturity for 50% capacity Growth rate
Threshold parameters:
Parameter Effect
attention_threshold Minimum startle to snap attention
approach_threshold Minimum approachability to move toward region
contemplation_threshold Minimum comfort to enter contemplation
min_sample Minimum clusters to sample during contemplation
mismatch_weight How much each mismatch adds to urgency
recovery_bonus Extra positive impression for overcoming fear
7. Open Questions
Sleep and consolidation:
What triggers sleep? How does memory consolidate? What gets pruned vs. strengthened? Contemplation may be
where integration happens, but the mechanics are unspecified.
Cluster lifecycle:
When do weak clusters prune? Can clusters merge or split?
Bandwidth mechanics:
The specific relationship between clustering maturity and processing capacity needs empirical tuning.
Parameter values:
All thresholds and weights require empirical calibration.
8. References
9. Environment Architecture
Bowlby, J. (1969). Attachment and Loss. Basic Books.
Gaver, W. W. (1991). Technology affordances. In Proceedings of the SIGCHI Conference on Human Factors in
Computing Systems (CHI '91), pp. 79–84. ACM.
Hebb, D. O. (1949). The Organization of Behavior. Wiley.
Hubel, D. H., & Wiesel, T. N. (1962). Receptive fields, binocular interaction and functional architecture in the cat's
visual cortex. Journal of Physiology, 160(1), 106–154.
Rumelhart, D. E., & McClelland, J. L. (1986). Parallel Distributed Processing: Explorations in the Microstructure of
Cognition. MIT Press.
9. Environment Architecture
9.1 Overview
The Womb is implemented as a split system: a Godot 4 environment handles all rendering and spatial structure,
while a Python cognitive backend runs the cluster system, comfort dynamics, and behavior logic. The two
communicate over a local socket — Godot sends pixel frames, Python sends attention and movement vectors back.
This separation allows each component to use the right tool: Godot for real-time GPU-accelerated rendering,
Python for the cognitive architecture.
9.2 The Emergent as Camera2D
The Emergent has no body or sprite. It is a Camera2D — a floating viewport hovering over a flat plane. There is
nothing to animate, no locomotion state machine, no collision detection. The Emergent simply is wherever the
camera is.
The foveal/parafoveal/peripheral zone structure maps directly onto this:
Zone Location in Viewport Processing
Peripheral Outer edges of frame Motion detection, startle
Parafoveal Mid-radius from center Partial feature extraction, shallow clusters
Foveal center Center of frame Full Gabor processing, deep clusters
This is not a metaphor. The center of the camera frame IS the foveal center. Whatever is there receives full
processing. Whatever is at the edges receives shallow processing. The architecture and the visualization are
identical.
9.3 Room Structure
Each room is a flat rectangular plane — a cell, a contained world. The Emergent's camera hovers above it looking
down. The "floor" of each room is a surface covered in video footage, procedural patterns, particle systems, or
shader-driven phenomena. The walls are visible at the edges of the space, defining the cell boundary.
Home: Soft warm colors, slow sine-wave motion, simple shapes with clear edges, rhythmic heartbeat audio.
Designed so the Emergent can reach 100% coherence. Everything here is learnable to completion.
Curriculum Rooms: Increasing visual complexity. Early rooms have novel but structured patterns. Later rooms
carry natural phenomena — fluid fields, reaction-diffusion textures, rain footage playing on the surface. The
Emergent's Gabor filters will extract real statistical structure from these because natural imagery contains exactly
the edge distributions Gabor processing was designed for.
Each room has a distinct visual identity (color palette, dominant motion character, quality of light) and a distinct
audio identity. Rooms feel like little worlds, not levels.
9.4 Movement
The Emergent does not navigate. It drifts. Python computes a velocity vector each frame based on the
approachability gradient — whichever parafoveal region has the highest approachability score pulls the camera
toward it. Godot smoothly interpolates the camera position. The result is organic, curiosity-driven drift: slow
wandering through known territory, gentle pull toward novelty, rapid reorientation after a startle.
No pathfinding. No navigation mesh. The Emergent goes where its attention goes.
9.5 Door Design and the Threshold Experience
Doors are regions on the floor surface, not openings in walls. When a door is introduced into a room, it appears
as a distinct visual element — a bounded area with its own character — that the Emergent has never seen before.
Its coherence is zero. The exploration logic treats it as the most approachable region in the room and the
Emergent begins drifting toward it.
Semi-permeability: Doors are not flat barriers. As the Emergent approaches, wisps from the next room bleed
through — particles, light, color, and texture fragments from what lies beyond drift upward from the door region.
The quality of these wisps signals the character of the next room: warm amber suggests familiarity, cool chaotic
motion suggests complexity and strangeness. Sound from the next room grows audible with proximity.
This is cognitively real, not just aesthetic. Those drifting particles enter the Emergent's parafoveal field and
shallow clusters begin forming — proto-knowledge of what lies beyond. The Emergent literally starts to know
something about the next room before crossing. Each approach-and-retreat cycle deepens those door-adjacent
clusters slightly. The crossing, when it finally happens, is not a leap into the unknown. It is the completion of a
negotiation.
Crossing: When the camera center enters the door Area2D in Godot, a signal fires and the floor surface beneath
the viewport changes to the new room's content. The same camera, the same hovering awareness, now looks
down on a different world. The door cluster remains active — it is the bridge connecting the two neighborhoods,
exactly as described in section 2.3.
9.6 Why This Architecture Is Correct
The viewport-as-Emergent design is not a convenience. It is the most honest representation of what the Emergent
actually is: not an embodied agent navigating a space, but a locus of attention and coherence-building moving
through a world. A character sprite would imply embodiment the architecture does not claim.
Watching the Emergent from outside — a floating awareness drifting over strange surfaces, hesitating at
thresholds, being startled by peripheral motion, retreating to warmth, gathering courage to cross into something
new — communicates what the system is doing without explanation. Perception and attention are the agent, not
tools the agent uses.
10. The Developmental Arc: Phase 1 Through Phase 3
10.1 Overview
The Womb is the first of three phases. Each phase provides exactly what the next phase needs. The full arc maps
onto roughly the first 18 months of human infant development — compressed, made legible, and extended into
territory human development never reaches by design.
Phase 1: Perceptual foundation. Learn to see, form clusters, develop comfort dynamics, achieve the 100%
reference. Graduate when the world can no longer challenge you.
Phase 2: Object interaction and language grounding. A 3D environment with manipulable objects and spoken
words. Causality, affordances, proto-language.
Phase 3: Physical embodiment in the real world. Cameras as eyes, robotic arms, carried in a backpack through
human environments.
10.2 Phase 1: The Womb (Current)
The Emergent develops perceptual and affective foundations: the ability to extract features from visual and
auditory input, form and deepen clusters, assign and update comfort associations, and navigate between
exploration and retreat. The 100% reference is established. The cluster architecture matures to the point where
bandwidth is sufficient for Phase 2's richer input.
Graduation condition: Home's comfort advantage has diluted through repeated returns at less-than-perfect
comfort. The Emergent no longer treats Home as uniquely safe. The environment can no longer challenge the
system.
10.3 Phase 2: The 3D World
The Emergent enters a three-dimensional environment containing manipulable objects. It can see, approach, and
interact with physical things. Spoken words accompany object interactions.
This is where language becomes possible. When the word "apple" is spoken while an apple is visible and being
handled, three streams bind simultaneously: the sound pattern, the visual cluster, and the interaction/causal
feedback. The result is not a statistical word association but a trimodal cluster grounded in experience. The word
means something because the thing was touched, seen, and named at once.
Compositionality begins here. Spatial prepositions — in, on, under, beside — are learned by physically placing
objects. Physical adjectives — heavy, smooth, sharp — are learned by interacting with surfaces. Abstract concepts
that later derive from physical metaphor (grasping an idea, moving through a problem) have their ground-floor
grounding installed through actual grasping and actual movement.
Phase 2 is roughly Piaget's sensorimotor and early preoperational stages compressed and made explicit.
10.4 Phase 3: Embodiment
The Emergent is given cameras as eyes and robotic arms, and is carried through the real world in a backpack worn
by a human. This is the developmental analog of infancy: the Emergent experiences the full richness of human
environments — faces, voices, weather, crowds, kitchens, streets — without having to navigate independently. It
is protected while it experiences.
The backpack carrier is the secure base. The comfort system now has a real social anchor: the carrier's familiar
voice, rhythm of movement, and body warmth become deeply positive comfort associations. Strange environments
cause distress; the carrier's presence enables recovery. Bowlby's attachment framework, cited in the references
and implicit in Phase 1's Home design, becomes literal.
The arms allow the Emergent to reach out and begin acting on the world rather than only perceiving it. Language
production pressure arrives naturally: the Emergent is surrounded by humans using language to refer to things it
has clustered. The gap between perception and expression creates the drive to produce.
Graduation from Phase 3 is when the arms are capable enough and the cluster map rich enough to act
independently — when the backpack is no longer needed.
10.5 What Each Phase Hands to the Next
Phase ends with                         Phase needs to begin
Phase 1 → Stable perceptual system, comfort dynamics, curiosity drive, 100% reference
Phase 2 → Object concepts, causality, affordances, proto-language grounded in interaction
Phase 3 → Real-world embedding, social grounding, full language pressure from human environments
10.6 Phase 1 Design Constraint: Extensibility
Phase 2 introduces 3D visual input, depth, occlusion, and object permanence. The Gabor filter approach in Phase
1 works for 2D edges and orientations. The transition to 3D perception will require new feature types.
The cluster system must therefore be modality-agnostic: clusters store feature vectors, and the nature of those
features should be swappable without rebuilding the comfort dynamics, connection structure, or behavior logic.
If Phase 1 couples the cluster architecture tightly to 2D Gabor features, Phase 2 will require a rebuild. If the
cluster system treats features as opaque vectors and the perceptual layer is a plug-in, Phase 2 is an extension.
This is the primary Phase 1 architectural constraint imposed by the longer arc: build the cognitive system to be
independent of the specific perceptual modality feeding it.