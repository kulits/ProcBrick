# Data

Part ids index BrickNet's catalog of 14,583 LDraw parts (`bricknet.load_catalog().id_to_stem`). Poses are 4 x 4
row-major float64 matrices in LDU, at each part's LDraw origin. Colors are LDraw color codes. The binary asset files
and the collider cache start with an 8-byte ASCII magic; integers and floats are little-endian.

## Released files (`python -m procbrick fetch`)

`fetch` downloads folders of the Hugging Face dataset kulits/BrickFlow, at the commit `procbrick.paths.REVISION`, into
the data directory, at their paths in the repo. Without names it fetches `assets`, then BrickNet's colliders with
`python -m bricknet fetch-meshes` unless BrickNet has them.

| name | files | bytes | contents |
| --- | --- | --- | --- |
| `assets` | 10 | 1,187,132,396 | the connection atlas and the files below |
| `train/pt` | 3 | 20,810,974,184 | BrickFlow's pretraining split: a directory of 100,000,000 trees of 100 parts |
| `train/ic` | 15,651 | 107,863,209,860 | BrickFlow's finetuning split: 1,000,000 renders |
| `val/procbrick` | 207 | 1,064,496,793 | BrickFlow's evaluation split: 9,900 renders |

The package holds the physical classes of parts, `procbrick/_data/classes.json.xz`: `class_ids`, the class of each
part id, and `representatives`, the part id of each class (-1: none).

## Assets (`assets/`)

Each (part, connector) has a connector profile: the atlas's entry for it, with a chart type, a class, and an index. A
joint domain is the grid of integer joint coordinates at which two profiles can mate; the atlas stores, per pair of
profiles, which of its coordinates leave the two parts free of collisions.

| file | contents |
| --- | --- |
| `charts` (`CHARTSET`) | per connector profile its type, class, and index; the group of each type pair; the joint domains of each class pair |
| `atlas.index` (`ATLASIDX`) | per (part, connector) its profile, ball-joint rows, and a flag per domain (some joint coordinate admitted) |
| `atlas.tiles` (`ATLASTIL`) | the domains' rows: the integer joint coordinates at which two mated parts do not collide |
| `pose_dictionary` (`POSEDICT`) | the connector frames and the joint rotations (below) |
| `parts.json` | part colliders, fixed connectors, connector lengths, and collider boxes (below) |
| `mates` (`MATESBIT`) | u64 n, the n covered part ids (u32, sorted), then n rows of ceil(n / 8) bytes: bit b % 8 of byte b / 8 of row a is set when the a-th and b-th covered parts have a connector pair that can mate at an admitted joint coordinate |
| `sampling_policy.json` | the sampler's part weights, axle seats, options, and controls (below) |
| `assembly_model` (`TWRMODEL`) | the coding model of trees, fitted to 2,000,000 trees of `train/pt` |
| `appearance_policy.npz` | the weights the appearance of renders is drawn from (below) |
| `appearance_model` (`APPMODEL`) | the coding model of the appearance of renders, fitted to `train/ic` |

`charts`, `atlas.index`, and `atlas.tiles` carry the same 32-byte atlas identity.

`parts.json`:

| key | contents |
| --- | --- |
| `geometry` | per collider: [collider id, SHA-256 of BrickNet's `<stem>.cvx.obj`] |
| `parts` | per part: [part id, collider id, fixed connectors], a fixed connector being [connector, subtype, polarity] |
| `lengths` | per connector profile: its length (LDU) |
| `bounds` | per collider, in `geometry`'s order: [[x, y, z] low, [x, y, z] high] (LDU, part frame), the box BrickNet's collision checks give the collider |

`pose_dictionary` holds the frames of the connectors, and the rotations of the joints at integer degrees:

| field | contents |
| --- | --- |
| header | `POSEDICT`, u64 F (frames), u64 C (connectors) |
| frames | F x 7 float64: translation (x, y, z), then the rotation as a quaternion (x, y, z, w) |
| rotations | 3 x 360 x 4 float64: the quaternions (x, y, z, w) of the rotations about x, y, and z by 0 to 359 degrees |
| connectors | C x 3 u32: part id, connector, frame index; sorted by (part id, connector) |
| checksum | the SHA-256 of everything before it |

`sampling_policy.json` (`procbrick.load_sampling_policy()`):

| key | contents |
| --- | --- |
| `part_weights`, `root_part_weights` | proposal weight of each part id, for added parts and for the first part; the pool is the 6,339 class representatives of positive weight |
| `axle_seats` | per axle connector: [part id, connector, pin direction (null: a socket), whether the pin has a collar, the socket's recess depth (LDU) on each side] |
| `parameters` | the sampler's options: backtracking, angle, connector, envelope, mass-growth, extra-connection, and axle-seating preferences |
| `continuous_controls` | three controls drawn per assembly (packing, repetition, growth), each a clipped normal: medians, spreads, the packing-growth correlation, and bounds |

`appearance_policy.npz` (`procbrick.load_appearance_policy()`, which adds `classes`, the class of each part id):

| key | contents |
| --- | --- |
| `variant_p` | float64 (14583,): the weight of each part id among the print variants of its class |
| `variant_strength` | float64 (C,): per class, the strength of the variant weights against the variants already drawn |
| `variant_size_power` | the power of size / 25 that scales the variant strengths |
| `material_p`, `materials` | float64 (M,), str (M,): the weight and BrickNet-Render name of each of the M = 260 materials |
| `colors` | int64 (M,): the LDraw color of each material, BrickNet-Render's code for it (the lowest of several); the 9 materials it gives no code have the direct color (0x02RRGGBB) of their base color |
| `material_reuse` | float64 (C,): per class, the reinforcement of materials already drawn for the class |
| `latent_parameters` | float64 (7,): an assembly's palette strength and reuse: log means (2), log spreads (2), correlation, size trends (2) |
| `material_inventory_profile` | float64 (99, 2, 129): per size 2 to 100 (row size - 2; larger sizes use 100), for classes without and with print variants, the target share of distinct-material pairs at 129 reuse scales |
| `palette_weights` | float64 (99, 40): per size 2 to 100 (row size - 2; larger sizes use 100), the weight of a candidate by its palette cell (distinct materials, largest material share) |
| `palette_candidates` | the candidate appearances drawn per assembly (16) |

## Collider cache (`colliders.cache`)

The first sampler (or `load_colliders()`) of a data directory parses BrickNet's colliders of the parts of
`parts.json`, each checked against its SHA-256 there, into `procbrick/colliders.cache`, which later samplers
memory-map. It is built again when `parts.json` changes.

| field | contents |
| --- | --- |
| header | `GEOMPACK`, the SHA-256 of `parts.json`, u64 C (colliders), u64 the index's offset, the index's SHA-256 |
| colliders | per collider u32 hulls, then per hull u32 V and V x 3 float64 vertices |
| index | per collider u32 id, u64 offset, u64 bytes (colliders with identical encodings share one record) |

## Trees (`Sampler.sample`, `load_trees`, `save_trees`, `generate`)

A sampled tree is a BrickNet `Tree` whose parts are colored 16. Edge k attaches part k + 1 (`child`) to an earlier
part (`parent`) through BrickNet's connectors (`parent_sub`, `parent_conn`, `child_sub`, `child_conn`), with the
joint's integer coordinates from the parent:

| family | `Tree` edge | joint J (degrees, LDU) |
| --- | --- | --- |
| stud | `StudEdge(yaw)` | Ry(yaw) |
| axle | `AxleEdge(flip, yaw, slide)` | [Ry(yaw) Rx(180)^flip \| (0, -slide, 0)] |
| ball | `BallEdge(rx, ry, rz)` | Rz(rz) Ry(ry) Rx(rx), from the connector on the `in` side |
| hinge | `HingeEdge(flip, yaw)` | Ry(yaw) Rx(180)^flip |
| fixed | `FixedEdge` | identity |

The child is at `pose[parent] @ F_parent @ J @ inverse(F_child)` (a ball edge whose parent is on the `on` side takes
the inverse of J), with `F` the connector frames and rotations of the pose dictionary at integer degrees; the first part
is at the identity. `bricknet.tree_to_graph` and `bricknet.decode_graph` give the same poses (to about 1e-12 LDU).

A directory of trees stores trees of class representatives (`procbrick.representatives()`) as records that decode to
the same parts, edges, and poses; `load_trees` colors every part 16.

| file | contents |
| --- | --- |
| `manifest.json` | `count` N, and `files`: per file (`data`, `index`) its `bytes` and `sha256` |
| `data` | one record per tree, under 64 KiB; a record ends where the next begins |
| `index` | per group of 64 trees: u64, the offset of its first record, then 64 u16, each record's offset from it modulo 2^16 (0 past the last tree) |

Trees decode with `assets/` (`pose_dictionary`, `charts`, `assembly_model`) and `procbrick.representatives()`; they
have 2 or more parts. Each record is entropy-coded (rANS) against `assembly_model`: the first part's class, then per
later part its parent (uniform from the 101st part), the parent's connector, its class, its connector, and the joint
coordinates, as the chart of the connector pair orders it. For an axle, the slide is the chart's first slide position
plus the coded slide index. A record returns to the coder's initial state only at its tree's end, which gives the
tree's parts. The record framing, the coded symbols, and the `assembly_model` layout are in `rust/trees.rs`.

Opening a directory of trees checks the manifest, index, and decoder files. While `generate` runs, `out/.blocks/`
holds its resume state (`plan.json`, `seeds`, and per finished block k `NNNNNN.data` and `NNNNNN.index`); it is
removed once every block is finished.

## Renders (`load_renders`, `save_renders`)

Render i shows the first `sizes[i]` parts of tree i; `Renders.read` gives each as a `Tree` whose parts have their
print variants and the `colors` of their materials, and `Renders.poses(i)` gives its poses.

| file | contents |
| --- | --- |
| `assemblies/` | a directory of the N trees |
| `appearance/` | N records in the layout of a directory of trees (`manifest.json`, `data`, `index`): per render, the part id with print variant and the material (an index of `load_appearance_policy()["materials"]`) of each of its parts |
| `sizes.npy` | unsigned (N,), u8 when every render has at most 255 parts, else u16: the parts of each render (2 or more); renders 2j and 2j + 1 are partners when they add up to 102 |
| `shard_ids.npy` | u32 (N,): the shard of each render |
| `frames.npy` | structured (N,): `offset` u64 and `bytes` u32, the image's extent in its shard; `camera_world`, `projection`, and `ldraw_to_world`, float64 4 x 4 |
| `shards/NNN/SSSSS.tar` | uncompressed tar shards of the renders of NNN parts (at least 3 digits), each image an AVIF (RGBA) member |

A point p of render i's tree (LDU) is at clip coordinates `projection @ inverse(camera_world) @ ldraw_to_world @ p`,
the matrices of `Renders.camera(i)`; image y runs down.

The appearance decodes with `assets/appearance_model`, `procbrick.class_ids()`, and the classes of the render's parts.
Each record is entropy-coded (rANS) against `appearance_model`: per part, its print variant (when its class has several
part ids) and its material, each new to the render or reused. The coded symbols and the `appearance_model` layout are in
`rust/appearance.rs`.
