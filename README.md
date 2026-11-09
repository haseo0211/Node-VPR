# Node-VPR

**Node-VPR Based Intelligent Indoor Localization & Navigation System**

*Node-Based Localization via Visual–Inertial Fusion and Topological Constraints*

---

## 1. Project Overview

Node-VPR is an indoor localization and navigation system based on **Visual Place Recognition (VPR)** and **topological node graphs**.

The system aims to identify a user's location and approach direction using visual information captured by a smartphone camera and the spatial connectivity of a predefined node graph.

The initial implementation is being developed and evaluated at Korea Aerospace University.

Future development will incorporate inertial measurements and graph-based constraints to improve localization robustness and support indoor navigation.

### Key Results & Specifications

| Item | Status / Result |
|---|---|
| Campus Node Map | Completed |
| Reference Visual Database | Constructed |
| VPR Model Benchmark | TBD |
| Selected VPR Model | TBD |
| Top-1 / Top-5 Recognition Accuracy | TBD |
| Average Query Processing Time | TBD |
| Localization Accuracy | TBD |

*Performance results and system specifications will be updated following experimental validation.*

## 2. System Architecture

**Status: TBD (To Be Determined)**

The system architecture will be documented as the VPR, localization, and navigation modules are developed and integrated.

## 3. Repository Structure

```text
Node-VPR/
├── Localization/       # Localization algorithms
├── Navigation/         # Route planning and navigation
├── NodeMap/            # Topological node graph and floor plans
├── NodeMapEditor/      # Node graph editing tools
├── VPR/                # Visual place recognition
│   ├── cli.py
│   ├── common/         # Database, retrieval, and evaluation
│   └── models/         # Model definitions and loaders
├── README.md
└── .gitignore
```

### Local-Only Directories

The following directories are maintained locally and excluded from the Git repository.

| Directory | Purpose |
|---|---|
| `Models/` | Pretrained model weights and caches |
| `VisualDB/` | Reference images and visual descriptors |
| `TestDB/` | Evaluation images and results |
| `Server/` | Local data collection server |
| `.venv/` | Python virtual environment |

## 4. Installation & Environment Setup

### Requirements

- Python 3.11
- PyTorch / TorchVision
- NumPy
- Pillow
- gdown

**Validated dependency versions: TBD**

### Environment Setup

Run the following commands from the project root.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install torch torchvision numpy Pillow gdown
```

Additional dependencies may be required. A complete and version-pinned dependency list will be provided after environment validation.

For GPU acceleration, install a PyTorch build compatible with the target CUDA environment.

## 5. Pretrained VPR Models

The current implementation supports four pretrained VPR configurations.

| Model | Backbone | Descriptor Dimension |
|---|---|---:|
| CosPlace | ResNet18 | 512 |
| EigenPlaces | ResNet18 | 256 |
| EigenPlaces | ResNet50 | 2048 |
| MixVPR | ResNet50 | 512 |

**Selected Model: TBD**

### Model Download

Pretrained model weights are not included in this repository.

The model loaders are configured to download the required pretrained implementations and checkpoints when initialized, if they are not already cached locally.

**CosPlace / EigenPlaces**

These models use PyTorch Hub to retrieve pretrained implementations and weights.

- [CosPlace](https://github.com/gmberton/CosPlace)
- [EigenPlaces](https://github.com/gmberton/eigenplaces)

**MixVPR**

The MixVPR loader retrieves its pretrained checkpoint using the source configured in `VPR/models/mixvpr_resnet50_512/model.py`.

- [MixVPR](https://github.com/amaralibey/MixVPR)

Downloaded model files are stored locally under `Models/`.

An internet connection is required for the initial download.

### Verify Model Loading

List the available models:

```powershell
python VPR/cli.py list
```

Run a basic model initialization and descriptor extraction test:

```powershell
python VPR/cli.py smoke --model cosplace_resnet18_512
```

The smoke test does not require the private VisualDB.

## 6. Visual Database & Recognition

### Database Structure

The Visual Database is organized by target nodes and incoming directions.

```text
VisualDB/
├── images/
│   ├── N1/
│   │   ├── from_N2/
│   │   └── from_E1/
│   └── N2/
│       └── from_N1/
│
└── descriptors/
    ├── cosplace_resnet18_512/
    ├── eigenplaces_resnet18_256/
    ├── eigenplaces_resnet50_2048/
    └── mixvpr_resnet50_512/
```

Directional labels distinguish visual observations according to the approach direction associated with each node.

### Database Construction

Generate directional image folders based on the Node Graph:

```powershell
python VPR/cli.py prepare-folders
```

Build visual descriptors:

```powershell
python VPR/cli.py build --models cosplace_resnet18_512
```

### Place Recognition

```powershell
python VPR/cli.py recognize --model cosplace_resnet18_512 --image path/to/query.jpg
```

The recognition pipeline retrieves ranked directed-edge candidates using visual descriptor similarity.

Recognition scores are not calibrated localization probabilities.

## 7. Model Benchmark & Evaluation

**Status: Evaluation in Progress**

The benchmark compares pretrained VPR models using reference images from VisualDB and separately collected query images from TestDB.

### Evaluation Metrics

- Top-1 Directed-Edge Accuracy
- Top-5 Directed-Edge Accuracy
- Micro / Macro Accuracy
- Average Query Processing Time
- Per-Edge Recognition Performance

### Evaluation Command

```powershell
python VPR/cli.py evaluate --models cosplace_resnet18_512 eigenplaces_resnet18_256 eigenplaces_resnet50_2048 mixvpr_resnet50_512
```

### Benchmark Results

| Model | Top-1 | Top-5 | Mean Time / Query |
|---|---|---|---|
| CosPlace ResNet18 | TBD | TBD | TBD |
| EigenPlaces ResNet18 | TBD | TBD | TBD |
| EigenPlaces ResNet50 | TBD | TBD | TBD |
| MixVPR ResNet50 | TBD | TBD | TBD |

### Evaluation Conditions

| Item | Specification |
|---|---|
| Evaluation Hardware | TBD |
| Reference Dataset Size | TBD |
| Query Dataset Size | TBD |
| Evaluation Environment | TBD |
| Model Selection Criteria | TBD |

Detailed experimental results and model selection rationale will be documented after evaluation.

## 8. Development Status

| Component | Status |
|---|---|
| Campus Node Map | Completed |
| Node Map Editor | Initial Implementation Completed |
| Reference Visual Database | Constructed |
| VPR Descriptor Extraction | Implemented |
| VPR Recognition Pipeline | Implemented — Validation Ongoing |
| VPR Model Benchmark | In Progress |
| Graph-Constrained Localization | Planned |
| Visual–Inertial Fusion | Planned |
| Navigation Integration | Planned |

Development status will be updated as implementation and validation progress.

## 9. Dataset & Privacy Policy

VisualDB and TestDB are excluded from this repository.

These datasets are collected for indoor localization development and may contain personally identifiable visual information.

Raw images, query data, and generated descriptor databases are maintained separately from the source code.

Any future dataset release will require an appropriate privacy review.

---

**Project:** Node-VPR Based Intelligent Indoor Localization & Navigation System

**Development Environment:** Korea Aerospace University

**Current Stage:** VPR Model Evaluation & Localization Development

**License:** TBD