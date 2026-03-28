# FUTURE.md

Dokumen ini merinci dua ide eksperimental utama yang belum pernah dicoba dalam riset deteksi kematangan TBS sawit 4-kelas ini, beserta analisis teknis, evidence pendukung dari literatur, dan roadmap implementasi.

---

## Executive Summary

Berdasarkan analysis terhadap [CONTEXT.md](CONTEXT.md), sistem 4-kelas TBS sawit mentok di ceiling ~0.24-0.27 mAP50-95 karena:

1. **B2/B3 Ambiguity**: Confusion tinggi (~34-35%) menunjukkan label ceiling atau ambiguitas definisi kelas
2. **B4 Small Object Burden**: Mean mAP50-95 hanya 0.085, sering missed ke background (41-43%)
3. **Domain Imbalance**: DAMIMAS 90.1% vs LONSUM 9.9%

Dua ide berikut menyerang akar masalah dengan pendekatan yang berbeda dari sekadar tuning recipe atau ganti arsitektur.

---

## IDE 1: Uncertainty-Aware Label Refinery (UALR)

### 1.1 Problem Statement

Dari CONTEXT.md Section 5.2, bottleneck B2/B3 kemungkinan **bukan** masalah backbone detector, melainkan:
- Ambiguity definisi kelas (B2 transisi, B3 full hitam, tapi overlap visual)
- Linear probe accuracy hanya 0.528 (Section 3.5), menunjukkan separability problem intrinsik
- Label smoothing statis (`label_smoothing=0.1` di Section 6.1) tidak membantu karena hard label tetap 0/1

### 1.2 Konsep Inti

**UALR mengubah paradigma dari hard label (0/1) ke soft label adaptif berbasis uncertainty model.**

```
Traditional:  Label [0, 1, 0, 0] → Loss calculation
UALR:         Label [0.15, 0.70, 0.10, 0.05] → Loss calculation
              ↑ uncertainty-aware, class-overlap conscious
```

#### 1.2.1 Confidence-Based Label Smoothing

**Mekanisme:**
1. Setiap epoch N, jalankan model pada validation set
2. Hitung confidence distribution per kelas:
   - High confidence (>0.9): Keep as hard label
   - Medium confidence (0.7-0.9): Generate soft label dengan temperature scaling
   - Low confidence (<0.7): Ambiguity region → soft label dengan entropy regularization

**Formula soft label generation:**
```python
# Untuk sampel dengan prediction p = [0.6, 0.35, 0.03, 0.02]
def generate_soft_label(pred_probs, temperature=0.5, ground_truth_class=0):
    # Blend prediction dengan ground truth
    alpha = confidence_weight(pred_probs[ground_truth_class])  # 0.3-0.7
    soft_label = alpha * one_hot(ground_truth_class) + (1-alpha) * pred_probs
    return soft_label
```

#### 1.2.2 Uncertainty-Weighted Loss

Berikan weight lebih rendah pada sampel dengan high uncertainty selama training:

```python
# Uncertainty diukur dari entropy atau variance prediksi
uncertainty = -sum(p * log(p) for p in pred_probs)  # Entropy
sample_weight = 1.0 / (1.0 + beta * uncertainty)    # Inverse uncertainty weighting
```

#### 1.2.3 Dynamic Class Boundary

Biarkan model belajar bahwa B2 dan B3 memiliki overlap region yang legitimate:
- Pada training, jika ground truth B2 tapi model confident B3 dengan prob > 0.3, **jangan** penalize 100%
- Gunakan curriculum: early epochs lebih toleran, late epochs lebih strict

### 1.3 Evidence dari Literatur

#### Paper 1: "Rethinking Pre-training and Self-training" (Zoph et al., NeurIPS 2020)
- **Link**: https://arxiv.org/abs/2006.06882
- **Relevansi**: Menunjukkan self-training dengan pseudo-labels adaptif lebih efektif daripada pre-training statis. Pada eksperimen COCO dengan 20% data, self-training memberi +3.4 AP improvement.
- **Key Insight**: Pseudo-labels dengan confidence threshold dinamis mengatasi label noise lebih baik daripada hard labels.

#### Paper 2: "Remix: Rebalanced Mixup" (Chou et al., ECCV 2020)
- **Link**: https://arxiv.org/abs/2007.03943
- **Relevansi**: Menunjukkan bahwa untuk class imbalance, mixing features dengan label yang disproportionately weighted ke minority class meningkatkan generalization.
- **Key Insight**: "By doing so, the classifier learns to push the decision boundaries towards the majority classes and balance the generalization error."
- **Aplikasi pada TBS**: B1 dan B4 (minority di beberapa domain) bisa di-weight lebih tinggi dalam soft label generation.

#### Paper 3: "SOLO: Segmenting Objects by Locations" (Wang et al., ECCV 2020)
- **Link**: https://arxiv.org/abs/1912.04488
- **Relevansi**: Mengubah instance segmentation menjadi classification problem dengan "instance categories". Konsep serupa bisa dipakai: B2/B3 sebagai "overlap category".
- **Key Insight**: "Instance categories nicely converts instance mask segmentation into a classification-solvable problem."

### 1.4 Perbedaan dengan yang Sudah Dicoba

| Approach | Yang Sudah Dicoba (Fail) | UALR (Proposal) |
|----------|-------------------------|-----------------|
| **Label Smoothing** | `label_smoothing=0.1` statis, global | Adaptive, per-sample, confidence-based |
| **Soft Labels** | Tidak pernah dicoba | Berbasis model prediction uncertainty |
| **Loss Weighting** | Focal loss (implisit) | Explicit uncertainty weighting |
| **B2/B3 Handling** | Hierarchical (gagal) | Dynamic boundary, soft overlap |

### 1.5 Expected Impact

Dari data CONTEXT.md Section 4.3:
- Current B2: 0.285 mAP50, 0.126 mAP50-95
- Current B3: 0.410 mAP50, 0.170 mAP50-95
- **Target**: B2 → 0.40+ mAP50, B3 → 0.50+ mAP50
- **Overall**: mAP50-95 naik dari 0.265 ke 0.30+

### 1.6 Implementation Roadmap

#### Phase 1: Baseline Modification (1-2 minggu)
```python
# Modifikasi YOLOv8/Ultralytics Loss
class UncertaintyAwareLoss:
    def __init__(self, uncertainty_threshold=0.7):
        self.threshold = uncertainty_threshold
    
    def forward(self, pred, target_hard, uncertainty_map):
        # Generate soft labels untuk region uncertain
        soft_targets = self.generate_soft_labels(pred, target_hard, uncertainty_map)
        
        # Weighted BCE/CE loss
        weights = 1.0 / (1.0 + uncertainty_map)
        loss = F.cross_entropy(pred, soft_targets, weight=weights)
        return loss
```

#### Phase 2: Confidence Calibration (2-3 minggu)
- Implementasi temperature scaling pada validation
- Build confidence-uncertainty mapping
- Curriculum learning schedule

#### Phase 3: Evaluation (1 minggu)
- Per-class metrics breakdown
- Confusion matrix analysis B2/B3
- Bandingkan dengan baseline AR29

---

## IDE 2: Multi-Resolution Feature Distillation with Objectness Prior (MRF-DOP)

### 2.1 Problem Statement

Dari CONTEXT.md:
- **B4 bottleneck**: Median rel_area 0.0072 (paling kecil), mean mAP50-95 cuma 0.085
- **Domain imbalance**: DAMIMAS 90.1% image, model belajar prior DAMIMAS
- **Architecture swap gagal**: YOLOv9e, YOLO11x, RT-DETR semua tidak memberi lompatan (Section 6.3)

### 2.2 Konsep Inti

**MRF-DOP menggunakan knowledge distillation dengan multi-scale features dan objectness-aware weighting untuk transfer knowledge dari domain mayoritas (DAMIMAS) ke minoritas (LONSUM), dengan fokus spesifik pada small objects (B4).**

#### 2.2.1 Teacher-Student Architecture

```
Teacher (Frozen): YOLOv8x / YOLO11x
├── Resolution: 1280x1280
├── Dataset: DAMIMAS-only (rich domain)
├── Knowledge: Strong objectness prior, robust feature representation
└── Output: Multi-scale feature maps [P3, P4, P5], objectness scores

Student (Trainable): YOLO11s (sama dengan baseline)
├── Resolution: 640x640
├── Dataset: Combined (DAMIMAS + LONSUM)
├── Receives: Distillation loss dari Teacher
└── Target: Match Teacher's P3 features for small objects
```

#### 2.2.2 Multi-Resolution Feature Distillation

**Why Multi-Resolution?**
- B4 adalah small object, muncul di P3 layer (highest resolution, 80x80)
- B1, B2 lebih besar, muncul di P4 (40x40) dan P5 (20x20)
- Distillation harus spesifik per-scale

**Distillation Loss:**
```python
# Feature distillation dengan attention weighting
def feature_distillation_loss(student_feat, teacher_feat, objectness_mask, scale='P3'):
    """
    scale: 'P3' untuk B4, 'P4' untuk B2/B3, 'P5' untuk B1
    objectness_mask: weight dari teacher objectness score
    """
    # Normalize features
    s_norm = F.normalize(student_feat, dim=1)
    t_norm = F.normalize(teacher_feat, dim=1)
    
    # Cosine similarity dengan objectness weighting
    similarity = (s_norm * t_norm).sum(dim=1)
    weighted_sim = similarity * objectness_mask
    
    # Scale-specific weight: P3 lebih penting untuk B4
    if scale == 'P3':
        scale_weight = 2.0
    elif scale == 'P4':
        scale_weight = 1.5
    else:
        scale_weight = 1.0
    
    loss = (1 - weighted_sim).mean() * scale_weight
    return loss
```

#### 2.2.3 Objectness-Aware Distillation

**Problem**: B4 sering missed ke background (41-43% di Section 4.4). Teacher dengan high objectness score pada B4 regions harus transfer knowledge lebih kuat.

```python
# Objectness mask dari teacher predictions
teacher_predictions = teacher(image)  # [batch, num_anchors, 4+1+num_classes]
objectness_scores = teacher_predictions[..., 4]  # Objectness confidence

# Weighting mask: higher objectness → higher weight dalam distillation
objectness_mask = torch.sigmoid(objectness_scores)  # [0, 1]

# B4 regions yang sering missed akan dapat attention lebih
```

#### 2.2.4 Domain-Aware Temperature Scaling

**Problem**: DAMIMAS dan LONSUM memiliki domain gap (scene-structure shift di Section 3.3).

```python
# Temperature yang berbeda per domain
def domain_aware_distillation(student_out, teacher_out, domain_label):
    """
    domain_label: 0 untuk DAMIMAS, 1 untuk LONSUM
    """
    # LONSUM pakai temperature lebih tinggi (softer distribution)
    # karena domain minoritas, jangan terlalu confident
    temperature = 2.0 if domain_label == 1 else 1.0
    
    student_prob = F.softmax(student_out / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_out / temperature, dim=-1)
    
    # KL divergence dengan domain-aware temperature
    kl_loss = (teacher_prob * (teacher_prob / student_prob).log()).sum()
    return kl_loss
```

### 2.3 Evidence dari Literatur

#### Paper 1: "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information" (Wang et al., 2024)
- **Link**: https://arxiv.org/abs/2402.13616
- **Relevansi**: YOLOv9 mengenalkan PGI (Programmable Gradient Information) untuk mengatasi information bottleneck. Teacher dalam MRF-DOP berfungsi sebagai "information source" yang kaya.
- **Key Insight**: "PGI can provide complete input information for the target task to calculate objective function, so that reliable gradient information can be obtained."
- **Aplikasi**: Teacher yang dilatih pada DAMIMAS (rich data) bisa supply "complete information" untuk student yang train pada combined dataset.

#### Paper 2: "YOLOv10: Real-Time End-to-End Object Detection" (Wang et al., NeurIPS 2024)
- **Link**: https://arxiv.org/abs/2405.14458
- **Relevansi**: YOLOv10 menggunakan NMS-free training dengan consistent dual assignments. Small object detection improved dengan eliminasi NMS.
- **Key Insight**: "The reliance on non-maximum suppression (NMS) for post-processing hampers the end-to-end deployment and adversely impacts the inference latency."
- **Aplikasi**: Student bisa dilatih dengan NMS-free objective untuk B4 recall improvement.

#### Paper 3: "A ConvNet for the 2020s" (Liu et al., CVPR 2022)
- **Link**: https://arxiv.org/abs/2201.03545
- **Relevansi**: ConvNeXt menunjukkan pure ConvNet bisa mengalahkan Swin Transformer dengan modernized design. Menunjukkan arsitektur swap bukan solusi; feature quality lebih penting.
- **Key Insight**: "Constructing entirely from standard ConvNet modules, ConvNeXts compete favorably with Transformers in terms of accuracy and scalability."
- **Aplikasi**: Knowledge distillation meningkatkan feature quality tanpa ganti backbone.

#### Paper 4: "FSD V2: Improving Fully Sparse 3D Object Detection with Virtual Voxels" (Fan et al., 2023)
- **Link**: https://arxiv.org/abs/2308.03755
- **Relevansi**: Virtual voxels mengatasi center feature missing problem dalam sparse detection. Konsep serupa: teacher bisa supply "virtual features" untuk region yang student struggle.
- **Key Insight**: "Virtual voxels not only address the notorious issue of the Center Feature Missing problem... but also endow the framework with a more elegant and streamlined approach."

### 2.4 Perbandingan dengan yang Sudah Dicoba

| Approach | Yang Sudah Dicoba (Fail) | MRF-DOP (Proposal) |
|----------|-------------------------|-------------------|
| **Two-Stage** | Detector + EfficientNet (gagal) | Single-stage dengan distillation |
| **Architecture Swap** | YOLOv9e, YOLO11x (diminishing returns) | Teacher architecture, student tetap YOLO11s |
| **Data Balancing** | Oversampling B1/B4 (gagal) | Knowledge transfer dari DAMIMAS ke LONSUM |
| **Resolution** | 1024 (gain kecil) | Multi-res distillation (1280→640) |

### 2.5 Expected Impact

Dari data CONTEXT.md:
- **Current B4**: 0.229 mAP50, 0.085 mAP50-95
- **Target B4**: 0.35+ mAP50, 0.15+ mAP50-95
- **Domain Gap**: LONSUM-only performance meningkat signifikan
- **Overall**: mAP50-95 naik dari 0.265 ke 0.30-0.32

### 2.6 Implementation Roadmap

#### Phase 1: Teacher Preparation (1-2 minggu)
```python
# Train teacher pada DAMIMAS-only dengan resolusi tinggi
from ultralytics import YOLO

teacher = YOLO('yolov8x.pt')  # atau yolo11x
teacher.train(
    data='damimas_only.yaml',
    imgsz=1280,
    epochs=100,
    batch=8,
    device='0'
)
```

#### Phase 2: Distillation Framework (2-3 minggu)
```python
class MultiResolutionDistillation:
    def __init__(self, teacher, student, distill_weights):
        self.teacher = teacher
        self.student = student
        self.weights = distill_weights  # {'P3': 2.0, 'P4': 1.5, 'P5': 1.0}
        
    def compute_loss(self, images, targets, domain_labels):
        # Forward pass
        with torch.no_grad():
            teacher_out = self.teacher(images, return_features=True)
        
        student_out = self.student(images, return_features=True)
        
        # Multi-scale feature distillation
        distill_loss = 0
        for scale in ['P3', 'P4', 'P5']:
            t_feat = teacher_out['features'][scale]
            s_feat = student_out['features'][scale]
            obj_mask = teacher_out['objectness'][scale]
            
            distill_loss += self.feature_distill_loss(
                s_feat, t_feat, obj_mask, scale
            ) * self.weights[scale]
        
        # Detection loss
        det_loss = self.detection_loss(student_out, targets)
        
        # Domain-aware temperature untuk LONSUM
        domain_loss = self.domain_aware_loss(student_out, teacher_out, domain_labels)
        
        total_loss = det_loss + 0.5 * distill_loss + 0.3 * domain_loss
        return total_loss
```

#### Phase 3: Training Strategy (2-3 minggu)
- **Stage 1 (Epochs 1-30)**: Freeze teacher, train student dengan high distillation weight
- **Stage 2 (Epochs 31-60)**: Reduce distillation weight, focus pada detection loss
- **Stage 3 (Epochs 61-100)**: Fine-tuning dengan curriculum pada B4 samples

#### Phase 4: Evaluation (1 minggu)
- Per-class metrics (terutama B4)
- Per-domain metrics (DAMIMAS vs LONSUM)
- Confusion matrix analysis
- Inference speed comparison

---

## Perbandingan Dua Ide

| Aspek | IDE 1: UALR | IDE 2: MRF-DOP |
|-------|-------------|----------------|
| **Target Masalah** | B2/B3 ambiguity, label ceiling | B4 small object, domain imbalance |
| **Approach** | Data-centric (label refinery) | Model-centric (knowledge transfer) |
| **Resource** | Low (modifikasi loss saja) | Medium (perlu train teacher) |
| **Risk** | Medium (bisa over-smooth) | Medium (distillation bisa noisy) |
| **Expected Gain** | B2/B3 +15-20% improvement | B4 +50-80% improvement |
| **Complexity** | Low-Medium | Medium |
| **Dependencies** | None | Perlu DAMIMAS-only teacher model |

---

## Rekomendasi Strategis

### Prioritas Implementasi

**Mulai dengan IDE 1 (UALR) karena:**
1. **Lower barrier to entry**: Tidak perlu train model baru, modifikasi loss function saja
2. **Falsifiable cepat**: Bisa di-test dalam 1-2 minggu
3. **Menyerang bottleneck utama**: B2/B3 confusion 34-35% adalah masalah paling kritis
4. **Foundation untuk IDE 2**: Kalau UALR menunjukkan promise pada B2/B3, MRF-DOP bisa fokus murni pada B4

**IDE 2 sebagai follow-up** jika:
- UALR tidak memberi gain signifikan pada B4
- Ada resource untuk train teacher model
- Target adalah generalisasi ke domain LONSUM

### Alternative: Hybrid Approach

**UALR + MRF-DOP Combined**
```python
# Training pipeline hybrid
for epoch in range(epochs):
    # 1. UALR: Generate soft labels dari teacher
    soft_labels = teacher.generate_uncertainty_aware_labels(images)
    
    # 2. Student training dengan soft labels + distillation
    student_out = student(images)
    
    # 3. Combined loss
    loss = detection_loss(student_out, soft_labels) + \
           distill_loss(student_out, teacher_out) + \
           uncertainty_weighting
```

**Keuntungan Hybrid:**
- Teacher menyediakan uncertainty estimation untuk UALR
- Distillation transfer feature knowledge
- Soft labels dari teacher lebih reliable daripada model student sendiri

---

## Success Criteria

### Minimum Viable Success (MVS)
- **UALR**: B2 mAP50 naik dari 0.285 ke 0.35+, B3 mAP50 naik dari 0.410 ke 0.48+
- **MRF-DOP**: B4 mAP50 naik dari 0.229 ke 0.30+, LONSUM-only mAP50 meningkat signifikan

### Stretch Goals
- Overall mAP50-95: 0.30+ (dari 0.265)
- B4 recall: 0.50+ (dari current yang lebih rendah)
- Cross-domain generalization: LONSUM mAP50 mendekati DAMIMAS

---

## Risks dan Mitigasi

| Risk | Probabilitas | Impact | Mitigasi |
|------|-------------|--------|----------|
| **UALR over-smoothing** | Medium | Model jadi "ragu-ragu", precision turun | Curriculum learning: gradually increase hardness |
| **MRF-DOP teacher overfit** | Low | Teacher terlalu spesifik DAMIMAS | Strong augmentation pada teacher training |
| **Distillation tidak transfer** | Medium | Student tidak belajar dari teacher | Feature-level distillation (bukan hanya output) |
| **Resource constraint** | Medium | Training teacher terlalu mahal | Gunakan pre-trained YOLOv8x sebagai teacher, fine-tune saja |
| **Implementation bug** | Medium | Loss tidak convergen | Extensive logging per-scale, per-class |

---

## Appendix: Paper Links Summary

### Knowledge Distillation & Transfer Learning
1. **YOLOv9**: https://arxiv.org/abs/2402.13616 - PGI dan information bottleneck
2. **YOLOv10**: https://arxiv.org/abs/2405.14458 - NMS-free dan efficiency
3. **ConvNeXt**: https://arxiv.org/abs/2201.03545 - Modernized ConvNet principles

### Label Noise & Uncertainty
4. **Self-training**: https://arxiv.org/abs/2006.06882 - Rethinking pre-training
5. **Remix**: https://arxiv.org/abs/2007.03943 - Rebalanced mixup untuk imbalance

### Object Detection Architecture
6. **SOLO**: https://arxiv.org/abs/1912.04488 - Instance categories concept
7. **FSD V2**: https://arxiv.org/abs/2308.03755 - Virtual voxels untuk sparse detection

---

## Catatan Implementasi Teknis

### Untuk YOLOv8/Ultralytics

**File yang perlu dimodifikasi:**
```
ultralytics/
├── nn/
│   ├── modules.py          # Add uncertainty head
│   └── loss.py             # Modify loss calculation
├── models/
│   └── yolo.py             # Add distillation hooks
└── engine/
    └── trainer.py          # Training loop modification
```

**Hook untuk Feature Extraction:**
```python
# Hook untuk capture intermediate features
class FeatureExtractor:
    def __init__(self, model):
        self.features = {}
        for name, module in model.named_modules():
            if name in ['model.15', 'model.18', 'model.21']:  # P3, P4, P5
                module.register_forward_hook(self._hook(name))
    
    def _hook(self, name):
        def hook_fn(module, input, output):
            self.features[name] = output
        return hook_fn
```

### Logging dan Monitoring

**Metrics yang harus di-track:**
```python
metrics = {
    'train/distill_loss': distill_loss.item(),
    'train/uncertainty_mean': uncertainty.mean().item(),
    'val/b2_mAP50': per_class_metrics['B2'],
    'val/b3_mAP50': per_class_metrics['B3'],
    'val/b4_mAP50': per_class_metrics['B4'],
    'val/b2_b3_confusion': confusion_matrix[1, 2] + confusion_matrix[2, 1],
    'val/b4_missed_to_bg': confusion_matrix[3, -1]  # B4 predicted as background
}
```

---

*Dokumen ini akan diupdate seiring dengan progress eksperimen. Last updated: 2025-01-XX*
