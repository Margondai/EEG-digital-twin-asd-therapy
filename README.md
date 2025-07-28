# EEG-digital-twin-asd-therapy
EEG-Driven Digital Twin Models for AI and VR-Based Language Therapy in Children with Autism Spectrum Disorder

Personalized neuroadaptive therapy system that uses real-time EEG data to deliver adaptive language interventions for children with Autism Spectrum Disorder through AI-powered virtual reality environments.

Abstract
This project develops an innovative EEG-driven digital twin framework that simulates cognitive responses during language tasks for children with ASD. The system integrates real-time brain activity monitoring with AI decision-making and VR therapy delivery to provide personalized, neuroadaptive interventions that adjust content based on engagement, anxiety, attention, and processing efficiency.

Key Features

Real-time EEG Processing: 11 validated neurophysiological features including TBR, spectral entropy, and connectivity measures
Digital Twin Architecture: Bidirectional LSTM with attention mechanisms for cognitive state prediction
Neuroadaptive AI: Deep Q-Network for intelligent therapy action selection
VR Therapy Interface: Immersive environment with real-time parameter adjustment
Clinical Validation: Tested on auditory evoked potential EEG datasets with significant engagement improvements

Results Highlights

79.1% High Engagement Rate (improved from 44.2%)
Real-time Adaptation with <500ms latency
Validated EEG Features from ASD research literature
Neuroadaptive Learning through reinforcement learning


Quick Start
Prerequisites
bashPython 3.8+
TensorFlow 2.x
PyGame
MNE-Python
scikit-learn
pandas
numpy
Installation
bash# Clone the repository
git clone https://github.com/yourusername/eeg-digital-twin-asd-therapy.git
cd eeg-digital-twin-asd-therapy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate eeg-asd-therapy
Quick Demo
bash# Run the main simulation
python src/main.py

# Or run with sample data
python scripts/run_simulation.py --use-sample-data

Usage
Basic Usage
pythonfrom src.models.digital_twin import build_asd_digital_twin
from src.data.eeg_loader import load_real_eeg_features
from src.vr_interface.pygame_interface import VRTherapyInterface

# Load EEG data
eeg_features, feature_names = load_real_eeg_features('data/sample_eeg.csv')

# Create and train digital twin
digital_twin = build_asd_digital_twin()
# ... training code ...

# Initialize VR therapy interface
vr_interface = VRTherapyInterface()

# Run neuroadaptive therapy session
# ... therapy loop ...
Advanced Configuration
python# Custom model configuration
config = {
    'lstm_units': [512, 256, 128],
    'dropout_rate': 0.3,
    'learning_rate': 1e-4,
    'therapy_weights': [4.0, 1.5, 4.0, 4.0]  # [engagement, efficiency, anxiety, attention]
}

digital_twin = build_asd_digital_twin(config)

Architecture
System Overview
[EEG Sensors] → [Signal Processing] → [Digital Twin] → [RL Agent] → [VR Environment]
      ↓              ↓                    ↓             ↓            ↓
   Raw EEG    Feature Extraction   Cognitive States  Actions   Therapy Delivery
Key Components

EEG Processing Pipeline: Real-time feature extraction and preprocessing
Digital Twin Model: Bidirectional LSTM with attention for cognitive state prediction
Reinforcement Learning Agent: DQN for adaptive therapy action selection
VR Interface: Real-time environment adjustment based on brain states


Dataset
EEG Data Sources

Primary: Auditory Evoked Potential EEG-Biometric Dataset
Channels: T7, F8, Cz, P4 (strategically selected for ASD research)
Sampling Rate: 256 Hz (resampled from 200 Hz)
Duration: 120 seconds per session

Features Extracted
FeatureFrequency BandClinical RelevanceDelta Power1-4 HzDeep processing statesTheta Power4-7 HzAttention, memoryAlpha Power8-12 HzRelaxed awarenessBeta Power13-30 HzActive cognitionGamma Power30-40 HzConscious processingTBRθ/β ratioADHD/attention markerFrontal AsymmetryF8-T7Emotional regulationConnectivityInter-channelNeural communication

Research Results
Therapy Effectiveness Metrics
MetricRun 1Run 2TargetStatusHigh Engagement (>0.6)44.2%79.1%70% AchievedLow Anxiety (<0.4)44.2%44.2%90%⚠ In ProgressGood Attention (>0.5)20.9%20.9%50%⚠ Needs ImprovementTherapy Outcome Score1.642.453.0+ Improving
Model Performance

Validation Accuracy: 85.3%
Real-time Latency: <500ms
Feature Importance: TBR and frontal asymmetry most predictive


 Conference Materials
MODSIM World 2025

Paper: Personalized EEG-Driven Digital Twin Models for AI and VR-Based Language Therapy in Children with Autism Spectrum Disorder
Presentation: MODSIM 2025 Slides
Poster: Research Poster
Demo Video: System Demonstration

Citation
bibtex@inproceedings{islam2025eeg,
  title={Personalized EEG-Driven Digital Twin Models for AI and VR-Based Language Therapy in Children with Autism Spectrum Disorder},
  author={Islam, Nikita and Margondai, Ancuta and Von Ahlefeldt, Cindy and Ezcurra, Valentina and Hani, Soraya and Willox, Sara and Diaz, Anamaria Acevedo and Antanavicius, Emma and Mouloua, Mustapha},
  booktitle={MODSIM World 2025},
  year={2025},
  organization={University of Central Florida}
}

Testing
Run the test suite:
bash# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_models.py -v
python -m pytest tests/test_data_processing.py -v
python -m pytest tests/test_vr_interface.py -v

Future Roadmap
Short-term (3-6 months)

 Advanced regularization for overfitting mitigation
 Multimodal sensor integration (HRV, eye tracking)
 Enhanced VR environment with Unity integration
 Individual baseline calibration

Medium-term (6-12 months)

 Transformer-based architecture implementation
 Clinical validation study design
 Real-time cloud deployment
 Therapist dashboard development

Long-term (1-2 years)

 FDA regulatory pathway exploration
 Multi-site clinical trials
 Insurance reimbursement strategy
 Commercial deployment


Contributing
We welcome contributions! Please see our Contributing Guidelines for details.
Development Setup
bash# Clone and setup development environment
git clone https://github.com/yourusername/eeg-digital-twin-asd-therapy.git
cd eeg-digital-twin-asd-therapy

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

License
This project is licensed under the MIT License - see the LICENSE file for details.

Authors
Research Team - University of Central Florida

Nikita Islam - Lead Developer & Biomedical Sciences
Ancuta Margondai - Human-AI Teaming Research
Cindy Von Ahlefeldt - Cognitive Psychology & Digital Mental Health
Valentina Ezcurra - Adolescent Psychology & Anxiety Research
Soraya Hani - Research Contributor
Sara Willox, Ph.D. - Business Research & Data Analytics
Anamaria Acevedo Diaz - Psychology & Neuroscience
Emma Antanavicius - ASD & Speech-Language Research
Mustapha Mouloua, Ph.D. - Human Factors & Cognitive Psychology


📞 Contact

Primary Contact: Nikita Islam - Ni836085@ucf.edu
Research Supervisor: Dr. Mustapha Mouloua - Mustapha.Mouloua@ucf.edu
Institution: University of Central Florida, Orlando, FL


🙏 Acknowledgments

University of Central Florida Human Factors & Cognitive Psychology Lab
MODSIM World 2025 Conference Organization
Auditory Evoked Potential EEG-Biometric Dataset Contributors
ASD Research Community
