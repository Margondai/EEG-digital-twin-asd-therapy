# EEG-digital-twin-asd-therapy

# EEG-Driven Digital Twin Models for AI and VR-Based Language Therapy in Children with Autism Spectrum Disorder

## Abstract

This project develops an innovative EEG-driven digital twin framework that simulates cognitive responses during language tasks for children with Autism Spectrum Disorder (ASD). The system integrates real-time brain activity monitoring with artificial intelligence decision-making and virtual reality therapy delivery to provide personalized, neuroadaptive interventions that adjust content based on engagement, anxiety, attention, and processing efficiency.

## Overview

Children with Autism Spectrum Disorder often face significant challenges in language development and social communication. Despite available therapies, traditional interventions often overlook the cognitive and neurological diversity within this population, resulting in limited long-term impact. This research addresses this gap by developing a neuroadaptive system that uses real-time EEG data to drive personalized therapy adjustments.

### Key Features

- **Real-time EEG Processing**: Extracts 11 validated neurophysiological features including theta/beta ratio, spectral entropy, and connectivity measures
- **Digital Twin Architecture**: Bidirectional LSTM neural network with attention mechanisms for cognitive state prediction
- **Neuroadaptive AI**: Deep Q-Network for intelligent therapy action selection based on brain states
- **VR Therapy Interface**: Immersive environment with real-time parameter adjustment
- **Clinical Validation**: Tested on auditory evoked potential EEG datasets with significant engagement improvements

## System Architecture

The system consists of five main components:

1. **EEG Data Acquisition**: Real-time processing of brain signals from T7, F8, Cz, and P4 channels
2. **Signal Processing & Feature Extraction**: Computation of spectral power, biomarkers, and connectivity measures
3. **Digital Twin Model Training**: Bidirectional LSTM with attention mechanism for cognitive state prediction
4. **Intelligent Decision Making**: DQN agent for therapy action selection
5. **VR Environment Adaptation**: Real-time adjustment of therapy parameters based on brain states

## Results

### Therapy Effectiveness Metrics

- **High Engagement Rate**: 79.1% (improved from 44.2% baseline)
- **Real-time Adaptation**: System responds with less than 500ms latency
- **Validated EEG Features**: Based on established ASD research literature
- **Neuroadaptive Learning**: Continuous improvement through reinforcement learning

The system demonstrated significant improvements in engagement prediction while maintaining consistent performance in anxiety detection and attention monitoring.

## Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- Required dependencies listed in `requirements.txt`

### Setup

```bash
git clone https://github.com/yourusername/eeg-digital-twin-asd-therapy.git
cd eeg-digital-twin-asd-therapy
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main simulation with demonstration data:

```bash
python src/main.py --demo
```

### Advanced Usage

Run with custom EEG files:

```bash
python src/main.py --eeg-files data/your_eeg_file.csv --epochs 100
```

Run without VR interface (headless mode):

```bash
python src/main.py --demo --no-vr
```

### Configuration

The system can be configured through YAML files in the `config/` directory. Key parameters include:

- EEG processing settings (sampling rate, frequency bands)
- Digital twin model architecture
- Therapy action weights
- VR interface parameters

## Dataset

### EEG Data Sources

The system has been tested with the Auditory Evoked Potential EEG-Biometric Dataset, using the following specifications:

- **Channels**: T7, F8, Cz, P4 (strategically selected for ASD research)
- **Sampling Rate**: 256 Hz (resampled from 200 Hz)  
- **Duration**: 120 seconds per session
- **Preprocessing**: 1-40 Hz bandpass filtering using FIR filter

### Feature Extraction

Eleven neurophysiological features are extracted:

| Feature | Frequency Band | Clinical Relevance |
|---------|---------------|-------------------|
| Delta Power | 1-4 Hz | Deep processing states |
| Theta Power | 4-7 Hz | Attention and memory |
| Alpha Power | 8-12 Hz | Relaxed awareness |
| Beta Power | 13-30 Hz | Active cognition |
| Gamma Power | 30-40 Hz | Conscious processing |
| Theta/Beta Ratio | - | ADHD/attention marker |
| Alpha/Theta Ratio | - | Cognitive efficiency |
| Gamma/Theta Ratio | - | Cognitive binding |
| Frontal Asymmetry | F8-T7 | Emotional regulation |
| Connectivity | Inter-channel | Neural communication |
| Spectral Entropy | - | Signal complexity |

## Methodology

### Digital Twin Model

The digital twin employs a bidirectional LSTM neural network architecture:

- **Input**: 10-second EEG windows (2560 timesteps × 11 features)
- **Architecture**: Three bidirectional LSTM layers (512, 256, 128 units)
- **Attention Mechanism**: Temporal dependency modeling
- **Output**: Four cognitive states (engagement, efficiency, anxiety, attention)
- **Training**: Custom loss function with therapy-specific weighting

### Reinforcement Learning Agent

A Deep Q-Network (DQN) selects optimal therapy actions:

- **State Space**: Four predicted cognitive states
- **Action Space**: Six therapy interventions
- **Reward Function**: Weighted combination prioritizing engagement and attention while minimizing anxiety
- **Learning**: Experience replay with epsilon-greedy exploration

### VR Therapy Interface

The virtual reality environment adapts in real-time:

- **Parameters**: Language complexity, sensory intensity, social interaction level
- **Update Frequency**: 0.5-second intervals
- **Feedback**: Visual and auditory cues based on cognitive states
- **Gamification**: Progress tracking and reward systems

## File Structure

```
eeg-digital-twin-asd-therapy/
├── src/
│   ├── main.py                 # Main application entry point
│   ├── data/                   # EEG data loading and preprocessing
│   ├── models/                 # Digital twin and RL models
│   ├── vr_interface/           # VR therapy interface
│   └── utils/                  # Configuration and utilities
├── data/
│   ├── sample_data/            # Demonstration datasets
│   └── processed/              # Processed EEG features
├── notebooks/                  # Jupyter analysis notebooks
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── results/                    # Experimental results
├── scripts/                    # Utility scripts
└── conference/                 # MODSIM 2025 materials
```

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

Run specific test categories:

```bash
python -m pytest tests/test_models.py -v
python -m pytest tests/test_data_processing.py -v
```

## Contributing

This is an academic research project. For contributions or collaborations, please contact the authors directly.

## Conference Presentation

This work was presented at MODSIM World 2025. Conference materials including the full paper, presentation slides, and supplementary materials are available in the `conference/` directory.

## Citation

If you use this software in your research, please cite:

```
Islam, N., Margondai, A., Von Ahlefeldt, C., Ezcurra, V., Hani, S., 
Willox, S., Diaz, A. A., Antanavicius, E., & Mouloua, M. (2025). 
Personalized EEG-Driven Digital Twin Models for AI and VR-Based 
Language Therapy in Children with Autism Spectrum Disorder. 
MODSIM World 2025, Orlando, FL.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about this research, please contact:

- Nikita Islam: Ni836085@ucf.edu
- Ancuta Margondai: Ancuta.Margondai@ucf.edu
- Dr. Mustapha Mouloua: Mustapha.Mouloua@ucf.edu

University of Central Florida  
Orlando, Florida
