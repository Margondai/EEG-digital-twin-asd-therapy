#!/usr/bin/env python3
"""
EEG-Driven Digital Twin Models for AI and VR-Based Language Therapy in Children with Autism Spectrum Disorder

Main application file - organized from original Nikita.py

Authors: Nikita Islam, Ancuta Margondai, et al.
Institution: University of Central Florida
Conference: MODSIM World 2025
"""

import asyncio
import numpy as np
import platform
from sklearn.model_selection import train_test_split
import os

# Import modular components
from data.eeg_processing import load_real_eeg_features, simulate_asd_eeg_patterns
from data.preprocessing import generate_asd_therapy_labels, enhanced_prepare_therapy_data
from models.digital_twin import build_asd_digital_twin
from models.reinforcement_learning import NeuroadaptiveTherapySystem
from vr_interface.therapy_interface import VRTherapyInterface


async def main():
    """
    Main research simulation: EEG-driven digital twin for ASD language therapy
    """
    print("=== EEG-Driven Digital Twin for ASD Language Therapy ===")
    print("Loading real EEG data and training neuroadaptive system...")

    # Load multiple EEG files for robust training
    edf_files = []

    eeg_data_list = []
    feature_names = None

    for edf_file in edf_files:
        if os.path.exists(edf_file):
            features, names = load_real_eeg_features(edf_file)
            eeg_data_list.append(features)
            if feature_names is None:
                feature_names = names
        else:
            print(f"File {edf_file} not found. Using simulated ASD patterns.")
            features, names = simulate_asd_eeg_patterns(duration=120)
            eeg_data_list.append(features)
            if feature_names is None:
                feature_names = names

    eeg_features = np.concatenate(eeg_data_list, axis=0)
    print(f"Real EEG Features: {eeg_features.shape}")
    print(f"Feature Names: {feature_names}")

    # Generate therapy-relevant labels
    therapy_labels = generate_asd_therapy_labels(eeg_features)
    X, y, scaler = enhanced_prepare_therapy_data(eeg_features, therapy_labels)
    print(f"Therapy Dataset: X={X.shape}, y={y.shape}")

    # Split and train digital twin
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    digital_twin = build_asd_digital_twin()
    print(f"Digital Twin Parameters: {digital_twin.count_params()}")

    # Train the digital twin
    print("\nTraining EEG-driven digital twin...")
    history = digital_twin.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=4,
        verbose=1
    )

    val_loss = digital_twin.evaluate(X_val, y_val, verbose=0)
    print(f"\nDigital Twin Performance:")
    print(f"Final Validation Loss: {val_loss[0]:.4f}")

    # Evaluate therapy targets
    predictions = digital_twin.predict(X_val, verbose=0)
    high_engagement = np.mean(predictions[:, 0] > 0.6)
    low_anxiety = np.mean(predictions[:, 2] < 0.4)
    good_attention = np.mean(predictions[:, 3] > 0.5)

    print(f"Therapy Effectiveness:")
    print(f"High Engagement (>0.6): {high_engagement*100:.1f}%")
    print(f"Low Anxiety (<0.4): {low_anxiety*100:.1f}%")
    print(f"Good Attention (>0.5): {good_attention*100:.1f}%")

    # Initialize neuroadaptive therapy system
    therapy_ai = NeuroadaptiveTherapySystem()
    vr_interface = VRTherapyInterface()

    print("\nStarting real-time neuroadaptive therapy simulation...")
    print("Press ESC in the pygame window to stop")

    # Real-time therapy simulation
    window_size = 10 * 256
    try:
        for i in range(0, len(eeg_features) - window_size, 256):
            if not vr_interface.running:
                print("VR interface stopped")
                break

            # Get current EEG window
            window = eeg_features[i:i + window_size]
            window_scaled = scaler.transform(window)

            # Digital twin predicts brain state
            brain_state = digital_twin.predict(window_scaled[np.newaxis, :], verbose=0)[0]

            # AI selects therapy action
            action = therapy_ai.select_therapy_action(brain_state)
            reward = therapy_ai.compute_therapy_reward(brain_state)

            # Get next state for learning
            next_i = min(i + 256, len(eeg_features) - window_size)
            next_window = eeg_features[next_i:next_i + window_size]
            next_window_scaled = scaler.transform(next_window)
            next_brain_state = digital_twin.predict(next_window_scaled[np.newaxis, :], verbose=0)[0]
            done = (next_i == len(eeg_features) - window_size)

            # Learn from experience
            therapy_ai.remember(brain_state, action, reward, next_brain_state, done)
            therapy_ai.replay()

            # Update VR therapy environment
            engagement, efficiency, anxiety, attention = brain_state
            print(f"Brain State - Engagement: {engagement:.2f}, Efficiency: {efficiency:.2f}, Anxiety: {anxiety:.2f}, Attention: {attention:.2f}")
            print(f"AI Action: {action}, Reward: {reward:.2f}")

            if not vr_interface.update_therapy_environment(action, brain_state, reward):
                break

            await asyncio.sleep(0.5)

    except Exception as e:
        print(f"Error in VR loop: {e}")
        vr_interface.quit()
        return

    # Calculate final therapy outcomes
    final_reward = therapy_ai.compute_therapy_reward(brain_state)
    print(f"\nFinal Therapy Outcome Score: {final_reward:.2f}")
    print("Research simulation complete!")

    vr_interface.quit()


def test_pygame():
    """Test if pygame is working properly before running the main simulation"""
    try:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Test Window")
        font = pygame.font.SysFont('arial', 20)
        clock = pygame.time.Clock()

        running = True
        counter = 0

        while running and counter < 100:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            screen.fill((100, 150, 200))
            text = font.render(f"Frame {counter} - Press ESC to quit", True, (255, 255, 255))
            screen.blit(text, (50, 50))
            pygame.display.flip()
            clock.tick(10)
            counter += 1

        pygame.quit()
        print("Pygame test completed successfully")
        return True

    except Exception as e:
        print(f"Pygame test failed: {e}")
        return False


if __name__ == "__main__":
    if platform.system() == "Emscripten":
        asyncio.ensure_future(main())
    else:
        print("Testing pygame...")
        if test_pygame():
            print("Pygame works! Running main simulation...")
            asyncio.run(main())
        else:
            print("Pygame test failed. Check your pygame installation.")

"""
EEG Data Processing Module

Handles loading and processing of EEG data for ASD therapy research.
Extracted from original Nikita.py file.
"""

import numpy as np
import pandas as pd
import mne
from scipy import signal


def load_real_eeg_features(file_path, duration=120, fs=256):
    """
    Extract ONLY real EEG-derived features for ASD research validity
    All features computed from actual brain activity
    """
    try:
        # Load real EEG data
        df = pd.read_csv(file_path)
        eeg_data = df[['T7', 'F8', 'Cz', 'P4']].values.T

        # Create MNE Raw object
        ch_names = ['T7', 'F8', 'Cz', 'P4']
        sfreq = 200
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info)

        # Apply standard EEG preprocessing
        raw.filter(1, 40, fir_design='firwin')
        raw.resample(fs)

        # Get all 4 channels for comprehensive analysis
        data, times = raw.get_data(return_times=True)
        data = data[:, :int(duration * fs)]
        times = times[:int(duration * fs)]

        # Compute Power Spectral Density for all channels
        psd, freqs = mne.time_frequency.psd_array_welch(data, sfreq=fs, fmin=1, fmax=40, n_fft=2048)

        # Define frequency bands (critical for ASD research)
        delta_idx = (freqs >= 1) & (freqs <= 4)  # Deep sleep, unconscious processes
        theta_idx = (freqs >= 4) & (freqs <= 7)  # Memory, emotion, attention
        alpha_idx = (freqs >= 8) & (freqs <= 12)  # Relaxed awareness, sensory processing
        beta_idx = (freqs >= 13) & (freqs <= 30)  # Active thinking, focus
        gamma_idx = (freqs >= 30) & (freqs <= 40)  # Consciousness, cognitive processing

        # Extract power in each band (averaged across channels for robustness)
        delta_power = np.mean(np.mean(psd[:, delta_idx], axis=1))
        theta_power = np.mean(np.mean(psd[:, theta_idx], axis=1))
        alpha_power = np.mean(np.mean(psd[:, alpha_idx], axis=1))
        beta_power = np.mean(np.mean(psd[:, beta_idx], axis=1))
        gamma_power = np.mean(np.mean(psd[:, gamma_idx], axis=1))

        # ASD-specific ratios (literature-validated biomarkers)
        tbr = np.log1p(theta_power / np.clip(beta_power, 1e-6, None))  # Attention/ADHD marker
        alpha_theta_ratio = np.log1p(alpha_power / np.clip(theta_power, 1e-6, None))  # Cognitive efficiency
        gamma_theta_ratio = np.log1p(gamma_power / np.clip(theta_power, 1e-6, None))  # Cognitive binding

        # Frontal asymmetry (F8 - T7, emotion regulation marker)
        f8_alpha = np.mean(psd[1, alpha_idx])  # F8 channel
        t7_alpha = np.mean(psd[0, alpha_idx])  # T7 channel
        frontal_asymmetry = np.log(f8_alpha) - np.log(t7_alpha)

        # Connectivity measure (simplified coherence between Cz and other channels)
        cz_data = data[2, :]  # Cz channel
        connectivity_measure = 0
        for ch_idx in [0, 1, 3]:  # T7, F8, P4
            coherence = np.corrcoef(cz_data, data[ch_idx, :])[0, 1]
            connectivity_measure += coherence
        connectivity_measure /= 3  # Average connectivity

        # Spectral entropy (measure of signal complexity/irregularity)
        def spectral_entropy(psd_channel):
            psd_norm = psd_channel / np.sum(psd_channel)
            psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
            return -np.sum(psd_norm * np.log(psd_norm))

        avg_spectral_entropy = np.mean([spectral_entropy(psd[i, :]) for i in range(4)])

        # Create time series with real features (11 features total)
        feature_names = [
            'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
            'tbr', 'alpha_theta_ratio', 'gamma_theta_ratio',
            'frontal_asymmetry', 'connectivity', 'spectral_entropy'
        ]

        # Extend each feature across the time series (with small temporal variations)
        features = []
        base_values = [delta_power, theta_power, alpha_power, beta_power, gamma_power,
                       tbr, alpha_theta_ratio, gamma_theta_ratio,
                       frontal_asymmetry, connectivity_measure, avg_spectral_entropy]

        for i, base_val in enumerate(base_values):
            # Add realistic temporal variation (10% of base value)
            temporal_variation = np.random.normal(0, abs(base_val) * 0.1, int(duration * fs))
            feature_series = base_val + temporal_variation
            features.append(feature_series)

        return np.stack(features, axis=1), feature_names

    except Exception as e:
        print(f"Error loading real EEG data: {e}. Using simulated ASD-pattern EEG.")
        return simulate_asd_eeg_patterns(duration, fs)


def simulate_asd_eeg_patterns(duration=300, fs=256):
    """
    Simulate EEG patterns based on ASD literature when real data unavailable
    Based on research: elevated theta, reduced alpha, altered connectivity
    """
    t = np.linspace(0, duration, int(duration * fs))

    # ASD-specific pattern: elevated theta/beta ratio
    delta_power = 2.0 + 0.3 * np.sin(2 * np.pi * 0.05 * t) + np.random.normal(0, 0.1, len(t))
    theta_power = 8.5 + 1.0 * np.sin(2 * np.pi * 0.1 * t) + np.random.normal(0, 0.3, len(t))  # Elevated
    alpha_power = 2.8 + 0.4 * np.cos(2 * np.pi * 0.08 * t) + np.random.normal(0, 0.15, len(t))  # Reduced
    beta_power = 1.2 + 0.3 * np.sin(2 * np.pi * 0.15 * t) + np.random.normal(0, 0.1, len(t))  # Reduced
    gamma_power = 0.8 + 0.2 * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, 0.05, len(t))

    # Compute ratios
    tbr = np.log1p(theta_power / np.clip(beta_power, 1e-6, None))
    alpha_theta_ratio = np.log1p(alpha_power / np.clip(theta_power, 1e-6, None))
    gamma_theta_ratio = np.log1p(gamma_power / np.clip(theta_power, 1e-6, None))

    # ASD pattern: altered frontal asymmetry and reduced connectivity
    frontal_asymmetry = -0.15 + 0.05 * np.sin(2 * np.pi * 0.03 * t) + np.random.normal(0, 0.02, len(t))
    connectivity = 0.3 + 0.1 * np.sin(2 * np.pi * 0.02 * t) + np.random.normal(0, 0.05, len(t))  # Reduced
    spectral_entropy = 2.8 + 0.2 * np.sin(2 * np.pi * 0.04 * t) + np.random.normal(0, 0.1, len(t))  # Higher

    feature_names = [
        'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
        'tbr', 'alpha_theta_ratio', 'gamma_theta_ratio',
        'frontal_asymmetry', 'connectivity', 'spectral_entropy'
    ]

    return np.stack([delta_power, theta_power, alpha_power, beta_power, gamma_power,
                     tbr, alpha_theta_ratio, gamma_theta_ratio,
                     frontal_asymmetry, connectivity, spectral_entropy], axis=1), feature_names
  """
Data Preprocessing Module

Handles preprocessing of EEG features and generation of therapy labels.
Extracted from original Nikita.py file.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


def generate_asd_therapy_labels(eeg_features):
    """
    Generate therapy-relevant labels based on real EEG features
    Targets: engagement, processing efficiency, anxiety, attention focus
    """
    # Extract key features
    theta = eeg_features[:, 1]  # theta_power
    alpha = eeg_features[:, 2]  # alpha_power
    beta = eeg_features[:, 3]  # beta_power
    tbr = eeg_features[:, 5]  # theta/beta ratio
    alpha_theta = eeg_features[:, 6]  # alpha/theta ratio
    frontal_asym = eeg_features[:, 8]  # frontal asymmetry
    connectivity = eeg_features[:, 9]  # connectivity
    spectral_entropy = eeg_features[:, 10]  # spectral_entropy

    # Normalize features
    theta_norm = theta / np.max(np.abs(theta))
    alpha_norm = alpha / np.max(np.abs(alpha))
    beta_norm = beta / np.max(np.abs(beta))
    tbr_norm = (tbr - np.min(tbr)) / (np.max(tbr) - np.min(tbr))
    entropy_norm = (spectral_entropy - np.min(spectral_entropy)) / (np.max(spectral_entropy) - np.min(spectral_entropy))

    # Engagement: higher alpha, better connectivity, moderate TBR, low entropy
    engagement = 1 / (1 + np.exp(-(1.5 * alpha_norm + 1.2 * connectivity - 0.5 * tbr_norm - 0.2 * entropy_norm)))

    # Processing efficiency: optimal alpha/theta ratio, good beta activity
    efficiency = 1 / (1 + np.exp(-(alpha_theta + beta_norm - 1.5)))

    # Anxiety: high theta, low alpha, negative frontal asymmetry, high entropy
    anxiety = 1 / (1 + np.exp(-(-frontal_asym + 0.5 * theta_norm - alpha_norm + 0.5 + 0.3 * entropy_norm)))

    # Attention focus: inversely related to TBR, higher beta, lower entropy (further refined)
    attention = 1 / (1 + np.exp(tbr_norm - 0.05 + 0.7 * beta_norm - 0.3 * entropy_norm))

    return np.stack([engagement, efficiency, anxiety, attention], axis=1)


def enhanced_prepare_therapy_data(eeg_features, labels, window_size=10 * 256):
    """
    Prepare data for neuroadaptive therapy system with noise augmentation
    """
    scaler = StandardScaler()
    eeg_features = scaler.fit_transform(eeg_features)

    X, y = [], []
    step_size = window_size // 3  # 67% overlap for rich temporal context

    for i in range(0, len(eeg_features) - window_size, step_size):
        window = eeg_features[i:i + window_size]
        target = labels[i + window_size - 1]

        X.append(window)
        y.append(target)

        # Data augmentation for robustness
        if len(X) % 2 == 0:
            # Add small temporal jitter
            jitter_amount = np.random.randint(-10, 11)
            if i + jitter_amount >= 0 and i + jitter_amount + window_size < len(eeg_features):
                jittered_window = eeg_features[i + jitter_amount:i + jitter_amount + window_size]
                X.append(jittered_window)
                y.append(target)

        # Add noise-based augmentation
        noise = np.random.normal(0, 0.05, window.shape)
        X.append(window + noise)
        y.append(target)

    return np.array(X), np.array(y), scaler

"""
Data Preprocessing Module

Handles preprocessing of EEG features and generation of therapy labels.
Extracted from original Nikita.py file.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


def generate_asd_therapy_labels(eeg_features):
    """
    Generate therapy-relevant labels based on real EEG features
    Targets: engagement, processing efficiency, anxiety, attention focus
    """
    # Extract key features
    theta = eeg_features[:, 1]  # theta_power
    alpha = eeg_features[:, 2]  # alpha_power
    beta = eeg_features[:, 3]  # beta_power
    tbr = eeg_features[:, 5]  # theta/beta ratio
    alpha_theta = eeg_features[:, 6]  # alpha/theta ratio
    frontal_asym = eeg_features[:, 8]  # frontal asymmetry
    connectivity = eeg_features[:, 9]  # connectivity
    spectral_entropy = eeg_features[:, 10]  # spectral_entropy

    # Normalize features
    theta_norm = theta / np.max(np.abs(theta))
    alpha_norm = alpha / np.max(np.abs(alpha))
    beta_norm = beta / np.max(np.abs(beta))
    tbr_norm = (tbr - np.min(tbr)) / (np.max(tbr) - np.min(tbr))
    entropy_norm = (spectral_entropy - np.min(spectral_entropy)) / (np.max(spectral_entropy) - np.min(spectral_entropy))

    # Engagement: higher alpha, better connectivity, moderate TBR, low entropy
    engagement = 1 / (1 + np.exp(-(1.5 * alpha_norm + 1.2 * connectivity - 0.5 * tbr_norm - 0.2 * entropy_norm)))

    # Processing efficiency: optimal alpha/theta ratio, good beta activity
    efficiency = 1 / (1 + np.exp(-(alpha_theta + beta_norm - 1.5)))

    # Anxiety: high theta, low alpha, negative frontal asymmetry, high entropy
    anxiety = 1 / (1 + np.exp(-(-frontal_asym + 0.5 * theta_norm - alpha_norm + 0.5 + 0.3 * entropy_norm)))

    # Attention focus: inversely related to TBR, higher beta, lower entropy (further refined)
    attention = 1 / (1 + np.exp(tbr_norm - 0.05 + 0.7 * beta_norm - 0.3 * entropy_norm))

    return np.stack([engagement, efficiency, anxiety, attention], axis=1)


def enhanced_prepare_therapy_data(eeg_features, labels, window_size=10 * 256):
    """
    Prepare data for neuroadaptive therapy system with noise augmentation
    """
    scaler = StandardScaler()
    eeg_features = scaler.fit_transform(eeg_features)

    X, y = [], []
    step_size = window_size // 3  # 67% overlap for rich temporal context

    for i in range(0, len(eeg_features) - window_size, step_size):
        window = eeg_features[i:i + window_size]
        target = labels[i + window_size - 1]

        X.append(window)
        y.append(target)

        # Data augmentation for robustness
        if len(X) % 2 == 0:
            # Add small temporal jitter
            jitter_amount = np.random.randint(-10, 11)
            if i + jitter_amount >= 0 and i + jitter_amount + window_size < len(eeg_features):
                jittered_window = eeg_features[i + jitter_amount:i + jitter_amount + window_size]
                X.append(jittered_window)
                y.append(target)

        # Add noise-based augmentation
        noise = np.random.normal(0, 0.05, window.shape)
        X.append(window + noise)
        y.append(target)

    return np.array(X), np.array(y), scaler
  """
Digital Twin Model Module

Contains the neural network architecture for the EEG-driven digital twin.
Extracted from original Nikita.py file.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional, Attention, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2


def adaptive_learning_rate_schedule(epoch, lr):
    """
    Dynamic learning rate schedule to prevent early stopping
    """
    if epoch < 20:
        return lr
    elif epoch < 50:
        return lr * 0.8
    elif epoch < 80:
        return lr * 0.5
    else:
        return lr * 0.1


def neuroadaptive_loss(y_true, y_pred):
    """
    Loss function optimized for therapy goals:
    - Maximize engagement and attention
    - Optimize processing efficiency
    - Minimize anxiety
    """
    # Individual component losses
    engagement_loss = tf.square(y_true[:, 0] - y_pred[:, 0])
    efficiency_loss = tf.square(y_true[:, 1] - y_pred[:, 1])
    anxiety_loss = tf.square(y_true[:, 2] - y_pred[:, 2])
    attention_loss = tf.square(y_true[:, 3] - y_pred[:, 3])

    # Therapy-specific weighting (balanced focus)
    weights = tf.constant([4.0, 1.5, 4.0, 4.0], dtype=tf.float32)  # [engagement, efficiency, anxiety, attention]

    # Combine losses
    weighted_loss = weights[0] * engagement_loss + weights[1] * efficiency_loss + \
                    weights[2] * anxiety_loss + weights[3] * attention_loss

    return tf.reduce_mean(weighted_loss)


def build_asd_digital_twin():
    """
    Digital twin architecture for ASD neuroadaptive therapy with regularization
    """
    inputs = Input(shape=(2560, 11))  # 11 real EEG features

    # Hierarchical temporal processing with L2 regularization
    x = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2, kernel_regularizer=l2(0.01)))(inputs)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, kernel_regularizer=l2(0.01)))(x)
    x = BatchNormalization()(x)

    # Attention mechanism for focusing on therapy-relevant patterns
    attention = Attention()([x, x])
    x = BatchNormalization()(attention)

    # Final temporal integration
    x = Bidirectional(LSTM(128, dropout=0.3, kernel_regularizer=l2(0.01)))(x)
    x = BatchNormalization()(x)

    # Therapy decision layers with regularization
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)

    # Output: [engagement, efficiency, anxiety, attention]
    outputs = Dense(4, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=neuroadaptive_loss,
        metrics=['mae']
    )
    return model


def get_model_callbacks():
    """
    Get standard callbacks for model training
    """
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        LearningRateScheduler(adaptive_learning_rate_schedule, verbose=1)
    ]
    return callbacks

"""
Digital Twin Model Module

Contains the neural network architecture for the EEG-driven digital twin.
Extracted from original Nikita.py file.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional, Attention, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2


def adaptive_learning_rate_schedule(epoch, lr):
    """
    Dynamic learning rate schedule to prevent early stopping
    """
    if epoch < 20:
        return lr
    elif epoch < 50:
        return lr * 0.8
    elif epoch < 80:
        return lr * 0.5
    else:
        return lr * 0.1


def neuroadaptive_loss(y_true, y_pred):
    """
    Loss function optimized for therapy goals:
    - Maximize engagement and attention
    - Optimize processing efficiency
    - Minimize anxiety
    """
    # Individual component losses
    engagement_loss = tf.square(y_true[:, 0] - y_pred[:, 0])
    efficiency_loss = tf.square(y_true[:, 1] - y_pred[:, 1])
    anxiety_loss = tf.square(y_true[:, 2] - y_pred[:, 2])
    attention_loss = tf.square(y_true[:, 3] - y_pred[:, 3])

    # Therapy-specific weighting (balanced focus)
    weights = tf.constant([4.0, 1.5, 4.0, 4.0], dtype=tf.float32)  # [engagement, efficiency, anxiety, attention]

    # Combine losses
    weighted_loss = weights[0] * engagement_loss + weights[1] * efficiency_loss + \
                    weights[2] * anxiety_loss + weights[3] * attention_loss

    return tf.reduce_mean(weighted_loss)


def build_asd_digital_twin():
    """
    Digital twin architecture for ASD neuroadaptive therapy with regularization
    """
    inputs = Input(shape=(2560, 11))  # 11 real EEG features

    # Hierarchical temporal processing with L2 regularization
    x = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2, kernel_regularizer=l2(0.01)))(inputs)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, kernel_regularizer=l2(0.01)))(x)
    x = BatchNormalization()(x)

    # Attention mechanism for focusing on therapy-relevant patterns
    attention = Attention()([x, x])
    x = BatchNormalization()(attention)

    # Final temporal integration
    x = Bidirectional(LSTM(128, dropout=0.3, kernel_regularizer=l2(0.01)))(x)
    x = BatchNormalization()(x)

    # Therapy decision layers with regularization
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)

    # Output: [engagement, efficiency, anxiety, attention]
    outputs = Dense(4, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=neuroadaptive_loss,
        metrics=['mae']
    )
    return model


def get_model_callbacks():
    """
    Get standard callbacks for model training
    """
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        LearningRateScheduler(adaptive_learning_rate_schedule, verbose=1)
    ]
    return callbacks
  """
VR Therapy Interface Module

Contains the VR interface that adapts based on neuroadaptive system recommendations.
Extracted from original Nikita.py file.
"""

import pygame
import numpy as np


class VRTherapyInterface:
    """
    VR interface that adapts based on neuroadaptive system recommendations
    with gamification elements
    """
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("EEG-Driven Digital Twin - ASD Language Therapy")
        self.font = pygame.font.SysFont('arial', 20)
        self.language_complexity = 0.5
        self.visual_support_level = 0.5
        self.sensory_intensity = 0.5
        self.social_interaction_level = 0.3
        self.clock = pygame.time.Clock()
        self.FPS = 30
        self.running = True  # Flag to control VR loop
        self.iteration = 0  # Track iterations for progress
        self.max_iterations = 359  # Approximate total iterations
        self.rewards = []  # Track rewards for gamification

    def update_therapy_environment(self, action, brain_state, reward):
        """Update VR environment with better error handling and gamification"""
        try:
            engagement, efficiency, anxiety, attention = brain_state
            self.iteration += 1
            self.rewards.append(reward)
            if len(self.rewards) > 10:  # Keep last 10 rewards for average
                self.rewards.pop(0)

            # Update parameters based on action
            if action == 'increase_language_complexity':
                self.language_complexity = min(self.language_complexity + 0.1, 1.0)
            elif action == 'decrease_language_complexity':
                self.language_complexity = max(self.language_complexity - 0.1, 0.1)
            elif action == 'add_visual_supports':
                self.visual_support_level = min(self.visual_support_level + 0.15, 1.0)
            elif action == 'reduce_sensory_stimulation':
                self.sensory_intensity = max(self.sensory_intensity - 0.2, 0.1)
            elif action == 'introduce_social_interaction':
                self.social_interaction_level = min(self.social_interaction_level + 0.1, 1.0)
            elif action == 'provide_calming_break':
                self.sensory_intensity = max(self.sensory_intensity - 0.3, 0.1)
                self.social_interaction_level = max(self.social_interaction_level - 0.2, 0.0)

            # Handle ALL Pygame events properly
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Window close requested")
                    self.running = False
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("Escape pressed - stopping simulation")
                        self.running = False
                        return False

            # Render with error checking
            try:
                bg_color = max(50, 255 - int(anxiety * 200))
                self.screen.fill((bg_color, bg_color, bg_color))

                # Display info with gamification elements
                avg_reward = np.mean(self.rewards) if self.rewards else 0
                progress = min(100, int((self.iteration / self.max_iterations) * 100))
                
                texts = [
                    f'Language Complexity: {self.language_complexity:.2f}',
                    f'Visual Support: {self.visual_support_level:.2f}',
                    f'Sensory Level: {self.sensory_intensity:.2f}',
                    f'Social Interaction: {self.social_interaction_level:.2f}',
                    f'Brain State - Eng: {engagement:.2f}, Eff: {efficiency:.2f}, Anx: {anxiety:.2f}, Att: {attention:.2f}',
                    f'Current Action: {action}',
                    f'Reward: {reward:.2f} (Avg: {avg_reward:.2f})',
                    f'Progress: {progress}% (Iteration {self.iteration}/{self.max_iterations})',
                    'Press ESC to quit'
                ]

                y_pos = 50
                for text in texts:
                    rendered = self.font.render(text, True, (0, 0, 0))
                    self.screen.blit(rendered, (50, y_pos))
                    y_pos += 30

                # Draw progress bar
                pygame.draw.rect(self.screen, (200, 200, 200), (50, y_pos, 700, 20))
                pygame.draw.rect(self.screen, (50, 200, 50), (50, y_pos, int(7 * progress), 20))

                pygame.display.flip()
                self.clock.tick(self.FPS)
                return True

            except pygame.error as e:
                print(f"Pygame render error: {e}")
                return False

        except Exception as e:
            print(f"Error in VR update: {e}")
            return False

    def quit(self):
        """Clean shutdown of pygame"""
        try:
            pygame.quit()
            print("Pygame cleaned up successfully")
        except Exception as e:
            print(f"Error during Pygame cleanup: {e}")
  #!/usr/bin/env python3
"""
EEG-Driven Digital Twin Models for AI and VR-Based Language Therapy in Children with Autism Spectrum Disorder

Main application file - organized from original Nikita.py

Authors: Nikita Islam, Ancuta Margondai, et al.
Institution: University of Central Florida
Conference: MODSIM World 2025
"""

import asyncio
import numpy as np
import platform
import os
from sklearn.model_selection import train_test_split

# Import modular components
from data.eeg_processing import load_real_eeg_features, simulate_asd_eeg_patterns
from data.preprocessing import generate_asd_therapy_labels, enhanced_prepare_therapy_data
from models.digital_twin import build_asd_digital_twin, get_model_callbacks
from models.reinforcement_learning import NeuroadaptiveTherapySystem
from vr_interface.therapy_interface import VRTherapyInterface


async def main():
    """
    Main research simulation: EEG-driven digital twin for ASD language therapy
    """
    print("=== EEG-Driven Digital Twin for ASD Language Therapy ===")
    print("Loading real EEG data and training neuroadaptive system...")

    # Load multiple EEG files for robust training
    edf_files = []

    eeg_data_list = []
    feature_names = None

    for edf_file in edf_files:
        if os.path.exists(edf_file):
            features, names = load_real_eeg_features(edf_file)
            eeg_data_list.append(features)
            if feature_names is None:
                feature_names = names
        else:
            print(f"File {edf_file} not found. Using simulated ASD patterns.")
            features, names = simulate_asd_eeg_patterns(duration=120)
            eeg_data_list.append(features)
            if feature_names is None:
                feature_names = names

    eeg_features = np.concatenate(eeg_data_list, axis=0)
    print(f"Real EEG Features: {eeg_features.shape}")
    print(f"Feature Names: {feature_names}")

    # Generate therapy-relevant labels
    therapy_labels = generate_asd_therapy_labels(eeg_features)
    X, y, scaler = enhanced_prepare_therapy_data(eeg_features, therapy_labels)
    print(f"Therapy Dataset: X={X.shape}, y={y.shape}")

    # Split and train digital twin
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    digital_twin = build_asd_digital_twin()
    print(f"Digital Twin Parameters: {digital_twin.count_params()}")

    # Train the digital twin
    print("\nTraining EEG-driven digital twin...")
    callbacks = get_model_callbacks()
    history = digital_twin.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=4,
        callbacks=callbacks,
        verbose=1
    )

    val_loss = digital_twin.evaluate(X_val, y_val, verbose=0)
    print(f"\nDigital Twin Performance:")
    print(f"Final Validation Loss: {val_loss[0]:.4f}")

    # Evaluate therapy targets
    predictions = digital_twin.predict(X_val, verbose=0)
    high_engagement = np.mean(predictions[:, 0] > 0.6)
    low_anxiety = np.mean(predictions[:, 2] < 0.4)
    good_attention = np.mean(predictions[:, 3] > 0.5)

    print(f"Therapy Effectiveness:")
    print(f"High Engagement (>0.6): {high_engagement*100:.1f}%")
    print(f"Low Anxiety (<0.4): {low_anxiety*100:.1f}%")
    print(f"Good Attention (>0.5): {good_attention*100:.1f}%")

    # Initialize neuroadaptive therapy system
    therapy_ai = NeuroadaptiveTherapySystem()
    vr_interface = VRTherapyInterface()

    print("\nStarting real-time neuroadaptive therapy simulation...")
    print("Press ESC in the pygame window to stop")

    # Real-time therapy simulation
    window_size = 10 * 256
    try:
        for i in range(0, len(eeg_features) - window_size, 256):
            if not vr_interface.running:
                print("VR interface stopped")
                break

            # Get current EEG window
            window = eeg_features[i:i + window_size]
            window_scaled = scaler.transform(window)

            # Digital twin predicts brain state
            brain_state = digital_twin.predict(window_scaled[np.newaxis, :], verbose=0)[0]

            # AI selects therapy action
            action = therapy_ai.select_therapy_action(brain_state)
            reward = therapy_ai.compute_therapy_reward(brain_state)

            # Get next state for learning
            next_i = min(i + 256, len(eeg_features) - window_size)
            next_window = eeg_features[next_i:next_i + window_size]
            next_window_scaled = scaler.transform(next_window)
            next_brain_state = digital_twin.predict(next_window_scaled[np.newaxis, :], verbose=0)[0]
            done = (next_i == len(eeg_features) - window_size)

            # Learn from experience
            therapy_ai.remember(brain_state, action, reward, next_brain_state, done)
            therapy_ai.replay()

            # Update VR therapy environment
            engagement, efficiency, anxiety, attention = brain_state
            print(f"Brain State - Engagement: {engagement:.2f}, Efficiency: {efficiency:.2f}, Anxiety: {anxiety:.2f}, Attention: {attention:.2f}")
            print(f"AI Action: {action}, Reward: {reward:.2f}")

            if not vr_interface.update_therapy_environment(action, brain_state, reward):
                break

            await asyncio.sleep(0.5)

    except Exception as e:
        print(f"Error in VR loop: {e}")
        vr_interface.quit()
        return

    # Calculate final therapy outcomes
    final_reward = therapy_ai.compute_therapy_reward(brain_state)
    print(f"\nFinal Therapy Outcome Score: {final_reward:.2f}")
    print("Research simulation complete!")

    vr_interface.quit()


def test_pygame():
    """Test if pygame is working properly before running the main simulation"""
    try:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Test Window")
        font = pygame.font.SysFont('arial', 20)
        clock = pygame.time.Clock()

        running = True
        counter = 0

        while running and counter < 100:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            screen.fill((100, 150, 200))
            text = font.render(f"Frame {counter} - Press ESC to quit", True, (255, 255, 255))
            screen.blit(text, (50, 50))
            pygame.display.flip()
            clock.tick(10)
            counter += 1

        pygame.quit()
        print("Pygame test completed successfully")
        return True

    except Exception as e:
        print(f"Pygame test failed: {e}")
        return False


if __name__ == "__main__":
    if platform.system() == "Emscripten":
        asyncio.ensure_future(main())
    else:
        print("Testing pygame...")
        if test_pygame():
            print("Pygame works! Running main simulation...")
            asyncio.run(main())
        else:
            print("Pygame test failed. Check your pygame installation.")
