# 📊 Auditory Evoked Potential EEG-Biometric Dataset

**Source:** [PhysioNet - Version 1.0.0](https://doi.org/10.13026/ps31-fc50)  
**Published by:** Nibras Abo Alzahab et al., 2021  
**Institution:** Marche Polytechnic University (UNIVPM)  
**License:** PhysioNet Contributor Review Health Data License 1.5.0

---

## 🧠 Overview

This dataset comprises over **240 EEG recordings** from **20 human subjects**, collected during **resting-state** and **auditory-stimuli experiments**. It supports research in EEG-based biometrics, user authentication, and auditory neurophysiology.

---

## 🎧 Experimental Setup

- **Resting State**: Eyes open and closed
- **Auditory Stimuli**:
  - Delivered via: in-ear headphones and bone-conducting transducers
  - Stimulus types:
    - Native-language song
    - Non-native-language song
    - Neutral instrumental music

---

## 🧪 EEG Acquisition

- **Device**: OpenBCI Ganglion Board  
- **Channels**: 4 (T7, F8, Cz, P4) — based on the 10/10 EEG placement system  
- **Sampling Rate**: 200 Hz  
- **Electrodes**: Gold-cup with Ten20 conductive paste  
- **Software**: OpenBCI GUI v5.0.3

---

## 📁 Data Organization

The dataset is provided in multiple versions:

- **Raw Data**: Unprocessed recordings
- **Segmented Data**: Noise-minimized 2-minute segments
- **Filtered Data**: Bandpass (1–40 Hz) and notch (50 Hz) filtered
- **Supplementary Files**:
  - `Subjects.csv`: Metadata and questionnaire responses
  - `Songs.csv`: Links to audio stimuli
  - `data_trim.csv`: Time indices for segmented portions

Each EEG file contains:
- **First 5 columns**: Sample index + 4 EEG channels
- **Filename format**: `sXX_exYY_sZZ.csv` (subject, experiment, session)

---

## 🎯 Applications

- EEG-based biometric identification
- Auditory stimulus response classification
- Language and conduction-method impact analysis
- Cognitive neuroscience and evoked potential studies

---

## ⚠️ Limitations

- Small number of channels (4)
- Limited to 20 subjects
- Fixed stimulus protocol

---

## 📚 Citation

When using this dataset, cite:

> Abo Alzahab, N., Di Iorio, A., Apollonio, L., et al. (2021). *Auditory evoked potential EEG-Biometric dataset (version 1.0.0)*. PhysioNet. https://doi.org/10.13026/ps31-fc50

Also include the PhysioNet standard citation:

> Goldberger, A. et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals*. *Circulation*, 101(23), e215–e220.
