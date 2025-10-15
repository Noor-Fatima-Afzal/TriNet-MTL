# 🧠 TriNet-MTL: A Multi-Branch Deep Learning Framework for Biometric Identification and Cognitive State Inference from Auditory-Evoked EEG

This repository contains the codebase for **TriNet-MTL**, a multi-task learning (MTL) Transformer-based architecture designed to simultaneously classify:

- 👤 **Biometric Identity** (20 subjects)
- 🗣️ **Language Condition** (Native vs. Non-native auditory stimulus)
- 🎧 **Device Modality** (In-ear vs. Bone-conduction headphones)

The model learns shared temporal representations from auditory-evoked EEG signals and branches into task-specific heads for joint optimization.

---

## 📊 Dataset

This work uses the **Auditory EEG dataset** from PhysioNet:

**Citation:**
```
Accou, B., Jalilpour Monesi, M., Montoya, J., Van hamme, H., & Francart, T. (2022). 
Auditory EEG dataset (version 1.0.0). PhysioNet. 
https://doi.org/10.13026/105p-ws26
```

**Dataset Access:** https://physionet.org/content/auditory-eeg/1.0.0/

**Description:** The dataset contains 64-channel EEG recordings from 20 subjects listening to auditory stimuli under different conditions (native/non-native language, in-ear/bone-conduction delivery). This multi-dimensional structure makes it ideal for multi-task learning approaches.

**PhysioNet Citation:**
```
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & 
Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new 
research resource for complex physiologic signals. Circulation, 101(23), e215-e220.
```

---


## 🖼️ Architecture

<p align="center">
  <img src="auditory.jpeg" alt="TriNet-MTL Architecture" width="700"/>
</p>

---

## 📦 Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/Noor-Fatima-Afzal/TriNet-MTL.git
cd eeg-multitask-transformer
pip install -r requirements.txt
```
---

## 🙏 Acknowledgments

- Dataset provided by PhysioNet
- Original dataset creators: Accou et al. (2022)

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [noorfatimaafzalbutt@gmail.com]
