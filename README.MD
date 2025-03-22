# Implemnetation of **DEGAST3D: Learning Deformable 3D Graph Similarity to Track Plant Cells in Unregistered Time Lapse Images (https://ieeexplore.ieee.org/document/10820545)**
# **3D Plant Cell Segmentation, Tracking, and Cell Division Detection**

## **Overview**
This project performs **3D segmentation, registration, tracking, and cell division detection** for plant cell images using deep learning-based approaches. It leverages **Cellpose** for segmentation and implements a **3D Graph Matching** approach for tracking plant cells and detecting cell division events.

## **Features**
✅ **Automated 3D segmentation** using **Cellpose**
✅ **Automated 3D registration**  
✅ **Supports multiple plant datasets** for microscopic image analysis  
✅ **Performs plant cell tracking** using **3D graph-based matching**  
✅ **Detects cell division events**  

---

## **Installation**
### **1. Install Dependencies**
Create the required Conda environment using:
```bash
conda env create -f environment.yml
```

---

## **Project Structure**
```
project_root/
├── segmentation.py         # 3D segmentation script using Cellpose
├── Registration.py         # Registration script for aligning time-lapse images
├── Inference_Tracking_Celldiv.py  # Combined script for registration, tracking & division detection
├── data_process.py         # Functions for data preprocessing
├── data/                   # Dataset directory (organized by plant)
│   ├── plant_1/
│   │   ├── microscopic_images/     # Raw images
│   │   ├── cellpose/               # Segmentation results
│   │   │   ├── seg/                # Segmented masks
│   │   |	├── stack_info/             # Processed stack information
│   │   |	├── Registration_info/      # Registration details
│   ├── plant_2/
│   ├── plant_4/
│   ├── plant_13/
│   ├── plant_15/
│   ├── plant_18/
│   ├── test_plant/
├── results/                # Output directory for segmentation, tracking, and cell division data
└── README.md               # Project documentation
```

---

## **Usage**
### **1. Define Plant Names**
Ensure the following plant names are set in `segmentation.py` and `Inference_Tracking_Celldiv.py`:
```python
plant_names = ['plant_1', 'plant_2', 'plant_4', 'plant_13', 'plant_15', 'plant_18', 'test_plant']
```

### **2. Prepare the Dataset**
Place your microscopic image dataset inside the `data` directory, following the structure:
```
data/
├── plant_1/
│   ├── microscopic_images/
│   ├── cellpose/
│   │   ├── seg/
│   |	├── stack_info/
│   |	├── Registration_info/
├── plant_2/
├── ...
```

### **3. Run Segmentation**
Execute the segmentation script by specifying the **dataset directory and plant index**: This script also includes data processing using `STACK_INF0_3D` and `STACK_INF0_Registration`. 
```bash
python segmentation.py --main_dir "data" --plant_idx 6 --segmentation "cellpose"
```
📌 **Output:** Segmented masks will be saved in the `seg/` directory inside the respective plant folder.
The processed data will also be saved in `stack_info` and  `Registration_info` sub-folder.


### **4. Run Registration, Tracking and Cell Division Detection for every pair of timepoint**
Perform **Registration , Tracking and Cell division detection** in one step:
```bash
python Inference_Tracking_Celldiv.py --main_dir "data" --plant_idx 6 --segmentation "cellpose"
```
📌 **Output:** Tracking and division results will be stored inside `results/{segmentation}_{plant_name}/`.

---

## **Customization**
- Change `--plant_idx` to select a different plant dataset.
- The framework supports **any segmentation algorithm**.

---

## **Acknowledgments**
This project utilizes:
- **[Cellpose](https://github.com/MouseLand/cellpose)** for deep learning-based cell segmentation.
- **3D Graph** was constructed based on this reposititory **[Deep Graphical Feature Learning for the Feature Matching Problem](https://github.com/zzhang1987/Deep-Graphical-Feature-Learning)**, ICCV2019

---

## **License**
📜 This project is licensed under the **MIT License**.

