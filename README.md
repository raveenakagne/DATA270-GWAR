# 🏭 Amazon Warehouse Bin Size Verification  
### Enhancing Inventory Accuracy Using Image Processing and Machine Learning  

## 📦 Project Overview
Accurate inventory management is essential for robotics-driven data warehouses. This project integrates **advanced image processing** and **machine learning** techniques to verify bin item sizes and count items in Amazon warehouse bins. The goal is to reduce inventory discrepancies and improve order fulfillment efficiency.

## 📁 Repository Contents
- `cleaned_data.csv` — Preprocessed image metadata  
- `metadata_file.csv` — Supporting metadata for bin images  
- `CNN_model/` — Convolutional Neural Network implementation  
- `R_CNN/` — Faster R-CNN model  
- `CLIP_model/` — Contrastive Language-Image Pre-Training model  
- `YOLO_model/` — YOLO-based object detection  
- `Data_Cleaning_&_Analysis/` — Image preprocessing and filtering  
- `README.md` — Project documentation  

## 🧪 Methodology
### 🖼️ Image Processing Techniques
- **CLAHE (Contrast-Limited Adaptive Histogram Equalization)**: Enhances image contrast  
- **Laplacian of Gaussian**: Highlights edges and contours  
- **High-Pass Filtering**: Extracts fine details  
- **Bilateral Filtering**: Reduces noise while preserving edges  

### 🤖 Machine Learning Models
| Model        | Accuracy | Notes |
|--------------|----------|-------|
| CNN          | 56%      | Baseline performance for item detection  
| Faster R-CNN | 56%      | Improved edge detection and localization  
| CLIP         | 60%      | Combines image and text embeddings  
| YOLO         | 70%      | Best performance, especially with higher item counts  

## 🎯 Objectives
- Detect and count items in warehouse bins using image data  
- Improve robotic inventory management accuracy  
- Reduce fulfillment errors through automated verification  
- Explore real-world applications of computer vision in logistics  

## 📊 Key Findings
- YOLO outperforms other models in high-density bin scenarios  
- Image preprocessing significantly improves model accuracy  
- Combining multiple models may enhance robustness across bin types  

## 🛠️ Tools & Technologies
- Python (OpenCV, NumPy, TensorFlow, PyTorch)  
- Jupyter Notebook  
- GitHub for version control  

## 🔗 Dataset & Resources
- Bin images sourced from Amazon Fulfillment Centers  
- [Amazon Bin Image Dataset on AWS](https://registry.opendata.aws/amazon-bin-imagery)  
- [Project Repository](https://github.com/raveenakagne/DATA270-GWAR.git)  

## 👩‍💻 Author
**Raveena Kagne**  
📎 [LinkedIn](https://www.linkedin.com/in/raveenakagne)  

---

📌 _This project demonstrates how image processing and machine learning can revolutionize warehouse automation by enabling accurate, scalable inventory verification._
