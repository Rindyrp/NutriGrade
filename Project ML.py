import streamlit as st
import cv2
import numpy as np
import pandas as pd
import re
from PIL import Image, ImageEnhance, ImageFilter
import paddleocr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import io
import base64
from dataclasses import dataclass
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfigurasi halaman
st.set_page_config(
    page_title="🏷️ NutriGrade Vision",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .grade-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .grade-a { background-color: #d4edda; color: #155724; }
    .grade-b { background-color: #fff3cd; color: #856404; }
    .grade-c { background-color: #f8d7da; color: #721c24; }
    .grade-d { background-color: #f5c6cb; color: #491217; }
    .warning-box {
        padding: 1rem;
        border-left: 4px solid #ff6b6b;
        background-color: #ffe0e0;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-left: 4px solid #4ecdc4;
        background-color: #e0f7fa;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class OCRConfig:
    """Configuration class for OCR system"""
    model_type: str = "paddleocr"
    language: List[str] = None
    use_gpu: bool = False  # Set to False for CPU-only
    confidence_threshold: float = 0.5
    preprocessing_enabled: bool = True
    postprocessing_enabled: bool = True
    
    def __post_init__(self):
        if self.language is None:
            self.language = ['id', 'en']  # Indonesian and English

class AdvancedImagePreprocessor:
    """Advanced image preprocessing for nutrition label OCR"""
    
    def __init__(self):
        self.setup_filters()
    
    def setup_filters(self):
        """Setup various filters and kernels"""
        self.sharpen_kernel = np.array([[-1,-1,-1],
                                       [-1, 9,-1],
                                       [-1,-1,-1]])
        
        self.edge_kernel = np.array([[-1,-1,-1,-1,-1],
                                    [-1, 2, 2, 2,-1],
                                    [-1, 2, 8, 2,-1],
                                    [-1, 2, 2, 2,-1],
                                    [-1,-1,-1,-1,-1]]) / 8.0
    
    def detect_document_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect corners of document/label for perspective correction"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 200, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                return approx.reshape(4, 2)
        
        return None
    
    def perspective_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply perspective correction to straighten the document"""
        corners = self.detect_document_corners(image)
        
        if corners is None:
            return image
        
        rect = np.zeros((4, 2), dtype="float32")
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]
        rect[3] = corners[np.argmax(diff)]
        
        width_a = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        width_b = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        height_b = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")
        
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        
        return warped
    
    def adaptive_lighting_correction(self, image: np.ndarray) -> np.ndarray:
        """Advanced lighting correction using multiple techniques"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            result = clahe.apply(image)
        
        return result
    
    def remove_glare_and_shadows(self, image: np.ndarray) -> np.ndarray:
        """Remove glare and shadows using morphological operations"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        result = cv2.absdiff(gray, opened)
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        
        return result
    
    def advanced_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced denoising techniques"""
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        return denoised
    
    def enhance_text_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance text contrast specifically"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
        adaptive_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
        combined = cv2.bitwise_and(otsu, cv2.bitwise_and(adaptive_mean, adaptive_gaussian))
        
        return combined
    
    def apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply intelligent sharpening"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        sharpened = cv2.filter2D(unsharp_mask, -1, self.sharpen_kernel)
        
        return sharpened
    
    def preprocess_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Complete preprocessing pipeline"""
        try:
            processed = image.copy()
            processed = self.perspective_correction(processed)
            processed = self.adaptive_lighting_correction(processed)
            processed = self.remove_glare_and_shadows(processed)
            processed = self.advanced_denoising(processed)
            processed = self.apply_sharpening(processed)
            processed = self.enhance_text_contrast(processed)
            logger.info("Preprocessing pipeline completed successfully")
            return processed
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return image

class NutrientAnalyzer:
    def __init__(self):
        self.nutrient_aliases = {
            'sugar': ['total sugar', 'total sugars', 'sucrose', 'glucose', 'fructose', 
                     'corn syrup', 'added sugar', 'madu', 'sirup', 'fruktosa', 
                     'glukosa', 'sukrosa', 'dextrose', 'maltose', 'gula'],
            'sodium': ['sodium', 'natrium', 'na', 'salt', 'garam', 
                      'monosodium glutamate', 'msg'],
            'saturated_fat': ['saturated fat', 'lemak jenuh', 'saturates', 
                            'lemak trans', 'trans fat']
        }
        
        self.thresholds = {
            'sugar': {'A': 1, 'B': 5, 'C': 10},
            'sodium': {'A': 300, 'B': 340, 'C': 370},
            'saturated_fat': {'A': 0.7, 'B': 1.2, 'C': 2.8}
        }
        
        self.who_limits = {
            'sugar': 25,
            'sodium': 2000,
            'saturated_fat': 20
        }
    
    def extract_nutrients_from_text(self, text: str) -> Dict[str, float]:
        """Ekstrak nilai nutrisi dari teks menggunakan regex"""
        nutrients = {'sugar': 0, 'sodium': 0, 'saturated_fat': 0}
        text = text.lower()
        
        patterns = {
            'sugar': r'(?:' + '|'.join(self.nutrient_aliases['sugar']) + r')\s*:?\s*(\d+(?:\.\d+)?)\s*g',
            'sodium': r'(?:' + '|'.join(self.nutrient_aliases['sodium']) + r')\s*:?\s*(\d+(?:\.\d+)?)\s*mg',
            'saturated_fat': r'(?:' + '|'.join(self.nutrient_aliases['saturated_fat']) + r')\s*:?\s*(\d+(?:\.\d+)?)\s*g'
        }
        
        for nutrient, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                nutrients[nutrient] = max([float(match) for match in matches])
        
        return nutrients
    
    def get_grade(self, nutrient: str, value: float) -> str:
        """Menentukan grade berdasarkan nilai nutrisi"""
        thresholds = self.thresholds[nutrient]
        
        if value <= thresholds['A']:
            return 'A'
        elif value <= thresholds['B']:
            return 'B'
        elif value <= thresholds['C']:
            return 'C'
        else:
            return 'D'
    
    def get_grade_color(self, grade: str) -> str:
        """Mengembalikan warna berdasarkan grade"""
        colors = {'A': '#28a745', 'B': '#ffc107', 'C': '#fd7e14', 'D': '#dc3545'}
        return colors.get(grade, '#6c757d')
    
    def get_grade_emoji(self, grade: str) -> str:
        """Mengembalikan emoji berdasarkan grade"""
        emojis = {'A': '✅', 'B': '🟡', 'C': '🟠', 'D': '🔴'}
        return emojis.get(grade, '⚪')
    
    def calculate_who_percentage(self, nutrient: str, value: float) -> float:
        """Menghitung persentase dari batas harian WHO"""
        return (value / self.who_limits[nutrient]) * 100
    
    def get_health_recommendation(self, nutrients: Dict[str, float], condition: str) -> List[str]:
        """Memberikan rekomendasi berdasarkan kondisi kesehatan"""
        recommendations = []
        
        if condition == "Diabetes Mellitus":
            if nutrients['sugar'] > 5:
                recommendations.append("⚠️ TIDAK DIREKOMENDASIKAN: Kandungan gula terlalu tinggi untuk penderita diabetes")
            elif nutrients['sugar'] > 1:
                recommendations.append("⚡ PERHATIAN: Konsumsi gula harus dibatasi")
        
        elif condition == "Hipertensi":
            if nutrients['sodium'] > 340:
                recommendations.append("⚠️ TIDAK DIREKOMENDASIKAN: Kandungan sodium terlalu tinggi untuk penderita hipertensi")
            elif nutrients['sodium'] > 300:
                recommendations.append("⚡ PERHATIAN: Konsumsi sodium harus dibatasi")
        
        elif condition == "Dislipidemia":
            if nutrients['saturated_fat'] > 1.2:
                recommendations.append("⚠️ TIDAK DIREKOMENDASIKAN: Kandungan lemak jenuh terlalu tinggi")
            elif nutrients['saturated_fat'] > 0.7:
                recommendations.append("⚡ PERHATIAN: Konsumsi lemak jenuh harus dibatasi")
        
        elif condition == "Anak-anak":
            total_bad_grades = sum(1 for nutrient in nutrients if self.get_grade(nutrient, nutrients[nutrient]) in ['C', 'D'])
            if total_bad_grades > 0:
                recommendations.append("⚠️ TIDAK DIREKOMENDASIKAN: Produk ini tidak cocok untuk anak-anak")
        
        return recommendations

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocessing gambar untuk meningkatkan akurasi OCR"""
    preprocessor = AdvancedImagePreprocessor()
    img_array = np.array(image)
    processed_array = preprocessor.preprocess_pipeline(img_array)
    return processed_array

def perform_ocr(image_array: np.ndarray) -> str:
    """Perform OCR using PaddleOCR"""
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    result = ocr.ocr(image_array, cls=True)
    text = ' '.join([line[1][0] for line in result[0]]) if result and result[0] else ""
    return text

def create_visualization(nutrients: Dict[str, float], analyzer: NutrientAnalyzer):
    """Membuat visualisasi hasil analisis"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('', 'WHO Daily Limit Comparison', 
                       'Nutrient Values', 'Health Risk Assessment'),
        specs=[[{"type": "indicator"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "pie"}]]
    )
    
    grades = [analyzer.get_grade(nutrient, value) for nutrient, value in nutrients.items()]
    grade_scores = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
    avg_score = sum(grade_scores[grade] for grade in grades) / len(grades)
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=avg_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Grade"},
            gauge={'axis': {'range': [1, 4]},
                  'bar': {'color': "darkblue"},
                  'steps': [{'range': [1, 2], 'color': "lightgray"},
                           {'range': [2, 3], 'color': "gray"}],
                  'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 3.5}}
        ),
        row=1, col=1
    )
    
    who_percentages = [analyzer.calculate_who_percentage(nutrient, value) 
                      for nutrient, value in nutrients.items()]
    
    fig.add_trace(
        go.Bar(
            x=['Sugar', 'Sodium', 'Saturated Fat'],
            y=who_percentages,
            marker_color=['red' if p > 100 else 'orange' if p > 50 else 'green' 
                         for p in who_percentages],
            text=[f'{p:.1f}%' for p in who_percentages],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=['Sugar (g)', 'Sodium (mg)', 'Sat Fat (g)'],
            y=[nutrients['sugar'], nutrients['sodium'], nutrients['saturated_fat']],
            mode='markers+text',
            marker=dict(size=20, color=[analyzer.get_grade_color(analyzer.get_grade(nutrient, value)) 
                                       for nutrient, value in nutrients.items()]),
            text=grades,
            textposition="middle center",
            textfont=dict(color="white", size=14)
        ),
        row=2, col=1
    )
    
    risk_levels = [grade for grade in grades]
    risk_counts = {level: risk_levels.count(level) for level in ['A', 'B', 'C', 'D']}
    
    fig.add_trace(
        go.Pie(
            labels=list(risk_counts.keys()),
            values=list(risk_counts.values()),
            marker_colors=[analyzer.get_grade_color(grade) for grade in risk_counts.keys()]
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="NutriGrade Analysis Dashboard")
    return fig

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🏷️ NutriGrade Vision</h1>
        <p>Aplikasi Pendeteksi Kandungan Gizi Negatif dari Kemasan Makanan atau Minuman</p>
    </div>
    """, unsafe_allow_html=True)
    
    analyzer = NutrientAnalyzer()
    
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        health_condition = st.selectbox(
            "Kondisi Kesehatan:",
            ["Tidak ada", "Diabetes Mellitus", "Hipertensi", "Dislipidemia", 
             "Anak-anak", "Ibu Hamil & Menyusui"]
        )
        input_method = st.radio(
            "Metode Input:",
            ["📁 Upload Gambar", "📸 Kamera", "📝 Input Manual"]
        )
        st.markdown("---")
        st.markdown("### 📊 Sistem Grading")
        st.markdown("""
        - **Grade A** ✅: Sangat Baik
        - **Grade B** 🟡: Baik
        - **Grade C** 🟠: Perlu Perhatian
        - **Grade D** 🔴: Tinggi
        """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📱 Input Data")
        nutrients = None
        
        if input_method == "📁 Upload Gambar":
            uploaded_file = st.file_uploader(
                "Pilih gambar label gizi:", 
                type=['jpg', 'jpeg', 'png'],
                help="Upload foto label nutrisi dari kemasan makanan atau minuman"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Gambar yang diupload", use_column_width=True)
                
                if st.button("🔍 Analisis Gambar"):
                    with st.spinner("Memproses gambar..."):
                        processed_image = preprocess_image(image)
                        extracted_text = perform_ocr(processed_image)
                        
                        st.subheader("📄 Teks yang Diekstrak:")
                        st.text_area("OCR Result:", extracted_text, height=100)
                        nutrients = analyzer.extract_nutrients_from_text(extracted_text)
        
        elif input_method == "📸 Kamera":
            st.info("💡 Fitur kamera akan tersedia dalam versi production dengan streamlit-camera-input")
            if st.button("📸 Simulasi Capture"):
                sample_text = "Nutrition Facts\nTotal Sugar: 8g\nSodium: 450mg\nSaturated Fat: 1.5g"
                nutrients = analyzer.extract_nutrients_from_text(sample_text)
        
        elif input_method == "📝 Input Manual":
            st.subheader("Input Nilai Gizi Manual")
            col_sugar, col_sodium, col_fat = st.columns(3)
            
            with col_sugar:
                sugar = st.number_input("Gula (gram):", min_value=0.0, step=0.1)
            with col_sodium:
                sodium = st.number_input("Sodium (mg):", min_value=0.0, step=1.0)
            with col_fat:
                sat_fat = st.number_input("Lemak Jenuh (gram):", min_value=0.0, step=0.1)
            
            if st.button("📊 Analisis Manual"):
                nutrients = {
                    'sugar': sugar,
                    'sodium': sodium,
                    'saturated_fat': sat_fat
                }
    
    with col2:
        st.header("ℹ️ Informasi")
        st.subheader("🧮 Simulasi Konsumsi")
        servings = st.number_input("Jumlah kemasan:", min_value=1, max_value=10, value=1)
        
        with st.expander("📚 Edukasi Gizi"):
            st.markdown("""
            **Nama Alternatif Gula:**
            - Sucrose, Glucose, Fructose
            - Corn Syrup, Dextrose, Maltose
            - Madu, Sirup
            
            **Nama Alternatif Sodium:**
            - Natrium, Na, Salt, Garam
            - MSG, Monosodium Glutamate
            
            **Lemak Jenuh:**
            - Saturated Fat, Lemak Jenuh
            - Trans Fat, Lemak Trans
            """)
    
    if nutrients:
        st.markdown("---")
        st.header("📊 Hasil Analisis")
        
        adjusted_nutrients = {k: v * servings for k, v in nutrients.items()}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            grade = analyzer.get_grade('sugar', adjusted_nutrients['sugar'])
            emoji = analyzer.get_grade_emoji(grade)
            st.markdown(f"""
            <div class="grade-card grade-{grade.lower()}">
                <h3>{emoji} Gula</h3>
                <h2>Grade {grade}</h2>
                <p>{adjusted_nutrients['sugar']:.1f}g</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            grade = analyzer.get_grade('sodium', adjusted_nutrients['sodium'])
            emoji = analyzer.get_grade_emoji(grade)
            st.markdown(f"""
            <div class="grade-card grade-{grade.lower()}">
                <h3>{emoji} Sodium</h3>
                <h2>Grade {grade}</h2>
                <p>{adjusted_nutrients['sodium']:.1f}mg</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            grade = analyzer.get_grade('saturated_fat', adjusted_nutrients['saturated_fat'])
            emoji = analyzer.get_grade_emoji(grade)
            st.markdown(f"""
            <div class="grade-card grade-{grade.lower()}">
                <h3>{emoji} Lemak Jenuh</h3>
                <h2>Grade {grade}</h2>
                <p>{adjusted_nutrients['saturated_fat']:.1f}g</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("📈 Perbandingan Batas Harian WHO")
        who_data = []
        for nutrient, value in adjusted_nutrients.items():
            percentage = analyzer.calculate_who_percentage(nutrient, value)
            who_data.append({
                'Nutrisi': nutrient.replace('_', ' ').title(),
                'Nilai': f"{value:.1f}{'g' if nutrient != 'sodium' else 'mg'}",
                'Batas WHO': f"{analyzer.who_limits[nutrient]}{'g' if nutrient != 'sodium' else 'mg'}",
                'Persentase': f"{percentage:.1f}%",
                'Status': '⚠️ Melebihi' if percentage > 100 else '✅ Aman'
            })
        
        df = pd.DataFrame(who_data)
        st.dataframe(df, use_container_width=True)
        
        if health_condition != "Tidak ada":
            recommendations = analyzer.get_health_recommendation(adjusted_nutrients, health_condition)
            if recommendations:
                st.subheader("🩺 Rekomendasi Kesehatan")
                for rec in recommendations:
                    st.markdown(f"""
                    <div class="warning-box">
                        <strong>{health_condition}:</strong><br>
                        {rec}
                    </div>
                    """, unsafe_allow_html=True)
        
        st.subheader("📊 Visualisasi Data")
        fig = create_visualization(adjusted_nutrients, analyzer)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("💾 Export Hasil")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Download PDF Report"):
                st.info("Fitur download PDF akan tersedia dalam versi production")
        with col2:
            if st.button("📊 Download Excel Data"):
                st.info("Fitur download Excel akan tersedia dalam versi production")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🏷️ NutriGrade Vision | Membantu Anda Membuat Pilihan Makanan atau Minuman yang Lebih Sehat</p>
        <p><small>Dikembangkan dengan ❤️ menggunakan Streamlit & Python</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
