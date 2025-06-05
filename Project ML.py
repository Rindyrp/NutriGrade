import streamlit as st
import cv2
import numpy as np
import pandas as pd
import re
from PIL import Image, ImageFilter, ImageEnhance
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Tuple, Optional
import io
import base64
import pytesseract  # Tambahkan import untuk pytesseract

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
    .camera-container {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class NutrientAnalyzer:
    def __init__(self):
        # Definisi nama alternatif nutrisi
        self.nutrient_aliases = {
            'sugar': ['total sugar', 'total sugars', 'sucrose', 'glucose', 'fructose', 
                     'corn syrup', 'added sugar', 'madu', 'sirup', 'fruktosa', 
                     'glukosa', 'sukrosa', 'dextrose', 'maltose', 'gula'],
            'sodium': ['sodium', 'natrium', 'na', 'salt', 'garam', 
                      'monosodium glutamate', 'msg'],
            'saturated_fat': ['saturated fat', 'lemak jenuh', 'saturates', 
                            'lemak trans', 'trans fat']
        }
        
        # Threshold untuk grading
        self.thresholds = {
            'sugar': {'A': 1, 'B': 5, 'C': 10},
            'sodium': {'A': 300, 'B': 340, 'C': 370},
            'saturated_fat': {'A': 0.7, 'B': 1.2, 'C': 2.8}
        }
        
        # Batas harian WHO
        self.who_limits = {
            'sugar': 25,  # gram
            'sodium': 2000,  # mg
            'saturated_fat': 20  # gram (untuk diet 2000 kcal)
        }
    
    def extract_nutrients_from_text(self, text: str) -> Dict[str, float]:
        """Ekstrak nilai nutrisi dari teks menggunakan regex"""
        nutrients = {'sugar': 0, 'sodium': 0, 'saturated_fat': 0}
        text = text.lower()
        
        # Pattern untuk menangkap angka dengan satuan
        patterns = {
            'sugar': r'(?:' + '|'.join(self.nutrient_aliases['sugar']) + r')\s*:?\s*(\d+(?:\.\d+)?)\s*g',
            'sodium': r'(?:' + '|'.join(self.nutrient_aliases['sodium']) + r')\s*:?\s*(\d+(?:\.\d+)?)\s*mg',
            'saturated_fat': r'(?:' + '|'.join(self.nutrient_aliases['saturated_fat']) + r')\s*:?\s*(\d+(?:\.\d+)?)\s*g'
        }
        
        for nutrient, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Ambil nilai tertinggi jika ada multiple matches
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

def advanced_preprocess_image(image):
    """
    Advanced preprocessing untuk meningkatkan akurasi OCR pada label nutrisi
    """
    # Convert PIL to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 1. Resize image jika terlalu kecil atau besar
    height, width = opencv_image.shape[:2]
    if width < 800:
        scale_factor = 800 / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        opencv_image = cv2.resize(opencv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    elif width > 2000:
        scale_factor = 2000 / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        opencv_image = cv2.resize(opencv_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # 3. Noise reduction dengan bilateral filter
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 4. Contrast enhancement menggunakan CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 5. Morphological operations untuk membersihkan noise
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # 6. Adaptive thresholding untuk binarization yang lebih baik
    binary = cv2.adaptiveThreshold(
        cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 7. Dilation untuk memperkuat teks
    kernel_dilate = np.ones((1,1), np.uint8)
    dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
    
    # Convert back to PIL
    processed_pil = Image.fromarray(dilated)
    
    return processed_pil

def detect_and_crop_nutrition_label(image):
    """
    Deteksi dan crop area label nutrisi secara otomatis
    """
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours berdasarkan area dan aspect ratio
    potential_labels = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Label nutrisi biasanya berbentuk persegi panjang vertikal atau horizontal
            if 0.3 < aspect_ratio < 3.0 and w > 200 and h > 200:
                potential_labels.append((x, y, w, h, area))
    
    if potential_labels:
        # Ambil kontour dengan area terbesar
        x, y, w, h, _ = max(potential_labels, key=lambda x: x[4])
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(opencv_image.shape[1] - x, w + 2*padding)
        h = min(opencv_image.shape[0] - y, h + 2*padding)
        
        # Crop image
        cropped = opencv_image[y:y+h, x:x+w]
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        
        return cropped_pil
    
    return image  # Return original if no suitable contour found

def enhance_text_regions(image):
    """
    Enhance regions yang kemungkinan mengandung teks
    """
    # Convert to PIL untuk enhancement
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Sharpen image
    sharpened = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(sharpened)
    contrasted = enhancer.enhance(1.5)
    
    # Enhance brightness jika gambar terlalu gelap
    enhancer = ImageEnhance.Brightness(contrasted)
    brightened = enhancer.enhance(1.1)
    
    return brightened

def perform_ocr_with_multiple_configs(image):
    """
    Perform OCR dengan berbagai konfigurasi untuk hasil terbaik
    """
    # Konfigurasi OCR yang berbeda
    configs = [
        '--oem 3 --psm 6',  # Default
        '--oem 3 --psm 4',  # Single column of text
        '--oem 3 --psm 8',  # Single word
        '--oem 3 --psm 13', # Raw line
        '--oem 3 --psm 11', # Sparse text
        '--oem 3 --psm 12', # Sparse text with OSD
    ]
    
    results = []
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(image, config=config, lang='ind+eng')
            if text.strip():
                results.append(text.strip())
        except Exception as e:
            st.warning(f"OCR config failed: {config} - {str(e)}")
            continue
    
    # Gabungkan hasil dan ambil yang terpanjang
    if results:
        # Ambil hasil terpanjang sebagai hasil utama
        best_result = max(results, key=len)
        return best_result
    else:
        return "Tidak dapat membaca teks dari gambar"

def preprocess_nutrition_label(image):
    """
    Pipeline lengkap preprocessing untuk label nutrisi
    """
    try:
        # Step 1: Detect and crop nutrition label area
        st.write("🔍 Mendeteksi area label nutrisi...")
        cropped_image = detect_and_crop_nutrition_label(image)
        
        # Step 2: Enhance text regions
        st.write("✨ Meningkatkan kualitas teks...")
        enhanced_image = enhance_text_regions(cropped_image)
        
        # Step 3: Advanced preprocessing
        st.write("🔧 Memproses gambar untuk OCR...")
        processed_image = advanced_preprocess_image(enhanced_image)
        
        return processed_image, cropped_image, enhanced_image
        
    except Exception as e:
        st.error(f"Error dalam preprocessing: {str(e)}")
        return image, image, image

def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocessing gambar untuk meningkatkan akurasi OCR"""
    # Convert ke array numpy
    img_array = np.array(image)
    
    # Convert ke grayscale jika berwarna
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Noise reduction
    denoised = cv2.medianBlur(gray, 3)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Thresholding
    _, threshold = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return Image.fromarray(threshold)

def simulate_ocr(image: Image.Image) -> str:
    """Simulasi OCR - dalam implementasi nyata, gunakan EasyOCR atau Tesseract"""
    # Ini adalah simulasi - replace dengan OCR engine sebenarnya
    sample_texts = [
        "Nutrition Facts\nTotal Sugar: 12g\nSodium: 850mg\nSaturated Fat: 3.2g",
        "Informasi Nilai Gizi\nGula Total: 8g\nNatrium: 450mg\nLemak Jenuh: 1.5g",
        "Kandungan Gizi\nSucrose: 15g\nGaram: 920mg\nLemak Trans: 2.1g"
    ]
    
    # Return random sample text for demo
    import random
    return random.choice(sample_texts)

def create_visualization(nutrients: Dict[str, float], analyzer: NutrientAnalyzer):
    """Membuat visualisasi hasil analisis"""
    
    # Buat subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('', 'WHO Daily Limit Comparison', 
                       'Nutrient Values', 'Health Risk Assessment'),
        specs=[[{"type": "indicator"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "pie"}]]
    )
    
    # Grade overview (indicator)
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
    
    # WHO comparison (bar chart)
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
    
    # Nutrient values (scatter)
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
    
    # Risk assessment (pie)
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
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏷️ NutriGrade Vision</h1>
        <p>Aplikasi Pendeteksi Kandungan Gizi Negatif dari Kemasan Makanan atau Minuman</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = NutrientAnalyzer()
    
    # Sidebar untuk konfigurasi
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        
        # Health condition selector
        health_condition = st.selectbox(
            "Kondisi Kesehatan:",
            ["Tidak ada", "Diabetes Mellitus", "Hipertensi", "Dislipidemia", 
             "Anak-anak", "Ibu Hamil & Menyusui"]
        )
        
        # Input method selector
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
    
    # Main content area
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
                        # Preprocess image
                        processed_image = preprocess_image(image)
                        
                        # OCR simulation
                        extracted_text = simulate_ocr(processed_image)
                        
                        st.subheader("📄 Teks yang Diekstrak:")
                        st.text_area("OCR Result:", extracted_text, height=100)
                        
                        # Extract nutrients
                        nutrients = analyzer.extract_nutrients_from_text(extracted_text)
        
        elif input_method == "📸 Kamera":
            # Updated camera section dengan preprocessing yang diperbaiki
            st.markdown("""
            <div class="camera-container">
                <h3>📸 Ambil Foto dengan Kamera</h3>
                <p>Gunakan kamera untuk mengambil foto label nutrisi</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Camera input
            camera_photo = st.camera_input("Ambil foto label nutrisi:")
            
            if camera_photo is not None:
                # Display original image
                image = Image.open(camera_photo)
                
                col1_inner, col2_inner = st.columns(2)
                with col1_inner:
                    st.subheader("📷 Foto Original")
                    st.image(image, caption="Foto dari kamera", use_column_width=True)
                
                # Analysis button
                if st.button("🔍 Analisis Foto Kamera", key="analyze_camera"):
                    with st.spinner("Memproses foto dengan AI..."):
                        try:
                            # Advanced preprocessing
                            processed_image, cropped_image, enhanced_image = preprocess_nutrition_label(image)
                            
                            # Show processing steps
                            with col2_inner:
                                st.subheader("🔧 Hasil Preprocessing")
                                
                                # Tabs untuk menampilkan berbagai tahap preprocessing
                                tab1, tab2, tab3 = st.tabs(["Cropped", "Enhanced", "Final"])
                                
                                with tab1:
                                    st.image(cropped_image, caption="Area label yang terdeteksi", use_column_width=True)
                                
                                with tab2:
                                    st.image(enhanced_image, caption="Gambar yang ditingkatkan", use_column_width=True)
                                
                                with tab3:
                                    st.image(processed_image, caption="Siap untuk OCR", use_column_width=True)
                            
                            # Perform real OCR
                            st.write("🔤 Melakukan OCR...")
                            extracted_text = perform_ocr_with_multiple_configs(processed_image)
                            
                            st.subheader("📄 Teks yang Diekstrak:")
                            st.text_area("OCR Result:", extracted_text, height=150, key="camera_ocr")
                            
                            # Extract nutrients
                            nutrients = analyzer.extract_nutrients_from_text(extracted_text)
                            
                            if nutrients:
                                st.subheader("🥗 Informasi Nutrisi yang Terdeteksi:")
                                
                                # Display nutrients in a nice format
                                col1_nut, col2_nut, col3_nut = st.columns(3)
                                
                                with col1_nut:
                                    if 'sugar' in nutrients:
                                        st.metric("Gula", f"{nutrients['sugar']} g")
                                
                                with col2_nut:
                                    if 'sodium' in nutrients:
                                        st.metric("Sodium", f"{nutrients['sodium']} mg")
                                
                                with col3_nut:
                                    if 'saturated_fat' in nutrients:
                                        st.metric("Lemak Jenuh", f"{nutrients['saturated_fat']} g")
                                
                                st.success("✅ Foto berhasil dianalisis dengan preprocessing yang ditingkatkan!")
                            else:
                                st.warning("⚠️ Tidak ada informasi nutrisi yang terdeteksi. Pastikan foto label nutrisi jelas dan terbaca.")
                                
                                # Show extracted text for debugging
                                if extracted_text and extracted_text != "Tidak dapat membaca teks dari gambar":
                                    st.info("💡 Teks yang diekstrak untuk debugging:")
                                    st.code(extracted_text)
                
                        except Exception as e:
                            st.error(f"❌ Error dalam analisis: {str(e)}")
                            st.info("💡 Tips: Pastikan Tesseract OCR sudah terinstall dengan benar")
            
            # Tips section
            st.markdown("""
            <div class="info-box">
                <strong>💡 Tips untuk Foto yang Optimal:</strong>
                <ul>
                    <li>📱 Pegang HP dengan stabil, hindari goyangan</li>
                    <li>💡 Pastikan pencahayaan cukup terang dan merata</li>
                    <li>🎯 Fokuskan kamera pada label nutrisi saja</li>
                    <li>📐 Posisikan kamera tegak lurus dengan label</li>
                    <li>🔍 Pastikan teks pada label terlihat jelas dan tajam</li>
                    <li>❌ Hindari bayangan atau pantulan cahaya</li>
                    <li>📏 Jarak ideal: 15-30cm dari label</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
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
        
        # Consumption simulator
        st.subheader("🧮 Simulasi Konsumsi")
        servings = st.number_input("Jumlah kemasan:", min_value=1, max_value=10, value=1)
        
        # Educational content
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
    
    # Results section
    if nutrients:
        st.markdown("---")
        st.header("📊 Hasil Analisis")
        
        # Adjust for multiple servings
        adjusted_nutrients = {k: v * servings for k, v in nutrients.items()}
        
        # Create grade cards
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
        
        # WHO Daily Limit Comparison
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
        
        # Health recommendations
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
        
        # Visualization
        st.subheader("📊 Visualisasi Data")
        fig = create_visualization(adjusted_nutrients, analyzer)
        st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        st.subheader("💾 Export Hasil")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Download PDF Report"):
                st.info("Fitur download PDF akan tersedia dalam versi production")
        
        with col2:
            if st.button("📊 Download Excel Data"):
                st.info("Fitur download Excel akan tersedia dalam versi production")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🏷️ NutriGrade Vision | Membantu Anda Membuat Pilihan Makanan atau Minuman yang Lebih Sehat</p>
        <p><small>Dikembangkan dengan ❤️ menggunakan Streamlit & Python</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
