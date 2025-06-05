import streamlit as st
import cv2
import numpy as np
import pandas as pd
import re
from PIL import Image, ImageEnhance, ImageFilter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Tuple, Optional
import io
import base64
import easyocr

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
        
        # Pattern untuk menangkap angka dengan satuan, termasuk desimal, variasi spasi, dan tanda baca
        patterns = {
            'sugar': r'(?:' + '|'.join(self.nutrient_aliases['sugar']) + r')\s*[:=]?\s*(\d+\.?\d*|\.\d+)\s*(?:g|gram|grams)?(?:\s|$|\b)',
            'sodium': r'(?:' + '|'.join(self.nutrient_aliases['sodium']) + r')\s*[:=]?\s*(\d+\.?\d*|\.\d+)\s*(?:mg|milligram|milligrams)?(?:\s|$|\b)',
            'saturated_fat': r'(?:' + '|'.join(self.nutrient_aliases['saturated_fat']) + r')\s*[:=]?\s*(\d+\.?\d*|\.\d+)\s*(?:g|gram|grams)?(?:\s|$|\b)'
        }
        
        for nutrient, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # Ambil nilai tertinggi, filter nilai yang valid
                    valid_matches = [float(match) for match in matches if match.strip()]
                    nutrients[nutrient] = max(valid_matches) if valid_matches else 0
                except ValueError:
                    nutrients[nutrient] = 0  # Default ke 0 jika konversi gagal
        
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

def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocessing gambar untuk meningkatkan akurasi OCR"""
    # Convert ke array numpy
    img_array = np.array(image)
    
    # Convert ke grayscale jika berwarna
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Noise reduction dengan Gaussian blur ringan
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Contrast enhancement menggunakan CLAHE dengan parameter yang disesuaikan
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12, 12))
    enhanced = clahe.apply(denoised)
    
    # Adaptive thresholding dengan parameter yang lebih sensitif
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 10
    )
    
    # Morphological operations untuk memperkuat teks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    return Image.fromarray(eroded)

def detect_and_crop_nutrition_label(image: Image.Image) -> Image.Image:
    """Deteksi dan crop area label nutrisi secara otomatis"""
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing untuk deteksi tepi yang lebih baik
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours berdasarkan area, aspect ratio, dan kepadatan teks
    potential_labels = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 20000:  # Area minimum lebih besar untuk menangkap label
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Label nutrisi biasanya persegi panjang, periksa kepadatan teks
            roi = gray[y:y+h, x:x+w]
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text_density = np.sum(binary == 255) / (w * h)
            
            if 0.1 < aspect_ratio < 5.0 and w > 300 and h > 300 and 0.05 < text_density < 0.5:
                potential_labels.append((x, y, w, h, area))
    
    if potential_labels:
        # Ambil kontour dengan area terbesar
        x, y, w, h, _ = max(potential_labels, key=lambda x: x[4])
        
        # Add padding yang lebih besar
        padding = 40
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(opencv_image.shape[1] - x, w + 2*padding)
        h = min(opencv_image.shape[0] - y, h + 2*padding)
        
        # Crop image
        cropped = opencv_image[y:y+h, x:x+w]
        return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    
    return image  # Return original jika tidak ada kontur yang cocok

def enhance_text_regions(image: Image.Image) -> Image.Image:
    """Enhance regions yang kemungkinan mengandung teks"""
    # Sharpen image dengan parameter yang lebih kuat
    sharpened = image.filter(ImageFilter.UnsharpMask(radius=3, percent=250, threshold=2))
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(sharpened)
    contrasted = enhancer.enhance(2.5)
    
    # Enhance brightness secara adaptif
    enhancer = ImageEnhance.Brightness(contrasted)
    brightened = enhancer.enhance(1.3)
    
    return brightened

def preprocess_nutrition_label(image: Image.Image) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """Pipeline lengkap preprocessing untuk label nutrisi"""
    try:
        # Step 1: Detect and crop nutrition label area
        st.write("🔍 Mendeteksi area label nutrisi...")
        cropped_image = detect_and_crop_nutrition_label(image)
        
        # Step 2: Enhance text regions
        st.write("✨ Meningkatkan kualitas teks...")
        enhanced_image = enhance_text_regions(cropped_image)
        
        # Step 3: Advanced preprocessing
        st.write("🔧 Memproses gambar untuk OCR...")
        processed_image = preprocess_image(enhanced_image)
        
        return processed_image, cropped_image, enhanced_image
    
    except Exception as e:
        st.error(f"Error dalam preprocessing: {str(e)}")
        return image, image, image

def perform_ocr(image: Image.Image) -> str:
    """Melakukan OCR menggunakan EasyOCR dengan pengaturan lanjutan"""
    try:
        # Inisialisasi EasyOCR reader dengan parameter optimal
        reader = easyocr.Reader(['en', 'id'], gpu=False)
        
        # Convert PIL Image ke numpy array
        img_array = np.array(image)
        
        # Lakukan OCR dengan pengaturan untuk teks kecil dan padat
        results = reader.readtext(
            img_array, 
            detail=0, 
            paragraph=True, 
            contrast_ths=0.1,  # Sensitivitas kontras rendah untuk teks redup
            adjust_contrast=0.5,  # Penyesuaian kontras otomatis
            text_threshold=0.7,  # Ambang batas deteksi teks
            low_text=0.3,  # Deteksi teks kecil
            mag_ratio=1.5  # Meningkatkan resolusi untuk teks kecil
        )
        
        # Gabungkan hasil OCR menjadi satu string
        extracted_text = " ".join(results)
        return extracted_text if extracted_text else "Tidak ada teks yang terdeteksi"
    
    except Exception as e:
        st.error(f"Error dalam OCR: {str(e)}")
        return "Gagal mengekstrak teks"

def create_visualization(nutrients: Dict[str, float], analyzer: NutrientAnalyzer):
    """Membuat visualisasi hasil analisis"""
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
                        processed_image, cropped_image, enhanced_image = preprocess_nutrition_label(image)
                        
                        # Show processing steps
                        st.subheader("🔧 Hasil Preprocessing")
                        tab1, tab2, tab3 = st.tabs(["Cropped", "Enhanced", "Final"])
                        with tab1:
                            st.image(cropped_image, caption="Area label yang terdeteksi", use_column_width=True)
                        with tab2:
                            st.image(enhanced_image, caption="Gambar yang ditingkatkan", use_column_width=True)
                        with tab3:
                            st.image(processed_image, caption="Siap untuk OCR", use_column_width=True)
                        
                        # Perform OCR
                        extracted_text = perform_ocr(processed_image)
                        
                        st.subheader("📄 Teks yang Diekstrak:")
                        st.text_area("OCR Result:", extracted_text, height=150)
                        
                        # Extract nutrients
                        nutrients = analyzer.extract_nutrients_from_text(extracted_text)
        
        elif input_method == "📸 Kamera":
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
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📷 Foto Original")
                    st.image(image, caption="Foto dari kamera", use_column_width=True)
                
                # Analysis button
                if st.button("🔍 Analisis Foto Kamera", key="analyze_camera"):
                    with st.spinner("Memproses foto dengan AI..."):
                        # Advanced preprocessing
                        processed_image, cropped_image, enhanced_image = preprocess_nutrition_label(image)
                        
                        # Show processing steps
                        with col2:
                            st.subheader("🔧 Hasil Preprocessing")
                            tab1, tab2, tab3 = st.tabs(["Cropped", "Enhanced", "Final"])
                            with tab1:
                                st.image(cropped_image, caption="Area label yang terdeteksi", use_column_width=True)
                            with tab2:
                                st.image(enhanced_image, caption="Gambar yang ditingkatkan", use_column_width=True)
                            with tab3:
                                st.image(processed_image, caption="Siap untuk OCR", use_column_width=True)
                        
                        # Perform OCR
                        extracted_text = perform_ocr(processed_image)
                        
                        st.subheader("📄 Teks yang Diekstrak:")
                        st.text_area("OCR Result:", extracted_text, height=150, key="camera_ocr")
                        
                        # Extract nutrients
                        nutrients = analyzer.extract_nutrients_from_text(extracted_text)
                        
                        if nutrients:
                            st.subheader("🥗 Informasi Nutrisi yang Terdeteksi:")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Gula", f"{nutrients['sugar']:.1f} g")
                            with col2:
                                st.metric("Sodium", f"{nutrients['sodium']:.1f} mg")
                            with col3:
                                st.metric("Lemak Jenuh", f"{nutrients['saturated_fat']:.1f} g")
                    
                    st.success("✅ Foto berhasil dianalisis dengan preprocessing yang ditingkatkan!")
            
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
