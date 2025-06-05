import streamlit as st
import cv2
import numpy as np
import pandas as pd
import re
from PIL import Image, ImageEnhance, ImageFilter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import easyocr
import io
import base64

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
    
    def extract_nutrients_from_text(self, text: str) -> dict:
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
                nutrients[nutrient] = max([float(match) for match in matches])
        
        # Validasi nilai yang diekstrak
        if nutrients['sugar'] > 1000 or nutrients['sodium'] > 10000 or nutrients['saturated_fat'] > 100:
            st.warning("⚠️ Nilai nutrisi yang diekstrak tampak tidak realistis.")
            return None
        
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
    
    def get_health_recommendation(self, nutrients: dict, condition: str) -> list:
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

def detect_and_crop_nutrition_label(image: Image.Image) -> Image.Image:
    """Deteksi dan crop area label nutrisi secara otomatis"""
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    potential_labels = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.3 < aspect_ratio < 3.0 and w > 200 and h > 200:
                potential_labels.append((x, y, w, h, area))
    
    if potential_labels:
        x, y, w, h, _ = max(potential_labels, key=lambda x: x[4])
        padding = 20
        x, y = max(0, x - padding), max(0, y - padding)
        w, h = min(opencv_image.shape[1] - x, w + 2*padding), min(opencv_image.shape[0] - y, h + 2*padding)
        cropped = opencv_image[y:y+h, x:x+w]
        return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    
    return image

def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocessing gambar untuk meningkatkan akurasi OCR"""
    gray = image.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(2.0)
    blurred = enhanced.filter(ImageFilter.GaussianBlur(radius=1))
    binarized = blurred.point(lambda p: 255 if p > 128 else 0)
    return binarized

def extract_text_from_image(image: Image.Image) -> str:
    """Ekstrak teks dari gambar menggunakan EasyOCR"""
    reader = easyocr.Reader(['en', 'id'])
    result = reader.readtext(np.array(image), detail=0)
    return ' '.join(result)

def create_visualization(nutrients: dict, analyzer: NutrientAnalyzer):
    """Membuat visualisasi hasil analisis"""
    fig = make_subplots(rows=2, cols=2, subplot_titles=('', 'WHO Daily Limit Comparison', 'Nutrient Values', 'Health Risk Assessment'),
                        specs=[[{"type": "indicator"}, {"type": "bar"}], [{"type": "scatter"}, {"type": "pie"}]])
    
    grades = [analyzer.get_grade(nutrient, value) for nutrient, value in nutrients.items()]
    grade_scores = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
    avg_score = sum(grade_scores[grade] for grade in grades) / len(grades)
    
    fig.add_trace(go.Indicator(mode="gauge+number+delta", value=avg_score, domain={'x': [0, 1], 'y': [0, 1]},
                  title={'text': "Overall Grade"}, gauge={'axis': {'range': [1, 4]}, 'bar': {'color': "darkblue"},
                  'steps': [{'range': [1, 2], 'color': "lightgray"}, {'range': [2, 3], 'color': "gray"}],
                  'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 3.5}}),
                  row=1, col=1)
    
    who_percentages = [analyzer.calculate_who_percentage(nutrient, value) for nutrient, value in nutrients.items()]
    fig.add_trace(go.Bar(x=['Sugar', 'Sodium', 'Saturated Fat'], y=who_percentages,
                         marker_color=['red' if p > 100 else 'orange' if p > 50 else 'green' for p in who_percentages],
                         text=[f'{p:.1f}%' for p in who_percentages], textposition='auto'), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=['Sugar (g)', 'Sodium (mg)', 'Sat Fat (g)'],
                            y=[nutrients['sugar'], nutrients['sodium'], nutrients['saturated_fat']],
                            mode='markers+text', marker=dict(size=20, color=[analyzer.get_grade_color(analyzer.get_grade(nutrient, value)) 
                                                                             for nutrient, value in nutrients.items()]),
                            text=grades, textposition="middle center", textfont=dict(color="white", size=14)), row=2, col=1)
    
    risk_counts = {level: grades.count(level) for level in ['A', 'B', 'C', 'D']}
    fig.add_trace(go.Pie(labels=list(risk_counts.keys()), values=list(risk_counts.values()),
                         marker_colors=[analyzer.get_grade_color(grade) for grade in risk_counts.keys()]), row=2, col=2)
    
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
        health_condition = st.selectbox("Kondisi Kesehatan:", ["Tidak ada", "Diabetes Mellitus", "Hipertensi", "Dislipidemia", "Anak-anak", "Ibu Hamil & Menyusui"])
        input_method = st.radio("Metode Input:", ["📁 Upload Gambar", "📸 Kamera", "📝 Input Manual"])
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
            uploaded_file = st.file_uploader("Pilih gambar label gizi:", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Gambar yang diupload", use_column_width=True)
                if st.button("🔍 Analisis Gambar"):
                    with st.spinner("Memproses gambar..."):
                        cropped_image = detect_and_crop_nutrition_label(image)
                        processed_image = preprocess_image(cropped_image)
                        extracted_text = extract_text_from_image(processed_image)
                        st.subheader("📄 Teks yang Diekstrak:")
                        st.text_area("OCR Result:", extracted_text, height=100)
                        nutrients = analyzer.extract_nutrients_from_text(extracted_text)
                        if nutrients is None:
                            st.error("Gagal mengekstrak nutrisi. Coba gambar lain atau input manual.")
        
        elif input_method == "📸 Kamera":
            camera_photo = st.camera_input("Ambil foto label nutrisi:")
            if camera_photo:
                image = Image.open(camera_photo)
                st.image(image, caption="Foto dari kamera", use_column_width=True)
                if st.button("🔍 Analisis Foto Kamera"):
                    with st.spinner("Memproses foto..."):
                        cropped_image = detect_and_crop_nutrition_label(image)
                        processed_image = preprocess_image(cropped_image)
                        extracted_text = extract_text_from_image(processed_image)
                        st.subheader("📄 Teks yang Diekstrak:")
                        st.text_area("OCR Result:", extracted_text, height=100)
                        nutrients = analyzer.extract_nutrients_from_text(extracted_text)
                        if nutrients is None:
                            st.error("Gagal mengekstrak nutrisi. Coba foto lain atau input manual.")
        
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
                nutrients = {'sugar': sugar, 'sodium': sodium, 'saturated_fat': sat_fat}
    
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
        for col, nutrient in zip([col1, col2, col3], ['sugar', 'sodium', 'saturated_fat']):
            with col:
                grade = analyzer.get_grade(nutrient, adjusted_nutrients[nutrient])
                emoji = analyzer.get_grade_emoji(grade)
                unit = 'mg' if nutrient == 'sodium' else 'g'
                st.markdown(f"""
                <div class="grade-card grade-{grade.lower()}">
                    <h3>{emoji} {nutrient.replace('_', ' ').title()}</h3>
                    <h2>Grade {grade}</h2>
                    <p>{adjusted_nutrients[nutrient]:.1f}{unit}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.subheader("📈 Perbandingan Batas Harian WHO")
        who_data = [{'Nutrisi': nutrient.replace('_', ' ').title(),
                     'Nilai': f"{value:.1f}{'g' if nutrient != 'sodium' else 'mg'}",
                     'Batas WHO': f"{analyzer.who_limits[nutrient]}{'g' if nutrient != 'sodium' else 'mg'}",
                     'Persentase': f"{analyzer.calculate_who_percentage(nutrient, value):.1f}%",
                     'Status': '⚠️ Melebihi' if analyzer.calculate_who_percentage(nutrient, value) > 100 else '✅ Aman'}
                    for nutrient, value in adjusted_nutrients.items()]
        st.dataframe(pd.DataFrame(who_data), use_container_width=True)
        
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
