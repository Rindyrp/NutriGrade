import streamlit as st
import cv2
import numpy as np
import pandas as pd
import re
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf
from tensorflow import keras
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import easyocr
import paddleocr
import pytesseract
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from skimage import filters, morphology, segmentation
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OCRConfig:
    """Konfigurasi sistem OCR"""
    model_type: str = "paddleocr"
    language: List[str] = None
    use_gpu: bool = False
    confidence_threshold: float = 0.5
    preprocessing_enabled: bool = True
    postprocessing_enabled: bool = True
    
    def __post_init__(self):
        if self.language is None:
            self.language = ['id', 'en']

class AdvancedImagePreprocessor:
    """Preprocessing gambar tingkat lanjut untuk OCR label gizi"""
    
    def __init__(self):
        self.setup_filters()
    
    def setup_filters(self):
        """Menyiapkan filter dan kernel"""
        self.sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        self.edge_kernel = np.array([[-1,-1,-1,-1,-1], [-1,2,2,2,-1], [-1,2,8,2,-1], [-1,2,2,2,-1], [-1,-1,-1,-1,-1]]) / 8.0
    
    def camera_calibration_correction(self, image, camera_matrix=None, dist_coeffs=None):
        if camera_matrix and dist_coeffs:
            h, w = image.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
            undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
            x, y, w, h = roi
            return undistorted[y:y+h, x:x+w]
        return image
    
    def detect_document_corners(self, image):
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
    
    def perspective_correction(self, image):
        corners = self.detect_document_corners(image)
        if corners is None:
            return image
        corners = self.order_corners(corners)
        width_a = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) + ((corners[2][1] - corners[3][1]) ** 2))
        width_b = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) + ((corners[1][1] - corners[0][1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        height_a = np.sqrt(((corners[1][0] - corners[2][0]) ** 2) + ((corners[1][1] - corners[2][1]) ** 2))
        height_b = np.sqrt(((corners[0][0] - corners[3][0]) ** 2) + ((corners[0][1] - corners[3][1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(corners.astype("float32"), dst)
        return cv2.warpPerspective(image, matrix, (max_width, max_height))
    
    def order_corners(self, corners):
        rect = np.zeros((4, 2), dtype="float32")
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]
        rect[3] = corners[np.argmax(diff)]
        return rect
    
    def adaptive_lighting_correction(self, image):
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def remove_glare_and_shadows(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        result = cv2.absdiff(gray, opened)
        return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    
    def advanced_denoising(self, image):
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        return cv2.bilateralFilter(denoised, 9, 75, 75)
    
    def enhance_text_contrast(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        adaptive_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return cv2.bitwise_and(otsu, cv2.bitwise_and(adaptive_mean, adaptive_gaussian))
    
    def apply_sharpening(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        return cv2.filter2D(unsharp_mask, -1, self.sharpen_kernel)
    
    def preprocess_pipeline(self, image, **kwargs):
        processed = image.copy()
        if 'camera_matrix' in kwargs and 'dist_coeffs' in kwargs:
            processed = self.camera_calibration_correction(processed, kwargs['camera_matrix'], kwargs['dist_coeffs'])
        processed = self.perspective_correction(processed)
        processed = self.adaptive_lighting_correction(processed)
        processed = self.remove_glare_and_shadows(processed)
        processed = self.advanced_denoising(processed)
        processed = self.apply_sharpening(processed)
        processed = self.enhance_text_contrast(processed)
        return processed

class MultiModelOCR:
    """Sistem OCR multi-model dengan kemampuan ensemble"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.initialize_models()
    
    def initialize_models(self):
        if self.config.model_type in ["paddleocr", "ensemble"]:
            self.models['paddleocr'] = paddleocr.PaddleOCR(use_angle_cls=True, lang=self.config.language[0], use_gpu=self.config.use_gpu, show_log=False)
        if self.config.model_type in ["easyocr", "ensemble"]:
            self.models['easyocr'] = easyocr.Reader(self.config.language, gpu=self.config.use_gpu)
        if self.config.model_type in ["tesseract", "ensemble"]:
            self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()%:-/ '
    
    def extract_with_paddleocr(self, image):
        results = self.models['paddleocr'].ocr(image, cls=True)
        extracted_data = []
        for line in results[0] if results[0] else []:
            bbox, text_info = line
            text, confidence = text_info
            if confidence >= self.config.confidence_threshold:
                extracted_data.append({'text': text, 'confidence': confidence, 'bbox': bbox, 'model': 'paddleocr'})
        return extracted_data
    
    def extract_with_easyocr(self, image):
        results = self.models ait['easyocr'].readtext(image)
        extracted_data = []
        for bbox, text, confidence in results:
            if confidence >= self.config.confidence_threshold:
                extracted_data.append({'text': text, 'confidence': confidence, 'bbox': bbox, 'model': 'easyocr'})
        return extracted_data
    
    def extract_with_tesseract(self, image):
        data = pytesseract.image_to_data(image, config=self.tesseract_config, output_type=pytesseract.Output.DICT)
        extracted_data = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) >= self.config.confidence_threshold * 100:
                text = data['text'][i].strip()
                if text:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    extracted_data.append({'text': text, 'confidence': int(data['conf'][i]) / 100.0, 'bbox': bbox, 'model': 'tesseract'})
        return extracted_data
    
    def ensemble_extraction(self, image):
        all_results = []
        if 'paddleocr' in self.models:
            all_results.extend(self.extract_with_paddleocr(image))
        if 'easyocr' in self.models:
            all_results.extend(self.extract_with_easyocr(image))
        if self.config.model_type in ["tesseract", "ensemble"]:
            all_results.extend(self.extract_with_tesseract(image))
        return self.combine_results(all_results)
    
    def combine_results(self, results):
        if not results:
            return []
        results.sort(key=lambda x: x['confidence'], reverse=True)
        final_results = []
        seen_texts = set()
        for result in results:
            text_clean = re.sub(r'\s+', ' ', result['text'].strip().lower())
            if not any(self.text_similarity(text_clean, seen) > 0.8 for seen in seen_texts):
                seen_texts.add(text_clean)
                final_results.append(result)
        return final_results
    
    def text_similarity(self, text1, text2):
        def levenshtein_distance(s1, s2):
            if len(s1) > len(s2):
                s1, s2 = s2, s1
            distances = range(len(s1) + 1)
            for i2, c2 in enumerate(s2):
                distances_ = [i2 + 1]
                for i1, c1 in enumerate(s1):
                    if c1 == c2:
                        distances_.append(distances[i1])
                    else:
                        distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
                distances = distances_
            return distances[-1]
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        distance = levenshtein_distance(text1, text2)
        return 1 - (distance / max_len)
    
    def extract_text(self, image):
        if self.config.model_type == "ensemble":
            return self.ensemble_extraction(image)
        elif self.config.model_type == "paddleocr":
            return self.extract_with_paddleocr(image)
        elif self.config.model_type == "easyocr":
            return self.extract_with_easyocr(image)
        elif self.config.model_type == "tesseract":
            return self.extract_with_tesseract(image)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

class NutritionPostProcessor:
    """Post-processing tingkat lanjut untuk data label gizi"""
    
    def __init__(self):
        self.setup_correction_rules()
        self.setup_nutrition_patterns()
    
    def setup_correction_rules(self):
        self.character_corrections = {
            'O': '0', 'o': '0', 'l': '1', 'I': '1', 'S': '5', 's': '5',
            'G': '6', 'g': '9', 'B': '8', 'Z': '2', 'z': '2',
            'mq': 'mg', 'qm': 'gm', 'cal': 'kcal', 'keal': 'kcal'
        }
        self.unit_corrections = {
            'mq': 'mg', 'qm': 'mg', 'mig': 'mg', 'mĝ': 'mg',
            'qg': 'g', 'gq': 'g', 'ĝ': 'g',
            'cal': 'kcal', 'keal': 'kcal', 'kCal': 'kcal'
        }
    
    def setup_nutrition_patterns(self):
        self.nutrition_patterns = {
            'calories': [r'(?:kalori|calories?|energi|energy)\s*:?\s*(\d+(?:\.\d+)?)\s*(?:kkal|kcal|cal)?', 
                         r'(\d+(?:\.\d+)?)\s*(?:kkal|kcal|cal)'],
            'total_fat': [r'(?:lemak\s*total|total\s*fat|fat)\s*:?\s*(\d+(?:\.\d+)?)\s*g', 
                          r'(?:lemak|fat)\s*:?\s*(\d+(?:\.\d+)?)\s*g'],
            'saturated_fat': [r'(?:lemak\s*jenuh|saturated\s*fat|lemak\s*trans|trans\s*fat)\s*:?\s*(\d+(?:\.\d+)?)\s*g'],
            'cholesterol': [r'(?:kolesterol|cholesterol)\s*:?\s*(\d+(?:\.\d+)?)\s*mg'],
            'sodium': [r'(?:sodium|natrium|garam|salt)\s*:?\s*(\d+(?:\.\d+)?)\s*mg'],
            'total_carbs': [r'(?:karbohidrat\s*total|total\s*carbohydrate|carbs?)\s*:?\s*(\d+(?:\.\d+)?)\s*g'],
            'dietary_fiber': [r'(?:serat\s*pangan|dietary\s*fiber|fiber)\s*:?\s*(\d+(?:\.\d+)?)\s*g'],
            'total_sugars': [r'(?:gula\s*total|total\s*sugar|sugar)\s*:?\s*(\d+(?:\.\d+)?)\s*g'],
            'added_sugars': [r'(?:gula\s*tambahan|added\s*sugar)\s*:?\s*(\d+(?:\.\d+)?)\s*g'],
            'protein': [r'(?:protein)\s*:?\s*(\d+(?:\.\d+)?)\s*g']
        }
    
    def correct_common_errors(self, text):
        corrected = text
        for wrong, correct in self.character_corrections.items():
            corrected = corrected.replace(wrong, correct)
        for wrong, correct in self.unit_corrections.items():
            corrected = re.sub(rf'\b{wrong}\b', correct, corrected, flags=re.IGNORECASE)
        return corrected
    
    def extract_nutrition_values(self, text_data):
        full_text = ' '.join([item['text'] for item in text_data])
        full_text = self.correct_common_errors(full_text.lower())
        nutrition_values = {}
        for nutrient, patterns in self.nutrition_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, full_text, re.IGNORECASE)
                if matches:
                    try:
                        value = float(matches[0])
                        nutrition_values[nutrient] = value
                        break
                    except (ValueError, IndexError):
                        continue
        return nutrition_values
    
    def validate_nutrition_values(self, values):
        ranges = {
            'calories': (0, 1000),
            'total_fat': (0, 100),
            'saturated_fat': (0, 50),
            'cholesterol': (0, 1000),
            'sodium': (0, 5000),
            'total_carbs': (0, 100),
            'dietary_fiber': (0, 50),
            'total_sugars': (0, 100),
            'added_sugars': (0, 100),
            'protein': (0, 100)
        }
        return {k: v for k, v in values.items() if k in ranges and ranges[k][0] <= v <= ranges[k][1]}
    
    def structure_output(self, nutrition_values, original_data):
        return {
            'nutrition_facts': nutrition_values,
            'raw_ocr_data': original_data,
            'extraction_metadata': {
                'total_text_items': len(original_data),
                'average_confidence': np.mean([item['confidence'] for item in original_data]) if original_data else 0,
                'models_used': list(set([item['model'] for item in original_data])) if original_data else [],
                'processing_timestamp': pd.Timestamp.now().isoformat()
            }
        }

class EnhancedNutritionOCR:
    """Kelas utama yang menggabungkan semua komponen"""
    
    def __init__(self, config=None):
        self.config = config or OCRConfig()
        self.preprocessor = AdvancedImagePreprocessor()
        self.ocr_engine = MultiModelOCR(self.config)
        self.postprocessor = NutritionPostProcessor()
    
    def process_image(self, image_path, **kwargs):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self.process_image_array(image, **kwargs)
    
    def process_image_array(self, image, **kwargs):
        if self.config.preprocessing_enabled:
            processed_image = self.preprocessor.preprocess_pipeline(image, **kwargs)
        else:
            processed_image = image
        ocr_results = self.ocr_engine.extract_text(processed_image)
        if not ocr_results:
            return self.empty_result()
        if self.config.postprocessing_enabled:
            nutrition_values = self.postprocessor.extract_nutrition_values(ocr_results)
            validated_values = self.postprocessor.validate_nutrition_values(nutrition_values)
            return self.postprocessor.structure_output(validated_values, ocr_results)
        return {'raw_ocr_data': ocr_results}
    
    def empty_result(self):
        return {
            'nutrition_facts': {},
            'raw_ocr_data': [],
            'extraction_metadata': {
                'total_text_items': 0,
                'average_confidence': 0,
                'models_used': [],
                'processing_timestamp': pd.Timestamp.now().isoformat()
            }
        }

def main():
    st.set_page_config(page_title="NutriGrade Vision", page_icon="🏷️", layout="wide")
    st.markdown("<h1 style='text-align: center;'>🏷️ NutriGrade Vision</h1>", unsafe_allow_html=True)
    
    config = OCRConfig(model_type="paddleocr", language=['id', 'en'], use_gpu=False)
    ocr_system = EnhancedNutritionOCR(config)
    
    input_method = st.sidebar.radio("Metode Input:", ["📁 Upload Gambar", "📸 Kamera", "📝 Input Manual"])
    
    if input_method == "📁 Upload Gambar":
        uploaded_file = st.file_uploader("Pilih gambar label gizi:", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diupload", use_column_width=True)
            if st.button("🔍 Analisis Gambar"):
                with st.spinner("Memproses gambar..."):
                    image_array = np.array(image)
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    result = ocr_system.process_image_array(image_array)
                    st.subheader("📄 Hasil Ekstraksi Awal:")
                    extracted_text = '\n'.join([item['text'] for item in result['raw_ocr_data']])
                    edited_text = st.text_area("Edit Hasil Ekstraksi:", value=extracted_text, height=200)
                    
                    if st.button("💾 Simpan Perubahan"):
                        edited_data = [{'text': line, 'confidence': 1.0, 'bbox': None, 'model': 'manual'} 
                                     for line in edited_text.split('\n') if line.strip()]
                        edited_result = ocr_system.postprocessor.structure_output(
                            ocr_system.postprocessor.extract_nutrition_values(edited_data),
                            edited_data
                        )
                        st.subheader("📊 Hasil Akhir:")
                        st.json(edited_result)
    
    elif input_method == "📸 Kamera":
        camera_image = st.camera_input("Ambil foto label gizi")
        if camera_image:
            image = Image.open(camera_image)
            st.image(image, caption="Foto dari kamera", use_column_width=True)
            if st.button("🔍 Analisis Foto"):
                with st.spinner("Memproses foto..."):
                    image_array = np.array(image)
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    result = ocr_system.process_image_array(image_array)
                    st.subheader("📄 Hasil Ekstraksi Awal:")
                    extracted_text = '\n'.join([item['text'] for item in result['raw_ocr_data']])
                    edited_text = st.text_area("Edit Hasil Ekstraksi:", value=extracted_text, height=200)
                    
                    if st.button("💾 Simpan Perubahan"):
                        edited_data = [{'text': line, 'confidence': 1.0, 'bbox': None, 'model': 'manual'} 
                                     for line in edited_text.split('\n') if line.strip()]
                        edited_result = ocr_system.postprocessor.structure_output(
                            ocr_system.postprocessor.extract_nutrition_values(edited_data),
                            edited_data
                        )
                        st.subheader("📊 Hasil Akhir:")
                        st.json(edited_result)
    
    elif input_method == "📝 Input Manual":
        st.subheader("Input Nilai Gizi Manual")
        manual_text = st.text_area("Masukkan teks label gizi:", height=200)
        if st.button("📊 Analisis Manual"):
            if manual_text:
                manual_data = [{'text': line, 'confidence': 1.0, 'bbox': None, 'model': 'manual'} 
                             for line in manual_text.split('\n') if line.strip()]
                result = ocr_system.postprocessor.structure_output(
                    ocr_system.postprocessor.extract_nutrition_values(manual_data),
                    manual_data
                )
                st.subheader("📊 Hasil Analisis:")
                st.json(result)

if __name__ == "__main__":
    main()
