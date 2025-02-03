import gradio as gr
import numpy as np
import cv2
import rembg
from PIL import Image
from io import BytesIO
import onnxruntime as ort
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import face_recognition


def detect_face_with_margins(image: np.ndarray):
    """Определяет лицо и рассчитывает необходимые отступы."""
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        raise ValueError("Лицо не обнаружено на изображении")

    top, right, bottom, left = face_locations[0]
    face_height = bottom - top

    # Находим верхнюю границу головы
    head_top = top
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    search_top = max(0, top - int(face_height * 0.8))  # Ищем выше лица

    for y in range(search_top, top):
        row = gray_image[y, max(0, left - 10) : min(right + 10, gray_image.shape[1])]
        if np.mean(row) > 240:  # Если строка почти белая
            head_top = y
            break

    face_center_x = (left + right) // 2
    face_center_y = (top + bottom) // 2

    return {
        "face_box": (top, right, bottom, left),
        "head_top": head_top,
        "face_center": (face_center_x, face_center_y),
        "face_height": face_height,
        "head_height": bottom - head_top,
    }


def calculate_target_positions(target_height: int = 720):
    """Рассчитывает целевые позиции на конечном изображении."""
    # Целевая высота головы (56.7% от высоты изображения)
    head_height = int(target_height * 0.567)
    # Отступ сверху (16.7% от высоты изображения)
    top_margin = int(target_height * 0.167)
    # Позиция центра лица (примерно посередине головы)
    face_center_y = top_margin + int(head_height * 0.5)

    return {
        "target_height": target_height,
        "target_width": 600,
        "head_height": head_height,
        "top_margin": top_margin,
        "face_center_y": face_center_y,
    }


def remove_background(image: Image.Image) -> Image.Image:
    """Удаляет фон, делает его белым."""
    image = image.convert("RGBA")
    data = rembg.remove(np.array(image))
    img_without_bg = Image.fromarray(data)
    white_bg = Image.new("RGBA", img_without_bg.size, "WHITE")
    white_bg.paste(img_without_bg, mask=img_without_bg)
    return white_bg.convert("RGB")


def align_and_scale_face(image: Image.Image) -> Image.Image:
    """Выравнивает и масштабирует изображение относительно центра лица."""
    np_image = np.array(image)
    face_data = detect_face_with_margins(np_image)
    target = calculate_target_positions()

    # Рассчитываем масштаб относительно требуемой высоты головы
    scale = target["head_height"] / face_data["head_height"]

    # Масштабируем изображение
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Создаем итоговое изображение
    final_image = Image.new(
        "RGB", (target["target_width"], target["target_height"]), "white"
    )

    # Рассчитываем смещения для правильного позиционирования
    scaled_face_center_y = int(face_data["face_center"][1] * scale)
    y_offset = target["face_center_y"] - scaled_face_center_y

    scaled_face_center_x = int(face_data["face_center"][0] * scale)
    x_offset = (target["target_width"] // 2) - scaled_face_center_x

    # Вставляем изображение
    final_image.paste(scaled_image, (x_offset, y_offset))

    return final_image


def create_print_sheet(image: Image.Image) -> str:
    """Создает лист с 4 фото и сохраняет в JPEG."""
    sheet = Image.new("RGB", (1200, 1440), "white")
    positions = [(0, 0), (600, 0), (0, 720), (600, 720)]
    for pos in positions:
        sheet.paste(image, pos)
    file_path = "document.jpg"
    sheet.save(file_path, format="JPEG", dpi=(305, 305))
    return file_path


def process_image(image: Image.Image):
    """Основная функция обработки изображения."""
    try:
        image = remove_background(image)
        image = align_and_scale_face(image)
        file_path = create_print_sheet(image)
        return image, file_path
    except ValueError as e:
        raise gr.Error(str(e))
    except Exception as e:
        raise gr.Error(f"Произошла ошибка при обработке изображения: {str(e)}")


demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(), gr.File(label="Скачать JPG")],
    title="Генератор фото на документы",
    description="Загрузи фото, получи готовый файл с 4 изображениями для печати с высотой лица 34 мм.",
)

demo.launch()
