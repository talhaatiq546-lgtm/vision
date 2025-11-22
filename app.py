import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------------------------------
# Load YOLOv8 model directly from root: yolov8m.pt
# -----------------------------------------------------
model = YOLO("yolov8m.pt")

# -----------------------------------------------------
# Object Detection Function
# -----------------------------------------------------
def detect(image):
    """
    image: PIL image from Gradio
    return: annotated image + list of detections
    """
    if image is None:
        return None, "No image provided."

    # Convert PIL → numpy
    img = np.array(image)

    # Run detection
    results = model(img)[0]

    objects = []

    # Draw boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)

        objects.append(f"{label} ({conf:.2f})")

    return img, "\n".join(objects)


# -----------------------------------------------------
# Gradio UI
# -----------------------------------------------------
with gr.Blocks(title="YOLOv8 Object Detection") as demo:

    gr.Markdown("<h2 style='text-align:center;'>YOLOv8 Object Detection</h2>")

    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload Image", source="upload")
        webcam_image = gr.Image(type="pil", label="Webcam", source="webcam")

    detect_button = gr.Button("Run Detection")

    output_image = gr.Image(label="Detection Result")
    output_objects = gr.Textbox(label="Detected Objects", lines=10)

    # choose webcam image if available
    def choose_image(upload_img, cam_img):
        return cam_img if cam_img is not None else upload_img

    detect_button.click(
        fn=lambda up, cam: detect(choose_image(up, cam)),
        inputs=[input_image, webcam_image],
        outputs=[output_image, output_objects]
    )

demo.launch()
