import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import urllib.request

class ImageComparisonApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Comparison App")
        self.image1_path = None
        self.image2_path = None
        self.create_widgets()

    def create_widgets(self):
        self.btn_image1 = tk.Button(self.master, text="Select Image 1", command=self.select_image1)
        self.btn_image1.pack()
        self.btn_image2 = tk.Button(self.master, text="Select Image 2", command=self.select_image2)
        self.btn_image2.pack()
        self.method_var = tk.StringVar(value="ssim")
        self.radio_ssim = tk.Radiobutton(self.master, text="SSIM", variable=self.method_var, value="ssim")
        self.radio_ssim.pack()
        self.radio_orb = tk.Radiobutton(self.master, text="ORB", variable=self.method_var, value="orb")
        self.radio_orb.pack()
        self.btn_process = tk.Button(self.master, text="Process Images", command=self.process_images)
        self.btn_process.pack()
        self.result_label = tk.Label(self.master, text="")
        self.result_label.pack()

    def select_image1(self):
        self.image1_path = filedialog.askopenfilename(title="Select Image 1")
        if self.image1_path:
            self.btn_image1.config(text="Image 1 Selected")

    def select_image2(self):
        self.image2_path = filedialog.askopenfilename(title="Select Image 2")
        if self.image2_path:
            self.btn_image2.config(text="Image 2 Selected")

    def download_yolo_files(self):
        yolo_files = {
            'yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights',
            'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
        }
        for file_name, url in yolo_files.items():
            try:
                urllib.request.urlretrieve(url, file_name)
                print(f"Successfully downloaded {file_name}")
            except Exception as e:
                print(f"Failed to download {file_name}: {str(e)}")
                return False
        return True

    def process_images(self):
        if not self.image1_path or not self.image2_path:
            self.result_label.config(text="Please select both images.")
            return

        os.makedirs('processed_images', exist_ok=True)

        method = self.method_var.get()
        if method == "ssim":
            self.ssim_comparison()
        elif method == "orb":
            self.orb_comparison()
            self.object_detection_on_orb()

        self.result_label.config(text=f"Processing complete. Results saved in 'processed_images' folder.")

    def ssim_comparison(self):
        img1 = img_as_float(io.imread(self.image1_path, as_gray=True))
        img2 = img_as_float(io.imread(self.image2_path, as_gray=True))
        if img1.shape != img2.shape:
            img2 = resize(img2, img1.shape, anti_aliasing=True)
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
        ssim_value, ssim_map = ssim(img1, img2, full=True, data_range=data_range)
        threshold = 0.8
        highlighted_map = np.where(ssim_map > threshold, ssim_map, 0)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        ax1.imshow(img1, cmap='gray')
        ax1.set_title('Image 1')
        ax1.axis('off')
        ax2.imshow(img2, cmap='gray')
        ax2.set_title('Image 2')
        ax2.axis('off')
        cmap = plt.cm.get_cmap('jet').copy()
        cmap.set_under('black')
        im = ax3.imshow(highlighted_map, cmap=cmap, vmin=threshold, vmax=1)
        ax3.set_title(f'Highlighted Object Similarities\nSSIM Value: {ssim_value:.4f}')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, label='Similarity')
        plt.tight_layout()
        plt.savefig('processed_images/ssim_visualization.png')
        plt.close()

    def orb_comparison(self):
        img1 = cv2.imread(self.image1_path)
        img2 = cv2.imread(self.image2_path)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        plt.imsave('processed_images/orb_visualization.png', img3)

    def object_detection_on_orb(self):
        if not os.path.exists('yolov3.weights') or not os.path.exists('yolov3.cfg'):
            if not self.download_yolo_files():
                self.result_label.config(text="Failed to download YOLO files. Object detection will not be performed.")
                return

        orb_result = cv2.imread('processed_images/orb_visualization.png')
        detected_image = self.object_detection(orb_result)
        cv2.imwrite('processed_images/orb_with_objects.png', detected_image)

    def object_detection(self, img):
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageComparisonApp(root)
    root.mainloop()
