import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import os
import urllib.request

class OptiChromApp:
    def __init__(self, master):
        self.master = master
        master.title("OptiChrom Image Comparison")
        
        self.image1_path = None
        self.image2_path = None
        
        self.create_widgets()
        self.download_coco_names()
    
    def create_widgets(self):
        self.image1_button = tk.Button(self.master, text="Select Image 1", command=self.select_image1)
        self.image1_button.pack()
        
        self.image2_button = tk.Button(self.master, text="Select Image 2", command=self.select_image2)
        self.image2_button.pack()
        
        self.method_var = tk.StringVar(value="ssim")
        self.ssim_radio = tk.Radiobutton(self.master, text="SSIM", variable=self.method_var, value="ssim")
        self.ssim_radio.pack()
        self.orb_radio = tk.Radiobutton(self.master, text="ORB", variable=self.method_var, value="orb")
        self.orb_radio.pack()
        
        self.process_button = tk.Button(self.master, text="Process Images", command=self.process_images)
        self.process_button.pack()
        
        self.result_label = tk.Label(self.master, text="")
        self.result_label.pack()
    
    def select_image1(self):
        self.image1_path = filedialog.askopenfilename()
        self.image1_button.config(text=f"Image 1: {self.image1_path.split('/')[-1]}")
    
    def select_image2(self):
        self.image2_path = filedialog.askopenfilename()
        self.image2_button.config(text=f"Image 2: {self.image2_path.split('/')[-1]}")
    
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
        
        self.object_detection(self.image1_path, 'processed_images/image1_objects.png')
        self.object_detection(self.image2_path, 'processed_images/image2_objects.png')
    
    def ssim_comparison(self):
        img1 = img_as_float(io.imread(self.image1_path))
        img2 = img_as_float(io.imread(self.image2_path))
        
        if img1.shape != img2.shape:
            img2 = resize(img2, img1.shape[:2], anti_aliasing=True, channel_axis=-1)
        
        ssim_value, ssim_map = ssim(img1, img2, multichannel=True, full=True)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        
        ax1.imshow(img1)
        ax1.set_title('Image 1')
        ax1.axis('off')
        
        ax2.imshow(img2)
        ax2.set_title('Image 2')
        ax2.axis('off')
        
        threshold = 0.7
        highlighted_map = np.ma.masked_where((ssim_map < threshold) & (ssim_map > -threshold), ssim_map)
        
        im = ax3.imshow(highlighted_map, cmap='RdYlBu', vmin=-1, vmax=1)
        ax3.set_title(f'Highlighted Object Similarities\nSSIM Value: {ssim_value:.4f}')
        ax3.axis('off')
        
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Similarity (Blue: Different, Red: Similar)', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig('processed_images/ssim_visualization.png')
        plt.close()
        
        self.result_label.config(text=f"SSIM visualization saved as 'processed_images/ssim_visualization.png'\nSSIM Value: {ssim_value:.4f}")
    
    def orb_comparison(self):
        img1 = cv2.imread(self.image1_path)
        img2 = cv2.imread(self.image2_path)
        
        img1 = cv2.resize(img1, (800, 600))
        img2 = cv2.resize(img2, (800, 600))
        
        orb = cv2.ORB_create()
        
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
        
        plt.figure(figsize=(16, 5))
        plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
        plt.title('ORB Feature Matching')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('processed_images/orb_visualization.png')
        plt.close()
        
        self.result_label.config(text="ORB visualization saved as 'processed_images/orb_visualization.png'")
    
    def object_detection(self, image_path, output_path):
        if not os.path.exists('yolov3.weights') or not os.path.exists('yolov3.cfg'):
            self.download_yolo_files()
        
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
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
        
        with open('coco.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(output_path, img)
    
    def download_yolo_files(self):
        yolo_files = {
            'yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights',
            'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
        }
        for file_name, url in yolo_files.items():
            if not os.path.exists(file_name):
                urllib.request.urlretrieve(url, file_name)
    
    def download_coco_names(self):
        url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
        filename = 'coco.names'
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)

if __name__ == "__main__":
    root = tk.Tk()
    app = OptiChromApp(root)
    root.mainloop()
