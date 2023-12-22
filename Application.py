import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

import numpy as np
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import tkinter.messagebox as messagebox


utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile


class VehicleDetectionSystem:
    def __init__(self, window):
            
        self.window = window
        self.window.title("Vehicle Detection System")
        self.window.configure(bg='#333')

        self.canvas = tk.Canvas(window, width=800, height=600, bg="gray", highlightthickness=0)
        self.canvas.pack(pady=10)
        
        button_style = {'padx': 10, 'pady': 5, 'bg': '#4CAF50', 'fg': 'white', 'borderwidth': 0, 'font': ('Arial', 12)}
        button_style2 = {'padx': 10, 'pady': 5, 'bg': '#758f8c', 'fg': 'black', 'borderwidth': 0, 'font': ('Arial', 12)}
        button_style3 = {'padx': 10, 'pady': 5, 'bg': '#758f8c', 'fg': 'black', 'borderwidth': 0, 'font': ('Arial', 12)}
        button_style4 = {'padx': 10, 'pady': 5, 'bg': '#d90d0d', 'fg': 'white', 'borderwidth': 0, 'font': ('Arial', 12)}


        self.btn_open = tk.Button(window, text="Open A Video", command=self.open_video, **button_style)
        self.btn_open.pack(side="left", padx=10)

        self.btn_detect = tk.Button(window, text="Detect Vehicles", command=self.detect_vehicles, **button_style2)
        self.btn_detect.pack(side="left", padx=10)

        self.btn_reset = tk.Button(window, text="Reset", command=self.reset, **button_style3)
        self.btn_reset.pack(side="left", padx=10)

        self.btn_exit = tk.Button(window, text="Exit", command=self.confirm_exit, **button_style4)
        self.btn_exit.pack(side="right", padx=10)

        self.video_capture = None
        self.video_running = False

        self.detection_model = None
        self.category_index = None

        self.show_placeholder_label()

    def show_placeholder_label(self, anchor='center'):
        placeholder_label = tk.Label(self.window, text="Vehicle Detection", font=('Helvetica', 64, 'bold'), fg='white', bg='#333')
        placeholder_label.place(relx=0.5, rely=0.5, anchor=anchor)
        
    def open_video(self):
        self.video_path = filedialog.askopenfilename()
        if self.video_path:        
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.video_running = True
            self.hide_placeholder_label()
            self.show_video()

    def hide_placeholder_label(self):
        for widget in self.window.winfo_children():
            if isinstance(widget, tk.Label) and widget.cget('text') == "Vehicle Detection":
                widget.destroy()

    def show_video(self):
        if self.video_running:
            ret, frame = self.video_capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = frame.resize((800, 600))
                frame_tk = ImageTk.PhotoImage(frame)
                self.canvas.create_image(0, 0, anchor="nw", image=frame_tk)
                self.canvas.image = frame_tk
                self.window.after(1, self.show_video)
            else:
                self.video_capture.release()
                self.video_running = False

    def detect_vehicles(self):
        if hasattr(self, 'video_path') and self.video_path:
            if not self.detection_model or not self.category_index:
                print("Please load the model and labelmap first.")
                return

            while self.video_running:
                ret, image_np = self.video_capture.read()
                if not ret:
                    break

                output_dict = self.run_inference_for_single_image(image_np)
                self.visualize_on_image(image_np, output_dict)
                                                
                cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    self.video_capture.release()
                    cv2.destroyAllWindows()
                    self.video_running = False
                    break
        else:
            messagebox.showinfo("Open Video", "Please open a video first.", icon=messagebox.WARNING)

    def count_vehicles(self):

        return
    
    def load_model(self, model_path):
        self.detection_model = tf.saved_model.load(model_path)

    def load_labelmap(self, labelmap_path):
        self.category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)

    def run_inference_for_single_image(self, image):
        image = np.asarray(image)
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]

        output_dict = self.detection_model(input_tensor)

        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        if 'detection_masks' in output_dict:
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        return output_dict

    def visualize_on_image(self, image_np, output_dict):
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)

    def reset(self):        
        self.canvas.delete("all")
        self.video_capture = None
        self.video_running = False
        self.show_placeholder_label()

    def confirm_exit(self):
        result = messagebox.askokcancel("Exit", "Are you sure you want to exit?", icon=messagebox.WARNING)
        if result:
            self.window.destroy()

if __name__ == "__main__":
    window = tk.Tk()
    window.geometry("1000x700")
    window.resizable(False, False)
    app = VehicleDetectionSystem(window)

    #model_path = r'D:/KhoaLuanTotNghiep/VehicleDetection/saved_model'
    #labelmap_path = r'D:/KhoaLuanTotNghiep/VehicleDetection/workspace/training_demo/annotations/labelmap.pbtxt'

    # Change the path to the address where you saved the project
    app.load_model('D:/KhoaLuanTotNghiep/VehicleDetection/saved_model/')
    app.load_labelmap('D:/KhoaLuanTotNghiep/VehicleDetection/workspace/training_demo/annotations/labelmap.pbtxt')

    window.mainloop()
