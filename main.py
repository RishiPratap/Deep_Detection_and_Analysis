import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
from tkinter import ttk
import time
import os
from rembg import remove
import datetime
import tensorflow as tf
import numpy as np
import cv2  # OpenCV
import skimage.exposure as exposure # For brightness correction
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ctypes  # For Windows-specific high DPI settings

# Load the saved model
loaded_model = tf.keras.models.load_model('dl-model/model/deep_fake_model_tf.h5')

global file_path

# Function to simulate processing commands
def process_image(status="Processing..."):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_textbox.insert(tk.END, f">> [{timestamp}] {status} \n")

def process_end(status="Processing Complete!"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time.sleep(2)  # Simulate processing time
    status_textbox.insert(tk.END, f">> [{timestamp}] {status} \n")

def upload_image():
    process_image("Image Uploading...")
    global file_path
    try:
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image.thumbnail((300, 300))  # Resize the image for display
            photo = ImageTk.PhotoImage(image)
            image_label.config(image=photo)
            image_label.photo = photo
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            process_end()
            status_textbox.insert(tk.END, f">> [{timestamp}] Image uploaded: {file_path}\n")
            # Simulate processing after image upload
        else:
            # Display "No image" text or a placeholder image
            no_image = Image.open("icon.png")  # Replace with the path to your placeholder image
            no_image.thumbnail((200, 200))
            no_image_photo = ImageTk.PhotoImage(no_image)
            image_label.config(image=no_image_photo)
            image_label.photo = no_image_photo
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_textbox.insert(tk.END, f">> [{timestamp}] No image uploaded.\n")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def remove_background():
    process_image("Background Removal started...")
    try:
        # Create an output directory
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)

        # Remove the background and save the result
        with open(file_path, "rb") as input_file:
            input_data = input_file.read()
            output_data = remove(input_data)
            output_file_path = os.path.join(output_dir, "background_removed.png")
            with open(output_file_path, "wb") as output_file:
                output_file.write(output_data)

        # Simulate processing after background removal

        # Update the status text box with the file path
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image = Image.open(output_file_path)
        image.thumbnail((300, 300))  # Resize the image for display
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.photo = photo
        status_textbox.insert(tk.END, f">> [{timestamp}] Background removed. Output file: {output_file_path}\n")
        return output_file_path

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def watermark_image():
    process_image("Watermarking started...")
    try:
        watermark_text = "Verified"
        img = Image.open(file_path)
        width, height = img.size
        font = ImageFont.load_default()
        text_width, text_height = font.getsize(watermark_text)
        margin = 10
        x = width - text_width - margin
        y = height - text_height - margin
        font = ImageFont.truetype("arial.ttf", 20)

        # Create an RGB version of the image without transparency
        rgb_img = img.convert("RGB")

        # Create a drawing context on the RGB image
        draw = ImageDraw.Draw(rgb_img)
        draw.text((x, y), watermark_text, fill=(255, 0, 0), font=font)

        # Save the watermarked image as JPEG
        watermarked_file_path = file_path.replace(".png", "_watermarked.jpg")
        rgb_img.save(watermarked_file_path, "JPEG")

        # Display the watermarked image
        image = Image.open(watermarked_file_path)
        image.thumbnail((300, 300))  # Resize the image for display
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.photo = photo
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_textbox.insert(tk.END, f">> [{timestamp}] Watermark added. Output file: {watermarked_file_path}\n")        

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    

def check_deepfake():
    try:
        # Load the single image
        image_path = file_path  # Replace with the path to your image
        image = Image.open(image_path)
        image_cv2 = cv2.imread(image_path)

        # Resize the image to match the model's input size (e.g., 128x128 pixels)
        image = image.resize((128, 128))

        # Convert the image to a NumPy array
        image_array = np.array(image) / 255.0  # Normalize pixel values to the [0, 1] range
        image_array = image_array.reshape(1, 128, 128, 3)  # Reshape to match the model's input shape

        # Make predictions on the single image
        prediction = loaded_model.predict(image_array)

        # Define thresholds for blurriness and lighting consistency
        threshold = 0.5  # Adjust as needed
        blurriness_threshold = 100  # Adjust as needed

        # Check the blurriness of the image
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Check the lighting consistency of the image
        hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s_std = np.std(s)
        v_std = np.std(v)
        print(f"Blurriness: {fm}")
        print(f"Lighting consistency (S): {s_std}")
        print(f"Lighting consistency (V): {v_std}")


        # Check the prediction result (assuming 0 represents fake and 1 represents real)
        if prediction[0][0] > threshold:
            print("Real Image")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_textbox.insert(tk.END, f">> [{timestamp}] Real Image\n")
        else:
            print("Fake Image")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_textbox.insert(tk.END, f">> [{timestamp}] Fake Image\n")

        # Check the blurriness of the image
        if fm < blurriness_threshold:
            print("Blurred Image")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_textbox.insert(tk.END, f">> [{timestamp}] Blurred Image\n")
        else:
            print("Clear Image")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_textbox.insert(tk.END, f">> [{timestamp}] Clear Image\n")

        # Check the lighting consistency of the image
        if s_std < 20 and v_std < 20:
            print("Lighting Inconsistent")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_textbox.insert(tk.END, f">> [{timestamp}] Lighting Inconsistent\n")
        else:
            print("Lighting Consistent")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_textbox.insert(tk.END, f">> [{timestamp}] Lighting Consistent\n")    
    
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        print(e)


def About():
    status_textbox.insert(tk.END, f">> About\n")

def Help():
    status_textbox.insert(tk.END, f">> Help\n")


# Function to analyze the image
def show_analysis_popup():
    try:
        # Load the single image
        image_path = file_path  # Replace with the path to your image
        image = Image.open(image_path)
        image_cv2 = cv2.imread(image_path)

        # Resize the image to match the model's input size (e.g., 128x128 pixels)
        image = image.resize((128, 128))

        # Convert the image to a NumPy array
        image_array = np.array(image) / 255.0  # Normalize pixel values to the [0, 1] range
        image_array = image_array.reshape(1, 128, 128, 3)  # Reshape to match the model's input shape

        # Make predictions on the single image
        prediction = loaded_model.predict(image_array)

        # Define thresholds for blurriness and lighting consistency
        threshold = 0.5  # Adjust as needed
        blurriness_threshold = 100  # Adjust as needed

        # Check the blurriness of the image
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Check the lighting consistency of the image
        hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s_std = np.std(s)
        v_std = np.std(v)

        # Define the labels for the pie chart
        labels = ['Real', 'Fake', 'Visibility', 'Clear', 'Inconsistent Lighting (S)', 'Consistent Lighting (S)', 'Inconsistent Lighting (V)', 'Consistent Lighting (V)']
        
        # Update the slices with the new values
        # Ensure non-negative values for blurriness and light consistency
        slices = [
            max(prediction[0][0] * 100, 0),  # Real
            max((1 - prediction[0][0]) * 100, 0),  # Fake
            max(fm, 0),  # Blurred
            max(100 - fm, 0),  # Clear
            max(s_std, 0),  # Inconsistent Lighting (S)
            max(100 - s_std, 0),  # Consistent Lighting (S)
            max(v_std, 0),  # Inconsistent Lighting (V)
            max(100 - v_std, 0)  # Consistent Lighting (V)
        ]
        
        # color for each label
        colors = ['r', 'y', 'g', 'b']
        
        # plotting the pie chart
        plt.pie(slices, labels=labels, colors=colors, 
                startangle=90, shadow=True, explode=(0, 0, 0.1, 0, 0, 0, 0.1, 0),
                radius=1.2, autopct='%1.1f%%')    
        # showing the plot
        plt.show()
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def reset_app():
    file_path = ""
    image_label.config(image="")
    status_textbox.delete(1.0, tk.END)
    no_image = Image.open("icon.png")  # Replace with the path to your placeholder image
    no_image.thumbnail((200, 200))
    no_image_photo = ImageTk.PhotoImage(no_image)
    image_label.config(image=no_image_photo)
    image_label.photo = no_image_photo
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_textbox.insert(tk.END, f">> [{timestamp}] App reset.\n")

# Create the main window
root = tk.Tk()
root.title("Deepfake Detection")
myappid = u'mycompany.myproduct.subproduct.version'
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
root.iconbitmap('favicon.ico')  # Replace with the path to your icon

# Set the theme style
style = ttk.Style()
style.theme_use("clam")  # You can change the theme here (e.g., 'clam', 'alt', 'default', 'vista', 'xpnative')

# Custom styling and colors
style.configure("TButton",
                foreground="white",
                background="#007acc",
                font=("Helvetica", 12),
                padding=10)
style.configure("TLabel",
                foreground="black",
                background="lightgray",
                font=("Helvetica", 18),
                padding=10)

# Get the screen width and height to center the window
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the window size and position it in the center of the screen
window_width = 800
window_height = 600
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Create a vertical scrollbar for the main window
scrollbar = ttk.Scrollbar(root, orient="vertical")
scrollbar.pack(side="right", fill="y")

# Create a main frame to hold all sections
main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True)

# Section 1: Navigation Menu
menu_bar = tk.Menu(main_frame)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Navigation", menu=file_menu)

file_menu.add_command(label="About", command=About)
file_menu.add_command(label="Help", command=Help)
file_menu.add_separator()  # Add a separator
file_menu.add_command(label="Upload", command=upload_image)  # Add Upload button
file_menu.add_command(label="Detect Deepfake", command=check_deepfake)  # Add Detect Deepfake button
file_menu.add_command(label="Analyze Image", command=show_analysis_popup)  # Add Analyze Image button
file_menu.add_command(label="Remove Background", command=remove_background)  # Add Remove Background button
file_menu.add_command(label="Add Watermark", command=watermark_image)  # Add Watermark button
file_menu.add_separator()  # Add a separator
file_menu.add_command(label="Reset", command=reset_app)  # Add Reset button
file_menu.add_command(label="Exit", command=root.quit)

# Section 2: Title
title_label = ttk.Label(main_frame, text="Deepfake Detection", font=("Helvetica", 24))
title_label.pack(pady=20)

# Section 3: Image Upload Modal
upload_modal = ttk.Frame(main_frame)
upload_modal.pack(expand=True, fill="both")

no_image = Image.open("icon.png")  # Replace with the path to your placeholder image
no_image.thumbnail((200, 200))
no_image_photo = ImageTk.PhotoImage(no_image)

image_label = ttk.Label(upload_modal, image=no_image_photo)
image_label.photo = no_image_photo
image_label.pack(pady=10)

# Section 4: Modal Text Box with Auto-scroll
status_textbox = tk.Text(main_frame, height=10, width=50, font=("Helvetica", 12))
status_textbox.pack(side="left", fill="both", expand=True, padx=5, pady=10)

# Add a scrollbar to the status_textbox
status_scrollbar = ttk.Scrollbar(main_frame, command=status_textbox.yview)
status_scrollbar.pack(side="right", fill="both")

status_textbox["yscrollcommand"] = status_scrollbar.set

# Start the application
root.mainloop()