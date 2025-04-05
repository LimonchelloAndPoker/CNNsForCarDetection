import os
import cv2
import requests
import numpy as np
import threading
from queue import Queue
from tqdm import tqdm

def download_image(url):
    """Downloads an image from a URL and returns it as a numpy array."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return None

def contains_face(image):
    """Detects faces in an image using OpenCV's Haar cascade."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

def save_image(image, path):
    """Saves an image to the specified path."""
    cv2.imwrite(path, image)

def worker(queue, save_dir, pbar):
    """Worker function to process images in a thread."""
    while True:
        attempts, image_count = queue.get()
        if image_count is None:
            queue.task_done()
            break
        
        url = f"https://picsum.photos/400/400?random={attempts}"
        image = download_image(url)
        
        if image is not None and not contains_face(image):
            save_path = os.path.join(save_dir, f"image_{image_count}.jpg")
            save_image(image, save_path)
            pbar.update(1)
        
        queue.task_done()

def scrape_images(target_count=7219, save_dir="nonhuman", num_threads=8):
    """Scrapes fully random images, ensuring no human faces are present, using multithreading."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    queue = Queue()
    threads = []
    
    with tqdm(total=target_count, desc="Downloading images") as pbar:
        for _ in range(num_threads):
            thread = threading.Thread(target=worker, args=(queue, save_dir, pbar))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        image_count = 0
        attempts = 0
        
        while image_count < target_count:
            attempts += 1
            queue.put((attempts, image_count + 1))
            image_count += 1
        
        queue.join()
        
        for _ in range(num_threads):
            queue.put((None, None))
        
        for thread in threads:
            thread.join()
    
    print(f"Successfully downloaded {target_count} images with no human faces.")

# Run the scraper
scrape_images()
