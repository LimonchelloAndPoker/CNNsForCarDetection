import os
import cv2
import requests
import numpy as np
import threading
import hashlib
from tqdm import tqdm

# Globaler Zähler für einzigartige Bilder
unique_count = 0

def download_image(url):
    """Lädt ein Bild von einer URL herunter und gibt es als numpy-Array zurück."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return None

def save_image(image, path):
    """Speichert ein Bild unter dem angegebenen Pfad."""
    cv2.imwrite(path, image)

def compute_hash(image):
    """Berechnet den SHA-256-Hash eines Bildes."""
    return hashlib.sha256(image.tobytes()).hexdigest()

def worker(save_dir, pbar, lock, hashes, target_count):
    """Worker-Funktion zum Herunterladen und Speichern einzigartiger Bilder."""
    global unique_count
    while True:
        # Prüfen, ob das Ziel erreicht ist
        with lock:
            if unique_count >= target_count:
                break
        
        # Bild herunterladen
        url = "https://thispersondoesnotexist.com/"
        image = download_image(url)
        if image is None:
            continue
        
        # Hash berechnen
        image_hash = compute_hash(image)
        
        # Bild speichern, wenn es einzigartig ist und das Ziel noch nicht erreicht ist
        with lock:
            if image_hash not in hashes and unique_count < target_count:
                hashes.add(image_hash)
                unique_count += 1
                save_path = os.path.join(save_dir, f"image_{unique_count}.jpg")
                save_image(image, save_path)
                pbar.update(1)

def scrape_images(target_count=7219, save_dir="human", num_threads=8):
    """Scrapt einzigartige menschliche Gesichter mit Multithreading."""
    # Verzeichnis erstellen, falls es nicht existiert
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Gemeinsame Ressourcen initialisieren
    lock = threading.Lock()
    hashes = set()
    global unique_count
    unique_count = 0
    
    # Fortschrittsbalken starten
    with tqdm(total=target_count, desc="Downloading images") as pbar:
        threads = []
        # Threads starten
        for _ in range(num_threads):
            thread = threading.Thread(target=worker, args=(save_dir, pbar, lock, hashes, target_count))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Auf das Beenden aller Threads warten
        for thread in threads:
            thread.join()
    
    print(f"Erfolgreich {unique_count} einzigartige Bilder mit menschlichen Gesichtern heruntergeladen.")

# Skript ausführen
scrape_images()