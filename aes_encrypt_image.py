from Crypto.Cipher import AES
from PIL import Image
import os

# --- Configurações ---
KEY = b'ThisIsASecretKey'  # 16 bytes = 128 bits
BLOCK_SIZE = 16  # AES block size (em bytes)

# --- Função para aplicar padding ao conteúdo ---
def pad(data):
    padding_len = BLOCK_SIZE - len(data) % BLOCK_SIZE
    return data + bytes([padding_len]) * padding_len

# --- Carregar imagem e converter para bytes ---
img = Image.open("Lenna.png").convert("RGB")
img_data = img.tobytes()

# --- Adicionar padding ---
padded_data = pad(img_data)

# --- Criptografar com AES ECB ---
cipher = AES.new(KEY, AES.MODE_ECB)
encrypted_data = cipher.encrypt(padded_data)

# --- Criar nova imagem criptografada ---
# Note: cortamos para o tamanho original, pois o padding pode ter adicionado bytes extras
encrypted_img = Image.frombytes("RGB", img.size, encrypted_data[:len(img_data)])
encrypted_img.save("lenna_aes.png")
encrypted_img.show()

print("Imagem criptografada com AES salva como 'lenna_aes.png'")
