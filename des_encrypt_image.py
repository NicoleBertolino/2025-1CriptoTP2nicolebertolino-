# -*- coding: utf-8 -*-
"""
Implementação do algoritmo DES para Criptografia de Imagens.

Os cenários de criptografia são:
1. DES Completo: Utilizando as 16 rodadas padrão.
2. DES com 0 Rodadas: Apenas as permutações inicial e final são aplicadas.
3. DES com 1 Rodada: Executando o algoritmo com apenas uma iteração.
4. DES com Chaves Nulas: Utilizando uma chave principal composta apenas por zeros.
"""

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Tabela de Permutação Inicial (IP)
IP = [58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6,
      64, 56, 48, 40, 32, 24, 16, 8,
      57, 49, 41, 33, 25, 17, 9, 1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7]

# Tabela de Permutação Final (FP), que é o inverso da IP
FP = [40, 8, 48, 16, 56, 24, 64, 32,
      39, 7, 47, 15, 55, 23, 63, 31,
      38, 6, 46, 14, 54, 22, 62, 30,
      37, 5, 45, 13, 53, 21, 61, 29,
      36, 4, 44, 12, 52, 20, 60, 28,
      35, 3, 43, 11, 51, 19, 59, 27,
      34, 2, 42, 10, 50, 18, 58, 26,
      33, 1, 41, 9, 49, 17, 57, 25]

# Tabela de Expansão (E) - para expandir 32 bits para 48 bits
E = [32, 1, 2, 3, 4, 5,
     4, 5, 6, 7, 8, 9,
     8, 9, 10, 11, 12, 13,
     12, 13, 14, 15, 16, 17,
     16, 17, 18, 19, 20, 21,
     20, 21, 22, 23, 24, 25,
     24, 25, 26, 27, 28, 29,
     28, 29, 30, 31, 32, 1]

# S-boxes (Caixas de Substituição)
S_BOX = [
    # S1
    [[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
     [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
     [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
     [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],
    # S2
    [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
     [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
     [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
     [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],
    # S3
    [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
     [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
     [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
     [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],
    # S4
    [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
     [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
     [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
     [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],
    # S5
    [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
     [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
     [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
     [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],
    # S6
    [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
     [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
     [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
     [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],
    # S7
    [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
     [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
     [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
     [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],
    # S8
    [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
     [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
     [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
     [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]
]

# Tabela de Permutação P (P-box)
P = [16, 7, 20, 21, 29, 12, 28, 17,
     1, 15, 23, 26, 5, 18, 31, 10,
     2, 8, 24, 14, 32, 27, 3, 9,
     19, 13, 30, 6, 22, 11, 4, 25]

# Permuted Choice 1 (PC-1) - para a chave de 64 bits
PC1 = [57, 49, 41, 33, 25, 17, 9,
       1, 58, 50, 42, 34, 26, 18,
       10, 2, 59, 51, 43, 35, 27,
       19, 11, 3, 60, 52, 44, 36,
       63, 55, 47, 39, 31, 23, 15,
       7, 62, 54, 46, 38, 30, 22,
       14, 6, 61, 53, 45, 37, 29,
       21, 13, 5, 28, 20, 12, 4]

# Permuted Choice 2 (PC-2) - para gerar sub-chaves de 48 bits
PC2 = [14, 17, 11, 24, 1, 5,
       3, 28, 15, 6, 21, 10,
       23, 19, 12, 4, 26, 8,
       16, 7, 27, 20, 13, 2,
       41, 52, 31, 37, 47, 55,
       30, 40, 51, 45, 33, 48,
       44, 49, 39, 56, 34, 53,
       46, 42, 50, 36, 29, 32]

# Tabela de rotação de bits para a geração de chaves

SHIFT_SCHEDULE = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

# --------------------------------------------------------------------------
# Funções Auxiliares
# --------------------------------------------------------------------------

def hex_to_bin(hex_string):
    """Converte uma string hexadecimal para uma string binária."""
    return bin(int(hex_string, 16))[2:].zfill(len(hex_string) * 4)

def bin_to_hex(bin_string):
    """Converte uma string binária para uma string hexadecimal."""
    return hex(int(bin_string, 2))[2:].zfill(len(bin_string) // 4)

def permute(block, table):
    """Aplica uma permutação a um bloco de bits."""
    return "".join(block[i - 1] for i in table)

def left_shift(bits, n):
    """Executa um shift circular para a esquerda."""
    return bits[n:] + bits[:n]

# --------------------------------------------------------------------------
# Parte 1: Geração de Sub-chaves
# --------------------------------------------------------------------------

def generate_round_keys(key_bin):
    """
    Gera as 16 sub-chaves de 48 bits a partir de uma chave principal de 64 bits.
    """
    if len(key_bin) != 64:
        raise ValueError("A chave principal deve ter 64 bits.")

    # 1. Aplicar PC-1 para obter uma chave de 56 bits
    key_56bit = permute(key_bin, PC1)

    # 2. Dividir em duas metades de 28 bits (C0 e D0)
    C = key_56bit[:28]
    D = key_56bit[28:]

    round_keys = []
    for i in range(16):
        # 3. Aplicar shifts circulares para a esquerda em C e D
        C = left_shift(C, SHIFT_SCHEDULE[i])
        D = left_shift(D, SHIFT_SCHEDULE[i])

        # 4. Combinar C e D e aplicar PC-2 para obter a sub-chave de 48 bits
        combined_cd = C + D
        round_key = permute(combined_cd, PC2)
        round_keys.append(round_key)

    return round_keys

# --------------------------------------------------------------------------
# Parte 2: Função de Feistel (Função F)
# --------------------------------------------------------------------------

def feistel_function(right_half_bin, round_key_bin):
    """
    Implementa a função F do DES (Função de Feistel).
    """
    # 1. Expandir a metade direita de 32 bits para 48 bits usando a tabela E
    expanded_right = permute(right_half_bin, E)

    # 2. XOR com a sub-chave da rodada
    xored = bin(int(expanded_right, 2) ^ int(round_key_bin, 2))[2:].zfill(48)

    # 3. Passar pelos S-Boxes
    s_box_output = ""
    for i in range(8):
        chunk = xored[i * 6:(i + 1) * 6]
        row = int(chunk[0] + chunk[5], 2)
        col = int(chunk[1:5], 2)
        s_box_output += bin(S_BOX[i][row][col])[2:].zfill(4)

    # 4. Permutar o resultado de 32 bits usando a P-box
    final_output = permute(s_box_output, P)

    return final_output

# --------------------------------------------------------------------------
# Parte 3: Processo de Criptografia DES
# --------------------------------------------------------------------------

def des_encrypt(block_bin, round_keys, num_rounds=16):
    """
    Criptografa um único bloco de 64 bits usando o DES.
    """
    if len(block_bin) != 64:
        raise ValueError("O bloco de dados para criptografia deve ter 64 bits.")

    # 1. Permutação Inicial (IP)
    permuted_block = permute(block_bin, IP)

    # 2. Divisão em metades Esquerda (L) e Direita (R)
    left_half = permuted_block[:32]
    right_half = permuted_block[32:]

    # 3. Loop de 16 Rodadas (ou o número especificado)
    for i in range(num_rounds):
        f_result = feistel_function(right_half, round_keys[i])
        new_left = right_half
        new_right = bin(int(left_half, 2) ^ int(f_result, 2))[2:].zfill(32)
        left_half, right_half = new_left, new_right

    # 4. Troca final das metades (swap)
    final_block_before_fp = right_half + left_half

    # 5. Permutação Final (FP)
    ciphertext_bin = permute(final_block_before_fp, FP)

    return ciphertext_bin

# --------------------------------------------------------------------------
# Funções de Manipulação de Imagem
# --------------------------------------------------------------------------

def pad_data(data):
    """Aplica padding PKCS7 aos dados para que sejam múltiplos de 8 bytes."""
    padding_len = 8 - (len(data) % 8)
    padding = bytes([padding_len] * padding_len)
    return data + padding

def encrypt_image_data(image_data, key_hex, num_rounds):
    """
    Criptografa os dados de uma imagem usando DES no modo ECB.
    """
    print(f"Iniciando criptografia com {num_rounds} rodada(s)...")
    
    # Prepara a chave e gera as sub-chaves
    key_bin = hex_to_bin(key_hex)
    if key_hex == '0' * 16: # Cenário especial com chaves nulas
        # Gera subchaves a partir de uma chave principal nula
        round_keys = generate_round_keys('0' * 64)
    else:
        round_keys = generate_round_keys(key_bin)

    # Adiciona padding aos dados da imagem
    padded_data = pad_data(image_data)
    
    encrypted_data = bytearray()

    # Processa a imagem em blocos de 8 bytes (64 bits)
    for i in range(0, len(padded_data), 8):
        block = padded_data[i:i+8]
        block_bin = ''.join(f'{byte:08b}' for byte in block)
        
        # Criptografa o bloco
        encrypted_block_bin = des_encrypt(block_bin, round_keys, num_rounds)
        
        # Converte o bloco binário criptografado de volta para bytes
        encrypted_block_bytes = int(encrypted_block_bin, 2).to_bytes(8, byteorder='big')
        encrypted_data.extend(encrypted_block_bytes)
        
    print("Criptografia concluída.")
    return bytes(encrypted_data)

# --------------------------------------------------------------------------
# Execução Principal
# --------------------------------------------------------------------------

def main():
    """Função principal para executar os cenários de criptografia."""
    image_filename = "Lenna.png"
    
    # Carregar imagem original a partir do arquivo local
    try:
        print(f"A carregar imagem '{image_filename}' do diretório local...")
        original_image = Image.open(image_filename).convert('RGB')
        original_data = original_image.tobytes()
        width, height = original_image.size
        print("Imagem carregada com sucesso.")
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{image_filename}' não foi encontrado.")
        print("Por favor, certifique-se de que a imagem está na mesma pasta que o script.")
        return
    except Exception as e:
        print(f"Não foi possível carregar a imagem '{image_filename}': {e}")
        return
        
    # Chave de 64 bits (16 caracteres hex) para os cenários A, B e C
    # Exemplo: '133457799BBCDFF1'
    key_hex_standard = '1A2B3C4D5E6F7890'

    # --- Cenário A: DES Completo (16 rodadas) ---
    print("\n--- Cenário A: DES Completo (16 rodadas) ---")
    data_a = encrypt_image_data(original_data, key_hex_standard, 16)
    img_a = Image.frombytes('RGB', (width, height), data_a[:len(original_data)])

    # --- Cenário B: DES sem nenhuma interação (0 rodadas) ---
    print("\n--- Cenário B: DES com 0 rodadas ---")
    data_b = encrypt_image_data(original_data, key_hex_standard, 0)
    img_b = Image.frombytes('RGB', (width, height), data_b[:len(original_data)])
    
    # --- Cenário C: DES com uma interação (1 rodada) ---
    print("\n--- Cenário C: DES com 1 rodada ---")
    data_c = encrypt_image_data(original_data, key_hex_standard, 1)
    img_c = Image.frombytes('RGB', (width, height), data_c[:len(original_data)])

    # --- Cenário D: DES com todas as chaves nulas ---
    print("\n--- Cenário D: DES com Chaves Nulas (16 rodadas) ---")
    data_d = encrypt_image_data(original_data, '0'*16, 16)
    img_d = Image.frombytes('RGB', (width, height), data_d[:len(original_data)])

    # --- Exibir resultados ---
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    
    axs[0].imshow(original_image)
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(img_a)
    axs[1].set_title('A: DES Completo (16 Rodadas)')
    axs[1].axis('off')

    axs[2].imshow(img_b)
    axs[2].set_title('B: DES (0 Rodadas)')
    axs[2].axis('off')
    
    axs[3].imshow(img_c)
    axs[3].set_title('C: DES (1 Rodada)')
    axs[3].axis('off')
    
    axs[4].imshow(img_d)
    axs[4].set_title('D: DES (Chaves Nulas)')
    axs[4].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
