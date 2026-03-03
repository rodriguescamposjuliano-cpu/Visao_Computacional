import json
import os
import shutil
import urllib.parse
from sklearn.model_selection import train_test_split
import re

# CONFIGURAÇÕES
json_dir = 'jsons_originais'
img_dir = 'imagens_originais' 
output_base = 'datasets/vaca_pose'
KPT_ORDER = ["withers", "back", "hook up", "hook down", "hip", "tail head", "pin up", "pin down"]

def extrair_apenas_nome_arquivo(caminho_sujo):
    # 1. Decodifica URL e normaliza barras
    path_decodificado = urllib.parse.unquote(caminho_sujo)
    nome_arquivo = path_decodificado.replace('\\', '/').split('/')[-1]
    
    # 2. Se houver parâmetro de URL (ex: file=nome.jpg), pega o final
    if '=' in nome_arquivo:
        nome_arquivo = nome_arquivo.split('=')[-1]

    
    # Se o nome começar com uma sequência de caracteres seguida por '-', 
    # e o que vem depois for uma data ou o padrão RLC/IPC
    padrao_uuid = re.compile(r'^[a-fA-F0-9]{8}-')
    
    if "RLC" in nome_arquivo:
        # Mantém sua lógica original para RLC
        posicao_rlc = nome_arquivo.find("RLC")
        nome_arquivo = nome_arquivo[posicao_rlc:]
    elif padrao_uuid.match(nome_arquivo):
        # Se começar com hash (como o seu exemplo 00720a5b-...), remove o hash e o hífen
        nome_arquivo = nome_arquivo.split('-', 1)[-1]

    return nome_arquivo
        
    return nome_arquivo

def limpar_dataset_antigo():
    if os.path.exists(output_base):
        print(f"Limpando arquivos antigos em {output_base}...")
        shutil.rmtree(output_base)
    os.makedirs(output_base, exist_ok=True)

def processar_e_dividir():
    limpar_dataset_antigo()
    json_files = []
    arquivos_com_erro = []

    for root, dirs, files in os.walk(json_dir):
        for f in files:
            if not f.startswith('.'):
                json_files.append(os.path.join(root, f))
    
    if not json_files:
        print(f"Nenhum arquivo encontrado em {json_dir}!")
        return

    train_list, val_list = train_test_split(json_files, test_size=0.20, random_state=42)

    def converter_batch(lista_arquivos, subset):
        sucessos = 0
        for j_path in lista_arquivos:
            try:
                subpasta = os.path.basename(os.path.dirname(j_path))
                with open(j_path, 'r') as f:
                    data = json.load(f)
                
                tasks = data if isinstance(data, list) else [data]
                
                for task in tasks:
                    # Tenta pegar o nome da imagem logo no início para usar nos logs de erro
                    img_raw = task['task']['data']['img']
                    img_nome_limpo = extrair_apenas_nome_arquivo(img_raw)
                    identificador = f"{img_nome_limpo} (Pasta: {subpasta})"

                    if 'result' not in task or not task['result']:
                        arquivos_com_erro.append(f"VAZIO: {identificador} -> JSON: {j_path}")
                        continue

                    # VALIDAR RETÂNGULO (BBOX)
                    bboxes = [r for r in task['result'] if r['type'] == 'rectanglelabels']
                    if not bboxes:
                        arquivos_com_erro.append(f"SEM BBOX (Retângulo): {identificador} -> JSON: {j_path}")
                        continue
                    
                    # VALIDAR KEYPOINTS (Contagem)
                    kpts_presentes = [r for r in task['result'] if r['type'] == 'keypointlabels']
                    if len(kpts_presentes) < 8:
                        arquivos_com_erro.append(f"PONTOS INCOMPLETOS ({len(kpts_presentes)}/8): {identificador} -> JSON: {j_path}")
                        # Não damos 'continue' aqui para que ele processe o que for possível, 
                        # mas o aviso aparecerá no final.
                    
                    # Extração de coordenadas
                    v = bboxes[0]['value']
                    bx, by = (v['x'] + v['width']/2)/100, (v['y'] + v['height']/2)/100
                    bw, bh = v['width']/100, v['height']/100
                    
                    kpts_dict = {r['value']['keypointlabels'][0]: r['value'] for r in kpts_presentes}
                    kpts_str = ""
                    for name in KPT_ORDER:
                        if name in kpts_dict:
                            kx, ky = kpts_dict[name]['x']/100, kpts_dict[name]['y']/100
                            kpts_str += f" {kx} {ky} 2"
                        else:
                            kpts_str += " 0 0 0"

                    # Caminhos de saída
                    novo_nome_base = f"{subpasta}_{os.path.splitext(img_nome_limpo)[0]}"
                    txt_name = novo_nome_base + ".txt"
                    ext_img = os.path.splitext(img_nome_limpo)[1]
                    
                    img_out_dir = os.path.join(output_base, subset, 'images')
                    lbl_out_dir = os.path.join(output_base, subset, 'labels')
                    os.makedirs(img_out_dir, exist_ok=True)
                    os.makedirs(lbl_out_dir, exist_ok=True)
                    
                    src_img_path = os.path.join(img_dir, img_nome_limpo)
                    if os.path.exists(src_img_path):
                        shutil.copy(src_img_path, os.path.join(img_out_dir, f"{novo_nome_base}{ext_img}"))
                        with open(os.path.join(lbl_out_dir, txt_name), 'w') as f_out:
                            f_out.write(f"0 {bx} {by} {bw} {bh}{kpts_str}\n")
                        sucessos += 1
                    else:
                        arquivos_com_erro.append(f"IMAGEM FÍSICA NÃO ENCONTRADA: {img_nome_limpo} -> JSON: {j_path}")

            except Exception as e:
                arquivos_com_erro.append(f"ERRO CRÍTICO no arquivo {j_path}, {identificador}: {str(e)}")
        
        return sucessos

    total_train = converter_batch(train_list, 'train')
    total_val = converter_batch(val_list, 'val')

    print("\n" + "="*80)
    print("                      RELATÓRIO DE AUDITORIA DE DADOS")
    print("="*80)
    print(f"Sucesso: {total_train + total_val} imagens sincronizadas.")
    print(f"Treino: {total_train} | Validação: {total_val}")
    print("-"*80)

    if arquivos_com_erro:
        print(f"ATENÇÃO! Foram encontrados {len(arquivos_com_erro)} imagens com menos de 8 key points:")
        for erro in arquivos_com_erro:
            print(f"  [!] {erro}")
        print("\nSugestão: Use os nomes das imagens acima para localizar e corrigir no Label Studio.")
    else:
        print("Todas as imagens estão devidamente marcadas com 8 keypoints.")
    print("="*80)

if __name__ == "__main__":
    processar_e_dividir()