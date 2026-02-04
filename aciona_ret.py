import pyautogui
import keyboard
import time

time.sleep(5)

def main():
    print("Iniciando... Pressione ESC para parar o script.")
    try:
        while True:
            # Verifica se a tecla ESC foi pressionada
            if keyboard.is_pressed("esc"):
                print("Tecla ESC pressionada. Encerrando o script...")
                break

            # Primeiro clique no local do campo numérico
            pyautogui.click(x=1143, y=698)  # Substitua com as coordenadas desejadas
            time.sleep(1)  # Aguarde um curto intervalo

            # Insere o valor numérico
            pyautogui.write("5")  # Substitua com o número desejado
            time.sleep(1)

            # Clica em outro ponto na tela
            pyautogui.click(x=1017, y=751)  # Substitua com as coordenadas desejadas
            time.sleep(10)  # Aguarde o tempo necessário antes de prosseguir
            pyautogui.press('enter')

            # Clica novamente no campo numérico
            pyautogui.click(x=1143, y=698)
            time.sleep(1)  # Substitua com as mesmas coordenadas do primeiro clique
            pyautogui.write("8")
            time.sleep(1)
            pyautogui.click(x=1017, y=751) 
            time.sleep(10)
            pyautogui.press('enter')

    except KeyboardInterrupt:
        print("Script interrompido manualmente.")

if __name__ == "__main__":
    main()
