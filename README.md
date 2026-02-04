# OpenCV Object Detection

Aplicacao de deteccao de objetos com OpenCV DNN, com opcao de usar webcam, tela do PC, video ou imagem.
Suporta varios modelos via `model_zoo.json` e permite escolher o modelo e a fonte no modo interativo.

## Requisitos

- Python 3.9+
- Dependencias do `requirements.txt`

Instalar:

```bash
pip install -r requirements.txt
```

## Baixar modelos

Lista de modelos disponiveis para download:

```bash
python download_models.py --list
```

Baixar um modelo:

```bash
python download_models.py --model yolov3
```

Baixar tudo:

```bash
python download_models.py --all
```

## Executar

Modo interativo (escolha de modelo e fonte):

```bash
python app.py
```

Webcam:

```bash
python app.py --source webcam --model yolov3 --show-fps
```

Tela do PC:

```bash
python app.py --source screen --model yolov3 --monitor 1 --show-fps
```

Listar monitores:

```bash
python app.py --list-monitors
```

Video:

```bash
python app.py --source video --input caminho\video.mp4 --model yolov3
```

Imagem:

```bash
python app.py --source image --input caminho\imagem.jpg --model yolov3 --save saida.png
```

## Filtrar classes (opcional)

Use `--classes` com nomes do COCO ou indices:

```bash
python app.py --source webcam --model yolov3 --classes person,car
python app.py --source webcam --model yolov3 --classes 0,2,3
```

## Adicionar novos modelos

Edite o `model_zoo.json` e adicione um novo item:

```json
{
  "meu-modelo": {
    "name": "Nome Amigavel",
    "type": "darknet",
    "config": "arquivo.cfg",
    "weights": "arquivo.weights",
    "names": "coco.names",
    "input_size": [416, 416]
  }
}
```

Para modelos ONNX YOLO, use `type: "onnx-yolo"` e `weights: "modelo.onnx"`.
