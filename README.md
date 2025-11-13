## 🚀 Instalación

### Método 1: Clonar Repositorio (Recomendado)

```bash
# 1. Clonar el repositorio
git clone https://github.com/rcornejom06/COMPONENTE_PRACTICO_IA.git
cd clasificador-residuos

# 2. Crear entorno virtual
#usar python version 11
python -m venv .venv

# 3. Activar entorno virtual
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt
pip install --upgrade streamlit



# 5. Ruta de los archivos entrenados
# models/
#   ├── mobilenetv2_residuos.h5
#   ├── alexnet_residuos.h5
#   └── densenet121_residuos.h5

# 6. Ejecutar aplicación
streamlit run app_residuos.py
```
### Hiperparámetros Usados

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 64  # GPU con 16GB
EPOCHS = 15
LEARNING_RATE = 0.001

# Data Augmentation
rotation_range = 30
width_shift_range = 0.2
height_shift_range = 0.2
zoom_range = 0.2
horizontal_flip = True

# Callbacks
EarlyStopping(patience=5)
ReduceLROnPlateau(factor=0.5, patience=3)
Model