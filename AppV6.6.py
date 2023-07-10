import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import Canvas
from tkinter import NW
import cv2

from tensorflow import keras
import numpy as np
from tensorflow.keras.applications import vgg19
import tensorflow as tf
from tkinter import ttk


from IPython.display import  display

img_nrows = 256
img_ncols = 256

total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

style_layer_names = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    
# The layer to use for the content loss.
content_layer_name = "block5_conv2"

class MainScreen(tk.Frame):
    
    def update_quality(self, value):
              # Actualizar la calidad de la imagen aquí
              img_nrows=value
              img_ncols=value
              self.value_label.config(text=f"Quality: {value}")
              
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.image = None
        self.pack()
        self.create_widgets()
        self.camera = cv2.VideoCapture(0) # inicializar la cámara

    def create_widgets(self):
        # Definir la fuente personalizada
        my_font = ("Garamond", 12)
        
        # Establecer la fuente predeterminada para todos los widgets de la aplicación
        self.master.option_add("*Font", my_font)
        
        
        
        self.create_title_label()

        # variable de control para el modelo seleccionado
        self.model_var = tk.StringVar(self)
        self.model_var.set("MODEL 1")

        # lista de imágenes para cada modelo, reescaladas a la mitad
        self.model_images = {
            "MODEL 1": tk.PhotoImage(file="model1.png").subsample(5),
            "MODEL 2": tk.PhotoImage(file="model2.png").subsample(7),
            "MODEL 3": tk.PhotoImage(file="model3.png").subsample(5)
        }

        # widget Label para mostrar la imagen
        self.model_image_label = tk.Label(self, image=self.model_images[self.model_var.get()], bg="white", font="Garamond" )
        self.model_image_label.pack()

        
        # función para actualizar la imagen cada vez que cambie el modelo seleccionado
        def update_image(*args):
            self.model_image_label.config(image=self.model_images[self.model_var.get()], bg="white")

        # actualizar la imagen al inicio
        update_image()

        # asociar la variable de control con el widget OptionMenu
        self.model_option_menu = tk.OptionMenu(self, self.model_var, "MODEL 1", "MODEL 2", "MODEL 3", command=update_image)
        self.model_option_menu.pack()

        self.image_label = tk.Label(self)
        self.image_label.pack()

        # Crear un Frame para los botones "Take Photo" y "Upload Image"
        button_frame = tk.Frame(self)
        button_frame.config(bg="white")
        button_frame.pack()

        # Crear el botón "Take Photo"
        self.photo_button = tk.Button(button_frame, text="TAKE PICTURE", font=("Garamond"), command=self.take_photo)
        image = tk.PhotoImage(file="camera.png")  
        image = image.subsample(15)
        self.photo_button.config(image=image, width=160, height=40, compound=tk.LEFT)
        self.photo_button.image = image 
        self.photo_button.pack(side=tk.LEFT, padx=5)


        # Crear el botón "Upload Image"
        self.upload_button = tk.Button(button_frame, command=self.upload_image, bg="white", fg="black", font="Garamond")
        upload_image = tk.PhotoImage(file="uploadImage.png")  # Reemplaza "ruta_de_la_imagen.png" con la ruta de tu imagen
        upload_image = upload_image.subsample(15)
        self.upload_button.config(image=upload_image, text="UPLOAD IMAGE", width=160, height=40, compound=tk.LEFT)  # Ajusta los valores de width y height según tus necesidades
        self.upload_button.image = upload_image
        self.upload_button.pack(side=tk.LEFT, padx=5)

        # Crear un Frame para los botones "Crop" y "Confirm"
        crop_confirm_frame = tk.Frame(self)
        crop_confirm_frame.config(bg="white")
        crop_confirm_frame.pack()
        
        # Crear un botón para iniciar el recorte
        self.crop_button = tk.Button(crop_confirm_frame,command=self.crop, bg="white", fg="black", font="Garamond")
        crop_image = tk.PhotoImage(file="crop.png")  
        crop_image = crop_image.subsample(15)
        self.crop_button.config(image=crop_image, text="CROP", width=100, height=40, compound=tk.LEFT)  # Ajusta los valores de width y height según tus necesidades
        self.crop_button.image = crop_image
        self.crop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Crear un botón para confirmar
        self.confirm_button = tk.Button(crop_confirm_frame, text="CONFIRM", command=self.confirm, bg="white", fg="black", font="Garamond")
        confirm_image = tk.PhotoImage(file="confirm.png")
        confirm_image = confirm_image.subsample(25)
        self.confirm_button.config(image=confirm_image, text="CONFIRM", width=120, height=40, compound=tk.LEFT)  # Ajusta los valores de width y height según tus necesidades
        self.confirm_button.image = confirm_image
        self.confirm_button.pack(side=tk.LEFT, padx=5) 
         
    

        # Crear el slider de calidad
        self.quality_scale_value = tk.IntVar()
        self.quality_scale_value.set(128)  # valor inicial del slider
        self.quality_scale = tk.Scale(self, from_=32, to=512, length=200,
                              variable=self.quality_scale_value, command=self.update_quality, orient=tk.HORIZONTAL)
        self.quality_scale.set(128)  # valor inicial del slider
        self.quality_scale.pack()

        # Crear la etiqueta para mostrar el valor del slider
        self.value_label = tk.Label(self, text="Quality: 128")
        self.value_label.pack()

        # Lista de valores fijos permitidos para el control deslizante
        fixed_values = [32, 64, 128, 192, 256, 512]

        def update_quality(self, value):
            # Redondear el valor del control deslizante al valor fijo más cercano
            closest_value = min(fixed_values, key=lambda x: abs(x - float(value)))
            self.quality_scale.set(closest_value)
            self.value_label.config(text=f"Quality: {closest_value}")
        
        # Inicializar la variable de instancia para la imagen recortada
        self.cropped_image = None

    def create_title_label(self):
        self.title_label = tk.Label(self, text="AVATAR STYLER", font=("Perpetua", 24), bg="white", fg="black")
        self.title_label.pack()


    def crop(self):

        if self.image is None:
            return
        
        # Crear una nueva ventana para el recorte
        self.crop_window = tk.Toplevel(self.master)

        # Mostrar la imagen en un Canvas para permitir que el usuario seleccione la zona de recorte
        self.canvas = Canvas(self.crop_window, width=self.image.width, height=self.image.height)
        self.canvas.pack()
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=NW)

        # Permitir que el usuario seleccione la zona de recorte
        self.crop_area = self.canvas.create_rectangle(0, 0, 0, 0, outline="red")
        self.canvas.bind("<ButtonPress-1>", self.start_crop)
        self.canvas.bind("<B1-Motion>", self.crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self.end_crop)

        # Crear un botón para confirmar el recorte
        self.confirm_button = tk.Button(self.crop_window, text="Confirmar", command=self.confirm_crop)
        self.confirm_button.pack()

    def start_crop(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def crop_drag(self, event):
        self.canvas.coords(self.crop_area, self.start_x, self.start_y, event.x, event.y)

    def end_crop(self, event):
        self.end_x = event.x
        self.end_y = event.y

    def confirm_crop(self):
        # Obtener las coordenadas de la zona de recorte
        x1 = min(self.start_x, self.end_x)
        y1 = min(self.start_y, self.end_y)
        x2 = max(self.start_x, self.end_x)
        y2 = max(self.start_y, self.end_y)

        # Recortar la imagen y almacenarla en la variable de instancia
        self.cropped_image = self.image.crop((x1, y1, x2, y2))

        # Actualizar la imagen en el label original
        self.photo = ImageTk.PhotoImage(self.cropped_image)
        self.image_label.configure(image=self.photo) #AQUIIIIIIIIII

        # Cerrar la ventana de recorte
        self.crop_window.destroy()


    def take_photo(self):
        ret, frame = self.camera.read()  # Capturar un frame de la cámara
        if ret:
            # Convertir el frame a una imagen PIL y mostrarla en el widget Label
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.image = Image.fromarray(image)
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)


    def upload_image(self):
        # Abrir un cuadro de diálogo para seleccionar un archivo de imagen
        file_path = filedialog.askopenfilename()
        if file_path:
            # Cargar la imagen seleccionada y redimensionarla
            self.image = Image.open(file_path)
            resized_image = self.image.resize((256, 256))  # Redimensionar a un tamaño fijo (256x256)
            
            # Mostrar la imagen redimensionada en el widget Label
            self.photo = ImageTk.PhotoImage(resized_image)
            self.image_label.config(image=self.photo)

        
        # # Abrir un cuadro de diálogo para seleccionar un archivo de imagen
        # file_path = filedialog.askopenfilename()
        # if file_path:
        #     # Cargar la imagen seleccionada y mostrarla en el widget Label
        #     self.image = Image.open(file_path)
        #     self.photo = ImageTk.PhotoImage(self.image)
        #     self.image_label.config(image=self.photo)





    def confirm(self):
        if self.cropped_image:
            image_path = "cropped_image.png"  # Ruta de archivo temporal para la imagen recortada
            self.cropped_image.save(image_path)  # Guardar la imagen recortada en un archivo temporal
        else:
            image_path = "uploaded_image.png"  # Ruta de archivo temporal para la imagen cargada
            self.image.save(image_path)  # Guardar la imagen cargada en un archivo temporal
        
        loading_screen = LoadingScreen(self)
        
        self.destroy()




class LoadingScreen(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.loading_label = tk.Label(self, text="Loading...")
        self.loading_label.pack()

        self.progress_bar = ttk.Progressbar(self, orient='horizontal', length=200, mode='determinate')

        self.progress_bar.pack()

        self.start_process()


    def start_process(self):
        # Obtén la imagen seleccionada por el usuario
        input_image = Image.open("torres.jpeg") # self.master.cropped_image if self.master.cropped_image else self.master.image
        input_image_path= "torres.jpeg"
        
        image_path = "uploaded_image.png" # cropped_image.png" if self.master.cropped_image else "uploaded_image.png"    
        
        # Obtén la ruta del archivo de imagen
        input_image.save(image_path)  # Guardar la imagen en un archivo temporal
        
        # Preprocesa la imagen de entrada para adaptarla al formato requerido por el modelo
        # preprocessed_image = self.preprocess_image(image_path)
        # preprocessed_image_path="preprocessed_image.png"
        # preprocessed_image.save(preprocessed_image_path)
        
        
        # Aplica el estilo artístico a la imagen mediante la transferencia de estilo
        # stylized_image = model.predict(preprocessed_image)
        
        
        style_transfer = StyleTransfer()
        style_image_path = "model3.png"
        output_image = style_transfer.aplica_estilo(image_path, input_image_path, style_image_path, 500)
        # output_image.save("result.png")


        #deprocessed_image = Image.open('ruta_de_tu_imagen_deprocesada.jpg')
        # 
        output_image = np.squeeze(output_image, axis=0)  # Elimina la dimensión adicional del lote
        
        print(output_image.shape)
        
        deprocessed_image = deprocess_image(output_image)
        
        final_image = Image.fromarray(deprocessed_image)
        final_image.save("result.png")
        # output_path = "imagen_estilizada.png"
        
        display(final_image)
        
        
        # # Guarda la imagen estilizada en un archivo
        # stylized_image = np.squeeze(stylized_image, axis=0)  # Elimina la dimensión adicional del lote
        
        # print(stylized_image.shape)
        
        # stylized_image = self.deprocess_image(stylized_image)  # Desprocesa la imagen
        # output_image = Image.fromarray(stylized_image)  # Crea una instancia de PIL.Image
        # output_path = "ruta/a/donde/guardar/imagen_estilizada.png"
        # output_image.save(output_path)  # Guarda la imagen estilizada en un archivo
        
        # Muestra la pantalla de resultados
        self.result_screen = ResultScreen(self.master)
        self.destroy()

class StyleTransfer:
    def __init__(self, img_nrows=256, img_ncols=256):
        self.img_nrows = img_nrows
        self.img_ncols = img_ncols
        self.total_variation_weight = 1e-6
        self.style_weight = 1e-6
        self.content_weight = 2.5e-8
        self.style_layer_names = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.content_layer_name = "block5_conv2"
        self.model = None
        self.feature_extractor = None
        self.optimizer = None

     #debe entrenarse en base a la imagen sino es muy dificil
    def aplica_estilo(self, image_path, input_image_path, model_path,iterations):
         
        # Build a VGG19 model loaded with pre-trained ImageNet weights
        model = vgg19.VGG19(weights="imagenet", include_top=False)

        # Get the symbolic outputs of each "key" layer (we gave them unique names).
        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

        # Set up a model that returns the activation values for every layer in
        # VGG19 (as a dict).
        feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

        # List of layers to use for the style loss.
        style_layer_names = [
             "block1_conv1",
             "block2_conv1",
             "block3_conv1",
             "block4_conv1",
             "block5_conv1",
        ]
        # The layer to use for the content loss.
        content_layer_name = "block5_conv2"
         
        optimizer = keras.optimizers.SGD(
            keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96)
        )

        base_image = preprocess_image(image_path)
        style_reference_image = preprocess_image(model_path)
        combination_image = tf.Variable(preprocess_image(input_image_path))

        if iterations<100:
            iterations= 100
             
        for i in range(1, iterations + 1):
            loss, grads = compute_loss_and_grads(feature_extractor,
               combination_image, base_image, style_reference_image)
            optimizer.apply_gradients([(grads, combination_image)])
            
        return combination_image
    

def preprocess_image(image_path):
    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)



def deprocess_image(x):
    # Util function to convert a tensor into a valid image
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

#utileria necesaria para el entrenamiento
# The gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


# The "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels**2) * (size**2))


# An auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))

# The 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))



def compute_loss(feature_extractor,combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )
    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # Add total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss
    
@tf.function
def compute_loss_and_grads(feature_extractor,combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(feature_extractor,combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads



class ResultScreen(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.result_label = tk.Label(self, text="Result:")
        self.result_label.pack()

        self.result_image = tk.PhotoImage(file="result.png")
        self.result_canvas = tk.Canvas(self, width=self.result_image.width(), height=self.result_image.height())
        self.result_canvas.create_image(0, 0, anchor=tk.NW, image=self.result_image)
        self.result_canvas.pack()

        self.download_button = tk.Button(self, text="Download Image", command=self.download)
        self.download_button.pack()

        self.return_button = tk.Button(self, text="Return to Main Screen", command=self.return_to_main)
        self.return_button.pack()

    def download(self):
        # TODO: Implement image downloading functionality
        pass

    def return_to_main(self):
        main_screen = MainScreen(self.master)
        self.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("400x400")
    main_screen = MainScreen(root)
    root.mainloop()
