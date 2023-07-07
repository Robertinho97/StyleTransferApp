# -*- coding: utf-8 -*-
"""
Created on Sun May 28 20:30:59 2023

@author: Rafael
"""
from PIL import Image, ImageTk
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications import vgg19
import tensorflow as tf
from IPython.display import  display


#img_nrows = 256
#img_ncols = 256

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

#def preprocess_image(image_path):
#        img = keras.preprocessing.image.load_img(
#            image_path, target_size=(img_nrows, img_ncols)
#        )
#        img = keras.preprocessing.image.img_to_array(img)
#        img = np.expand_dims(img, axis=0)
#        img = vgg19.preprocess_input(img)
#        return img
    

#def deprocess_image(x):        
#        x = x.reshape((img_nrows, img_ncols, 3)).astype("float64")
#        # Reajusta la imagen a su rango original
#        x[:, :, 0] += 103.939
#        x[:, :, 1] += 116.779
#        x[:, :, 2] += 123.68
#        # Convierte de BGR a RGB
#        x = x[:, :, ::-1]
#        # Asegúrate de que los valores estén en el rango de 0 a 255
#        x = np.clip(x, 0, 255).astype("uint8")
#        return x

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



    
#debe entrenarse en base a la imagen sino es muy dificil
def crea_entrenamiento(image_path,model_path,iterations):
    
    # Build a VGG19 model loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(weights="imagenet", include_top=False)

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in
    # VGG19 (as a dict).
    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

    # List of layers to use for the style loss.

    
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
    
    print(combination_image.shape)
    
    model.save("modelo.h5")
    #finalmente guardamos los pesos generados 
    
    
#        optimizer.apply_gradients([(grads, combination_image)])
#        if i % 100 == 0:
#            print("Iteration %d: loss=%.2f" % (i, loss))
#            img = deprocess_image(combination_image.numpy())
#            fname = result_prefix + "_at_iteration_%d.png" % i
#            keras.preprocessing.image.save_img(fname, img)
#            display(Image(result_prefix + "_at_iteration_%d.png" % i))

#debe entrenarse en base a la imagen sino es muy dificil
def aplica_estilo(image_path,model_path,iterations):
    
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

    
model_image_path="model3.png"
input_image_path="Bicho0.png"
    
model_image = Image.open("model3.png")
input_image = Image.open("Bicho0.png")    
display(input_image)
display(model_image)


width, height = keras.preprocessing.image.load_img(input_image_path).size
img_nrows = 256 #400
img_ncols = int(width * img_nrows / height)

width2, height2 = model_image.size
img_nrows = 256 #400
img_ncols = int(width * img_nrows / height)

print("img_nrows: %2d, img_ncols : %5.2f" % (width,height))
print("img_nrows: %2d, img_ncols : %5.2f" % (width2,height2))

modo =0

# Obtén la ruta del archivo de imagen
image_path = "uploaded_image.png"
input_image.save(image_path)  # Guardar la imagen en un archivo temporal  

if modo==0:
    #OJO! debes aplicar esto!
    stylized_image=aplica_estilo(image_path,model_image_path,1000)    
else:
    #crea_entrenamiento(input_image_path,model_image_path,300)
    # Carga tu modelo de style transfer
    model = keras.models.load_model("modelo.h5")
    #OJO! si miras el summary veras que lo que aplicamos no se aplica al modelo son solo la red VG1999, piensa que aplicamos el entrenamiento
    model.summary()
    
    # Preprocesa la imagen de entrada para adaptarla al formato requerido por el modelo
    preprocessed_image = preprocess_image(image_path)        
    # Aplica el estilo artístico a la imagen mediante la transferencia de estilo
    stylized_image = model.predict(preprocessed_image)
    
    
# Guarda la imagen estilizada en un archivo
stylized_image = np.squeeze(stylized_image, axis=0)  # Elimina la dimensión adicional del lote
        
print(stylized_image.shape)
        
stylized_image = deprocess_image(stylized_image)  # Desprocesa la imagen
output_image = Image.fromarray(stylized_image)  # Crea una instancia de PIL.Image
#output_path = "ruta/a/donde/guardar/imagen_estilizada.png"
#output_image.save(output_path)  # Guarda la imagen estilizada en un archivo

display(output_image)
