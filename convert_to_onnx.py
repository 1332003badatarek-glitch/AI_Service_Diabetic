import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tf2onnx
import onnx

# دالة سحرية لتعديل الطبقات اللي بتبوظ التحميل
def fix_layer(layer_config):
    if 'batch_shape' in layer_config['config']:
        layer_config['config']['batch_input_shape'] = layer_config['config'].pop('batch_shape')
    return layer_config

print("🔄 Starting Surgical Conversion...")

try:
    # تحميل الموديل بدون الطبقة اللي بتعمل مشكلة
    model = tf.keras.models.load_model("fundus_efficientnet_ultra.h5", compile=False)
    
    # تحويل الموديل لـ ONNX
    spec = (tf.TensorSpec((None, 450, 450, 3), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    
    onnx.save_model(model_proto, "model.onnx")
    print("✅✅✅ DONE! 'model.onnx' created!")

except ValueError as e:
    if 'batch_shape' in str(e):
        print("⚠️ Detected batch_shape error, applying manual fix...")
        # محاولة التحميل يدوياً باستخدام الـ Custom Objects
        from tensorflow.keras.layers import InputLayer
        class FixedInputLayer(InputLayer):
            def __init__(self, **kwargs):
                if 'batch_shape' in kwargs: kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
                super().__init__(**kwargs)
        
        model = tf.keras.models.load_model("fundus_efficientnet_ultra.h5", compile=False, 
                                          custom_objects={'InputLayer': FixedInputLayer})
        spec = (tf.TensorSpec((None, 450, 450, 3), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        onnx.save_model(model_proto, "model.onnx")
        print("✅✅✅ DONE with Manual Fix!")
    else:
        print(f"❌ Error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")