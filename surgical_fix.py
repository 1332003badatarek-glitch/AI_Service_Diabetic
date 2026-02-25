import h5py
import json

filename = 'fundus_efficientnet_ultra.h5'
new_filename = 'fixed_model.h5'

print("🏥 Starting Surgical Fix on H5 file...")

with h5py.File(filename, 'r+') as f:
    # قراءة إعدادات الموديل من الـ Metadata
    model_config_str = f.attrs.get('model_config')
    if model_config_str is None:
        print("❌ Could not find model_config in the file.")
    else:
        # تحويل النص لـ JSON عشان نعدله
        if isinstance(model_config_str, bytes):
            model_config_str = model_config_str.decode('utf-8')
        
        # 1. إصلاح مشكلة batch_shape و DTypePolicy
        fixed_config = model_config_str.replace('"batch_shape"', '"batch_input_shape"')
        fixed_config = fixed_config.replace('"dtype_policy": "DTypePolicy"', '"dtype": "float32"')
        fixed_config = fixed_config.replace('"dtype_policy": {"class_name": "DTypePolicy", "config": {"name": "float32"}}', '"dtype": "float32"')
        
        # حفظ التعديلات في ملف جديد
        with h5py.File(new_filename, 'w') as f_new:
            for key in f.keys():
                f.copy(key, f_new)
            f_new.attrs.update(f.attrs)
            f_new.attrs['model_config'] = fixed_config.encode('utf-8')
            
        print("✅✅ Done! Created 'fixed_model.h5'")
        print("Now try running 'main.py' using the NEW 'fixed_model.h5'")