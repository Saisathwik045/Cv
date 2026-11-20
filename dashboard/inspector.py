"""Synthetic-only view of the exact original preprocessing; never loads a model."""
import io

SHAPES = {'Landscape · 320 × 240': (320, 240), 'Portrait · 240 × 320': (240, 320),
          'Square · 320 × 320': (320, 320), 'Wide · 640 × 80': (640, 80)}


def fixture(shape, eye):
    from PIL import Image, ImageDraw
    if shape not in SHAPES or eye not in ('Left', 'Right'):
        raise ValueError('Use a declared synthetic fixture')
    width, height = SHAPES[shape]
    image = Image.new('RGB', (width, height), (18, 22, 28))
    draw = ImageDraw.Draw(image)
    draw.ellipse((width*.12, height*.1, width*.88, height*.9),
                 fill=(100, 35, 20) if eye == 'Left' else (65, 35, 100))
    draw.line((width*.15, height*.8, width*.85, height*.2), fill=(220, 140, 50), width=4)
    draw.text((8, 8), 'SYNTHETIC NON-PATIENT', fill='white')
    stream = io.BytesIO();image.save(stream, format='PNG')
    return stream.getvalue()


def inspect(content):
    import numpy as np
    from dataclasses import asdict
    from trustretina.inference.bundle import verified_contract
    from trustretina.preprocessing.shared import PreprocessConfig, preprocess, letterbox_rgb
    from trustretina.inference.service import decode_image
    contract = verified_contract()  # Public committed contract only, never the bundle.
    config = PreprocessConfig()
    if config.fingerprint() != contract['preprocessing_sha256'] or config.size != 224:
        raise ValueError('Original frozen preprocessing identity required')
    source = decode_image(content)
    canvas, geometry = letterbox_rgb(source, config)
    tensor, metadata = preprocess(source, config, role='inference')
    if (tuple(tensor.shape) != (3, 224, 224) or str(tensor.device) != 'cpu'
            or metadata['horizontal_flip'] or metadata['preprocessing_sha256'] != config.fingerprint()):
        raise ValueError('Unexpected original preprocessing output')
    array = tensor.numpy()
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite preprocessing output')
    reconstructed = array.transpose(1, 2, 0) * np.array(config.std, np.float32) + np.array(config.mean, np.float32)
    if not np.allclose(reconstructed, np.asarray(canvas, np.float32)/255, rtol=0, atol=1e-6):
        raise ValueError('Canvas and normalized tensor disagree')
    for key, value in geometry.items():
        if metadata[key] != value:
            raise ValueError('Geometry and tensor metadata disagree')
    fraction = 1 - (geometry['resized_size'][0]*geometry['resized_size'][1])/(224*224)
    return {'source': source, 'canvas': canvas, 'geometry': geometry,
            'tensor_shape': list(tensor.shape), 'tensor_dtype': str(tensor.dtype),
            'padding_fraction': fraction, 'config': asdict(config),
            'preprocessing_sha256': config.fingerprint(),
            'padding_rgb': [round(v*255) for v in config.mean],
            'channel_statistics': [{'Channel': name, 'Minimum': float(x.min()),
                                    'Maximum': float(x.max()), 'Mean': float(x.mean())}
                                   for name, x in zip(('R','G','B'), array)]}


def boundary_overlay(canvas, geometry):
    from PIL import ImageDraw
    overlay = canvas.copy()
    left, top, right, bottom = geometry['padding']
    ImageDraw.Draw(overlay).rectangle((left, top, overlay.width-right-1, overlay.height-bottom-1),
                                     outline=(0, 255, 255), width=2)
    return overlay


def render_inspector_page():
    import pandas as pd
    import streamlit as st
    st.title('Preprocessing inspector')
    st.warning('SYNTHETIC NON-PATIENT INPUTS · REVIEW REQUIRED · This page runs no model and provides no medical prediction.')
    st.write('Compare a generated geometric source with the exact original 224 px letterbox canvas and normalized tensor statistics. This uses the existing frozen preprocessing implementation, not an approximation or a new model setting.')
    st.caption('No uploads, private records, bundle access, downloads of weights or inference. These generated shapes have no retinal ground truth. The paired predictor still requires both eyes; this page inspects one image transformation at a time.')
    if st.button('Reset inspector'):
        generation = st.session_state.get('inspector_generation', 0) + 1
        for key in list(st.session_state):
            if key.startswith('inspector_'):
                st.session_state.pop(key)
        st.session_state['inspector_generation'] = generation
        st.rerun()
    generation = st.session_state.get('inspector_generation', 0)
    shape = st.selectbox('Synthetic frame geometry', list(SHAPES), key=f'inspector_shape_{generation}')
    eye = st.selectbox('Synthetic fixture colour', ['Left', 'Right'], key=f'inspector_eye_{generation}')
    show_boundary = st.checkbox('Show educational content-boundary overlay', key=f'inspector_boundary_{generation}')
    try:
        data = inspect(fixture(shape, eye))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        st.error('The verified original preprocessing could not be inspected. No substitute canvas or assumed tensor is displayed.')
        return
    source_col, canvas_col = st.columns(2)
    source_col.image(data['source'], caption='Generated source after EXIF correction / RGB decode', width=320)
    canvas_col.image(data['canvas'], caption='Actual 224 × 224 RGB canvas before normalization; display enlarged only', width=320)
    if show_boundary:
        st.image(boundary_overlay(data['canvas'], data['geometry']), width=320,
                 caption='Educational copy only: cyan marks the resized content boundary. This annotation is never passed to the model.')
    g = data['geometry']
    st.subheader('Geometry from the frozen pipeline')
    st.dataframe(pd.DataFrame([{'Stage': 'Decoded source', 'Width': g['original_size'][0], 'Height': g['original_size'][1]},
                               {'Stage': 'Bilinear resized content', 'Width': g['resized_size'][0], 'Height': g['resized_size'][1]},
                               {'Stage': 'Letterbox canvas', 'Width': 224, 'Height': 224}]), hide_index=True, width='stretch')
    st.write(dict(zip(('Padding left', 'Padding top', 'Padding right', 'Padding bottom'), g['padding'])))
    st.metric('Canvas occupied by padding', f"{100*data['padding_fraction']:.2f}%")
    st.caption('Aspect ratio is preserved subject to integer rounding; there is no inference crop or random flip. Resizing may lose detail. Padding fraction describes geometry, not retinal coverage, image quality or clinical suitability.')
    st.subheader('What enters the model: normalized float32 tensor')
    st.write(f"Shape: {data['tensor_shape']} (channels, height, width) · dtype: {data['tensor_dtype']} · CPU preprocessing only")
    st.code('x = RGB_uint8 / 255\nnormalized[c] = (x[c] - mean[c]) / std[c]', language='python')
    st.write({'RGB mean': data['config']['mean'], 'RGB std': data['config']['std'], 'Padding RGB uint8': data['padding_rgb']})
    st.dataframe(pd.DataFrame(data['channel_statistics']), hide_index=True, width='stretch',
                 column_config={k: st.column_config.NumberColumn(format='%.6f') for k in ('Minimum','Maximum','Mean')})
    st.caption('Negative normalized values are expected. The RGB preview is before normalization; it is not a rendering of negative tensor values. Padding uses rounded mean×255 bytes, so its normalized values are close to, not necessarily exactly, zero. The inspector verifies that tensor de-normalization reproduces the canvas within float32 tolerance.')
    st.code(data['preprocessing_sha256'], language='text')
    st.caption('Preprocessing SHA256 matches the committed original inference-v8 contract. No model parameters, temperature, thresholds or scientific results are changed. No images or tensors are cached, exported or written to disk by this page; session reset is not a secure-erasure guarantee.')
