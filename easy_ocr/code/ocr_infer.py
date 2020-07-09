import json
import logging
import os
import easyocr
import numpy as np

JSON_CONTENT_TYPE = 'application/json'

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    logger.info('Creating the reader.')
    reader = easyocr.Reader(['ko','en'])

    return reader

def input_fn(serialized_input_data, content_type='application/x-image'):
    logger.info('Deserializing the input data.')
    if content_type == 'application/x-image':
        input_data = bytes(serialized_input_data)
        return input_data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


#https://stackoverflow.com/questions/50916422
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output, cls=NpEncoder), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def predict_fn(input_byte_data, reader):
    logger.info('Generating text based on input parameters.')
    output = reader.readtext(input_byte_data)
    return output

# For testing, make sure you run the infer.py and store the model outputs to S3
if __name__ == '__main__':
    out = [([[135, 21], [223, 21], [223, 61], [135, 61]], '김진중', 0.9717709422111511), ([[138, 68], [206, 68], [206, 94], [138, 94]], '6 hrs', 0.7103054523468018), ([[799, 151], [937, 151], [937, 189], [799, 189]], 'OBS@', 0.18651865422725677), ([[779, 435], [815, 435], [815, 473], [779, 473]], 'N', 0.9430522322654724), ([[256, 614], [758, 614], [758, 670], [256, 670]], '잘맞지 않는 사람들 밑에서', 0.10900411009788513), ([[360, 670], [656, 670], [656, 726], [360, 726]], '일하지는 마세요', 0.5090770125389099), ([[274, 800], [344, 800], [344, 824], [274, 824]], 'OB5HD', 0.504237174987793), ([[624, 800], [678, 800], [678, 824], [624, 824]], 'OB5}', 0.1399403065443039), ([[964, 800], [1012, 800], [1012, 824], [964, 824]], 'OB5', 0.572525680065155), ([[383, 1029], [623, 1029], [623, 1065], [383, 1065]], '출근할 때기분이 좋은 N', 0.07733354717493057), ([[712, 1030], [942, 1030], [942, 1062], [712, 1062]], '본인의 열정을 따르는 게{', 0.1905403733253479), ([[44, 1032], [220, 1032], [220, 1062], [44, 1062]], '미믐이 끌리는 일을', 0.07591995596885681), ([[246, 1044], [272, 1044], [272, 1076], [246, 1076]], 'N', 0.6398029923439026), ([[410, 1058], [560, 1058], [560, 1090], [410, 1090]], '회사여야 합니다', 0.2557936906814575), ([[79, 1059], [183, 1059], [183, 1089], [79, 1089]], '선택하세요', 0.8828091621398926), ([[746, 1060], [898, 1060], [898, 1090], [746, 1090]], '기장 중요합L다', 0.16542577743530273), ([[239, 1073], [269, 1073], [269, 1093], [239, 1093]], '', 0.049899641424417496)]
    print(output_fn(out))

    reader = model_fn(model_dir=None)
    print(predict_fn('example.jpg', reader))

    img = open('example.jpg', 'rb').read()
    input = bytearray(img)
    output = predict_fn(input_fn(input), reader)
    print(type(output), output)
    print(output_fn(output))

