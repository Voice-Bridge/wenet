# -*- coding:utf-8 -*-
#
#   author: iflytek
#
#  本demo测试时运行的环境为：Windows + Python3.7
#  本demo测试成功运行时所安装的第三方库及其版本如下，您可自行逐一或者复制到一个新的txt文件利用pip一次性安装：
#   cffi==1.12.3
#   gevent==1.4.0
#   greenlet==0.4.15
#   pycparser==2.19
#   six==1.12.0
#   websocket==0.2.1
#   websocket-client==0.56.0
#
#  语音听写流式 WebAPI 接口调用示例 接口文档（必看）：https://doc.xfyun.cn/rest_api/语音听写（流式版）.html
#  webapi 听写服务参考帖子（必看）：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=38947&extra=
#  语音听写流式WebAPI 服务，热词使用方式：登陆开放平台https://www.xfyun.cn/后，找到控制台--我的应用---语音听写（流式）---服务管理--个性化热词，
#  设置热词
#  注意：热词只能在识别的时候会增加热词的识别权重，需要注意的是增加相应词条的识别率，但并不是绝对的，具体效果以您测试为准。
#  语音听写流式WebAPI 服务，方言试用方法：登陆开放平台https://www.xfyun.cn/后，找到控制台--我的应用---语音听写（流式）---服务管理--识别语种列表
#  可添加语种或方言，添加后会显示该方言的参数值
#  错误码链接：https://www.xfyun.cn/document/error-code （code返回错误码时必看）
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
from pydub import AudioSegment
from threading import Thread, Lock
import re
import json
import numpy as np
STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识

global_sentence = ''


class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, AudioFile):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.AudioFile = AudioFile

        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {"domain": "iat", "language": "zh_cn", "accent": "mandarin", "vinfo":1, "vad_eos":10000,"nbest": 5}

    # 生成url
    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        # print("date: ",date)
        # print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        # print('websocket url :', url)
        return url


class SpeechToText:
    def __init__(self, app_id: str, api_key: str, api_secret: str):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.ws_url = None
        self.wsParam = None
        self.ws = None
        self.global_sentence = ''
        self.message = []
        self.lock = Lock()  # 用于线程安全的锁

    @classmethod
    def load_model(cls, api_name: str) -> 'SpeechToText':
        """Load the model with predefined API credentials."""
        # 读取api-key
        with open('C://Users//fanchenghao//Desktop//语雀//xunfei_apikey.json') as config_file:
            config = json.load(config_file)
        api_credentials = {
            'xunfei_api': {
                'app_id': config.get('api_id'),
                'api_key': config.get('api_key'),
                'api_secret': config.get('api_secret'),
            }
            # You can add more APIs here with their respective credentials
        }

        if api_name not in api_credentials:
            raise ValueError(f"Unsupported API name: {api_name}")

        return cls(**api_credentials[api_name])

    def transcribe(self, audio_file: str, confidence: float = 0.0) -> dict:
        wav_file = audio_file
        audio = AudioSegment.from_file(wav_file)
        # 转换为单声道,降频
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export("D://sample_audios//tmpfile.wav", format="wav")
        # global wsParam
        self.wsParam = Ws_Param(APPID='c5966e64', APISecret='NmQyNDdjNjVkNWQ2Y2ZlMGJmZDgwZWIx',
                           APIKey='9b24f7330fc0923186ae95e0da03ccbf',
                           AudioFile="D://sample_audios//tmpfile.wav")
        websocket.enableTrace(False)
        wsUrl = self.wsParam.create_url()
        ws = websocket.WebSocketApp(wsUrl, on_message=self.on_message, on_error=self.on_error, on_close=self.on_close)
        ws.on_open = self.on_open
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        # 使用正则表达式去除标点符号
        cleaned_sentence = re.sub(r'[, . ! ^，。！…]', '', global_sentence)
        print("cleaned_sentence",cleaned_sentence)
        result = {
            'confidence': '无返回值',
            'text': cleaned_sentence
        }
        # print(result)
        return result


    def transcribe_with_labels(self, audio_file: str, target: str) -> dict:
        # wav_file =
        wav_file = audio_file
        audio = AudioSegment.from_file(wav_file)
        # 转换为单声道,降频
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export("D://sample_audios//tmpfile.wav", format="wav")
        # global wsParam
        self.wsParam = Ws_Param(APPID='c5966e64', APISecret='NmQyNDdjNjVkNWQ2Y2ZlMGJmZDgwZWIx',
                           APIKey='9b24f7330fc0923186ae95e0da03ccbf',
                           AudioFile="D://sample_audios//tmpfile.wav")
        websocket.enableTrace(False)
        wsUrl = self.wsParam.create_url()
        ws = websocket.WebSocketApp(wsUrl, on_message=self.on_message, on_error=self.on_error, on_close=self.on_close)
        ws.on_open = self.on_open
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        # 使用正则表达式去除标点符号
        cleaned_sentence = re.sub(r'[,.!^ 。…]', '', global_sentence)
        # print("cleaned_sentence",cleaned_sentence)
        message = self.message[0]
        # print(message)
        # 提取 ws 中的内容
        if 'data' in message and 'result' in message['data'] and 'ws' in message['data']['result']:
            ws_list = message['data']['result']['ws'][0].get('cw', [])
        else:
            ws_list = []

        # 提取词汇
        words = [item['w'] for item in ws_list]
        # print("words",words)

        # 如果没有词汇，直接返回空结果
        if not words:
            print("没有词汇返回")
            result = {
                'max_word': '未识别为任何词汇',
                'max_score': 0,
                'target_word': target,
                'target_score': 0

            }
            return {}
        else:
            # 动态生成权重列表，越靠前的词汇权重越高
            scores = list(range(len(words) - 1, -1, -1))  # [len(words)-1, len(words)-2, ..., 0]

            # 定义 softmax 函数
            def softmax(x):
                """Compute softmax values for each set of scores in x."""
                e_x = np.exp(x - np.max(x))  # 防止数值溢出
                return e_x / e_x.sum()

            # 计算 softmax 置信度
            confidence_scores = softmax(scores)

            # 构造结果
            result = [{'word': word, 'confidence': confidence} for word, confidence in zip(words, confidence_scores)]

            if target in words:
                target_index = words.index(target)  # 找到目标词的索引
                target_score = result[target_index]['confidence']  #
            else:
                target_score = 0
            # 输出结果
            # for item in result:
            #     print(f"词: {item['word']}, 置信度: {item['confidence']:.4f}")

        result = {
            'max_word' : result[0]['word'],
            'max_score': result[0]['confidence'],
            'target_word': target,
            'target_score': target_score

        }
        # print(result)
        return result

    # 收到websocket消息的处理
    def on_message(self, ws, message):
        try:
            code = json.loads(message)["code"]
            sid = json.loads(message)["sid"]
            if code != 0:
                errMsg = json.loads(message)["message"]
                print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))

            else:
                data = json.loads(message)["data"]["result"]["ws"]
                self.message.append(json.loads(message))
                # print("message",json.loads(message))
                result = ""
                for i in data:
                    for w in i["cw"]:
                        result += w["w"]
                # print("sid:%s call success!,data is:%s" % (sid, json.dumps(data, ensure_ascii=False)))
                sentence = ''.join([item['cw'][0]['w'] for item in data])
                # print("sentence:",sentence)
                global global_sentence
                global_sentence += sentence
        except Exception as e:
            print("receive msg,but parse exception:", e)

    # 收到websocket错误的处理
    def on_error(self, ws, error):
        pass
        # print("### error:", error)

    # 收到websocket关闭的处理
    def on_close(self, ws, a, b):
        pass
        # print("### closed ###")

    # 收到websocket连接建立的处理
    def on_open(self, ws):
        def run(*args):
            frameSize = 8000  # 每一帧的音频大小
            intervel = 0.04  # 发送音频间隔(单位:s)
            status = STATUS_FIRST_FRAME  # 音频的状态信息，标识音频是第一帧，还是中间帧、最后一帧

            with open(self.wsParam.AudioFile, "rb") as fp:
                while True:
                    buf = fp.read(frameSize)
                    # 文件结束
                    if not buf:
                        status = STATUS_LAST_FRAME
                    # 第一帧处理
                    # 发送第一帧音频，带business 参数
                    # appid 必须带上，只需第一帧发送
                    if status == STATUS_FIRST_FRAME:

                        d = {"common": self.wsParam.CommonArgs,
                             "business": self.wsParam.BusinessArgs,
                             "data": {"status": 0, "format": "audio/L16;rate=16000",
                                      "audio": str(base64.b64encode(buf), 'utf-8'),
                                      "encoding": "raw"}}
                        d = json.dumps(d)
                        ws.send(d)
                        status = STATUS_CONTINUE_FRAME
                    # 中间帧处理
                    elif status == STATUS_CONTINUE_FRAME:
                        d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                                      "audio": str(base64.b64encode(buf), 'utf-8'),
                                      "encoding": "raw"}}
                        ws.send(json.dumps(d))
                    # 最后一帧处理
                    elif status == STATUS_LAST_FRAME:
                        d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                                      "audio": str(base64.b64encode(buf), 'utf-8'),
                                      "encoding": "raw"}}
                        ws.send(json.dumps(d))
                        time.sleep(1)
                        break
                    # 模拟音频采样间隔
                    time.sleep(intervel)
            ws.close()

        thread.start_new_thread(run, ())

if __name__ == "__main__":
    # Load the model with predefined API credentials
    model = SpeechToText.load_model('xunfei_api')
    # Transcribe an audio file
    result = model.transcribe_with_labels('D://sample_audios/47_112.225.101.110_20241119074110_跛脚.wav','跛脚')
    print(result)
