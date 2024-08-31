import logging
import os
import socket
import sys
import threading
import numpy as np
import torch
import math

from pathlib import Path
from threading import Event, Thread
from queue import Queue
from time import perf_counter
from rich.console import Console

from funasr import AutoModel

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer,
)

# from parler_tts import (
#     ParlerTTSForConditionalGeneration,
#     ParlerTTSStreamer
# )

from nltk.tokenize import sent_tokenize

from utils import (
    VADIterator,
    int2float,
    next_power_of_2
)

import ChatTTS
import torchaudio
import random

# chunk size 不能大于1024 否则vad处理会出错
chunk_size_cfg=1024

# listen all request
listen_ip = '0.0.0.0'
receive_port = "9527"
send_port = "2795"

console = Console()


class BaseHandler:
    """
    Base class for pipeline parts. Each part of the pipeline has an input and an output queue.
    The `setup` method along with `setup_args` and `setup_kwargs` can be used to address the specific requirements of the implemented pipeline part.
    To stop a handler properly, set the stop_event and, to avoid queue deadlocks, place b"END" in the input queue.
    Objects placed in the input queue will be processed by the `process` method, and the yielded results will be placed in the output queue.
    The cleanup method handles stopping the handler, and b"END" is placed in the output queue.
    """

    def __init__(self, stop_event, queue_in, queue_out, setup_args=(), setup_kwargs={}):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.setup(*setup_args, **setup_kwargs)
        self._times = []

    def setup(self):
        pass

    def process(self):
        raise NotImplementedError

    def run(self):
        while not self.stop_event.is_set():
            input = self.queue_in.get()
            if isinstance(input, bytes) and input == b'END':
                # sentinelle signal to avoid queue deadlock
                logger.debug("Stopping thread")
                break
            start_time = perf_counter()
            for output in self.process(input):
                self._times.append(perf_counter() - start_time)
                logger.debug(f"{self.__class__.__name__}: {self.last_time: .3f} s")
                self.queue_out.put(output)
                start_time = perf_counter()

        self.cleanup()
        self.queue_out.put(b'END')

    @property
    def last_time(self):
        return self._times[-1]

    def cleanup(self):
        pass


class ThreadManager:
    """
    Manages multiple threads used to execute given handler tasks.
    """

    def __init__(self, handlers):
        self.handlers = handlers
        self.threads = []

    def start(self):
        for handler in self.handlers:
            thread = threading.Thread(target=handler.run)
            self.threads.append(thread)
            thread.start()

    def stop(self):
        for handler in self.handlers:
            handler.stop_event.set()
        for thread in self.threads:
            thread.join()



class SocketReceiver:
    """
    Handles reception of the audio packets from the client.
    """

    def __init__(
        self, 
        stop_event,
        queue_out,
        should_listen,
        host=listen_ip, 
        port=receive_port,
        chunk_size=chunk_size_cfg,
    ):  
        self.stop_event = stop_event
        self.queue_out = queue_out
        self.should_listen = should_listen
        self.chunk_size = chunk_size
        self.host = host
        self.port = port

    def receive_full_chunk(self, conn, chunk_size):
        data = b''
        while len(data) < chunk_size:
            packet = conn.recv(chunk_size - len(data))
            if not packet:
                # connection closed
                return None  
            data += packet
        return data

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info('Receiver waiting to be connected...')
        self.conn, _ = self.socket.accept()
        logger.info("receiver connected")

        self.should_listen.set()
        while not self.stop_event.is_set():
            audio_chunk = self.receive_full_chunk(self.conn, self.chunk_size)
            if audio_chunk is None:
                # connection closed
                self.queue_out.put(b'END')
                break
            if self.should_listen.is_set():
                self.queue_out.put(audio_chunk)
        self.conn.close()
        logger.info("Receiver closed")

class SocketSender:
    """
    Handles sending generated audio packets to the clients.
    """

    def __init__(
        self, 
        stop_event,
        queue_in,
        host=listen_ip, 
        port=send_port,
    ):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.host = host
        self.port = port
        

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info('Sender waiting to be connected...')
        self.conn, _ = self.socket.accept()
        logger.info("sender connected")

        while not self.stop_event.is_set():
            audio_chunk = self.queue_in.get()
            self.conn.sendall(audio_chunk)
            if isinstance(audio_chunk, bytes) and audio_chunk == b'END':
                break
        self.conn.close()
        logger.info("Sender closed")


class VADHandler(BaseHandler):
    """
    Handles voice activity detection. When voice activity is detected, audio will be accumulated until the end of speech is detected and then passed
    to the following part.
    ref: https://huggingface.co/spaces/vishnuverse-in/silero-vad/blob/main/app.py#L10
    """

    def setup(
            self, 
            should_listen,
            thresh=0.3, 
            sample_rate=16000, 
            min_silence_ms=250,
            min_speech_ms=750, 
            max_speech_ms=float('inf'),
            speech_pad_ms=30,

        ):
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        # self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad',force_reload=True)

        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                            model='silero_vad',
                                            onnx=False)

        # print("VADIterator", self.model)
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )


    def process(self, audio_chunk):
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = int2float(audio_int16)
        vad_output = self.iterator(torch.from_numpy(audio_float32))
        if vad_output is not None and len(vad_output) != 0:
            logger.info("VAD: end of speech detected")
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.info(f"audio input of duration: {len(array) / self.sample_rate}s, skipping")
            else:
                self.should_listen.clear()
                logger.info("Stop listening")
                yield array


class SenseVoiceSTTHandler(BaseHandler):
    """
    Handles the Speech To Text generation using the SenseVoiceSmall model.
    """

    def setup(
            self,
            model_name = "FunAudioLLM/SenseVoiceSmall",
            device="cuda:1",
            hub="hf",  
            chunk_size = [0, 10, 5], #[0, 10, 5] 600ms, [0, 8, 4] 480ms
            encoder_chunk_look_back = 4, #number of chunks to lookback for encoder self-attention
            decoder_chunk_look_back = 1, #number of encoder chunks to lookback for decoder cross-attention
            disable_update = True,
        ): 
        
        self.model_name = model_name
        # self.model_revision = model_revision

        # 让系统选择模型
        if device is None:
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.hub = hub
        self.chunk_size = chunk_size
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back



        logger.info("load funasr model")
        # self.model = AutoModel(model=model_name,device=self.device, hub=hub, model_revision=model_revision, disable_update=disable_update)
        self.model = AutoModel(model=model_name,device=self.device, hub=hub, disable_update=disable_update,weights_only=True)
        logger.info("load funasr model complete")

        

    def process(self, audio_chunk):

        print("audio_chunk",audio_chunk)
        logger.info("infering SenseVoice...")

        global pipeline_start
        pipeline_start = perf_counter()


        cache = {}
        chunk_stride =  self.chunk_size[1] * 960  # 600ms
        total_chunk_num = int(len(audio_chunk) / chunk_stride)

        combined_text = ""
        for i in range(total_chunk_num):
            start_idx = i * chunk_stride
            end_idx = start_idx + chunk_stride
            speech_chunk = audio_chunk[start_idx:end_idx]
            is_final = i == total_chunk_num - 1
            
            res = self.model.generate(input=speech_chunk, 
                                    cache=cache, 
                                    is_final=is_final, 
                                    chunk_size= self.chunk_size, 
                                    encoder_chunk_look_back=self.encoder_chunk_look_back, 
                                    decoder_chunk_look_back=self.decoder_chunk_look_back,
                                    language="zn")
            
            pred_text = res[0]['text']
            logger.info(pred_text)
            # 获取 pred_text 的最后一个词
            # 使用关键词找到截取起点
            start_point = pred_text.rfind("<|woitn|>")
            if start_point != -1:
                start_point += len("<|woitn|>")  # 跳过关键词本身
                relevant_text = pred_text[start_point:]  # 截取关键词之后的文本
            else:
                relevant_text = pred_text  # 如果没找到关键词，使用整个文本
    
            # 截取的文本拼接到 combined_text
            combined_text += relevant_text
            
        logger.info("Finished inference for current chunk")
        console.print(f"[yellow]USER: {combined_text}")
        yield combined_text


# distil-whisper/distil-large-v3 是 Whisper 模型的蒸馏版本, 使用下来只输出英文
# 使用下来openai/whisper-large-v3的效果最好，能够准确识别中英文
# SenseVoice 支持多语言，但是没有使用 transfomers 进行加载和管理，需要用funsr，支持多语言，同时支持输出情绪参数
# 但是容易识别错误成其他语言
class WhisperSTTHandler(BaseHandler):
    """
    Handles the Speech To Text generation using a Whisper model.
    ref :https://huggingface.co/openai/whisper-large-v3
    """

    def setup(
            self,
            # model_name="distil-whisper/distil-large-v3",
            model_name="openai/whisper-large-v3",
            device="cuda:1",  
            torch_dtype="float16",  
            compile_mode=None,
            gen_kwargs={}
        ): 
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.compile_mode=compile_mode
        self.gen_kwargs = gen_kwargs

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
        ).to(device)
        
        # compile
        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(self.model.forward, mode=self.compile_mode, fullgraph=True)
        self.warmup()
    
    def prepare_model_inputs(self, spoken_prompt):
        input_features = self.processor(
            spoken_prompt, sampling_rate=16000, return_tensors="pt"
        ).input_features
        input_features = input_features.to(self.device, dtype=self.torch_dtype)

        return input_features
        
    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture 
        n_steps = 1 if self.compile_mode == "default" else 2
        dummy_input = torch.randn(
            (1,  self.model.config.num_mel_bins, 3000),
            dtype=self.torch_dtype,
            device=self.device
        ) 
        if self.compile_mode not in (None, "default"):
            # generating more tokens than previously will trigger CUDA graphs capture
            # one should warmup with a number of generated tokens above max tokens targeted for subsequent generation
            warmup_gen_kwargs = {
                "min_new_tokens": self.gen_kwargs["max_new_tokens"],
                "max_new_tokens": self.gen_kwargs["max_new_tokens"],
                **self.gen_kwargs
            }
        else:
            warmup_gen_kwargs = self.gen_kwargs

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(n_steps):
            _ = self.model.generate(dummy_input, **warmup_gen_kwargs)
        end_event.record()
        torch.cuda.synchronize()

        logger.info(f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")

    def process(self, spoken_prompt):
        logger.debug("infering whisper...")

        global pipeline_start
        pipeline_start = perf_counter()

        input_features = self.prepare_model_inputs(spoken_prompt)
        pred_ids = self.model.generate(input_features, **self.gen_kwargs)
        pred_text = self.processor.batch_decode(
            pred_ids, 
            skip_special_tokens=True,
            decode_with_timestamps=False
        )[0]

        logger.debug("finished whisper inference")
        console.print(f"[yellow]USER: {pred_text}")

        yield pred_text

class Chat:
    """
    Handles the chat using to avoid OOM issues.
    避免内存被耗尽
    """

    def __init__(self, size):
        self.size = size
        self.init_chat_message = None
        # maxlen is necessary pair, since a each new step we add an prompt and assitant answer
        self.buffer = []

    def append(self, item):
        # 向buffer中添加新消息
        self.buffer.append(item)
        # 检查 buffer 的长度
        # 如果超过了允许的大小（2 * (self.size + 1)），则删除最旧的两条消息（一个用户输入和一个助手回复）
        # 确保了聊天记录不会无限增长，避免占用过多内存
        if len(self.buffer) == 2 * (self.size + 1):
            self.buffer.pop(0)
            self.buffer.pop(0)   

    def init_chat(self, init_chat_message):
        # 初始化一个空的聊天消息
        self.init_chat_message = init_chat_message

    def to_list(self):
        # 模型记忆短期记忆（聊天上下文）
        # 返回列表，包含初始聊天消息和当前的聊天历史记录。
        if self.init_chat_message:
            return [self.init_chat_message] + self.buffer
        else:
            return self.buffer


class LargeLanguageModelHandler(BaseHandler):
    """
    Handles the language model part. 
    Here I use Qwen2-7B
    """

    def setup(
            self,
            model_name ="Qwen/Qwen2-7B-Instruct",
            device = None,  # let system select
            torch_dtype = "float16",
            # gen_kwargs={'return_full_text': False, 'temperature': 0.7, 'do_sample': False},
            gen_kwargs={'return_full_text': False,'do_sample': True},
            user_role="user",
            chat_size=1,
            init_chat_role=None, 
            init_chat_prompt="你是一个AI助手.",
        ):

        # 让系统选择模型
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

        # 加载与指定模型对应的分词器（Tokenizer）（从 Hugging Face 的模型库或本地路径）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 加载模型，指定模型的计算精度（数据类型），在指定设备上运行
        logger.info("load llm")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(device)
        logger.info("module load complete")
        
        # 文本生成管道
        self.pipe = pipeline( 
            "text-generation", 
            device=device,
            model=self.model, 
            tokenizer=self.tokenizer, 
            torch_dtype="auto", 
            max_new_tokens = 1024,  # 设置生成的新 maxtoken 数量
            min_new_tokens = 10  # 设置生成的新 mintoken 数量
        ) 

        # 流式输出，处理生成的文本
        # 用于逐步处理文本生成任务中的输出。在文本生成的过程中逐个处理生成的 token，不是等待整个文本生成完毕。
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True, #跳过初始的提示部分（用户输入的文本），只返回生成的内容。
            skip_special_tokens=True,  #跳过特殊 token（如 <|endoftext|> 等），使输出的文本更干净
        )
        self.gen_kwargs = {
            "streamer": self.streamer,
            "return_full_text": False,
            **gen_kwargs
        }

        # 初始化Chat 对象，并根据条件设置初始聊天信息和用户角色， 
        # chat_size 聊天记录的大小，（限制缓冲区中存储的对话轮数）
        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(f"An initial promt needs to be specified when setting init_chat_role.")
            self.chat.init_chat(
                {"role": init_chat_role, "content": init_chat_prompt}
            )
        self.user_role = user_role

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = "我是一个语音机器人"
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]
        warmup_gen_kwargs = {
            "min_new_tokens": 20,
            "max_new_tokens": 50,
            **self.gen_kwargs
        }

        # warmup_gen_kwargs {'min_new_tokens': 64, 'max_new_tokens': 64, 'streamer': <transformers.generation.streamers.TextIteratorStreamer object at 0x7f7464088040>, 'return_full_text': False, 'temperature': 0.0, 'do_sample': False}

        n_steps = 2

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(n_steps):
            thread = Thread(target=self.pipe, args=(dummy_chat,), kwargs=warmup_gen_kwargs)
            thread.start()
            for _ in self.streamer: 
                pass    
        end_event.record()
        torch.cuda.synchronize()

        logger.info(f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")

    def process(self, prompt):
        logger.info("infering language model...")

        self.chat.append(
            {"role": self.user_role, "content": prompt}
        )
        thread = Thread(target=self.pipe, args=(self.chat.to_list(),), kwargs=self.gen_kwargs)
        thread.start()

        generated_text, printable_text = "", ""
        for new_text in self.streamer:
            generated_text += new_text
            printable_text += new_text
            sentences = sent_tokenize(printable_text)
            if len(sentences) > 1:
                yield(sentences[0])
                printable_text = new_text

        self.chat.append(
            {"role": "assistant", "content": generated_text}
        )

        # don't forget last sentence
        # print(printable_text)
        yield printable_text


class ChatTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="cuda:1", 
        compile_mode=False,
        temperature = .3,
        top_P = 0.7,        # top P decode
        top_K = 20,         # top K decode
        description=(
            "TTS"
        ),
    ):
        self.should_listen = should_listen
        self.model =  ChatTTS.Chat(logger)
        logger.info("load ChatTTS")
        # use force_redownload=True if the weights have been updated.
        # self.model.load(source="huggingface")
        # chat.load(source='local') same source not set   
        self.model.load(compile=compile_mode, device=device) # Set to compile =  True for better performance
        
        ###################################
        # Sample a speaker from Gaussian.
        # https://huggingface.co/spaces/taa/ChatTTS_Speaker  音色控制
        rand_spk = self.model.sample_random_speaker()
        # print(rand_spk) # save it for later timbre recovery

        self.params_infer_code = self.model.InferCodeParams(
            spk_emb = rand_spk, # add sampled speaker 
            temperature = temperature,   # using custom temperature
            top_P = top_P,        # top P decode
            top_K = top_K,         # top K decode
        )

        # use oral_(0-9), laugh_(0-2), break_(0-7) 
        # to generate special token in text to synthesize.
        # self.params_refine_text = self.model.RefineTextParams(
        #     prompt='[oral_2][laugh_0][break_6]',
        # )


    # stream状态更新。数据量不足的stream，先存一段时间，直到拿到足够数据，监控小块数据情况
    @staticmethod
    def _update_stream(history_stream_wav, new_stream_wav, thre):
        if history_stream_wav is not None:
            result_stream = np.concatenate([history_stream_wav, new_stream_wav], axis=1)
            is_keep_next = result_stream.shape[0] * result_stream.shape[1] < thre
            if random.random() > 0.1:
                print(
                    "update_stream",
                    is_keep_next,
                    [i.shape if i is not None else None for i in result_stream],
                )
        else:
            result_stream = new_stream_wav
            is_keep_next = result_stream.shape[0] * result_stream.shape[1] < thre

        return result_stream, is_keep_next

    # 已推理batch数据保存
    @staticmethod
    def _accum(accum_wavs, stream_wav):
        if accum_wavs is None:
            accum_wavs = stream_wav
        else:
            accum_wavs = np.concatenate([accum_wavs, stream_wav], axis=1)
        return accum_wavs

    # batch stream数据格式转化
    @staticmethod
    def batch_stream_formatted(stream_wav, output_format="PCM16_byte"):
        if output_format in ("PCM16_byte", "PCM16"):
            format_data = ChatTTSHandler.float_to_int16(stream_wav)
        else:
            format_data = stream_wav
        return format_data

    # 数据格式转化
    @staticmethod
    def formatted(data, output_format="PCM16_byte"):
        if output_format == "PCM16_byte":
            format_data = data.astype("<i2").tobytes()
        else:
            format_data = data
        return format_data

    # 检查声音是否为空
    @staticmethod
    def checkvoice(data):
        if np.abs(data).max() < 1e-6:
            return False
        else:
            return True

    # 将声音进行适当拆分返回
    @staticmethod
    def _subgen(data, thre=12000):
        for stard_idx in range(0, data.shape[0], thre):
            end_idx = stard_idx + thre
            yield data[stard_idx:end_idx]

    def float_to_int16(audio: np.ndarray) -> np.ndarray:
        am = int(math.ceil(float(np.abs(audio).max())) * 32768)
        am = 32767 * 32768 // am
        return np.multiply(audio, am).astype(np.int16)

    # 流式数据获取，支持获取音频编码字节流
    def generate(self, streamchat, output_format=None):
        assert output_format in ("PCM16_byte", "PCM16", None)
        curr_sentence_index = 0
        history_stream_wav = None
        article_streamwavs = None
        for stream_wav in streamchat:
            print(np.abs(stream_wav).max(axis=1))
            n_texts = len(stream_wav)
            n_valid_texts = (np.abs(stream_wav).max(axis=1) > 1e-6).sum()
            if n_valid_texts == 0:
                continue
            else:
                block_thre = n_valid_texts * 8000
                stream_wav, is_keep_next = ChatTTSHandler._update_stream(
                    history_stream_wav, stream_wav, block_thre
                )
                # 数据量不足，先保存状态
                if is_keep_next:
                    history_stream_wav = stream_wav
                    continue
                # 数据量足够，执行写入操作
                else:
                    history_stream_wav = None
                    stream_wav = ChatTTSHandler.batch_stream_formatted(
                        stream_wav, output_format
                    )
                    article_streamwavs = ChatTTSHandler._accum(
                        article_streamwavs, stream_wav
                    )
                    # 写入当前句子
                    if ChatTTSHandler.checkvoice(stream_wav[curr_sentence_index]):
                        for sub_wav in ChatTTSHandler._subgen(
                            stream_wav[curr_sentence_index]
                        ):
                            if ChatTTSHandler.checkvoice(sub_wav):
                                yield ChatTTSHandler.formatted(sub_wav, output_format)
                    # 当前句子已写入完成，直接写下一个句子已经推理完成的部分
                    elif curr_sentence_index < n_texts - 1:
                        curr_sentence_index += 1
                        print("add next sentence")
                        finish_stream_wavs = article_streamwavs[curr_sentence_index]

                        for sub_wav in ChatTTSHandler._subgen(finish_stream_wavs):
                            if ChatTTSHandler.checkvoice(sub_wav):
                                yield ChatTTSHandler.formatted(sub_wav, output_format)

                    # streamchat遍历完毕，在外层把剩余结果写入
                    else:
                        break
        # 本轮剩余最后一点数据写入
        if is_keep_next:
            if len(list(filter(lambda x: x is not None, stream_wav))) > 0:
                stream_wav = ChatTTSHandler.batch_stream_formatted(
                    stream_wav, output_format
                )
                if ChatTTSHandler.checkvoice(stream_wav[curr_sentence_index]):

                    for sub_wav in ChatTTSHandler._subgen(
                        stream_wav[curr_sentence_index]
                    ):
                        if ChatTTSHandler.checkvoice(sub_wav):
                            yield ChatTTSHandler.formatted(sub_wav, output_format)
                    article_streamwavs = ChatTTSHandler._accum(
                        article_streamwavs, stream_wav
                    )
        # 把已经完成推理的下几轮剩余数据写入
        for i_text in range(curr_sentence_index + 1, n_texts):
            finish_stream_wavs = article_streamwavs[i_text]

            for sub_wav in ChatTTSHandler._subgen(finish_stream_wavs):
                if ChatTTSHandler.checkvoice(sub_wav):
                    yield ChatTTSHandler.formatted(sub_wav, output_format)


    def process(self,llm_sentence):

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        streamchat = self.model.infer(
            llm_sentence,
            skip_refine_text=True,
            stream=True,
            params_infer_code=self.params_infer_code,
        )

        streamer = ChatTTSHandler.generate(self, streamchat, output_format=None)

        for i, audio_chunk in enumerate(streamer):
            if i == 0:
                logger.info(f"Time to first audio: {perf_counter() - pipeline_start:.3f}")
            # 将音频数据从浮点数格式转换为 16 位整数格式
            # 这里不处理会出现很多背景噪声
            audio_chunk = np.int16(audio_chunk * 32767)
            yield audio_chunk

        # self.should_listen.set()
        # thread = Thread(target=ChatTTSHandler.generate, kwargs={"self":self,"streamchat":streamchat, "output_format":None})
        # thread.start()
        logger.info("wav_chunk tts processed")
        self.should_listen.set() # 设置监听状态


class ParlerTTSHandler(BaseHandler):
    def setup(
            self,
            should_listen,
            model_name="ylacombe/parler-tts-mini-jenny-30H",
            device="cuda:1", 
            torch_dtype="float16",
            compile_mode=None,
            gen_kwargs={},
            max_prompt_pad_length=8,
            description=(
                "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. "
                "She speaks very fast."
            ),
            play_steps_s=1
        ):
        self.should_listen = should_listen
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.gen_kwargs = gen_kwargs
        self.compile_mode = compile_mode
        self.max_prompt_pad_length = max_prompt_pad_length
        self.description = description

        self.description_tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype
        ).to(device)
        
        framerate = self.model.audio_encoder.config.frame_rate
        self.play_steps = int(framerate * play_steps_s)

        if self.compile_mode not in (None, "default"):
            logger.warning("Torch compilation modes that captures CUDA graphs are not yet compatible with the STT part. Reverting to 'default'")
            self.compile_mode = "default"

        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(self.model.forward, mode=self.compile_mode, fullgraph=True)

        self.warmup()

    def prepare_model_inputs(
        self,
        prompt,
        max_length_prompt=50,
        pad=False,
    ):
        pad_args_prompt = {"padding": "max_length", "max_length": max_length_prompt} if pad else {}

        tokenized_description = self.description_tokenizer(self.description, return_tensors="pt")
        input_ids = tokenized_description.input_ids.to(self.device)
        attention_mask = tokenized_description.attention_mask.to(self.device)

        tokenized_prompt = self.prompt_tokenizer(prompt, return_tensors="pt", **pad_args_prompt)
        prompt_input_ids = tokenized_prompt.input_ids.to(self.device)
        prompt_attention_mask = tokenized_prompt.attention_mask.to(self.device)

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            **self.gen_kwargs
        }

        return gen_kwargs
    
    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture 
        n_steps = 1 if self.compile_mode == "default" else 2

        torch.cuda.synchronize()
        start_event.record()
        if self.compile_mode:
            pad_lengths = [2**i for i in range(2, self.max_prompt_pad_length)]
            for pad_length in pad_lengths[::-1]:
                model_kwargs = self.prepare_model_inputs(
                    "dummy prompt", 
                    max_length_prompt=pad_length,
                    pad=True
                )
                for _ in range(n_steps):
                    _ = self.model.generate(**model_kwargs)
                logger.info(f"Warmed up length {pad_length} tokens!")
        else:
            model_kwargs = self.prepare_model_inputs("dummy prompt")
            for _ in range(n_steps):
                    _ = self.model.generate(**model_kwargs)
                
        end_event.record() 
        torch.cuda.synchronize()
        logger.info(f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")


    def process(self, llm_sentence):
        console.print(f"[green]ASSISTANT: {llm_sentence}")
        nb_tokens = len(self.prompt_tokenizer(llm_sentence).input_ids)

        pad_args = {}
        if self.compile_mode:
            # pad to closest upper power of two
            pad_length = next_power_of_2(nb_tokens)
            logger.debug(f"padding to {pad_length}")
            pad_args["pad"] = True
            pad_args["max_length_prompt"] = pad_length
    
        tts_gen_kwargs = self.prepare_model_inputs(
            llm_sentence,
            **pad_args,
        )

        streamer = ParlerTTSStreamer(self.model, device=self.device, play_steps=self.play_steps)
        tts_gen_kwargs = {
            "streamer": streamer,
            **tts_gen_kwargs
        }
        torch.manual_seed(0)
        thread = Thread(target=self.model.generate, kwargs=tts_gen_kwargs)
        thread.start()

        for i, audio_chunk in enumerate(streamer):
            if i == 0:
                logger.info(f"Time to first audio: {perf_counter() - pipeline_start:.3f}")
            audio_chunk = np.int16(audio_chunk * 32767)
            yield audio_chunk

        self.should_listen.set()

    

# Main
def main():
     
    
    #create log process
    global logger
    logging.basicConfig(
        filename='server.log',  # logfile
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger = logging.getLogger("Server_logger")
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 测试日志是否正常打印
    logger.info("安装funasr后需要使用自定义日志记录器")

    #Build the pipeline
    stop_event = Event()
    # used to stop putting received audio chunks in queue until all setences have been processed by the TTS
    should_listen = Event() 
    recv_audio_chunks_queue = Queue()
    send_audio_chunks_queue = Queue()
    spoken_prompt_queue = Queue() 
    text_prompt_queue = Queue()
    lm_response_queue = Queue()

    # create voice data receiver instance
    recv_handler = SocketReceiver(
        stop_event, 
        recv_audio_chunks_queue, 
        should_listen,
        host='0.0.0.0',
        port=9527,
        chunk_size=chunk_size_cfg,
    )

    # create voice data send instance
    send_handler = SocketSender(
        stop_event, 
        send_audio_chunks_queue,
        host='0.0.0.0',
        port=2795,
    )

    # create vad instance
    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
    )

    #create sensevoice stt instance
    sersevoice_stt = SenseVoiceSTTHandler(
        stop_event,
        queue_in=spoken_prompt_queue,
        queue_out=text_prompt_queue,
        # setup_args=(should_listen,),
    )

    #create whisper stt instance
    whisper_stt = WhisperSTTHandler(
        stop_event,
        queue_in=spoken_prompt_queue,
        queue_out=text_prompt_queue,
    )

    # create LLM instance
    llm = LargeLanguageModelHandler(
        stop_event,
        queue_in=text_prompt_queue,
        queue_out=lm_response_queue,
    )

    # create TTS instance
    chat_tts = ChatTTSHandler(
        stop_event,
        queue_in=lm_response_queue,
        queue_out=send_audio_chunks_queue,
        setup_args=(should_listen,),
    )

    # Run the pipeline
    try:
        # pipeline_manager = ThreadManager([vad, sersevoice_stt, llm, chat_tts, recv_handler, send_handler])
        pipeline_manager = ThreadManager([vad, whisper_stt, llm, chat_tts, recv_handler, send_handler])
        pipeline_manager.start()

    except KeyboardInterrupt:
        pipeline_manager.stop()


if __name__ == "__main__":

    main()
